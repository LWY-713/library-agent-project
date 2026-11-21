# rag_module.py
"""Milvus 向量库模块：文档切分 → 向量化 → 入库 → 相似度检索。

首次运行前请在 .env 里填好 SILICONFLOW_API_KEY，然后执行：
    python rag_module.py
会把 data/library_knowledge.txt 灌入本地 Milvus Lite 数据库（milvus_library.db）。
"""
from __future__ import annotations

import argparse
from pathlib import Path

# 注意：config 要最先导入。它会读取 .env 并处理 MILVUS_URI，
# 而 pymilvus 自己也会读环境变量 MILVUS_URI 并要求是 URL，顺序反了会报 Illegal uri。
from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    EMBEDDING_API_KEY,
    EMBEDDING_BASE_URL,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    KNOWLEDGE_FILE,
    METRIC_TYPE,
    MILVUS_URI,
    TOP_K,
)

from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pymilvus import MilvusClient


class EmbeddingNotConfigured(RuntimeError):
    """没有配置向量化服务的密钥。"""


# ---------------------------------------------------------------- 客户端
_client: MilvusClient | None = None


def get_client() -> MilvusClient:
    """懒加载 Milvus 客户端，避免 import 阶段就连库。"""
    global _client
    if _client is None:
        _client = MilvusClient(MILVUS_URI)
    return _client


def get_embeddings() -> OpenAIEmbeddings:
    """懒加载 embedding 客户端；单独成函数，测试时可以整体替换。"""
    if not EMBEDDING_API_KEY:
        raise EmbeddingNotConfigured(
            "未配置向量化密钥：请在项目根目录 .env 中填写 SILICONFLOW_API_KEY。\n"
            "    也可以改用其它 OpenAI 兼容的向量服务，设置 EMBEDDING_API_KEY / "
            "EMBEDDING_BASE_URL / EMBEDDING_MODEL / EMBEDDING_DIM 即可。"
        )
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=EMBEDDING_API_KEY,
        base_url=EMBEDDING_BASE_URL,
        # bge-m3 / 百炼等非 OpenAI 模型必须传原始字符串，不能传 token id 数组
        check_embedding_ctx_length=False,
        # 控制单次请求的文本条数（百炼上限 10 条）
        chunk_size=EMBEDDING_BATCH_SIZE,
    )


# ---------------------------------------------------------------- 建库
def _ensure_collection(client: MilvusClient) -> None:
    if not client.has_collection(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            dimension=EMBEDDING_DIM,
            metric_type=METRIC_TYPE,
        )


def _row_count(client: MilvusClient) -> int:
    if not client.has_collection(COLLECTION_NAME):
        return 0
    try:
        stats = client.get_collection_stats(COLLECTION_NAME)
        return int(stats.get("row_count", 0))
    except Exception:
        return 0


def _ensure_loaded(client: MilvusClient) -> None:
    """Milvus Lite 进程重启后集合会变成 released 状态，检索前必须先 load。"""
    try:
        client.load_collection(COLLECTION_NAME)
    except Exception:
        # 已经加载时部分版本会抛异常，忽略即可，真正的错误会在 search 时暴露
        pass


def load_chunks():
    """读取知识库文件并按课程参数切分。"""
    path = Path(KNOWLEDGE_FILE)
    if not path.exists():
        raise FileNotFoundError(f"知识库文件不存在：{path}")

    text = path.read_text(encoding="utf-8")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        # 用正则以兼容分隔线长度不一致的情况（原代码写死了 30 个等号，和文件对不上）
        separators=[r"\n=+\n", "\n\n", "\n", "。", "！", "？", "；", " ", ""],
        is_separator_regex=True,
    )
    chunks = splitter.create_documents([text], metadatas=[{"source": str(path)}])
    return [chunk for chunk in chunks if chunk.page_content.strip()]


def build_index(rebuild: bool = True, verbose: bool = True) -> int:
    """切分 + 向量化 + 写入 Milvus，返回集合行数。"""
    # 先确认向量化密钥可用，避免删掉旧集合后才发现没配置
    embeddings = get_embeddings()

    client = get_client()
    if rebuild and client.has_collection(COLLECTION_NAME):
        client.drop_collection(COLLECTION_NAME)
    _ensure_collection(client)

    chunks = load_chunks()
    if not chunks:
        raise ValueError("知识库为空，没有可入库的文本")
    if verbose:
        print(f"📄 文档切分完成，共 {len(chunks)} 个 chunk，开始向量化 …")

    texts = [chunk.page_content for chunk in chunks]
    vectors: list[list[float]] = []
    for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch = texts[start:start + EMBEDDING_BATCH_SIZE]
        vectors.extend(embeddings.embed_documents(batch))
        if verbose:
            print(f"   向量化进度 {min(start + len(batch), len(texts))}/{len(texts)}")

    if len(vectors) != len(texts):
        raise RuntimeError(f"向量数量（{len(vectors)}）与文本数量（{len(texts)}）不一致")
    if vectors and len(vectors[0]) != EMBEDDING_DIM:
        raise RuntimeError(
            f"向量维度 {len(vectors[0])} 与配置的 EMBEDDING_DIM={EMBEDDING_DIM} 不一致，"
            "请检查 EMBEDDING_MODEL 与 EMBEDDING_DIM 是否匹配"
        )

    data = []
    for idx, (chunk, vector) in enumerate(zip(chunks, vectors)):
        data.append(
            {
                "id": idx,
                "vector": vector,
                "text": chunk.page_content,
                "source": chunk.metadata.get("source", str(KNOWLEDGE_FILE)),
                "chunk_id": idx,
            }
        )

    client.upsert(collection_name=COLLECTION_NAME, data=data)
    client.flush(COLLECTION_NAME)
    count = _row_count(client)
    if verbose:
        print(f"✅ 已写入 Milvus，当前集合行数：{count}")
    return count


def ensure_index(verbose: bool = True) -> bool:
    """集合为空时自动建库；返回是否执行了建库。"""
    client = get_client()
    if client.has_collection(COLLECTION_NAME) and _row_count(client) > 0:
        return False
    build_index(rebuild=True, verbose=verbose)
    return True


# ---------------------------------------------------------------- 检索
def search_knowledge(query: str, limit: int | None = None) -> list[dict]:
    """检索知识库，返回 [{text, chunk_id, source, score}, ...]。"""
    query = str(query or "").strip()
    if not query:
        return []

    client = get_client()
    if not client.has_collection(COLLECTION_NAME) or _row_count(client) == 0:
        return []

    _ensure_loaded(client)
    query_vector = get_embeddings().embed_query(query)
    raw = client.search(
        collection_name=COLLECTION_NAME,
        data=[query_vector],
        limit=limit or TOP_K,
        output_fields=["text", "chunk_id", "source"],
    )

    hits: list[dict] = []
    for item in raw[0] if raw else []:
        entity = item.get("entity") or {}
        hits.append(
            {
                "text": entity.get("text", ""),
                "chunk_id": entity.get("chunk_id"),
                "source": entity.get("source", ""),
                "score": float(item.get("distance", 0.0)),
            }
        )
    return hits


# ---------------------------------------------------------------- 命令行入口
def main() -> None:
    parser = argparse.ArgumentParser(description="构建 / 查询图书馆知识库")
    parser.add_argument("--check", action="store_true", help="只查看集合状态，不重建")
    parser.add_argument("--query", "-q", default="图书馆是什么时候建立的", help="建库后执行的测试问题")
    parser.add_argument("--top-k", type=int, default=TOP_K, help="测试检索返回条数")
    args = parser.parse_args()

    if args.check:
        client = get_client()
        exists = client.has_collection(COLLECTION_NAME)
        print(f"Milvus URI   : {MILVUS_URI}")
        print(f"集合          : {COLLECTION_NAME}（{'存在' if exists else '不存在'}）")
        print(f"行数          : {_row_count(client)}")
        return

    try:
        build_index(rebuild=True)
    except EmbeddingNotConfigured as exc:
        print(f"❌ {exc}")
        raise SystemExit(1) from None

    hits = search_knowledge(args.query, limit=args.top_k)
    print(f"\n======== 🔍 测试检索：{args.query} ========")
    if not hits:
        print("没有命中任何结果。")
        return
    for index, hit in enumerate(hits, start=1):
        print(f"[{index}] score={hit['score']:.4f}  chunk={hit['chunk_id']}")
        print(hit["text"])
        print("-" * 50)


if __name__ == "__main__":
    main()
