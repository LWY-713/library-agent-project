# config.py
"""集中配置：参数统一从环境变量 / 项目根目录 .env 读取，代码里不再散落魔法值。"""
from __future__ import annotations  #向前注解，可以在类定义之前就提前写类型提示

import os
import sys #获取标准输入输出流 stdout stderr
from pathlib import Path #现代Python文件路径处理，比os.path 更优雅

from dotenv import load_dotenv

# Windows 控制台默认 GBK，输出 emoji 会抛 UnicodeEncodeError；这里退化为替换字符
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(errors="replace")
        except Exception:
            pass

BASE_DIR = Path(__file__).resolve().parent

# 只补充缺失的环境变量，不覆盖进程里已有的值（方便测试时临时改配置）
load_dotenv(BASE_DIR / ".env")


def _env(name: str, default: str = "") -> str:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return value.strip()


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)))
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    return _env(name, "true" if default else "false").lower() in {"1", "true", "yes", "y", "on"}


def _resolve_path(raw: str) -> str:
    if "://" in raw or raw.startswith(("tcp:", "unix:")):
        return raw
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = BASE_DIR / path
    return str(path)


def _env_path(name: str, default: Path) -> str:
    """路径型配置：相对路径按项目根目录解析。"""
    raw = _env(name)
    return _resolve_path(raw) if raw else str(default)


# ======================= LLM（对话模型） =======================
DEEPSEEK_API_KEY = _env("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = _env("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
LLM_MODEL_NAME = _env("LLM_MODEL_NAME", "deepseek-chat")
LLM_TEMPERATURE = _env_float("LLM_TEMPERATURE", 0.6)

# ======================= Embedding（向量化） =======================
# 默认硅基流动 SiliconFlow 的 BAAI/bge-m3（1024 维，OpenAI 兼容接口）
EMBEDDING_API_KEY = _env("EMBEDDING_API_KEY") or _env("SILICONFLOW_API_KEY")
EMBEDDING_BASE_URL = (
    _env("EMBEDDING_BASE_URL")
    or _env("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")
)
EMBEDDING_MODEL = _env("EMBEDDING_MODEL", "BAAI/bge-m3")
EMBEDDING_DIM = _env_int("EMBEDDING_DIM", 1024)
# 单次请求最多送几条文本：百炼 text-embedding-v4 上限是 10，硅基流动可以更大
EMBEDDING_BATCH_SIZE = _env_int("EMBEDDING_BATCH_SIZE", 10)

# ======================= Milvus / RAG =======================
# 默认本地文件库（Milvus Lite），不需要额外启动 Milvus 服务。
# 如果你有 Milvus 服务端，在 .env 里设置 MILVUS_URI=http://localhost:19530 即可。
_uri = _env("MILVUS_URI")
if _uri and "://" not in _uri:
    # 兼容旧写法：MILVUS_URI 写成文件路径时，pymilvus 自身读同名环境变量会报 Illegal uri
    os.environ.pop("MILVUS_URI", None)
    MILVUS_URI = _resolve_path(_uri)
elif _uri:
    MILVUS_URI = _uri
else:
    MILVUS_URI = _env_path("MILVUS_DB_PATH", BASE_DIR / "milvus_library.db")

COLLECTION_NAME = _env("MILVUS_COLLECTION_NAME", "library_rag_collection")
METRIC_TYPE = _env("METRIC_TYPE", "COSINE")
TOP_K = _env_int("TOP_K", 3)
# 检索相关度下限：0 表示不启用（实测正确/错误检索分数重叠，阈值法不可靠）
SEARCH_MIN_SCORE = _env_float("SEARCH_MIN_SCORE", 0.0)
CHUNK_SIZE = _env_int("CHUNK_SIZE", 200)
CHUNK_OVERLAP = _env_int("CHUNK_OVERLAP", 80)
KNOWLEDGE_FILE = _env_path("KNOWLEDGE_FILE", BASE_DIR / "data" / "library_knowledge.txt")

# ======================= 借阅业务 =======================
CURRENT_USER_ID = _env("CURRENT_USER_ID", "2023001")
LOANS_FILE = _env_path("LOANS_FILE", BASE_DIR / "data" / "loans.json")

# ======================= 其它 =======================
MAX_CHAR_THRESHOLD = _env_int("MAX_CHAR_THRESHOLD", 300)
MAX_INPUT_LENGTH = _env_int("MAX_INPUT_LENGTH", 1000)
DEBUG = _env_bool("DEBUG", True)

# ======================= LangGraph 显式图 =======================
# agent ↔ tools 最大循环轮数（一轮 = 模型发起一次工具调用并拿到结果）。
# 超过后进入 fallback 节点兜底，防止死循环。
MAX_TOOL_ROUNDS = _env_int("MAX_TOOL_ROUNDS", 12)
# 是否开启流式输出（逐 token 打印最终回答）。默认关闭，保持原行为不变。
STREAM_OUTPUT = _env_bool("STREAM_OUTPUT", False)
