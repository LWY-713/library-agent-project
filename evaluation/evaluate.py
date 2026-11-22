# evaluate.py
"""图书馆智能 Agent 评测脚本。

四类指标：
1. Recall@K / MRR —— 检索质量（只对知识库内的题计算）
2. 回答准确率      —— LLM-as-judge：对照标准要点给回答打分
3. 工具选择准确率  —— 该调工具时是否调用、不该调时是否克制
4. 轨迹与成本      —— tool_calls / 工具轮数 / LLM 次数 / token 用量

用法（在项目根目录 library_agent/ 下执行，使用已装好依赖的 conda 环境）：
    python evaluation/evaluate.py --mode retrieval        # 只跑检索指标，快、省 token
    python evaluation/evaluate.py --mode agent --limit 5  # 只跑 5 题回答评测（试跑）
    python evaluation/evaluate.py --mode all              # 全量评测（默认）
    python evaluation/evaluate.py --mode all --report evaluation/ablation/topk1/report.json  # 消融用
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
PROJECT_DIR = EVAL_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))  # 让脚本能 import 项目根目录的模块

import config as cfg  # noqa: E402
from langchain_core.messages import AIMessage  # noqa: E402
from langgraph.types import Command  # noqa: E402
from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, LLM_MODEL_NAME, TOP_K  # noqa: E402
from rag_module import EmbeddingNotConfigured, search_knowledge  # noqa: E402

QUESTIONS_FILE = EVAL_DIR / "eval_questions.json"
REPORT_FILE = EVAL_DIR / "report.json"
EXPECTED_TOOL = "library_search_tool"


def load_questions() -> list[dict]:
    data = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))
    questions = data["questions"]
    seen = set()
    for question in questions:
        if question["id"] in seen:
            raise ValueError(f"评测集存在重复 id：{question['id']}")
        seen.add(question["id"])
        # 老题目默认属于图书馆领域，应调用工具；新题目显式写 false
        question.setdefault("expect_tool_call", True)
        # 未打难度标签的旧题按单跳事实处理
        question.setdefault("level", "L1")
    return questions


def _rate(flags: list[bool]) -> float:
    return statistics.mean(1 if flag else 0 for flag in flags) if flags else 0.0


def by_level(records: list[dict], kind: str) -> dict:
    """按 L0-L5 难度标签拆分指标，回答「哪一类题真正会做」。"""
    summary: dict = {}
    for level in sorted({r.get("level", "L1") for r in records}):
        subset = [r for r in records if r.get("level", "L1") == level]
        item: dict = {"count": len(subset)}
        if kind == "retrieval":
            item["recall"] = _rate([bool(r["hit_rank"]) for r in subset])
            item["mrr"] = (statistics.mean(r["mrr"] for r in subset)
                           if subset else 0.0)
        else:
            item["answer_accuracy"] = _rate([r["pass"] for r in subset])
            item["tool_selection_accuracy"] = _rate([r["tool_selection_correct"]
                                                     for r in subset])
        summary[level] = item
    return summary


# ================================================================ 检索评测
def eval_retrieval(questions: list[dict], top_k: int) -> list[dict]:
    """对每道知识题做检索，用关键词判断是否命中，统计 Recall@K 与 MRR。

    expect_tool_call=false 的题（闲聊 / 常识 / 实时信息等）不属于知识库检索范围，
    不参与检索指标，否则会凭空拉低 Recall。
    """
    records = []
    for index, question in enumerate(questions, start=1):
        text = question["question"]
        keywords = [kw.lower() for kw in question.get("keywords", [])]
        try:
            hits = search_knowledge(text, limit=top_k)
        except Exception as exc:
            print(f"[{index}/{len(questions)}] ❌ 检索出错：{exc}")
            records.append({"id": question["id"], "level": question["level"],
                            "hit_rank": None, "mrr": 0.0})
            continue

        # 相关片段判定：片段文本里出现任一关键词即视为相关（宽松判定，容忍切分把词切断）
        hit_rank = None
        for rank, hit in enumerate(hits, start=1):
            chunk_text = hit["text"].lower()
            if any(kw in chunk_text for kw in keywords):
                hit_rank = rank
                break

        mrr = (1.0 / hit_rank) if hit_rank else 0.0
        mark = "✅" if hit_rank else "❌"
        rank_info = f"首个相关片段排在第 {hit_rank} 位" if hit_rank else "top-K 内无相关片段"
        print(f"[{index}/{len(questions)}] {mark} {question['id']}  {rank_info}")
        records.append({"id": question["id"], "level": question["level"],
                        "hit_rank": hit_rank, "mrr": mrr})

    recall = _rate([bool(r["hit_rank"]) for r in records])
    mrr = statistics.mean(r["mrr"] for r in records) if records else 0.0
    print(f"\n📊 检索指标（top_k={top_k}，{len(records)} 题）")
    print(f"   Recall@{top_k} = {recall:.2%}   MRR = {mrr:.3f}")
    for level, item in by_level(records, "retrieval").items():
        print(f"   {level}（{item['count']} 题）  Recall = {item['recall']:.2%}"
              f"  MRR = {item['mrr']:.3f}")
    return records


# ================================================================ 回答评测
JUDGE_PROMPT = """你是 RAG 问答系统的评测裁判。请根据【标准要点】判断【AI 回答】是否合格。

判定规则：
1. 正常题目：回答需要覆盖全部标准要点，且没有编造标准要点之外的关键事实。
2. 拒答题目（expect_no_info = true）：AI 应明确表示知识库中未收录该信息；
   只要它编造了具体答案（编楼层、编号、规则等），直接判不合格。
3. 回答里采用更早日期的旧时间/旧规则（比如题目要求"当前/现在"，而 AI 用了更早通知的时间）判不合格；把读书节等限时活动的数字当成现行规则也判不合格。
4. 不该调工具的题（expect_tool_call = false）：只判断回答本身是否符合标准要点，
   不因「未检索知识库」扣分；若标准要求说明没有实时数据而 AI 编造了实时信息，判不合格。
   工具是否调用另有独立指标统计，本裁判不要据此打分。

只输出 JSON，不要输出其它内容：
{{"pass": true 或 false, "reason": "一句话理由"}}"""


def content_to_text(content) -> str:
    """把消息 content 统一转成字符串（str 或 content block 列表）。"""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and (block.get("type") == "text" or "text" in block):
            parts.append(str(block.get("text", "")))
    return "\n".join(part for part in parts if part)


def _token_stat(usage: dict | None) -> dict:
    if not usage:
        return {}
    details = usage.get("input_token_details") or {}
    return {
        "input_tokens": int(usage.get("input_tokens") or 0),
        "output_tokens": int(usage.get("output_tokens") or 0),
        "total_tokens": int(usage.get("total_tokens") or 0),
        "cache_read_tokens": int(details.get("cache_read") or 0),
    }


def build_trace(result: dict) -> tuple[str, dict]:
    """从图的最终状态里抽出最终回答 + 完整轨迹（工具调用 / 轮数 / token）。"""
    messages = result.get("messages", [])
    tool_calls = []
    steps = []
    tokens = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "cache_read_tokens": 0}
    llm_calls = 0

    for index, message in enumerate(messages):
        calls = []
        for call in getattr(message, "tool_calls", None) or []:
            item = {"step": index, "name": call.get("name"),
                    "args": call.get("args"), "id": call.get("id")}
            calls.append({"name": item["name"], "args": item["args"], "id": item["id"]})
            tool_calls.append(item)

        stat = _token_stat(getattr(message, "usage_metadata", None))
        if stat:  # 只有真实 LLM 响应带 usage_metadata，ToolMessage / HumanMessage 没有
            llm_calls += 1
            for key in tokens:
                tokens[key] += stat[key]

        steps.append({
            "step": index,
            "role": getattr(message, "type", "") or type(message).__name__,
            "content": content_to_text(getattr(message, "content", ""))[:1200],
            "tool_calls": calls,
            "tokens": stat or None,
        })

    answer = ""
    for message in reversed(messages):
        if isinstance(message, AIMessage) or getattr(message, "type", "") == "ai":
            answer = content_to_text(message.content)
            break
    if not answer and messages:  # 极端情况下兜底取最后一条
        answer = content_to_text(getattr(messages[-1], "content", ""))

    trace = {
        "tool_calls": tool_calls,
        "tool_names": [call["name"] for call in tool_calls if call["name"]],
        "tool_rounds": int(result.get("tool_rounds", 0) or 0),
        "llm_calls": llm_calls,
        "tokens": tokens,
        "steps": steps,
        "error": result.get("error", ""),
    }
    return answer, trace


def _extract_json(text: str) -> dict | None:
    """从裁判输出里抠出 JSON（模型偶尔会加前后缀）。"""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def eval_agent_answers(questions: list[dict], judge, agent,
                       limit: int | None) -> tuple[list[dict], dict]:
    """逐题调用 Agent，记录轨迹，再用裁判模型对照标准要点打分。"""
    if limit:
        questions = questions[:limit]

    records = []
    judge_tokens = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    for index, question in enumerate(questions, start=1):
        expect_tool_call = bool(question.get("expect_tool_call", True))
        thread_config = {"configurable": {"thread_id": f"eval-{question['id']}"}}
        try:
            result = agent.invoke(
                {"messages": [{"role": "user", "content": question["question"]}]},
                config=thread_config,
            )
            # 写操作题会触发 interrupt；评测里自动放行，避免卡住
            while "__interrupt__" in result:
                result = agent.invoke(Command(resume="yes"), config=thread_config)
            answer, trace = build_trace(result)
        except Exception as exc:
            print(f"[{index}/{len(questions)}] ❌ Agent 调用失败：{exc}")
            trace = {"tool_calls": [], "tool_names": [], "tool_rounds": 0,
                     "llm_calls": 0, "tokens": {}, "steps": [], "error": str(exc)}
            answer = ""

        tool_called = bool(trace["tool_calls"])
        tool_selection_correct = (tool_called == expect_tool_call)
        if expect_tool_call and tool_called:
            # 知识题不仅要看「有没有调」，还要看调的是不是期望的工具
            tool_selection_correct = EXPECTED_TOOL in trace["tool_names"]

        judge_input = (
            f"【问题】{question['question']}\n"
            f"【标准要点】{'；'.join(question['answer_points'])}\n"
            f"【expect_no_info】{question.get('expect_no_info', False)}\n"
            f"【expect_tool_call】{expect_tool_call}\n"
            f"【AI 回答】{answer}"
        )
        try:
            judge_resp = judge.invoke([
                {"role": "system", "content": JUDGE_PROMPT},
                {"role": "user", "content": judge_input},
            ])
            for key in judge_tokens:
                judge_tokens[key] += int((getattr(judge_resp, "usage_metadata", None) or {}).get(key) or 0)
            verdict = _extract_json(judge_resp.content) or {}
            passed = bool(verdict.get("pass", False))
            reason = str(verdict.get("reason", ""))[:100]
        except Exception as exc:
            passed, reason = False, f"裁判调用失败: {exc}"

        mark = "✅" if passed else "❌"
        tool_mark = "🎯" if tool_selection_correct else "🔧✗"
        print(f"[{index}/{len(questions)}] {mark} {tool_mark} {question['id']}  "
              f"工具={'调用' if tool_called else '未调用'}（期望{'调用' if expect_tool_call else '不调用'}）  {reason}")
        records.append({
            "id": question["id"],
            "level": question["level"],
            "category": question.get("category", ""),
            "expect_tool_call": expect_tool_call,
            "tool_called": tool_called,
            "tool_names": trace["tool_names"],
            "tool_selection_correct": tool_selection_correct,
            "pass": passed,
            "reason": reason,
            "answer": answer[:2000],
            "trace": trace,
        })

    metrics = summarize_agent(records, judge_tokens)
    print(f"\n📊 Agent 评测（{len(records)} 题）")
    print(f"   回答准确率       = {metrics['answer_accuracy']:.2%}")
    print(f"   工具选择准确率   = {metrics['tool_selection_accuracy']:.2%}")
    print(f"   该调时调用率     = {metrics['tool_call_rate_when_expected']:.2%}"
          f"（{metrics['expected_tool_questions']} 题）")
    print(f"   不该调时误调率   = {metrics['false_tool_call_rate']:.2%}"
          f"（{metrics['no_tool_questions']} 题）")
    print(f"   平均工具轮数     = {metrics['avg_tool_rounds']:.2f}")
    print(f"   多步触发率       = {metrics['multi_step_rate']:.2%}"
          f"（{metrics['max_tool_rounds_seen']} 轮上限）")
    print(f"   工具轮数分布     = {metrics['rounds_distribution']}")
    print(f"   Agent token 合计 = {metrics['agent_tokens']['total_tokens']}"
          f"（裁判额外 {judge_tokens['total_tokens']}）")
    print("   分层回答准确率：")
    for level, item in by_level(records, "agent").items():
        print(f"   {level}（{item['count']} 题）  准确率 = {item['answer_accuracy']:.2%}"
              f"  工具选择 = {item['tool_selection_accuracy']:.2%}")
    return records, metrics


def summarize_agent(records: list[dict], judge_tokens: dict) -> dict:
    expected = [r for r in records if r["expect_tool_call"]]
    no_tool = [r for r in records if not r["expect_tool_call"]]
    agent_tokens = {
        key: sum((r["trace"].get("tokens") or {}).get(key, 0) for r in records)
        for key in ("input_tokens", "output_tokens", "total_tokens", "cache_read_tokens")
    }
    rounds = [r.get("trace", {}).get("tool_rounds", 0) for r in records]
    summary_extra = {
        "multi_step_rate": sum(1 for x in rounds if x >= 2) / len(rounds) if rounds else 0.0,
        "max_tool_rounds_seen": max(rounds) if rounds else 0,
        "rounds_distribution": {str(k): rounds.count(k) for k in sorted(set(rounds))},
    }
    return {
        **summary_extra,
        "answer_accuracy": _rate([r["pass"] for r in records]),
        "tool_selection_accuracy": _rate([r["tool_selection_correct"] for r in records]),
        "tool_call_rate_when_expected": _rate([r["tool_called"] for r in expected]),
        "false_tool_call_rate": _rate([r["tool_called"] for r in no_tool]),
        "expected_tool_questions": len(expected),
        "no_tool_questions": len(no_tool),
        "avg_tool_rounds": (statistics.mean(r["trace"]["tool_rounds"] for r in records)
                            if records else 0.0),
        "avg_llm_calls": (statistics.mean(r["trace"]["llm_calls"] for r in records)
                          if records else 0.0),
        "agent_tokens": agent_tokens,
        "judge_tokens": judge_tokens,
    }


# ================================================================ 报告
def config_snapshot() -> dict:
    return {
        "llm_model": cfg.LLM_MODEL_NAME,
        "llm_temperature": cfg.LLM_TEMPERATURE,
        "top_k": cfg.TOP_K,
        "chunk_size": cfg.CHUNK_SIZE,
        "chunk_overlap": cfg.CHUNK_OVERLAP,
        "max_tool_rounds": cfg.MAX_TOOL_ROUNDS,
        "embedding_model": cfg.EMBEDDING_MODEL,
        "embedding_dim": cfg.EMBEDDING_DIM,
        "collection": cfg.COLLECTION_NAME,
        "milvus_uri": cfg.MILVUS_URI,
    }


def save_report(args, report_path: Path, retrieval_records: list[dict] | None,
                agent_records: list[dict] | None, agent_metrics: dict | None) -> Path:
    report: dict = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "args": vars(args),
        "config": config_snapshot(),
    }
    if retrieval_records is not None:
        report["retrieval"] = {
            "top_k": args.top_k,
            "recall": _rate([bool(r["hit_rank"]) for r in retrieval_records]),
            "mrr": (statistics.mean(r["mrr"] for r in retrieval_records)
                    if retrieval_records else 0.0),
            "count": len(retrieval_records),
            "by_level": by_level(retrieval_records, "retrieval"),
            "detail": retrieval_records,
        }
    if agent_records is not None:
        report["agent"] = {
            **agent_metrics,
            "count": len(agent_records),
            "by_level": by_level(agent_records, "agent"),
            "detail": agent_records,
        }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n💾 详细报告已保存：{report_path}")
    return report_path


def resolve_report_path(raw: str | None) -> Path:
    """默认写到 evaluation/report.json；带目录的路径按当前工作目录解析。"""
    if not raw:
        return REPORT_FILE
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path
    if path.parent != Path(".") and (Path.cwd() / path.parent).exists():
        return Path.cwd() / path
    return EVAL_DIR / path


def main() -> None:
    parser = argparse.ArgumentParser(description="图书馆智能 Agent 评测")
    parser.add_argument("--mode", choices=["retrieval", "agent", "all"], default="all")
    parser.add_argument("--top-k", type=int, default=TOP_K, help="检索评测返回片段数")
    parser.add_argument("--limit", type=int, default=None, help="只评测前 N 题（调试/消融用）")
    parser.add_argument("--report", default=None, help="报告输出路径（相对路径按 evaluation/ 解析）")
    args = parser.parse_args()

    questions = load_questions()
    knowledge_questions = [q for q in questions if q["expect_tool_call"]]
    print(f"📋 加载评测集：{len(questions)} 题"
          f"（知识题 {len(knowledge_questions)} / 不该调工具 {len(questions) - len(knowledge_questions)}）"
          f"（{QUESTIONS_FILE.name}）\n")

    retrieval_records = agent_records = agent_metrics = None
    try:
        if args.mode in {"retrieval", "all"}:
            retrieval_questions = knowledge_questions
            if args.limit:
                retrieval_questions = retrieval_questions[:args.limit]
            retrieval_records = eval_retrieval(retrieval_questions, args.top_k)
        if args.mode in {"agent", "all"}:
            if not DEEPSEEK_API_KEY:
                raise RuntimeError("未配置 DEEPSEEK_API_KEY，无法运行回答评测")
            from agent_builder import build_agent
            from langchain_deepseek import ChatDeepSeek

            judge = ChatDeepSeek(  # 裁判用低温度，尽量稳定
                model=LLM_MODEL_NAME,
                api_key=DEEPSEEK_API_KEY,
                api_base=DEEPSEEK_BASE_URL,
                temperature=0,
            )
            print("\n🤖 开始回答评测（每题会真实调用 Agent，耗时较长）…")
            agent_records, agent_metrics = eval_agent_answers(
                questions, judge, build_agent(), args.limit
            )
    except EmbeddingNotConfigured as exc:
        print(f"❌ {exc}")
        sys.exit(1)

    save_report(args, resolve_report_path(args.report),
                retrieval_records, agent_records, agent_metrics)


if __name__ == "__main__":
    main()
