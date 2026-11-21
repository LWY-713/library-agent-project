# ablation.py
"""参数消融：用子进程逐组覆盖 config.py 里的环境变量，真实跑评测并汇总对比。

为什么用子进程：config.py 在 import 时就把 TOP_K / LLM_TEMPERATURE / CHUNK_SIZE
等读成模块常量了，同一进程里改环境不会生效；子进程可以让每组参数完全隔离。

用法（项目根目录，conda 环境）：
    python evaluation/ablation.py                      # 默认 6 组参数，全量评测
    python evaluation/ablation.py --limit 12           # 快速消融：每组只跑前 12 题
    python evaluation/ablation.py --variants-file my_variants.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
PROJECT_DIR = EVAL_DIR.parent
PYTHON = sys.executable
EVAL_SCRIPT = EVAL_DIR / "evaluate.py"
BUILD_SCRIPT = PROJECT_DIR / "rag_module.py"

# 每组变体只写 env，不直接改源码；rebuild=True 表示要用新的切分参数重建独立向量库。
DEFAULT_VARIANTS = [
    {"name": "baseline", "env": {}},
    {"name": "topk_1", "env": {"TOP_K": "1"}},
    {"name": "topk_5", "env": {"TOP_K": "5"}},
    {"name": "temp_0", "env": {"LLM_TEMPERATURE": "0"}},
    {
        "name": "chunk_400",
        "env": {"CHUNK_SIZE": "400", "CHUNK_OVERLAP": "80",
                "MILVUS_DB_PATH": "./eval_ablation/chunk_400.db"},
        "rebuild": True,
    },
    {
        "name": "chunk_100",
        "env": {"CHUNK_SIZE": "100", "CHUNK_OVERLAP": "40",
                "MILVUS_DB_PATH": "./eval_ablation/chunk_100.db"},
        "rebuild": True,
    },
]


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("_") or "variant"


def load_variants(path: str | None) -> list[dict]:
    if not path:
        return DEFAULT_VARIANTS
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    variants = raw["variants"] if isinstance(raw, dict) else raw
    if not isinstance(variants, list) or not variants:
        raise ValueError("variants 文件必须是非空列表，或包含 variants 数组的对象")
    return variants


def prepare_db_dir(env: dict) -> None:
    """重建类变体用独立 db 文件，先确保父目录存在。"""
    raw = env.get("MILVUS_DB_PATH", "")
    if not raw or "://" in raw:
        return
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = PROJECT_DIR / path
    path.parent.mkdir(parents=True, exist_ok=True)


def run_step(label: str, cmd: list[str], env: dict) -> int:
    print(f"\n{'=' * 60}\n▶ {label}\n  {' '.join(cmd)}\n{'=' * 60}", flush=True)
    started = time.perf_counter()
    result = subprocess.run(cmd, cwd=PROJECT_DIR, env=env, check=False)
    print(f"⏱ {label} 完成，用时 {time.perf_counter() - started:.1f}s，退出码 {result.returncode}",
          flush=True)
    return result.returncode


def extract_summary(variant: dict, report: dict, report_path: Path) -> dict:
    retrieval = report.get("retrieval") or {}
    agent = report.get("agent") or {}
    return {
        "variant": variant["name"],
        "env": variant.get("env", {}),
        "rebuild": bool(variant.get("rebuild", False)),
        "config": report.get("config", {}),
        "report_path": str(report_path.relative_to(PROJECT_DIR)),
        "timestamp": report.get("timestamp"),
        "retrieval": ({
            "recall": retrieval.get("recall"),
            "mrr": retrieval.get("mrr"),
            "count": retrieval.get("count"),
        } if retrieval else None),
        "agent": ({
            "count": agent.get("count"),
            "answer_accuracy": agent.get("answer_accuracy"),
            "tool_selection_accuracy": agent.get("tool_selection_accuracy"),
            "tool_call_rate_when_expected": agent.get("tool_call_rate_when_expected"),
            "false_tool_call_rate": agent.get("false_tool_call_rate"),
            "avg_tool_rounds": agent.get("avg_tool_rounds"),
            "avg_llm_calls": agent.get("avg_llm_calls"),
            "agent_tokens": agent.get("agent_tokens"),
            "judge_tokens": agent.get("judge_tokens"),
        } if agent else None),
    }


def delta_vs_baseline(summary: dict, baseline: dict | None) -> dict:
    if not baseline:
        return {}
    delta = {}
    for section in ("retrieval", "agent"):
        current = summary.get(section) or {}
        base = baseline.get(section) or {}
        for key in ("recall", "mrr", "answer_accuracy", "tool_selection_accuracy",
                    "avg_tool_rounds", "avg_llm_calls"):
            if key in current and key in base:
                delta[f"{section}.{key}"] = round(float(current[key]) - float(base[key]), 4)
        cur_tokens = (current.get("agent_tokens") or {}).get("total_tokens")
        base_tokens = (base.get("agent_tokens") or {}).get("total_tokens")
        if cur_tokens is not None and base_tokens:
            delta["agent.tokens_change_pct"] = round(
                (cur_tokens - base_tokens) / base_tokens * 100, 1
            )
    return delta


def print_table(summaries: list[dict]) -> None:
    print("\n" + "=" * 100)
    print("📊 消融对比（→ 表示相对 baseline 的变化）")
    print("=" * 100)
    header = f"{'变体':<14}{'Recall':>10}{'MRR':>9}{'回答准确率':>12}{'工具选择':>11}{'轮数':>8}{'token':>10}"
    print(header)
    print("-" * 100)
    for item in summaries:
        ret = item.get("retrieval") or {}
        agent = item.get("agent") or {}
        tokens = (agent.get("agent_tokens") or {}).get("total_tokens", "-")
        print(f"{item['variant']:<14}"
              f"{_pct(ret.get('recall')):>10}"
              f"{_num(ret.get('mrr')):>9}"
              f"{_pct(agent.get('answer_accuracy')):>12}"
              f"{_pct(agent.get('tool_selection_accuracy')):>11}"
              f"{_num(agent.get('avg_tool_rounds')):>8}"
              f"{str(tokens):>10}")
    print("-" * 100)


def _pct(value) -> str:
    return f"{value:.1%}" if isinstance(value, (int, float)) else "-"


def _num(value) -> str:
    return f"{value:.3f}" if isinstance(value, (int, float)) else "-"


def main() -> None:
    parser = argparse.ArgumentParser(description="Library RAG Agent 参数消融")
    parser.add_argument("--limit", type=int, default=None, help="每组只评测前 N 题（快速对比）")
    parser.add_argument("--mode", choices=["retrieval", "agent", "all"], default="all")
    parser.add_argument("--variants-file", default=None, help="自定义变体 JSON 文件")
    parser.add_argument("--output-dir", default=str(EVAL_DIR / "ablation"), help="各组报告目录")
    parser.add_argument("--output", default=str(EVAL_DIR / "ablation_report.json"), help="汇总报告路径")
    parser.add_argument("--only", nargs="*", default=None, help="只跑指定名称的变体")
    args = parser.parse_args()

    variants = load_variants(args.variants_file)
    if args.only:
        variants = [v for v in variants if v["name"] in args.only]
    if not variants:
        raise SystemExit("没有可运行的变体")

    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = PROJECT_DIR / output_dir
    output_path = Path(args.output).expanduser()
    if not output_path.is_absolute():
        output_path = PROJECT_DIR / output_path
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    started = time.perf_counter()
    for index, variant in enumerate(variants, start=1):
        name = safe_name(variant["name"])
        print(f"\n🧪 消融 {index}/{len(variants)}：{variant['name']}  env={variant.get('env', {})}",
              flush=True)
        env = os.environ.copy()
        env.update({str(key): str(value) for key, value in (variant.get("env") or {}).items()})
        prepare_db_dir(env)

        if variant.get("rebuild"):
            code = run_step(f"重建向量库 [{name}]", [PYTHON, str(BUILD_SCRIPT)], env)
            if code != 0:
                print(f"❌ {name} 建库失败，跳过该组")
                summaries.append({"variant": variant["name"], "error": f"build_index exit {code}"})
                continue

        report_path = output_dir / name / "report.json"
        cmd = [PYTHON, str(EVAL_SCRIPT), "--mode", args.mode, "--report", str(report_path)]
        if args.limit:
            cmd += ["--limit", str(args.limit)]
        code = run_step(f"评测 [{name}]", cmd, env)
        if code != 0:
            print(f"❌ {name} 评测失败，跳过该组")
            summaries.append({"variant": variant["name"], "error": f"evaluate exit {code}"})
            continue

        report = json.loads(report_path.read_text(encoding="utf-8"))
        summary = extract_summary(variant, report, report_path)
        baseline = next((s for s in summaries if s["variant"] == "baseline"), None)
        summary["delta_vs_baseline"] = (
            {} if variant["name"] == "baseline" else delta_vs_baseline(summary, baseline)
        )
        summaries.append(summary)

        # 每组结束就落盘，中途失败也保留已有结果
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "mode": args.mode,
            "limit": args.limit,
            "variants_file": args.variants_file,
            "elapsed_seconds": round(time.perf_counter() - started, 1),
            "summaries": summaries,
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    print_table(summaries)
    print(f"\n✅ 消融完成，用时 {time.perf_counter() - started:.1f}s")
    print(f"💾 汇总报告：{output_path}")
    print(f"💾 各组详细报告：{output_dir}/<变体>/report.json")


if __name__ == "__main__":
    main()
