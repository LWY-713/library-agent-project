# scripts/reset_demo_data.py
"""重置演示用借阅台账：所有日期都相对「今天」生成，避免演示数据随日期失效。

背景：data/loans.json 里的日期一旦写死，过几天再演示就会出现
「README 说逾期 12 天、实际逾期 20 天」，预借的取书期限也会过期。
演示前（或 clone 后第一次跑）执行一次即可：

    python scripts/reset_demo_data.py

脚本只写 data/loans.json 与清空 data/events.jsonl，不动知识库和向量库。
"""
from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import EVENTS_FILE, LOANS_FILE  # noqa: E402

TODAY = date.today()


def _d(offset: int) -> str:
    """相对今天偏移 N 天的 ISO 日期（负数表示过去）。"""
    return (TODAY + timedelta(days=offset)).isoformat()


def build_loans() -> dict:
    """构造与 README 演示脚本一致的台账。

    2023001：3 本在借，其中《数据库系统概论》逾期 12 天（停借）
    2023002：1 本在借，用来对比「不同用户看到不同台账」
    2024071184：1 本预借待取，取书期限还有 2 天
    """
    return {
        "2023001": {
            "《机器学习》": {
                "copy": "B0001",
                "status": "borrowed",
                "borrow_date": _d(-23),
                "due_date": _d(37),          # 已续借一次，借期延长 30 天
                "renewed": True,
                "returned": False,
                "fine_due": 0.0,
            },
            "《数据库系统概论》": {
                "copy": "B0002",
                "status": "borrowed",
                "borrow_date": _d(-72),
                "due_date": _d(-12),         # 逾期 12 天 → 账户停借
                "renewed": True,
                "returned": False,
                "fine_due": 0.0,
            },
            "《深度学习》": {
                "copy": "B0003",
                "status": "returned",
                "borrow_date": _d(-43),
                "due_date": _d(-13),
                "renewed": False,
                "returned": True,
                "returned_at": _d(-2),
                "fine_due": 1.1,             # 逾期 11 天，0.1 元/天
            },
            "《数据科学导论》": {
                "copy": "B0004",
                "status": "borrowed",
                "borrow_date": _d(-2),
                "due_date": _d(28),
                "renewed": False,
                "returned": False,
                "fine_due": 0.0,
            },
        },
        "2023002": {
            "《数据挖掘导论》": {
                "copy": "B0005",
                "status": "borrowed",
                "borrow_date": _d(-9),
                "due_date": _d(21),
                "renewed": False,
                "returned": False,
                "fine_due": 0.0,
            }
        },
        "2024071184": {
            "《机器学习》": {
                "copy": "B0006",
                "status": "reserved",
                "reserved_at": _d(-1),
                "pickup_deadline": _d(2),
                "borrow_date": None,
                "due_date": None,
                "renewed": False,
                "fine_due": 0.0,
                "returned": False,
            }
        },
    }


def main() -> int:
    Path(LOANS_FILE).write_text(
        json.dumps(build_loans(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    # 设备日志是运行产物，重置台账时一并清空，避免旧扫码记录对不上
    Path(EVENTS_FILE).write_text("", encoding="utf-8")

    print(f"✅ 已重置演示台账：{LOANS_FILE}")
    print(f"   基准日期 {TODAY.isoformat()}")
    print("   2023001：3 本在借（《数据库系统概论》逾期 12 天）")
    print("   2023002：1 本在借")
    print("   2024071184：1 本预借待取（取书期限还有 2 天）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
