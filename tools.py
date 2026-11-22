# tools.py
"""对外暴露给 Agent 的工具（结构化返回：ok / code / data）。"""
from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from typing import Literal

from langchain_core.tools import tool

from config import CURRENT_USER_ID, LOANS_FILE, SEARCH_MIN_SCORE, TOP_K
from rag_module import search_knowledge


def _err(code: str, msg: str, **extra) -> str:
    payload = {"ok": False, "code": code, "msg": msg, **extra}
    return json.dumps(payload, ensure_ascii=False)


@tool
def library_search_tool(query: str) -> str:
    """查询校园中心图书馆知识库。

    只要问题与图书馆有关（开放时间、馆藏图书、书架位置、借阅规则、研讨室预约、
    活动、设备报修、失物招领等），就应该调用本工具，再依据返回片段回答。

    返回 JSON：
    - {"ok": true,  "code": "OK",              "data": [{index, chunk_id, score, text}, ...]}
    - {"ok": false, "code": "NO_HIT",          "hint": "换说法重试的建议"}
    - {"ok": false, "code": "SEARCH_ERROR",    "msg": "..."}

    Args:
        query: 检索词。用书名全称、正式术语，不要原样照抄用户口语句子。
    """
    try:
        hits = search_knowledge(query, limit=TOP_K)
    except Exception as exc:
        return _err("SEARCH_ERROR", f"检索失败：{exc}")

    if not hits:
        return _err(
            "NO_HIT", "知识库没有召回任何片段",
            hint="换一种说法再检索：用书名全称代替外号，用正式术语代替口语，或补充作者/学科/楼层",
        )

    top_score = max(hit["score"] for hit in hits)
    if SEARCH_MIN_SCORE > 0 and top_score < SEARCH_MIN_SCORE:
        return _err(
            "LOW_RELEVANCE", "召回结果相关度过低",
            top_score=round(top_score, 4),
            hint="更换检索词后重试",
        )

    data = [
        {
            "index": i,
            "chunk_id": hit["chunk_id"],
            "score": round(hit["score"], 4),
            "text": hit["text"],
        }
        for i, hit in enumerate(hits, start=1)
    ]
    return json.dumps({"ok": True, "code": "OK", "data": data}, ensure_ascii=False)


def _load_loans() -> dict:
    path = Path(LOANS_FILE)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _save_loans(data: dict) -> None:
    Path(LOANS_FILE).write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )


@tool
def borrow_manage_tool(
    action: Literal["borrow", "renew", "return"],
    book_title: str,
) -> str:
    """办理图书借阅 / 续借 / 归还（写操作）。

    仅用于执行借还操作。查询借阅规则、开放时间、馆藏位置请用 library_search_tool。
    当前用户身份由系统注入，无需也无法指定。

    Args:
        action: borrow=借书，renew=续借，return=还书
        book_title: 书名，需与被借阅图书一致，例如《机器学习》
    """
    loans = _load_loans()
    key = book_title.strip()
    rec = loans.get(key)

    if action == "borrow":
        if rec and not rec.get("returned"):
            return _err("ALREADY_BORROWED", f"你已借阅{key}，不能重复借阅")
        loans[key] = {
            "user": CURRENT_USER_ID,
            "borrow_date": date.today().isoformat(),
            "due_date": (date.today() + timedelta(days=30)).isoformat(),
            "renewed": False,
            "returned": False,
        }
        _save_loans(loans)
        return json.dumps(
            {"ok": True, "code": "BORROWED",
             "data": {"book": key, "due_date": loans[key]["due_date"]}},
            ensure_ascii=False,
        )

    if not rec or rec.get("returned"):
        return _err("NOT_BORROWED", f"你名下没有借阅中的{key}")

    if action == "renew":
        due = date.fromisoformat(rec["due_date"])
        overdue_days = max(0, (date.today() - due).days)
        # 返回里必须带上到期日与逾期天数：只给一句“已续借过”时，模型无法判断
        # 这本书是不是已经超期，容易编出“尽快归还以免逾期”这种与事实相反的答复。
        if rec.get("renewed"):
            return _err(
                "ALREADY_RENEWED", "每本书只能续借一次，该书已经续借过",
                due_date=rec["due_date"], overdue_days=overdue_days,
            )
        if due < date.today():
            return _err(
                "OVERDUE", f"该书已于 {rec['due_date']} 超期，需先归还并缴清罚款",
                due_date=rec["due_date"], overdue_days=overdue_days,
            )
        rec["renewed"] = True
        rec["due_date"] = (due + timedelta(days=30)).isoformat()
        _save_loans(loans)
        return json.dumps(
            {"ok": True, "code": "RENEWED",
             "data": {"book": key, "due_date": rec["due_date"]}},
            ensure_ascii=False,
        )

    # action == "return"
    rec["returned"] = True
    due = date.fromisoformat(rec["due_date"])
    overdue_days = max(0, (date.today() - due).days)
    fine = min(overdue_days * 0.1, 10.0)      # 0.1 元/天，单本上限 10 元
    _save_loans(loans)
    return json.dumps(
        {"ok": True, "code": "RETURNED",
         "data": {"book": key, "overdue_days": overdue_days, "fine": round(fine, 2)}},
        ensure_ascii=False,
    )
