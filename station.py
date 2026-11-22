"""模拟图书馆的自助设备：借书机与还书机。

为什么单独一层？
真实图书馆里，借出与归还是「实物事件」——设备扫到册条码那一刻，系统才改状态。
本模块只由 HTTP 设备端点（api.py 的 /api/station/*）调用，**不注册为 LLM 工具**，
所以模型无法凭一句话把书借走或销账；它最多能替用户"预借"和"登记归还"。

幂等性（设备集成的第一个坑）：
自助机可能重复扫码、网络重试导致同一请求打两次。这里不靠请求 ID，而靠**状态本身**做幂等——
记录已经是 returned，第二次扫描返回 ALREADY_RETURNED 且不改任何数据。
"""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path

from config import EVENTS_FILE
from tools import (
    BORROWED, CANCELLED, LOAN_DAYS, PENDING_RETURN, RESERVED, RETURNED,
    _fine_of, _load_loans, _save_loans, _status, find_by_copy,
)


def _log(event: str, **fields) -> None:
    """往设备日志追加一行。审计和排障靠它，不参与业务判断。"""
    path = Path(EVENTS_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = {"ts": datetime.now().isoformat(timespec="seconds"), "event": event, **fields}
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(line, ensure_ascii=False) + "\n")


def scan_borrow(copy: str) -> dict:
    """自助借书机：扫一本预约架上的书，完成借出。"""
    loans = _load_loans()
    hit = find_by_copy(loans, copy)
    if not hit:
        return {"ok": False, "code": "NO_SUCH_COPY",
                "msg": f"馆藏里没有册号 {copy}，请确认条码"}
    user_id, book, record = hit
    status = _status(record)

    if status == BORROWED:
        # 重复扫码：不改数据，直接回报当前状态
        return {"ok": False, "code": "ALREADY_BORROWED",
                "msg": f"{book}（{record.get('copy')}）已经借出，无需重复扫码",
                "book": book, "due_date": record.get("due_date")}
    if status == PENDING_RETURN:
        return {"ok": False, "code": "PENDING_RETURN",
                "msg": f"{book} 已登记归还、还没过还书机，不能借出", "book": book}
    if status in {RETURNED, CANCELLED}:
        return {"ok": False, "code": "NOT_RESERVED",
                "msg": f"{book}（{record.get('copy')}）不在预约架上，无法取书", "book": book}

    deadline = record.get("pickup_deadline")
    if deadline and date.today() > date.fromisoformat(deadline):
        record["status"] = CANCELLED
        record["cancelled_at"] = date.today().isoformat()
        _save_loans(loans)
        _log("reservation_expired", user=user_id, book=book, copy=record.get("copy"))
        return {"ok": False, "code": "PICKUP_EXPIRED",
                "msg": f"预借已于 {deadline} 过期，本次预借自动作废，请重新预借",
                "book": book, "user": user_id}

    record["status"] = BORROWED
    record["borrow_date"] = date.today().isoformat()
    record["due_date"] = (date.today() + timedelta(days=LOAN_DAYS)).isoformat()
    record["renewed"] = False
    record.pop("pickup_deadline", None)
    _save_loans(loans)
    _log("borrow", user=user_id, book=book, copy=record.get("copy"), due=record["due_date"])
    return {"ok": True, "code": "BORROWED",
            "data": {"user": user_id, "book": book, "copy": record.get("copy"),
                     "due_date": record["due_date"]}}


def scan_return(copy: str) -> dict:
    """自助还书机：扫一本读者塞进来的书，销账并结算逾期罚款。"""
    loans = _load_loans()
    hit = find_by_copy(loans, copy)
    if not hit:
        return {"ok": False, "code": "NO_SUCH_COPY",
                "msg": f"馆藏里没有册号 {copy}，请确认条码"}
    user_id, book, record = hit
    status = _status(record)

    if status == RETURNED:
        # 幂等：重复扫码不重复结算，也不再加罚款
        return {"ok": False, "code": "ALREADY_RETURNED",
                "msg": f"{book}（{record.get('copy')}）已经归还，本次扫码未做任何变更",
                "book": book, "returned_at": record.get("returned_at")}
    if status == RESERVED:
        return {"ok": False, "code": "NOT_BORROWED",
                "msg": f"{book} 还在预约架/馆内，未处于借出状态，无需归还", "book": book}
    if status == CANCELLED:
        return {"ok": False, "code": "NOT_BORROWED",
                "msg": f"{book} 的预借已作废，无需归还", "book": book}

    overdue = max(0, (date.today() - date.fromisoformat(record["due_date"])).days)
    fine = _fine_of(record)

    record["status"] = RETURNED
    record["returned_at"] = date.today().isoformat()
    record["fine_due"] = round(float(record.get("fine_due") or 0) + fine, 2)
    record.pop("pending_since", None)
    _save_loans(loans)
    _log("return", user=user_id, book=book, copy=record.get("copy"),
         overdue_days=overdue, fine=fine)

    return {"ok": True, "code": "RETURNED",
            "data": {"user": user_id, "book": book, "copy": record.get("copy"),
                     "overdue_days": overdue, "fine": fine,
                     "fine_due": record["fine_due"],
                     "note": "已销账；有罚款请到服务台或线上缴清，欠费期间无法预借新书"}}


def pending_returns(user_id: str) -> list[dict]:
    """当前用户已登记但还没过还书机的书（前端"模拟还书机"面板用）。"""
    loans = _load_loans()
    out = []
    for book, record in (loans.get(user_id) or {}).items():
        if isinstance(record, dict) and _status(record) == PENDING_RETURN:
            out.append({
                "book": book, "copy": record.get("copy"),
                "due_date": record.get("due_date"),
                "overdue_days": max(
                    0, (date.today() - date.fromisoformat(record["due_date"])).days
                ) if record.get("due_date") else 0,
                "fine_estimate": _fine_of(record),
            })
    return out


def reserved_items(user_id: str) -> list[dict]:
    """当前用户已预借、等待取书的书（前端"模拟借书机"面板用）。"""
    loans = _load_loans()
    out = []
    for book, record in (loans.get(user_id) or {}).items():
        if isinstance(record, dict) and _status(record) == RESERVED:
            out.append({
                "book": book, "copy": record.get("copy"),
                "pickup_deadline": record.get("pickup_deadline"),
            })
    return out
