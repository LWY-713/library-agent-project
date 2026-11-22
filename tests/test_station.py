"""设备层（自助借书机 / 还书机）的测试。

这一层锁的是「实物事件」的语义：
- 预借后必须扫码才能借出；扫码后才能销账；
- 同一册重复扫码是幂等的，不重复加罚款；
- 预借超期自动作废，不能取书；
- 设备只认册条码，不认学号（现实的机器就是这样）。
"""
from __future__ import annotations

import itertools
import json
from datetime import date, timedelta

import pytest

import station
import tools

_copy_seq = itertools.count(8001)


def _rec(status: str, due_offset: int = 20, pickup_offset: int = 2) -> dict:
    today = date.today()
    return {
        "copy": f"B{next(_copy_seq):04d}",
        "status": status,
        "borrow_date": (today - timedelta(days=10)).isoformat(),
        "due_date": (today + timedelta(days=due_offset)).isoformat(),
        "renewed": False,
        "returned": status == "returned",
        "pickup_deadline": (today + timedelta(days=pickup_offset)).isoformat(),
    }


@pytest.fixture
def books(tmp_path, monkeypatch):
    """返回 (写台账, 读台账)，并把设备日志指向临时文件。"""
    path = tmp_path / "loans.json"
    monkeypatch.setattr(tools, "LOANS_FILE", str(path))
    monkeypatch.setattr(station, "EVENTS_FILE", str(tmp_path / "events.jsonl"))

    def seed(data: dict) -> None:
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    def read() -> dict:
        return json.loads(path.read_text(encoding="utf-8"))

    return seed, read


def test_scan_borrow_turns_reservation_into_loan(books):
    seed, read = books
    record = _rec("reserved")
    seed({"2023001": {"《机器学习》": record}})

    out = station.scan_borrow(record["copy"])

    assert out["ok"] is True
    assert out["data"]["due_date"] == (date.today() + timedelta(days=30)).isoformat()
    saved = read()["2023001"]["《机器学习》"]
    assert saved["status"] == "borrowed"
    assert saved["borrow_date"] == date.today().isoformat()


def test_scan_borrow_twice_is_idempotent(books):
    seed, read = books
    record = _rec("reserved")
    seed({"2023001": {"《机器学习》": record}})

    station.scan_borrow(record["copy"])
    first_due = read()["2023001"]["《机器学习》"]["due_date"]
    out = station.scan_borrow(record["copy"])

    assert out["ok"] is False and out["code"] == "ALREADY_BORROWED"
    assert read()["2023001"]["《机器学习》"]["due_date"] == first_due


def test_scan_borrow_rejects_expired_pickup(books):
    seed, read = books
    record = _rec("reserved", pickup_offset=-1)
    seed({"2023001": {"《机器学习》": record}})

    out = station.scan_borrow(record["copy"])

    assert out["code"] == "PICKUP_EXPIRED"
    assert read()["2023001"]["《机器学习》"]["status"] == "cancelled"


def test_scan_borrow_rejects_unreserved_copy(books):
    seed, _ = books
    record = _rec("borrowed")
    seed({"2023001": {"《机器学习》": record}})

    out = station.scan_borrow(record["copy"])

    assert out["code"] == "ALREADY_BORROWED"


def test_scan_return_settles_fine_and_marks_returned(books):
    seed, read = books
    record = _rec("pending_return", due_offset=-11)
    seed({"2023001": {"《深度学习》": record}})

    out = station.scan_return(record["copy"])

    assert out["ok"] is True
    assert out["data"]["overdue_days"] == 11
    assert out["data"]["fine"] == pytest.approx(1.1)
    saved = read()["2023001"]["《深度学习》"]
    assert saved["status"] == "returned"
    assert saved["returned"] is True
    assert saved["fine_due"] == pytest.approx(1.1)


def test_scan_return_twice_does_not_double_charge(books):
    """重复扫码是设备集成最常见的坑：不能重复结算罚款。"""
    seed, read = books
    record = _rec("pending_return", due_offset=-11)
    seed({"2023001": {"《深度学习》": record}})

    station.scan_return(record["copy"])
    out = station.scan_return(record["copy"])

    assert out["ok"] is False and out["code"] == "ALREADY_RETURNED"
    assert read()["2023001"]["《深度学习》"]["fine_due"] == pytest.approx(1.1)


def test_scan_return_on_reserved_copy_is_rejected(books):
    seed, _ = books
    record = _rec("reserved")
    seed({"2023001": {"《机器学习》": record}})

    out = station.scan_return(record["copy"])

    assert out["code"] == "NOT_BORROWED"


def test_scan_unknown_copy(books):
    seed, _ = books
    seed({})

    assert station.scan_return("B9999")["code"] == "NO_SUCH_COPY"
    assert station.scan_borrow("B9999")["code"] == "NO_SUCH_COPY"


def test_pending_lists_only_matching_status(books):
    seed, _ = books
    seed({"2023001": {
        "《待还的书》": _rec("pending_return"),
        "《在借的书》": _rec("borrowed"),
        "《待取的书》": _rec("reserved"),
    }})

    pending = station.pending_returns("2023001")
    reserved = station.reserved_items("2023001")

    assert [item["book"] for item in pending] == ["《待还的书》"]
    assert [item["book"] for item in reserved] == ["《待取的书》"]


def test_event_log_appended(books):
    seed, _ = books
    record = _rec("reserved")
    seed({"2023001": {"《机器学习》": record}})

    station.scan_borrow(record["copy"])

    with open(station.EVENTS_FILE, encoding="utf-8") as fh:
        lines = fh.read().strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["event"] == "borrow"
