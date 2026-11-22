"""borrow_manage_tool 的行为测试。

这些用例锁的是「业务规则」而不是实现细节：预借而非借出、续借一次、超期拦截、
停借规则、馆藏校验。改 prompt 或换模型都不该让它们变红。

注意语义：action="borrow" 只做预借（RESERVED），action="return_request" 只登记
（PENDING_RETURN）——真正的借出与销账由设备完成，见 test_station.py。
"""
from __future__ import annotations

import itertools
import json
from datetime import date, timedelta

import pytest

import station
import tools

_copy_seq = itertools.count(9001)


def _record(due_offset: int, renewed: bool = False, returned: bool = False) -> dict:
    """构造一条借阅记录；due_offset 是相对今天的到期天数（负数=已逾期）。"""
    today = date.today()
    return {
        "copy": f"B{next(_copy_seq):04d}",
        "borrow_date": (today - timedelta(days=30)).isoformat(),
        "due_date": (today + timedelta(days=due_offset)).isoformat(),
        "renewed": renewed,
        "returned": returned,
    }


@pytest.fixture
def loans_file(tmp_path, monkeypatch):
    """把台账与设备日志都指向临时文件，测试不碰仓库里的 data/。"""
    path = tmp_path / "loans.json"
    monkeypatch.setattr(tools, "LOANS_FILE", str(path))
    monkeypatch.setattr(station, "EVENTS_FILE", str(tmp_path / "events.jsonl"))

    def seed(data: dict, user_id: str = "2023001") -> None:
        path.write_text(
            json.dumps({user_id: data}, ensure_ascii=False), encoding="utf-8"
        )

    return seed


def _call(action: str, book: str, user_id: str = "2023001") -> dict:
    return json.loads(
        tools.borrow_manage_tool.invoke(
            {"action": action, "book_title": book},
            config={"configurable": {"user_id": user_id}},
        )
    )


def test_renew_extends_due_date_by_30_days(loans_file):
    loans_file({"《机器学习》": _record(due_offset=20)})

    out = _call("renew", "《机器学习》")

    assert out["ok"] is True
    assert out["code"] == "RENEWED"
    assert date.fromisoformat(out["data"]["due_date"]) == date.today() + timedelta(days=50)


def test_second_renew_blocked_and_reports_overdue(loans_file):
    """已续借过的书要拦，但必须告诉模型它是否已经超期。"""
    loans_file({"《数据库系统概论》": _record(due_offset=-12, renewed=True)})

    out = _call("renew", "《数据库系统概论》")

    assert out["ok"] is False
    assert out["code"] == "ALREADY_RENEWED"
    assert out["overdue_days"] == 12
    assert out["due_date"] == (date.today() - timedelta(days=12)).isoformat()


def test_renew_blocked_when_overdue(loans_file):
    loans_file({"《数据挖掘导论》": _record(due_offset=-3)})

    out = _call("renew", "《数据挖掘导论》")

    assert out["code"] == "OVERDUE"
    assert out["overdue_days"] == 3
    assert out["fine_estimate"] == pytest.approx(0.3)


def test_borrow_reserves_instead_of_lending(loans_file):
    """borrow 只预借：拿到册号和取书期限，状态是 reserved，不是已借出。"""
    loans_file({})

    out = _call("borrow", "《数据挖掘导论》")

    assert out["code"] == "RESERVED"
    assert out["data"]["copy"].startswith("B")
    assert date.fromisoformat(out["data"]["pickup_deadline"]) == date.today() + timedelta(days=3)
    assert "自助借书机" in out["data"]["next_step"]


def test_borrow_rejects_book_not_in_collection(loans_file):
    """馆藏里没有的书不能预借：否则谁报个书名都能“借成功”，台账里全是虚构记录。"""
    loans_file({})

    out = _call("borrow", "《哈利波特与魔法石》")

    assert out["ok"] is False
    assert out["code"] == "NOT_IN_COLLECTION"
    assert "现有馆藏" in out["hint"]


def test_borrow_accepts_author_suffix_and_returns_location(loans_file):
    """知识库写《机器学习》（周志华），用户可能连带作者一起说，要能对上同一本。"""
    loans_file({})

    out = _call("borrow", "《机器学习》（周志华）")

    assert out["code"] == "RESERVED"
    assert out["data"]["book"] == "《机器学习》"
    assert out["data"]["location"] == "四楼 408 号书架 C 区"


def test_borrow_blocked_when_all_copies_held(loans_file, tmp_path):
    """《深度学习》馆藏 4 本，4 本都在别人手上时第 5 个人借不到。"""
    loans = {f"u{i}": {"《深度学习》": _record(due_offset=10)} for i in range(4)}
    (tmp_path / "loans.json").write_text(
        json.dumps(loans, ensure_ascii=False), encoding="utf-8"
    )

    out = _call("borrow", "《深度学习》")

    assert out["ok"] is False
    assert out["code"] == "NO_COPY_AVAILABLE"
    assert out["total"] == 4 and out["occupied"] == 4


def test_reserve_twice_blocked(loans_file):
    loans_file({"《机器学习》": _record(due_offset=20)})

    out = _call("borrow", "《机器学习》")

    assert out["code"] == "ALREADY_BORROWED"


def test_account_frozen_when_overdue_book_exists(loans_file):
    """有逾期未还的书 → 停借，不能预借新书。"""
    loans_file({"《数据库系统概论》": _record(due_offset=-12)})

    out = _call("borrow", "《数据挖掘导论》")

    assert out["code"] == "ACCOUNT_FROZEN"
    assert "《数据库系统概论》" in out["msg"]
    assert out["fine_due"] == 0.0


def test_return_request_only_marks_pending(loans_file, tmp_path):
    """登记归还只改状态，不算罚款、不算销账——销账要等设备扫码。"""
    loans_file({"《深度学习》": _record(due_offset=-11)})

    out = _call("return_request", "《深度学习》")

    assert out["code"] == "PENDING_RETURN"
    assert out["data"]["overdue_days"] == 11
    assert out["data"]["fine_estimate"] == pytest.approx(1.1)
    saved = json.loads((tmp_path / "loans.json").read_text(encoding="utf-8"))
    record = saved["2023001"]["《深度学习》"]
    assert record["status"] == "pending_return"
    assert record["returned"] is False
    assert record.get("fine_due") in (0, 0.0, None)


def test_return_request_twice_is_reported(loans_file):
    loans_file({"《深度学习》": _record(due_offset=-11)})

    _call("return_request", "《深度学习》")
    out = _call("return_request", "《深度学习》")

    assert out["code"] == "ALREADY_PENDING_RETURN"
    assert out["copy"]


def test_renew_unknown_book(loans_file):
    loans_file({})

    out = _call("renew", "《不存在的书》")

    assert out["code"] == "NOT_BORROWED"
