"""borrow_manage_tool 的行为测试。

这些用例锁住的是「业务规则」而不是实现细节：续借一次、超期拦截、
罚款计算与上限、重复借阅。改 prompt 或换模型都不该让它们变红。
"""
from __future__ import annotations

import json
from datetime import date, timedelta

import pytest

import tools


def _record(due_offset: int, renewed: bool = False, returned: bool = False) -> dict:
    """构造一条借阅记录；due_offset 是相对今天的到期天数（负数=已逾期）。"""
    today = date.today()
    return {
        "user": "2023001",
        "borrow_date": (today - timedelta(days=30)).isoformat(),
        "due_date": (today + timedelta(days=due_offset)).isoformat(),
        "renewed": renewed,
        "returned": returned,
    }


@pytest.fixture
def loans_file(tmp_path, monkeypatch):
    """把台账指向临时文件，测试不碰仓库里的 data/loans.json。"""
    path = tmp_path / "loans.json"
    monkeypatch.setattr(tools, "LOANS_FILE", str(path))

    def seed(data: dict) -> None:
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    return seed


def _call(action: str, book: str) -> dict:
    return json.loads(
        tools.borrow_manage_tool.invoke({"action": action, "book_title": book})
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
    loans_file({"《算法导论》": _record(due_offset=-3)})

    out = _call("renew", "《算法导论》")

    assert out["code"] == "OVERDUE"
    assert out["overdue_days"] == 3


def test_return_computes_fine_and_persists(loans_file, tmp_path):
    loans_file({"《深度学习》": _record(due_offset=-11)})

    out = _call("return", "《深度学习》")

    assert out["code"] == "RETURNED"
    assert out["data"]["overdue_days"] == 11
    assert out["data"]["fine"] == pytest.approx(1.1)
    saved = json.loads((tmp_path / "loans.json").read_text(encoding="utf-8"))
    assert saved["《深度学习》"]["returned"] is True


def test_return_fine_is_capped(loans_file):
    """本科生单本上限 10 元：逾期 200 天也只收 10 元。"""
    loans_file({"《存在与时间》": _record(due_offset=-200)})

    out = _call("return", "《存在与时间》")

    assert out["data"]["fine"] == pytest.approx(10.0)


def test_borrow_twice_blocked(loans_file):
    loans_file({"《机器学习》": _record(due_offset=20)})

    out = _call("borrow", "《机器学习》")

    assert out["code"] == "ALREADY_BORROWED"


def test_borrow_new_book_sets_30_day_term(loans_file):
    loans_file({})

    out = _call("borrow", "《编译原理》")

    assert out["code"] == "BORROWED"
    assert date.fromisoformat(out["data"]["due_date"]) == date.today() + timedelta(days=30)


def test_renew_unknown_book(loans_file):
    loans_file({})

    out = _call("renew", "《不存在的书》")

    assert out["code"] == "NOT_BORROWED"
