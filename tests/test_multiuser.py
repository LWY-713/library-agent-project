"""多用户隔离与越权拦截。

这一组用例锁的是「身份边界」而不是业务规则：
- 不同用户看到各自的台账，互不串数据；
- 同名图书允许被不同人各借一本；
- 续借 / 归还别人的书必须被拦下；
- 没有身份时一切个人业务都拒绝；
- 旧版扁平台账能被自动迁移。
"""
from __future__ import annotations

import json
from datetime import date, timedelta

import pytest

import tools


@pytest.fixture
def loans_file(tmp_path, monkeypatch):
    path = tmp_path / "loans.json"
    monkeypatch.setattr(tools, "LOANS_FILE", str(path))

    def seed(data: dict) -> None:
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    return seed


def _rec(due_offset: int = 20, renewed: bool = False, returned: bool = False) -> dict:
    today = date.today()
    return {
        "borrow_date": (today - timedelta(days=10)).isoformat(),
        "due_date": (today + timedelta(days=due_offset)).isoformat(),
        "renewed": renewed,
        "returned": returned,
    }


def _call(action: str, book: str = "", user_id: str = "2023001") -> dict:
    return json.loads(
        tools.borrow_manage_tool.invoke(
            {"action": action, "book_title": book},
            config={"configurable": {"user_id": user_id}},
        )
    )


def test_list_only_returns_own_books(loans_file):
    loans_file({
        "2023001": {"《机器学习》": _rec(), "《数据库系统概论》": _rec()},
        "2023002": {"《算法导论》": _rec()},
    })

    a = _call("list", user_id="2023001")
    b = _call("list", user_id="2023002")

    assert a["data"]["count"] == 2
    assert b["data"]["count"] == 1
    assert {item["book"] for item in b["data"]["items"]} == {"《算法导论》"}


def test_list_skips_returned_books(loans_file):
    loans_file({
        "2023001": {"《已还的书》": _rec(returned=True), "《在借的书》": _rec()},
    })

    out = _call("list")

    assert out["data"]["count"] == 1
    assert out["data"]["items"][0]["book"] == "《在借的书》"


def test_same_title_can_be_borrowed_by_two_users(loans_file):
    """真实图书馆有多册复本：A 借着《机器学习》，不妨碍 B 也预借一本。"""
    loans_file({"2023001": {"《机器学习》": _rec()}})

    out = _call("borrow", "《机器学习》", user_id="2023002")

    assert out["code"] == "RESERVED"
    assert out["data"]["copy"] != "B0001"


def test_cannot_renew_someone_elses_book(loans_file):
    loans_file({"2023001": {"《机器学习》": _rec()}})

    out = _call("renew", "《机器学习》", user_id="2023002")

    assert out["ok"] is False
    assert out["code"] == "NOT_OWNED"


def test_cannot_return_someone_elses_book(loans_file, tmp_path):
    loans_file({"2023001": {"《机器学习》": _rec()}})

    out = _call("return_request", "《机器学习》", user_id="2023002")

    assert out["code"] == "NOT_OWNED"
    saved = json.loads((tmp_path / "loans.json").read_text(encoding="utf-8"))
    assert saved["2023001"]["《机器学习》"]["returned"] is False


def test_missing_identity_is_blocked(loans_file):
    loans_file({})

    out = json.loads(
        tools.borrow_manage_tool.invoke({"action": "list"}, config={"configurable": {}})
    )

    assert out["code"] == "NEED_IDENTITY"


def test_old_flat_format_is_migrated(loans_file):
    """旧版 {书名: {user: 学号, ...}} 读进来要自动归到对应用户名下。"""
    loans_file({"《机器学习》": {"user": "2023009", **_rec()}})

    out = _call("list", user_id="2023009")

    assert out["data"]["count"] == 1
    assert out["data"]["user"] == "2023009"
