# tests/test_history_repair.py
"""历史修复：悬空的 tool_calls 必须被补上，否则整段会话对模型不可用。

修复前的真实故障：图停在 interrupt（等学号 / 等确认）时用户又发了一条消息，
这条 human 消息被追加到 AI(tool_calls) 之后，下一次调模型 DeepSeek 直接 400：
    "An assistant message with 'tool_calls' must be followed by tool messages..."
"""
import json

import api
from agent_builder import _sanitize_history, fallback_node
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage


def _ai_with(*ids: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": "library_search_tool", "args": {"query": "x"}, "id": i} for i in ids],
    )


def _tool(call_id: str) -> ToolMessage:
    return ToolMessage(content='{"ok": true}', tool_call_id=call_id)


def test_dangling_calls_are_filled_right_after_the_ai_message():
    history = [HumanMessage("我借了几本"), _ai_with("c1", "c2"), HumanMessage("图书馆几点关门")]

    out = _sanitize_history(history)

    kinds = [m.type for m in out]
    # 占位消息必须插在 AI 与后续 human 之间，追加到末尾同样非法
    assert kinds == ["human", "ai", "tool", "tool", "human"]
    assert [m.tool_call_id for m in out[2:4]] == ["c1", "c2"]
    assert out[-1].content == "图书馆几点关门"


def test_valid_history_is_left_untouched():
    history = [HumanMessage("q"), _ai_with("c1", "c2"), _tool("c1"), _tool("c2")]

    out = _sanitize_history(history)

    assert len(out) == 4
    assert out == history


def test_partially_answered_calls_only_fill_the_gap():
    history = [HumanMessage("q"), _ai_with("c1", "c2"), _tool("c1")]

    out = _sanitize_history(history)

    assert [m.type for m in out] == ["human", "ai", "tool", "tool"]
    assert out[-1].tool_call_id == "c2"


def test_fallback_records_aborted_result_before_replying():
    """超轮数 / LLM 报错走 fallback 时，既要有兜底回复，也要把 tool_calls 收尾。"""
    state = {"messages": [HumanMessage("q"), _ai_with("c9")], "error": ""}

    out = fallback_node(state)

    kinds = [m.type for m in out["messages"]]
    assert kinds == ["tool", "ai"]
    payload = json.loads(out["messages"][0].content)
    assert payload["code"] == "ABORTED"
    assert out["messages"][0].tool_call_id == "c9"


def test_fallback_without_dangling_calls_only_replies():
    state = {"messages": [HumanMessage("q"), AIMessage("答案")], "error": ""}

    out = fallback_node(state)

    assert [m.type for m in out["messages"]] == ["ai"]


# ---------------- 接口层：挂起期间不许再灌新消息 ----------------
class _Snapshot:
    def __init__(self, interrupts, values):
        self.interrupts = interrupts
        self.values = values


class _Interrupt:
    def __init__(self, value):
        self.value = value


class _FakeAgent:
    """只用来验证「挂起时不该进图」：一旦被 invoke 就直接失败。"""

    def __init__(self, payload=None, values=None):
        self.payload = payload
        self.values = values or {}

    def get_state(self, config):
        items = (_Interrupt(self.payload),) if self.payload else ()
        return _Snapshot(items, self.values)

    def invoke(self, *args, **kwargs):  # pragma: no cover - 断言用
        raise AssertionError("挂起期间不应该再向图里灌新消息")


def test_chat_while_identify_pending_does_not_touch_the_graph(monkeypatch):
    monkeypatch.setattr(api, "get_agent", lambda: _FakeAgent({"type": "identify"}, {}))
    api._agent = object()              # 让 get_agent 的懒加载分支不触发

    res = api.chat(api.ChatRequest(session_id="s1", message="图书馆几点关门"))

    assert res["status"] == "ok"
    assert "学号" in res["answer"]


def test_chat_while_confirm_pending_asks_user_to_decide(monkeypatch):
    monkeypatch.setattr(api, "get_agent", lambda: _FakeAgent({"type": "confirm"}, {"user_id": "2023001"}))
    api._agent = object()

    res = api.chat(api.ChatRequest(session_id="s2", message="随便问点别的"))

    assert res["status"] == "ok"
    assert "确认" in res["answer"]
    assert res["user_id"] == "2023001"


def test_confirm_without_pending_interrupt_is_a_noop(monkeypatch):
    monkeypatch.setattr(api, "get_agent", lambda: _FakeAgent(None, {"user_id": "2023001"}))
    api._agent = object()

    res = api.confirm(api.ConfirmRequest(session_id="s3", approved=True))

    assert res["status"] == "ok"
    assert "没有待确认" in res["answer"]
