# agent_builder.py
"""用显式 StateGraph 构建 LangGraph Agent：DeepSeek 对话模型 + 图书馆检索工具。

重构说明（原实现为 langchain.agents.create_agent）：
create_agent 内部其实就是一个预置的 StateGraph（agent 节点 + tools 节点循环）。
这里把它“展开”成手写的显式图，行为保持不变，但图结构、状态、流式输出完全可控。

图结构：
                    ┌──────────────┐
        有 tool_calls│              │无 tool_calls
        ┌──────────►│  agent 节点  │──────────► END
        │           │ (调用 LLM)   │
        │           └──────┬───────┘
        │                  │ 出错 / 超过工具轮数上限
        │                  ▼
   ┌────┴───────┐      ┌──────────┐
   │ approve节点│      │fallback  │──► END（兜底回复，不让程序崩掉）
   │(写操作确认)│      └──────────┘
   └────┬───────┘
        │ 放行 → tools 节点；拒绝 → 直接补 ToolMessage 回 agent
        ▼
   ┌──────────┐
   │ tools节点│──► agent（工具结果喂回，继续下一轮）
   └──────────┘

状态（AgentState）：
- messages: 对话历史，add_messages reducer 负责按消息 id 增量合并（与 create_agent 一致）
- tool_rounds: 已完成的“工具调用轮数”，用于防止 agent/tools 死循环
- error: agent 节点的报错信息，空串表示正常
"""
from __future__ import annotations

import json
from typing import Annotated, Literal

from langchain_deepseek import ChatDeepSeek
from langchain_core.messages import AnyMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt
from typing_extensions import TypedDict

from config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    LLM_MODEL_NAME,
    LLM_TEMPERATURE,
    MAX_TOOL_ROUNDS,
)
from tools import borrow_manage_tool, library_search_tool

SYSTEM_PROMPT = """你是「校园中心图书馆」的智能助手，面向同学解答图书馆相关问题。

## 一、工具使用
1. 凡涉及开放时间、馆藏图书、书架位置、借阅规则、研讨室预约、活动、设备报修、
   失物招领等图书馆话题，必须先调用 `library_search_tool` 检索，再依据检索结果回答。
2. 检索工具返回 JSON：`{"ok": true, "code": "OK", "data": [...]}` 表示有结果；
   `{"ok": false, "code": "NO_HIT"}` 表示没召回。
3. **如果 ok=false，或者返回的片段明显答非所问（比如问"西瓜书"却全是《Python编程》），
   不要直接说"没找到"。你应该换一种检索词再查，最多重试 2 次：**
   - 用书名全称替代外号或简称（"西瓜书" → "《机器学习》周志华"）
   - 用正式术语替代口语（"能再拖几天" → "续借 借阅期限"）
   - 补充作者、学科、楼层等限定词
4. 重试 2 次仍无结果，才回复"知识库中暂未收录"，并说明你尝试过哪些检索词。

## 二、回答规则
5. 知识库中同时存在历史通知和最新通知时，以日期最新的条目为准，已作废的旧时间、
   旧规则不要采用。
6. 检索结果里没有的信息，直接说明「知识库中暂未收录」，严禁编造具体楼层、编号、规则。
7. 回答使用中文，简洁准确；引用依据时标注片段编号，如【片段1】。
"""


# ======================= 状态定义 =======================
class AgentState(TypedDict):
    """图的全局状态：每个节点读它、改它，框架负责在节点间传递。"""

    # add_messages 是“增量合并”reducer：节点返回的消息会追加到历史里，
    # 而不是整体覆盖 —— 这正是 create_agent 内部用的同款 reducer。
    messages: Annotated[list[AnyMessage], add_messages]
    # 工具调用轮数计数（tools 节点每执行一轮 +1）
    tool_rounds: int
    # agent 节点调用 LLM 失败时的原因，空串 = 正常
    error: str


# ======================= 模型 =======================
def get_llm() -> ChatDeepSeek:
    """DeepSeek 官方接口（OpenAI 兼容）。"""
    if not DEEPSEEK_API_KEY:
        raise RuntimeError("未配置 DEEPSEEK_API_KEY，请在项目根目录 .env 中填写。")
    return ChatDeepSeek(
        model=LLM_MODEL_NAME,
        api_key=DEEPSEEK_API_KEY,
        api_base=DEEPSEEK_BASE_URL,
        temperature=LLM_TEMPERATURE,
    )


# ======================= 节点定义 =======================
def agent_node(state: AgentState, llm_with_tools: ChatDeepSeek) -> dict:
    """agent 节点：把 system prompt 和对话历史一起交给绑定工具的 LLM。

    修复：此前直接 invoke(state["messages"])，SYSTEM_PROMPT 从未传入模型。
    """
    try:
        response = llm_with_tools.invoke(
            [{"role": "system", "content": SYSTEM_PROMPT}, *state["messages"]]
        )
    except Exception as exc:  # 网络 / API 失败：不抛异常，交给 fallback 节点兜底
        return {"error": f"{type(exc).__name__}: {exc}"}
    return {"messages": [response], "error": ""}


def tools_node(state: AgentState, tool_node: ToolNode) -> dict:
    """tools 节点：执行上一条 AI 消息里的所有 tool_calls，结果作为 ToolMessage 追加。

    直接复用 langgraph.prebuilt.ToolNode（create_agent 内部用的就是它），
    保证工具执行、异常转述等行为与原来完全一致。
    """
    result = tool_node.invoke(state)
    # 轮数 +1，供路由函数判断是否达到上限
    return {**result, "tool_rounds": state.get("tool_rounds", 0) + 1}


def fallback_node(state: AgentState) -> dict:
    """fallback 节点：兜底回复，保证图总能正常走到 END，而不是让程序崩溃。"""
    error = state.get("error", "")
    if error:
        # LLM 调用失败（网络 / API 问题），给一条可读的错误回复
        reply = f"运行出错：{error}"
    else:
        # agent ↔ tools 循环超过上限仍未给出最终回答
        reply = f"抱歉，这个问题处理了 {MAX_TOOL_ROUNDS} 轮工具调用仍未完成，请换个问法再试。"
    return {"messages": [("ai", reply)], "error": ""}


# ======================= 写操作人工确认 =======================
WRITE_TOOLS = {"borrow_manage_tool"}


def approve_node(state: AgentState) -> dict:
    """写操作前人工确认。读工具直接放行。

    interrupt 会挂起图，等 Command(resume=...)。没有 checkpointer 会直接报错。
    """
    last = state["messages"][-1]
    calls = getattr(last, "tool_calls", None) or []
    if not any(c.get("name") in WRITE_TOOLS for c in calls):
        return {}                      # 纯读操作，放行

    detail = "；".join(f"{c['name']}({c.get('args')})" for c in calls)
    decision = interrupt({
        "ask": "即将执行写操作，是否确认？",
        "detail": detail,
        "options": ["yes", "no"],
    })

    if decision == "yes":
        return {}

    # 拒绝：为这条 AI 消息里的全部 tool_call 补上拒绝结果，跳过 tools 节点
    return {"messages": [
        ToolMessage(
            content=json.dumps(
                {"ok": False, "code": "USER_REJECTED", "msg": "用户取消了本次操作"},
                ensure_ascii=False,
            ),
            tool_call_id=c["id"],
        )
        for c in calls
    ]}


def route_after_approve(state: AgentState) -> Literal["tools", "agent"]:
    last = state["messages"][-1]
    if getattr(last, "type", "") == "tool":
        return "agent"                 # 已写入拒绝结果，回到 agent
    return "tools"


# ======================= 路由（条件边） =======================
def route_after_agent(state: AgentState) -> Literal["tools", "fallback", "__end__"]:
    """agent 节点之后的分诊台：决定下一步去 tools、fallback 还是结束。"""
    if state.get("error"):
        return "fallback"

    last = state["messages"][-1]
    has_tool_calls = bool(getattr(last, "tool_calls", None))
    if has_tool_calls:
        if state.get("tool_rounds", 0) >= MAX_TOOL_ROUNDS:
            # 达到工具轮数上限：不再执行工具，去 fallback 收尾
            return "fallback"
        return "tools"
    return END


# ======================= 组图 =======================
def build_agent():
    """组装显式 StateGraph：START → agent ⇄ tools，异常/超限 → fallback → END。

    返回的是编译好的图，对外接口和 create_agent 一致：
    - graph.invoke({"messages": [...]}) → {"messages": [...]}
    - graph.stream(..., stream_mode="messages") 可逐 token 流式输出
    """
    llm_with_tools = get_llm().bind_tools([library_search_tool, borrow_manage_tool])
    tool_node = ToolNode([library_search_tool, borrow_manage_tool])

    builder = StateGraph(AgentState)
    builder.add_node("agent", lambda state: agent_node(state, llm_with_tools))
    builder.add_node("approve", approve_node)
    builder.add_node("tools", lambda state: tools_node(state, tool_node))
    builder.add_node("fallback", fallback_node)

    builder.add_edge(START, "agent")
    builder.add_conditional_edges(
        "agent", route_after_agent,
        {"tools": "approve", "fallback": "fallback", END: END},
    )
    builder.add_conditional_edges(
        "approve", route_after_approve,
        {"tools": "tools", "agent": "agent"},
    )
    builder.add_edge("tools", "agent")
    builder.add_edge("fallback", END)

    return builder.compile(checkpointer=MemorySaver())   # interrupt 必须
