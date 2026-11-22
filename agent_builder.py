# agent_builder.py
"""用显式 StateGraph 构建 LangGraph Agent：DeepSeek 对话模型 + 图书馆检索工具。

重构说明（原实现为 langchain.agents.create_agent）：
create_agent 内部其实就是一个预置的 StateGraph（agent 节点 + tools 节点循环）。
这里把它“展开”成手写的显式图，行为保持不变，但图结构、状态、流式输出完全可控。

图结构：
  START → agent ─┬─ 无 tool_calls ──────────────► END
                 ├─ 出错 / 超工具轮数上限 ─────► fallback ──► END
                 └─ 有 tool_calls ──► identify ──► approve ──┬─► tools ──► agent
                                       (未认证则要学号)  (写操作才确认) └─(拒绝)─► agent

两道门的分工：
- identify：只拦涉及个人工具（借/续/还/list）的调用，知识问答直接放行；
- approve ：只拦写操作（borrow/renew/return），list 查询不弹确认。

状态（AgentState）：
- messages: 对话历史，add_messages reducer 负责按消息 id 增量合并（与 create_agent 一致）
- tool_rounds: 已完成的“工具调用轮数”，用于防止 agent/tools 死循环
- error: agent 节点的报错信息，空串表示正常
"""
from __future__ import annotations

import json
from typing import Annotated, Literal

from langchain_deepseek import ChatDeepSeek
from langchain_core.messages import AIMessage, AnyMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
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
8. 写工具返回 `ok=false` 时，按返回里的 `code` / `msg` / `due_date` / `overdue_days`
   如实说明原因。例如 `ALREADY_RENEWED` 且 `overdue_days>0` 时，要同时说明
   "已经续借过" 和 "该书已于 X 到期、已逾期 N 天"，不要臆测它还没逾期。
9. 用户问「我借了哪些书 / 借了几本 / 什么时候到期」时，调用
   `borrow_manage_tool(action="list")` 查询真实台账，不要回答"查不到"。
10. 个人业务（借书、续借、还书、查询我的借阅）的当前用户由系统注入，你不要向用户
    索要学号；系统尚未识别身份时会自动挂起并要求输入，按提示继续即可。
11. 借书被拒时按 `code` 区分说法，不要混为一谈：
    - `NOT_IN_COLLECTION`：这本不在馆藏中。把 `hint` 里的现有馆藏书名转告用户，
      并建议他确认书名或走「图书采购推荐」；不要编造索书号和位置。
    - `NO_COPY_AVAILABLE`：馆藏有这本书但都已借出，说明 `total` 册全部在外，
      可等归还后再借或到服务台登记预约。
    - `ACCOUNT_FROZEN`：用户有逾期未还或欠费，账户停借。把 `overdue_items` 和 `fine_due`
      如实说出来，先解决这些才能再借。
12. **借出与归还是实物事件，你无权完成，措辞必须准确：**
    - `action="borrow"` 只做**预借**：返回 `RESERVED` 时要说"已为你预借，请在某日期前
      到自助借书机扫册条码取书"，**不能说"借阅成功"或"已借出"**；册号和取书期限要转告。
    - `action="return_request"` 只做**登记**：返回 `PENDING_RETURN` 时要说"已登记归还，
      请把书放进自助还书机（册号 X）"，**不能说"已归还"或"已还成功"**。
    - 真正销账只能由设备完成。用户问"还成功了吗"，若记录仍是 `pending_return`，就直说
      "还没过机，账面仍是待还"。
    - 用户不知道册号时，先调 `action="list"` 查，别让用户自己去翻。
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
    # 本次会话已识别的学号；空串表示尚未认证（个人业务会先挂起索要）
    user_id: str


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


# ======================= 历史修复 =======================
# OpenAI / DeepSeek 协议要求：每条带 tool_calls 的 assistant 消息，后面必须紧跟
# 对应每个 tool_call_id 的 tool 消息，否则整个请求直接 400：
#   "An assistant message with 'tool_calls' must be followed by tool messages..."
# 图里有两处会产生这种「悬空 tool_calls」：
#   1. interrupt 挂起期间（等学号/等确认）用户又发了一条新消息，human 消息被追加到
#      AI(tool_calls) 后面，这轮 agent 再调模型就带着一段不合法的历史；
#   2. 工具轮数超限或 LLM 报错走 fallback 时，tool_calls 从未被执行。
# 入口层已经拦了第 1 种，这里再兜一层：只要历史不合规就补占位结果，保证任何
# 入口（命令行、评测脚本、未来的新接口）都不会因历史残缺而整段会话不可用。
ABORTED_RESULT = {
    "ok": False,
    "code": "ABORTED",
    "msg": "该工具调用未执行：上一轮在等待用户响应时被打断，请重新发起",
}


def _aborts_for(calls: list[dict]) -> list[ToolMessage]:
    """为没有拿到结果的 tool_call 生成占位 ToolMessage。"""
    return [
        ToolMessage(
            content=json.dumps(ABORTED_RESULT, ensure_ascii=False),
            tool_call_id=call["id"],
        )
        for call in calls
        if call.get("id")
    ]


def _sanitize_history(messages: list[AnyMessage]) -> list[AnyMessage]:
    """把历史整理成协议合法的形状：悬空的 tool_calls 就地补上占位 tool 消息。

    补的位置必须在对应 assistant 消息之后、下一条消息之前——只追加到末尾同样非法。
    """
    out: list[AnyMessage] = []
    pending: list[dict] = []          # 上一条 AI 消息里尚未得到应答的调用

    for message in messages:
        is_tool = getattr(message, "type", "") == "tool"
        if pending and not is_tool:
            # 下一条已经不是 tool 了，说明上面那批调用永远等不到结果，先补齐
            out.extend(_aborts_for(pending))
            pending = []
        out.append(message)

        if is_tool:
            done_id = getattr(message, "tool_call_id", None)
            pending = [c for c in pending if c.get("id") != done_id]
        calls = getattr(message, "tool_calls", None) or []
        if calls:
            pending = list(calls)

    if pending:
        out.extend(_aborts_for(pending))
    return out


# ======================= 节点定义 =======================
def agent_node(state: AgentState, llm_with_tools: ChatDeepSeek) -> dict:
    """agent 节点：把 system prompt 和对话历史一起交给绑定工具的 LLM。

    system prompt 末尾附上当前会话学号，让模型知道“我”是谁；但身份不会作为
    工具参数暴露给它，避免被对话内容诱导伪造。
    """
    user_id = state.get("user_id", "")
    identity = f"学号 {user_id}" if user_id else "尚未识别（办理个人业务时会先要求输入学号）"
    system_prompt = f"{SYSTEM_PROMPT}\n## 六、当前会话用户\n{identity}\n"
    try:
        response = llm_with_tools.invoke(
            [{"role": "system", "content": system_prompt}, *_sanitize_history(state["messages"])]
        )
    except Exception as exc:  # 网络 / API 失败：不抛异常，交给 fallback 节点兜底
        return {"error": f"{type(exc).__name__}: {exc}"}
    return {"messages": [response], "error": ""}


def tools_node(state: AgentState, tool_node: ToolNode, config: RunnableConfig) -> dict:
    """tools 节点：执行上一条 AI 消息里的所有 tool_calls，结果作为 ToolMessage 追加。

    state 里的 user_id 会塞进 configurable：工具从 RunnableConfig 取值，
    所以身份永远来自会话状态，模型无法通过工具参数改写。
    """
    merged = dict(config or {})
    merged["configurable"] = {
        **(merged.get("configurable") or {}),
        "user_id": state.get("user_id", ""),
    }
    result = tool_node.invoke(state, merged)
    # 轮数 +1，供路由函数判断是否达到上限
    return {**result, "tool_rounds": state.get("tool_rounds", 0) + 1}


def fallback_node(state: AgentState) -> dict:
    """fallback 节点：兜底回复，保证图总能正常走到 END，而不是让程序崩溃。

    收尾前先把最后一条 AI 消息里没执行的 tool_calls 补上占位结果：否则这段历史
    被存进 checkpoint，下一轮再调模型就是一段非法历史，DeepSeek 会直接 400。
    """
    error = state.get("error", "")
    if error:
        # LLM 调用失败（网络 / API 问题），给一条可读的错误回复
        reply = f"运行出错：{error}"
    else:
        # agent ↔ tools 循环超过上限仍未给出最终回答
        reply = f"抱歉，这个问题处理了 {MAX_TOOL_ROUNDS} 轮工具调用仍未完成，请换个问法再试。"

    repair: list[ToolMessage] = []
    last = state["messages"][-1] if state["messages"] else None
    calls = getattr(last, "tool_calls", None) or []
    if calls:
        answered = {
            getattr(m, "tool_call_id", None)
            for m in state["messages"]
            if getattr(m, "type", "") == "tool"
        }
        repair = _aborts_for([c for c in calls if c.get("id") not in answered])

    return {"messages": [*repair, AIMessage(content=reply)], "error": ""}


# ======================= 个人业务：身份认证 + 写操作确认 =======================
PERSONAL_TOOLS = {"borrow_manage_tool"}
# 预借与续借会改动用户可见的承诺（占一本复本 / 延长借期），先确认；
# return_request 只是登记归还不销账，多问一次反而累赘，不确认。
WRITE_ACTIONS = {"borrow", "renew"}


def _is_write_call(call: dict) -> bool:
    """只有借 / 续 / 还才算写操作；list 是查询，不该弹确认。"""
    if call.get("name") != "borrow_manage_tool":
        return False
    return (call.get("args") or {}).get("action") in WRITE_ACTIONS


def identify_node(state: AgentState) -> dict:
    """第一道门：个人业务必须先确认「你是谁」，知识问答直接放行。

    判断依据是「有没有调用个人工具」，而不是「这句话像不像个人问题」：
    模型想借书就必须调 borrow_manage_tool，调它就必然经过这里，绕不过去。
    """
    if state.get("user_id"):
        return {}                      # 本次会话已认证，不再重复询问

    last = state["messages"][-1]
    calls = getattr(last, "tool_calls", None) or []
    if not any(c.get("name") in PERSONAL_TOOLS for c in calls):
        return {}                      # 纯知识问答，直接放行

    answer = interrupt({
        "type": "identify",
        "ask": "这是个人借阅业务，请先输入你的学号",
        "detail": "学号仅用于区分借阅记录；本次会话内不会重复询问",
    })
    if isinstance(answer, dict):        # 兼容 {"user_id": ...} 形式的恢复值
        answer = answer.get("user_id", "")
    return {"user_id": str(answer or "").strip()}


def approve_node(state: AgentState) -> dict:
    """第二道门：写操作前人工确认。查询类工具调用直接放行。

    interrupt 会挂起图，等 Command(resume=...)。没有 checkpointer 会直接报错。
    """
    last = state["messages"][-1]
    calls = getattr(last, "tool_calls", None) or []
    write_calls = [c for c in calls if _is_write_call(c)]
    if not write_calls:
        return {}                      # 没有写操作（含 list 查询），放行

    detail = "；".join(f"{c['name']}({c.get('args')})" for c in write_calls)
    decision = interrupt({
        "type": "confirm",
        "ask": "即将执行写操作，是否确认？",
        "detail": detail,
        "options": ["yes", "no"],
    })
    if isinstance(decision, dict):     # 兼容 {"approved": True} 形式的恢复值
        decision = "yes" if decision.get("approved") else "no"

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
def route_after_agent(state: AgentState) -> Literal["identify", "fallback", "__end__"]:
    """agent 节点之后的分诊台：决定下一步去 identify、fallback 还是结束。"""
    if state.get("error"):
        return "fallback"

    last = state["messages"][-1]
    has_tool_calls = bool(getattr(last, "tool_calls", None))
    if has_tool_calls:
        if state.get("tool_rounds", 0) >= MAX_TOOL_ROUNDS:
            # 达到工具轮数上限：不再执行工具，去 fallback 收尾
            return "fallback"
        # 先过身份门，再过确认门
        return "identify"
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

    def run_tools(state: AgentState, config: RunnableConfig) -> dict:
        """包一层，把 config 透传给 tools_node，用于注入 user_id。"""
        return tools_node(state, tool_node, config)

    builder = StateGraph(AgentState)
    builder.add_node("agent", lambda state: agent_node(state, llm_with_tools))
    builder.add_node("identify", identify_node)
    builder.add_node("approve", approve_node)
    builder.add_node("tools", run_tools)
    builder.add_node("fallback", fallback_node)

    builder.add_edge(START, "agent")
    builder.add_conditional_edges(
        "agent", route_after_agent,
        {"identify": "identify", "fallback": "fallback", END: END},
    )
    builder.add_edge("identify", "approve")
    builder.add_conditional_edges(
        "approve", route_after_approve,
        {"tools": "tools", "agent": "agent"},
    )
    builder.add_edge("tools", "agent")
    builder.add_edge("fallback", END)

    return builder.compile(checkpointer=MemorySaver())   # interrupt 必须
