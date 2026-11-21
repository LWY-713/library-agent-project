# main.py
"""项目主入口：命令行里和图书馆 Agent 对话。"""
from __future__ import annotations

import sys

from langchain_core.messages import AIMessageChunk
from langgraph.types import Command

from agent_builder import build_agent
from config import DEBUG, STREAM_OUTPUT
from middleware import log_middleware, request_check_middleware
from rag_module import EmbeddingNotConfigured, ensure_index


def content_to_text(content) -> str:
    """把 AIMessage.content 统一转成字符串（可能是 str，也可能是 content block 列表）。"""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(str(block.get("text", "")))
    return "\n".join(part for part in parts if part)


@log_middleware
def run_agent(agent, user_input: str):
    # 显式 StateGraph 与 create_agent 的输入格式一致：messages 列表
    return agent.invoke({"messages": [{"role": "user", "content": user_input}]})


def run_with_approval(agent, user_input: str, config: dict) -> str:
    """跑一轮对话；遇到写操作确认就停下来问用户，再恢复。"""
    result = agent.invoke(
        {"messages": [{"role": "user", "content": user_input}]}, config=config
    )

    while "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        print(f"\n⚠️  {payload['ask']}")
        print(f"    {payload['detail']}")
        answer = input("    确认执行？(yes/no) ").strip().lower()
        if answer not in {"yes", "no"}:
            answer = "no"
        result = agent.invoke(Command(resume=answer), config=config)

    messages = result.get("messages", [])
    return content_to_text(messages[-1].content) if messages else ""


@log_middleware
def run_agent_stream(agent, user_input: str) -> str:
    """流式运行：stream_mode="messages" 会把图中每一步产生的消息块实时吐出来。

    - AI 消息块：逐 token 实时打印（这就是可控的流式输出）
    - AI 消息带 tool_calls：打印工具调用事件（DEBUG 时）
    - ToolMessage 块：工具结果，只在 DEBUG 时打印前 100 字预览
    返回拼接出的最终回答文本。
    """
    printed = ""  # 累计已经实时打印的内容（最终回答 = 最后一轮模型输出）
    seen_calls = set()  # 已打印过的工具调用 id：流式下 tool_calls 分片到达，需去重
    print("🤖 Agent：", end="", flush=True)

    for chunk, _metadata in agent.stream(
        {"messages": [{"role": "user", "content": user_input}]},
        stream_mode="messages",
    ):
        # 注意：流式模式下 AI 输出的类型是 AIMessageChunk（其 .type 也是
        # "AIMessageChunk" 而非 "ai"），所以用 isinstance 判断更可靠
        if isinstance(chunk, AIMessageChunk):
            # 工具调用事件：模型决定要调工具时先在这里出现。
            # 只在参数解析完整（args 非空）且没打印过时打印一次
            for call in getattr(chunk, "tool_calls", None) or []:
                call_id = call.get("id") or ""
                if call.get("name") and call.get("args") and call_id not in seen_calls:
                    seen_calls.add(call_id)
                    if DEBUG:
                        print(f"\n   🔧 调用工具 {call.get('name')}，参数 {call.get('args')}")

            # 模型正在生成的文本：逐块实时打印
            text = content_to_text(chunk.content)
            if text:
                print(text, end="", flush=True)
                printed += text

        elif getattr(chunk, "type", "") == "tool" and DEBUG:
            text = content_to_text(chunk.content)
            if text:
                print(f"\n   📄 工具返回（前 100 字）：{text[:100]}...")

    print()  # 流式内容结束后换行
    return printed


def print_tool_calls(messages) -> None:
    for message in messages:
        for call in getattr(message, "tool_calls", None) or []:
            print(f"   🔧 调用工具 {call.get('name')}，参数 {call.get('args')}")


def main() -> int:
    print("===== 📚 图书馆智能 RAG Agent =====")

    try:
        if ensure_index(verbose=DEBUG):
            print("✅ 首次运行，已自动建库")
        else:
            print("✅ 已加载已有向量库")
    except EmbeddingNotConfigured as exc:
        print(f"❌ {exc}")
        return 1
    except Exception as exc:
        print(f"❌ 初始化向量库失败：{exc}")
        return 1

    try:
        agent = build_agent()
        config = {"configurable": {"thread_id": "cli-session-1"}}
    except Exception as exc:
        print(f"❌ 初始化对话模型失败：{exc}")
        return 1
    print("输入问题开始对话，输入 exit / quit / q 退出。\n")

    while True:
        try:
            user_input = input("👤 你：")
        except (EOFError, KeyboardInterrupt):
            print("\n👋 退出程序")
            break

        if user_input.strip().lower() in {"exit", "quit", "q"}:
            print("👋 退出程序")
            break

        ok, message = request_check_middleware(user_input)
        if not ok:
            print(f"⚠️ {message}")
            continue

        try:
            if STREAM_OUTPUT:
                # 流式模式：回答边生成边打印（内部已换行），不再单独打印最终结果
                answer = run_agent_stream(agent, user_input)
                if not answer.strip():
                    print("🤖 Agent：（没有返回内容）")
                continue

            answer = run_with_approval(agent, user_input, config)
        except Exception as exc:
            print(f"❌ 运行出错：{exc}")
            continue

        print(f"🤖 Agent：{answer or '（没有返回内容）'}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
