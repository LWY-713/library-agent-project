# api.py
"""把命令行 Agent 包成 HTTP 服务，供网页 demo 调用。

启动：
    uvicorn api:app --reload --port 8000
然后浏览器打开 http://127.0.0.1:8000

设计要点：
- 一个进程内只编译一次图（build_agent），用 thread_id 区分会话；
  MemorySaver 是进程内检查点，所以 uvicorn 必须单 worker 跑，
  多 worker 会把同一个会话的 interrupt 状态拆到不同进程里，恢复不了。
- 写操作走 interrupt：/api/chat 返回 status=confirm，前端确认后调
  /api/confirm，后端用 Command(resume=...) 继续跑同一张图。
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from langgraph.types import Command
from pydantic import BaseModel, Field

from agent_builder import build_agent
from main import content_to_text
from middleware import request_check_middleware
from rag_module import ensure_index

WEB_DIR = Path(__file__).resolve().parent / "web"

app = FastAPI(title="图书馆智能 Agent", version="1.0")

_agent = None


def get_agent():
    """懒加载：第一次请求时才连向量库、建图，避免 import 阶段就依赖密钥。"""
    global _agent
    if _agent is None:
        ensure_index(verbose=False)
        _agent = build_agent()
    return _agent


class ChatRequest(BaseModel):
    session_id: str = Field(default="web-demo", max_length=64)
    message: str = ""


class ConfirmRequest(BaseModel):
    session_id: str = Field(default="web-demo", max_length=64)
    approved: bool = False


def _config(session_id: str) -> dict:
    return {"configurable": {"thread_id": session_id}}


def _shape(result: dict) -> dict:
    """把图的返回值统一成前端能直接渲染的三种状态。"""
    if "__interrupt__" in result:
        payload = result["__interrupt__"][0].value or {}
        return {
            "status": "confirm",
            "ask": payload.get("ask", "即将执行写操作，是否确认？"),
            "detail": payload.get("detail", ""),
        }
    messages = result.get("messages", [])
    answer = content_to_text(messages[-1].content) if messages else ""
    return {"status": "ok", "answer": answer or "（没有返回内容）"}


def _explain(exc: Exception) -> str:
    """把初始化异常翻译成用户能看懂的一句话。"""
    text = f"{type(exc).__name__}: {exc}"
    if "LOCK" in text or "Permission denied" in text:
        return (
            "向量库被另一个进程占用。milvus-lite 是单进程独占的本地库，"
            "通常是还开着命令行 main.py 的会话——请先关掉它，再刷新本页重试。"
        )
    return f"初始化失败：{text}"


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "agent_ready": _agent is not None}


@app.post("/api/chat")
def chat(req: ChatRequest) -> dict:
    ok, message = request_check_middleware(req.message)
    if not ok:
        return {"status": "error", "answer": message}

    try:
        agent = get_agent()
    except Exception as exc:  # 向量库被占用 / 未配置密钥等
        return {"status": "error", "answer": _explain(exc)}

    result = agent.invoke(
        {"messages": [{"role": "user", "content": req.message}]},
        config=_config(req.session_id),
    )
    return _shape(result)


@app.post("/api/confirm")
def confirm(req: ConfirmRequest) -> dict:
    try:
        agent = get_agent()
    except Exception as exc:
        return {"status": "error", "answer": _explain(exc)}

    result = agent.invoke(
        Command(resume="yes" if req.approved else "no"),
        config=_config(req.session_id),
    )
    return _shape(result)
