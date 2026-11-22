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

import re
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from langgraph.types import Command
from pydantic import BaseModel, Field

from agent_builder import build_agent
from main import content_to_text
from middleware import request_check_middleware
from rag_module import ensure_index
import station

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


class IdentifyRequest(BaseModel):
    session_id: str = Field(default="web-demo", max_length=64)
    user_id: str = Field(default="", max_length=32)


class StationRequest(BaseModel):
    """自助设备扫码请求：只认册条码，不认学号——现实里的机器就是这么工作的。"""
    barcode: str = Field(min_length=3, max_length=16)


# 学号只做最基本的形态校验：4-20 位数字/字母。
# 这不是身份认证，只是挡住空值和明显非法输入进入会话状态。
IDENTITY_PATTERN = re.compile(r"[0-9A-Za-z]{4,20}")


def _config(session_id: str) -> dict:
    return {"configurable": {"thread_id": session_id}}


# 挂起时用户完全可以直接在输入框里回一句话，这里认这些说法
YES_WORDS = {"yes", "y", "ok", "确认", "确认执行", "是", "好", "好的"}
NO_WORDS = {"no", "n", "取消", "否", "不用", "算了"}


def _pending(agent, session_id: str) -> tuple[dict, dict]:
    """读 checkpoint：这张图是否正停在一个 interrupt 上。

    返回 (挂起载荷, 当前状态值)。没有挂起时载荷为 {}。
    用途：interrupt 期间不能再往图里灌新的 human 消息，否则那条消息会接在
    AI(tool_calls) 后面，模型收到的历史就不合法（DeepSeek 400）。
    """
    try:
        snapshot = agent.get_state(_config(session_id))
    except Exception:  # 线程不存在 / 检查点读失败：当作没有挂起
        return {}, {}
    for item in getattr(snapshot, "interrupts", ()) or ():
        return (item.value or {}), (getattr(snapshot, "values", {}) or {})
    return {}, (getattr(snapshot, "values", {}) or {})


def _resume(agent, session_id: str, value) -> dict:
    return _shape(agent.invoke(Command(resume=value), config=_config(session_id)))


def _pending_hint(payload: dict, values: dict) -> dict:
    """挂起期间收到不相干的输入：不碰图，直接告诉用户先把这一步处理完。"""
    kind = payload.get("type")
    user_id = values.get("user_id", "")
    if kind == "identify":
        answer = (
            "这条消息我先记下了，但现在还没确认你是哪位——"
            "请在上面的学号卡片里填入学号（也可以直接回一句话输入学号），我再继续办理。"
        )
    elif kind == "confirm":
        answer = (
            "上一个操作还在等你确认。请在卡片上点「确认执行」或「取消」，"
            "也可以直接回复「确认」或「取消」。"
        )
    else:
        answer = "上一步还没有处理完，请先完成页面上的提示。"
    return {"status": "ok", "answer": answer, "user_id": user_id}


def _shape(result: dict) -> dict:
    """把图的返回值统一成前端能直接渲染的几种状态。"""
    user_id = result.get("user_id", "")

    if "__interrupt__" in result:
        payload = result["__interrupt__"][0].value or {}
        if payload.get("type") == "identify":
            return {
                "status": "identify",
                "ask": payload.get("ask", "请先输入你的学号"),
                "detail": payload.get("detail", ""),
                "user_id": user_id,
            }
        return {
            "status": "confirm",
            "ask": payload.get("ask", "即将执行写操作，是否确认？"),
            "detail": payload.get("detail", ""),
            "user_id": user_id,
        }

    messages = result.get("messages", [])
    answer = content_to_text(messages[-1].content) if messages else ""
    return {
        "status": "ok",
        "answer": answer or "（没有返回内容）",
        "user_id": user_id,
    }


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

    # 图还停在 interrupt 上时，绝不能把这条消息当新输入塞进去：
    # 它会被追加到 AI(tool_calls) 之后，模型看到的历史就不合法（400 insufficient
    # tool messages）。这里改成「把话当答复」或「提示先处理完」。
    payload, values = _pending(agent, req.session_id)
    if payload:
        text = req.message.strip()
        kind = payload.get("type")
        if kind == "identify" and IDENTITY_PATTERN.fullmatch(text):
            return _resume(agent, req.session_id, text)
        if kind == "confirm" and text.lower() in YES_WORDS | NO_WORDS:
            return _resume(agent, req.session_id, "yes" if text.lower() in YES_WORDS else "no")
        return _pending_hint(payload, values)

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

    payload, values = _pending(agent, req.session_id)
    if payload.get("type") != "confirm":
        # 没有待确认的写操作（重复点击 / 页面放太久）：别去 resume 一个不存在的挂起点
        return {"status": "ok", "answer": "当前没有待确认的操作，直接发消息就行。",
                "user_id": values.get("user_id", "")}

    return _resume(agent, req.session_id, "yes" if req.approved else "no")


@app.post("/api/identify")
def identify(req: IdentifyRequest) -> dict:
    """用户提交学号：恢复被 identify 挂起的图。格式不对在进图前就挡掉。"""
    user_id = req.user_id.strip()
    if not IDENTITY_PATTERN.fullmatch(user_id):
        return {"status": "error", "answer": "学号格式不对：应为 4-20 位数字或字母，例如 2023001"}

    try:
        agent = get_agent()
    except Exception as exc:
        return {"status": "error", "answer": _explain(exc)}

    payload, values = _pending(agent, req.session_id)
    if payload.get("type") != "identify":
        return {"status": "ok", "answer": "身份已确认，直接说你要办什么就行。",
                "user_id": values.get("user_id", "")}

    return _resume(agent, req.session_id, user_id)


# ======================= 设备端点（不是 LLM 工具） =======================
# 现实里借出与归还由自助机完成：机器只扫册条码，不对话。
# 这几个接口单独暴露给前端的「设备模拟」面板，模型无法调用，也就无法凭一句话销账。
@app.post("/api/station/borrow")
def station_borrow(req: StationRequest) -> dict:
    return station.scan_borrow(req.barcode)


@app.post("/api/station/return")
def station_return(req: StationRequest) -> dict:
    return station.scan_return(req.barcode)


@app.get("/api/station/pending")
def station_pending(user_id: str = "") -> dict:
    """给前端面板用：这个学号有哪些「已登记待还」和「已预借待取」的书。"""
    uid = user_id.strip()
    if not IDENTITY_PATTERN.fullmatch(uid):
        return {"ok": False, "msg": "请先确定当前学号"}
    return {
        "ok": True,
        "user": uid,
        "pending_return": station.pending_returns(uid),
        "reserved": station.reserved_items(uid),
    }
