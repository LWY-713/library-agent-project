# tools.py
"""对外暴露给 Agent 的工具（结构化返回：ok / code / data）。

多用户设计：
- 台账按「学号 → 书名 → 记录」两层存储，用户之间互不可见，同名图书可被不同人各借一本；
- 当前学号通过 RunnableConfig 注入。工具签名里虽然带 ``config``，但该参数不会出现在
  工具 schema 中（模型看不到），所以身份只能来自会话状态，无法从对话内容里伪造；
- 借/续/还都会校验记录归属，操作别人的书会被 NOT_OWNED 拦下。

权限边界（关键设计）：
- Agent 只能做「无实物依赖」的动作：查询、预借、续借、**登记归还**。
- 借出与销账必须由设备（自助借书机 / 还书机，见 station.py）扫册条码触发。
  现实里书还在读者手上时，系统不能把记录改成已归还——所以本模块的 ``return_request``
  只把记录置为 ``pending_return``，真正销账在 ``station.scan_return``。
"""
from __future__ import annotations

import json
import re
from datetime import date, timedelta
from pathlib import Path
from typing import Literal

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from config import BOOKS_FILE, CURRENT_USER_ID, LOANS_FILE, SEARCH_MIN_SCORE, TOP_K
from rag_module import search_knowledge

# ---------------------------------------------------------------- 状态机
RESERVED = "reserved"              # 已预借，等待到馆取书
BORROWED = "borrowed"              # 已借出（设备确认后）
PENDING_RETURN = "pending_return"  # 已登记归还，书还没过机
RETURNED = "returned"              # 已销账（设备确认后）
CANCELLED = "cancelled"            # 预借超期未取，自动作废

# 这些状态占用一本实物副本
OCCUPYING = {RESERVED, BORROWED, PENDING_RETURN}

PICKUP_DAYS = 3          # 预借后保留天数
LOAN_DAYS = 30           # 借期
RENEW_DAYS = 30          # 单次续借延长天数
FINE_PER_DAY = 0.1       # 逾期罚款（元/天，本科生档）
FINE_CAP = 10.0          # 单本上限


def _err(code: str, msg: str, **extra) -> str:
    payload = {"ok": False, "code": code, "msg": msg, **extra}
    return json.dumps(payload, ensure_ascii=False)


def _current_user(config: RunnableConfig | None) -> str:
    """从运行时上下文取当前用户学号；取不到返回空串。"""
    if not config:
        return ""
    configurable = config.get("configurable") or {}
    return str(configurable.get("user_id") or "").strip()


def _status(record: dict) -> str:
    """取记录状态；老数据只有 returned 布尔值，按它推断。"""
    status = record.get("status")
    if status:
        return status
    return RETURNED if record.get("returned") else BORROWED


def _overdue_days(record: dict) -> int:
    due = record.get("due_date")
    if not due:
        return 0
    return max(0, (date.today() - date.fromisoformat(due)).days)


def _fine_of(record: dict) -> float:
    return round(min(_overdue_days(record) * FINE_PER_DAY, FINE_CAP), 2)


@tool
def library_search_tool(query: str) -> str:
    """查询校园中心图书馆知识库。

    只要问题与图书馆有关（开放时间、馆藏图书、书架位置、借阅规则、研讨室预约、
    活动、设备报修、失物招领等），就应该调用本工具，再依据返回片段回答。

    返回 JSON：
    - {"ok": true,  "code": "OK",              "data": [{index, chunk_id, score, text}, ...]}
    - {"ok": false, "code": "NO_HIT",          "hint": "换说法重试的建议"}
    - {"ok": false, "code": "SEARCH_ERROR",    "msg": "..."}

    Args:
        query: 检索词。用书名全称、正式术语，不要原样照抄用户口语句子。
    """
    try:
        hits = search_knowledge(query, limit=TOP_K)
    except Exception as exc:
        return _err("SEARCH_ERROR", f"检索失败：{exc}")

    if not hits:
        return _err(
            "NO_HIT", "知识库没有召回任何片段",
            hint="换一种说法再检索：用书名全称代替外号，用正式术语代替口语，或补充作者/学科/楼层",
        )

    top_score = max(hit["score"] for hit in hits)
    if SEARCH_MIN_SCORE > 0 and top_score < SEARCH_MIN_SCORE:
        return _err(
            "LOW_RELEVANCE", "召回结果相关度过低",
            top_score=round(top_score, 4),
            hint="更换检索词后重试",
        )

    data = [
        {
            "index": i,
            "chunk_id": hit["chunk_id"],
            "score": round(hit["score"], 4),
            "text": hit["text"],
        }
        for i, hit in enumerate(hits, start=1)
    ]
    return json.dumps({"ok": True, "code": "OK", "data": data}, ensure_ascii=False)


# ---------------------------------------------------------------- 台账读写
def _load_loans() -> dict:
    """读台账，统一成 {学号: {书名: 记录}}。

    兼容两代旧格式：
    - 扁平格式 {书名: {user: 学号, ...}}：按 user 字段拆到各学号下；
    - 无 status/copy 的记录：补 status，并分配册条码。
    """
    path = Path(LOANS_FILE)
    if not path.exists():
        return {}

    raw = json.loads(path.read_text(encoding="utf-8"))
    if not raw:
        return {}

    if all(isinstance(rec, dict) and "user" in rec for rec in raw.values()):
        migrated: dict[str, dict] = {}
        for book, rec in raw.items():
            record = dict(rec)
            owner = str(record.pop("user", "") or CURRENT_USER_ID)
            migrated.setdefault(owner, {})[book] = record
        raw = migrated

    seq = 0
    for user_id, books in raw.items():
        if not isinstance(books, dict):
            continue
        for book, record in books.items():
            if not isinstance(record, dict):
                continue
            record.setdefault("status", RETURNED if record.get("returned") else BORROWED)
            if not record.get("copy"):
                seq += 1
                record["copy"] = f"B{seq:04d}"
                record["_dirty"] = True     # 下次保存时落盘
    return raw


def _save_loans(data: dict) -> None:
    for books in data.values():
        if not isinstance(books, dict):
            continue
        for record in books.values():
            if isinstance(record, dict):
                record.pop("_dirty", None)
                # 保留 returned 布尔值，老接口/老数据仍可读
                record["returned"] = _status(record) == RETURNED
    Path(LOANS_FILE).write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def find_by_copy(loans: dict, copy: str) -> tuple[str, str, dict] | None:
    """按册条码定位记录，返回 (学号, 书名键, 记录)。设备扫码走这条路。"""
    target = (copy or "").strip().upper()
    if not target:
        return None
    for user_id, books in loans.items():
        if not isinstance(books, dict):
            continue
        for book, record in books.items():
            if isinstance(record, dict) and str(record.get("copy", "")).upper() == target:
                return user_id, book, record
    return None


def _next_copy(loans: dict) -> str:
    """分配一个新的册条码：全台账最大值 +1。"""
    nums = [
        int(m.group(1))
        for books in loans.values() if isinstance(books, dict)
        for record in books.values() if isinstance(record, dict)
        for m in [re.fullmatch(r"B(\d+)", str(record.get("copy", "")))] if m
    ]
    return f"B{(max(nums) if nums else 0) + 1:04d}"


def _account_block(loans: dict, user_id: str) -> dict | None:
    """停借检查：有逾期未还或欠费未缴，就不给借新书。返回阻断原因或 None。"""
    overdue, owed = [], 0.0
    for book, record in (loans.get(user_id) or {}).items():
        if not isinstance(record, dict):
            continue
        if _status(record) in {BORROWED, PENDING_RETURN} and _overdue_days(record) > 0:
            overdue.append(f"{book}（逾期 {_overdue_days(record)} 天）")
        owed += float(record.get("fine_due") or 0)
    if not overdue and owed <= 0:
        return None
    msg = "你的借阅账户处于停借状态"
    if overdue:
        msg += f"：{'、'.join(overdue)} 尚未归还"
    if owed > 0:
        msg += f"；另有未缴罚款 {round(owed, 2)} 元"
    return {"msg": msg + "。请先归还/缴清后再预借。", "overdue_items": overdue, "fine_due": round(owed, 2)}


# ---------------------------------------------------------------- 馆藏
_BRACKET_RE = re.compile(r"[（(][^）)]*[）)]")


def _norm_title(raw: str) -> str:
    """书名归一化：去书名号、去作者括注、去空白。

    知识库里写《机器学习》（周志华），台账里写《机器学习》，用户还可能只打「机器学习」，
    比较前统一压成「机器学习」，避免同一本书被当成两本。
    """
    text = (raw or "").strip()
    text = _BRACKET_RE.sub("", text)      # 先删作者括注，避免《书》（作者）末尾的 》 被漏掉
    return re.sub(r"\s+", "", text.strip().strip("《》"))


def _load_books() -> list[dict]:
    path = Path(BOOKS_FILE)
    if not path.exists():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    return raw.get("books", []) if isinstance(raw, dict) else raw


def _find_book(title: str) -> dict | None:
    """在馆藏清单里查书，查不到返回 None（= 这本不在馆藏中）。"""
    target = _norm_title(title)
    if not target:
        return None
    for entry in _load_books():
        if _norm_title(entry.get("title", "")) == target:
            return entry
    return None


def _display_title(entry: dict) -> str:
    return f"《{entry['title']}》"


def _occupied(loans: dict, norm: str) -> int:
    """全馆范围内这本书被占用的册数（预借 + 在借 + 待还，都占着实物）。"""
    return sum(
        1
        for books in loans.values() if isinstance(books, dict)
        for book, record in books.items()
        if _norm_title(book) == norm
        and isinstance(record, dict)
        and _status(record) in OCCUPYING
    )


def _held_by_other(loans: dict, user_id: str, norm: str) -> bool:
    """这本书是否正被其他用户占着（不返回是谁，避免泄露他人借阅信息）。"""
    for other_id, books in loans.items():
        if other_id == user_id or not isinstance(books, dict):
            continue
        for book, record in books.items():
            if (_norm_title(book) == norm and isinstance(record, dict)
                    and _status(record) in OCCUPYING):
                return True
    return False


def _match_held(mine: dict, title: str) -> str | None:
    """在本人记录里按归一化书名找已存在的键。"""
    norm = _norm_title(title)
    for book in mine:
        if _norm_title(book) == norm:
            return book
    return None


# ---------------------------------------------------------------- 个人业务
@tool
def borrow_manage_tool(
    action: Literal["borrow", "renew", "return_request", "list"],
    config: RunnableConfig,
    book_title: str = "",
) -> str:
    """办理图书预借 / 续借 / 登记归还，或查询当前用户的借阅清单。

    重要：借出与归还是「实物事件」，本工具不能直接完成。
    - action=borrow：只做预借（在馆藏里锁一本，保留 3 天），用户需到自助借书机取书；
    - action=return_request：只登记归还意向，用户需把书放进自助还书机才算销账；
    - 销账与借出由设备完成，你无法代替。

    仅用于个人借阅业务。查询借阅规则、开放时间、馆藏位置请用 library_search_tool。
    当前用户身份由系统注入，无需也无法指定；若系统还没识别身份，会先要求输入学号。

    Args:
        action: borrow=预借（到馆取书）、renew=续借、return_request=登记归还、list=查询清单
        book_title: 书名，例如《机器学习》；action=list 时留空
    """
    user_id = _current_user(config)
    if not user_id:
        return _err("NEED_IDENTITY", "还没有识别你的学号，请先提供学号再办理个人业务")

    loans = _load_loans()
    mine = loans.setdefault(user_id, {})
    key = book_title.strip()

    if action == "list":
        items = []
        for book, record in mine.items():
            if not isinstance(record, dict):
                continue
            status = _status(record)
            if status in {RETURNED, CANCELLED}:
                continue
            items.append({
                "book": book,
                "copy": record.get("copy"),
                "status": status,
                "borrow_date": record.get("borrow_date"),
                "due_date": record.get("due_date"),
                "pickup_deadline": record.get("pickup_deadline"),
                "renewed": bool(record.get("renewed")),
                "overdue_days": _overdue_days(record),
                "fine_due": round(float(record.get("fine_due") or 0), 2),
            })
        items.sort(key=lambda item: (item["status"], item["due_date"] or ""))
        return json.dumps(
            {"ok": True, "code": "OK",
             "data": {"user": user_id, "count": len(items), "items": items}},
            ensure_ascii=False,
        )

    if not key:
        return _err("BAD_REQUEST", "请告诉我具体是哪本书，例如《机器学习》")

    if action == "borrow":
        entry = _find_book(key)
        if entry is None:
            titles = "、".join(_display_title(b) for b in _load_books())
            return _err(
                "NOT_IN_COLLECTION", f"馆藏里没有{key}，无法办理预借",
                hint=f"现有馆藏：{titles}" if titles else "馆藏清单为空",
            )

        key = _display_title(entry)
        existing = _match_held(mine, key)
        if existing:
            key = existing
            status = _status(mine[key])
            if status == RESERVED:
                return _err(
                    "ALREADY_RESERVED", f"你已经预借了{key}，还没到馆取书",
                    copy=mine[key].get("copy"),
                    pickup_deadline=mine[key].get("pickup_deadline"),
                )
            if status in {BORROWED, PENDING_RETURN}:
                return _err("ALREADY_BORROWED", f"你已借阅{key}，不能重复借阅",
                            copy=mine[key].get("copy"))

        block = _account_block(loans, user_id)
        if block:
            return _err("ACCOUNT_FROZEN", block["msg"],
                        overdue_items=block["overdue_items"], fine_due=block["fine_due"])

        total = int(entry.get("total") or 0)
        occupied = _occupied(loans, _norm_title(key))
        if occupied >= total:
            return _err(
                "NO_COPY_AVAILABLE", f"{key}的 {total} 册已全部借出",
                total=total, occupied=occupied,
                hint="可过几天再试；需要的话可以让馆员帮你登记预约排队",
            )

        copy = _next_copy(loans)
        pickup = date.today() + timedelta(days=PICKUP_DAYS)
        mine[key] = {
            "copy": copy,
            "status": RESERVED,
            "reserved_at": date.today().isoformat(),
            "pickup_deadline": pickup.isoformat(),
            "borrow_date": None,
            "due_date": None,
            "renewed": False,
            "fine_due": 0.0,
            "returned": False,
        }
        _save_loans(loans)
        return json.dumps(
            {"ok": True, "code": "RESERVED",
             "data": {"book": key, "copy": copy,
                      "pickup_deadline": pickup.isoformat(),
                      "location": entry.get("location", ""),
                      "remaining": total - occupied - 1,
                      "next_step": "请在取书期限内到自助借书机扫册条码取书，逾期预借自动作废"}},
            ensure_ascii=False,
        )

    # ---- 以下两个动作都需要本人名下存续的记录 ----
    held = _match_held(mine, key)
    record = mine.get(held) if held else None
    if not record or _status(record) in {RETURNED, CANCELLED}:
        if _held_by_other(loans, user_id, _norm_title(key)):
            return _err("NOT_OWNED", f"{key}不在你的借阅记录中，无法操作")
        return _err("NOT_BORROWED", f"你名下没有借阅中的{key}")
    key = held

    if action == "renew":
        status = _status(record)
        if status == RESERVED:
            return _err("NOT_BORROWED", f"{key}还没取书，预借状态无法续借",
                        copy=record.get("copy"), pickup_deadline=record.get("pickup_deadline"))
        if status == PENDING_RETURN:
            return _err("PENDING_RETURN", f"{key}已登记归还，等还书机确认后才能再续借",
                        copy=record.get("copy"))
        # 返回里必须带上到期日与逾期天数：只给一句“已续借过”时，模型无法判断
        # 这本书是不是已经超期，容易编出“尽快归还以免逾期”这种与事实相反的答复。
        if record.get("renewed"):
            return _err(
                "ALREADY_RENEWED", "每本书只能续借一次，该书已经续借过",
                due_date=record["due_date"], overdue_days=_overdue_days(record),
            )
        if _overdue_days(record) > 0:
            return _err(
                "OVERDUE", f"该书已于 {record['due_date']} 超期，需先归还并缴清罚款",
                due_date=record["due_date"], overdue_days=_overdue_days(record),
                fine_estimate=_fine_of(record),
            )
        record["renewed"] = True
        record["due_date"] = (date.fromisoformat(record["due_date"])
                              + timedelta(days=RENEW_DAYS)).isoformat()
        _save_loans(loans)
        return json.dumps(
            {"ok": True, "code": "RENEWED",
             "data": {"book": key, "due_date": record["due_date"]}},
            ensure_ascii=False,
        )

    # ---- action == "return_request"：只登记，不销账 ----
    status = _status(record)
    if status == RESERVED:
        return _err("NOT_BORROWED",
                    f"{key}是预借状态、书还在预约架上，你现在没有这本书可还",
                    copy=record.get("copy"), pickup_deadline=record.get("pickup_deadline"))
    if status == PENDING_RETURN:
        return _err("ALREADY_PENDING_RETURN", f"{key}已经登记过归还，去还书机扫一下就好",
                    copy=record.get("copy"), pending_since=record.get("pending_since"))

    record["status"] = PENDING_RETURN
    record["pending_since"] = date.today().isoformat()
    _save_loans(loans)
    return json.dumps(
        {"ok": True, "code": "PENDING_RETURN",
         "data": {"book": key, "copy": record.get("copy"),
                  "overdue_days": _overdue_days(record),
                  "fine_estimate": _fine_of(record),
                  "next_step": "把书放进自助还书机（或交给一楼服务台），扫册条码后才算归还完成；"
                               "在此之前记录仍是待还状态"}},
        ensure_ascii=False,
    )
