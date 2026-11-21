# middleware.py
"""输入校验与调用日志中间件。"""
from __future__ import annotations

import time
from functools import wraps
from typing import Callable, TypeVar

from config import DEBUG, MAX_INPUT_LENGTH

F = TypeVar("F", bound=Callable)


def log_middleware(func: F) -> F:
    """装饰器：打印函数调用耗时（只在 DEBUG 打开时输出）。"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            if DEBUG:
                print(f"[中间件] {func.__name__} 耗时 {time.perf_counter() - start:.2f}s")

    return wrapper  # type: ignore[return-value]


def request_check_middleware(user_input: str) -> tuple[bool, str]:
    """过滤空输入和超长输入。"""
    user_input = (user_input or "").strip()
    if not user_input:
        return False, "输入不能为空，请重新输入"
    if len(user_input) > MAX_INPUT_LENGTH:
        return False, f"输入过长（{len(user_input)} 字），请控制在 {MAX_INPUT_LENGTH} 字以内"
    return True, ""
