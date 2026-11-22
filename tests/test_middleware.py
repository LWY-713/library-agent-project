"""输入校验中间件测试：空输入与超长输入必须在进模型前被挡掉。"""
from __future__ import annotations

from config import MAX_INPUT_LENGTH
from middleware import request_check_middleware


def test_empty_input_rejected():
    assert request_check_middleware("")[0] is False
    assert request_check_middleware("   \n ")[0] is False


def test_too_long_input_rejected():
    ok, message = request_check_middleware("啊" * (MAX_INPUT_LENGTH + 1))

    assert ok is False
    assert str(MAX_INPUT_LENGTH) in message


def test_normal_input_passes():
    assert request_check_middleware("图书馆几点关门")[0] is True
