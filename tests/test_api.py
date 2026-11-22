"""HTTP 接口冒烟测试：只覆盖不触发模型调用的路径，可在 CI 里无密钥跑。"""
from __future__ import annotations

from fastapi.testclient import TestClient

import api

client = TestClient(api.app)


def test_health_ok():
    body = client.get("/api/health").json()

    assert body["ok"] is True
    assert "agent_ready" in body


def test_index_serves_page():
    response = client.get("/")

    assert response.status_code == 200
    assert "图书馆智能 Agent" in response.text


def test_blank_message_rejected_before_model_call():
    response = client.post("/api/chat", json={"session_id": "t", "message": "   "})

    assert response.json()["status"] == "error"
