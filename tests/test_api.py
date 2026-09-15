"""Public API tests, driven through the real ASGI app.

The backend is the fake llama-server, so these exercise the whole path: app →
proxy → a real child process on a real port.
"""

from __future__ import annotations

import json

import httpx
import pytest

from huddle.app import create_app
from huddle.config import HuddleConfig

CHAT_BODY = {"model": "tiny", "messages": [{"role": "user", "content": "hi"}]}


async def test_health_reports_loaded_model(client: httpx.AsyncClient) -> None:
    response = await client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model"] == "tiny.gguf"
    assert body["node"] == "testnode"


async def test_models_endpoint_proxies_backend(client: httpx.AsyncClient) -> None:
    response = await client.get("/v1/models")
    assert response.status_code == 200
    assert response.json()["data"][0]["id"] == "tiny"


async def test_chat_completion(client: httpx.AsyncClient) -> None:
    response = await client.post("/v1/chat/completions", json=CHAT_BODY)
    assert response.status_code == 200
    content = response.json()["choices"][0]["message"]["content"]
    assert content == "huddle fake response"


async def test_chat_completion_streams_sse(client: httpx.AsyncClient) -> None:
    """Tokens must arrive as SSE events, not as one buffered blob."""
    chunks: list[str] = []
    async with client.stream(
        "POST", "/v1/chat/completions", json={**CHAT_BODY, "stream": True}
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        async for line in response.aiter_lines():
            if line.startswith("data: "):
                chunks.append(line.removeprefix("data: "))

    assert chunks[-1] == "[DONE]"
    deltas = [json.loads(c)["choices"][0]["delta"]["content"] for c in chunks[:-1]]
    assert "".join(deltas).strip() == "huddle fake response"
    assert len(deltas) > 1, "streaming collapsed into a single chunk"


async def test_invalid_json_rejected(client: httpx.AsyncClient) -> None:
    response = await client.post(
        "/v1/chat/completions",
        content=b"not json",
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 400


async def test_requires_api_key_when_configured(huddle_config: HuddleConfig) -> None:
    huddle_config.api.api_key = "s3cret"
    app = create_app(huddle_config)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://huddle.test") as http:
            unauthorized = await http.post("/v1/chat/completions", json=CHAT_BODY)
            assert unauthorized.status_code == 401

            authorized = await http.post(
                "/v1/chat/completions",
                json=CHAT_BODY,
                headers={"Authorization": "Bearer s3cret"},
            )
            assert authorized.status_code == 200


async def test_503_when_backend_is_not_running(huddle_config: HuddleConfig) -> None:
    """A stopped backend must say so, not hang or 500."""
    huddle_config.backend.autostart = False
    app = create_app(huddle_config)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://huddle.test") as http:
            response = await http.post("/v1/chat/completions", json=CHAT_BODY)
            assert response.status_code == 503
            assert "not running" in response.json()["detail"]


@pytest.mark.parametrize("path", ["/v1/completions", "/v1/embeddings"])
async def test_other_openai_routes_round_trip(client: httpx.AsyncClient, path: str) -> None:
    """Both routes must reach the backend and return its real response."""
    response = await client.post(path, json={"model": "tiny", "prompt": "hi", "input": "hi"})
    assert response.status_code == 200
    assert response.json()
