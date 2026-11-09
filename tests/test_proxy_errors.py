"""When the model server cannot answer, the caller must be told why.

The failure that matters is the race after a worker or head dies: the process
table still says the backend is up, so the request is sent, and the connection
fails. That used to surface as a bare 500 with no body the OpenAI SDKs could
read, and a streamed reply that simply stopped. These tests fail the proxy's
HTTP client in specific ways while the fake backend keeps the cluster "up".
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import AsyncIterator, Callable
from typing import Any

import httpx
import pytest
from fastapi import FastAPI

from huddle.app import create_app
from huddle.config import HuddleConfig
from huddle.coordinator.service import ClusterStatus
from tests.conftest import wait_until_running

CHAT = {"model": "tiny", "messages": [{"role": "user", "content": "hi"}]}
Handler = Callable[[httpx.Request], httpx.Response]


@contextlib.asynccontextmanager
async def running_app(
    config: HuddleConfig, monkeypatch: pytest.MonkeyPatch, handler: Handler
) -> AsyncIterator[tuple[FastAPI, httpx.AsyncClient]]:
    """A running app whose proxy talks to ``handler`` instead of the backend."""
    monkeypatch.setattr(
        "huddle.app.make_client",
        lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    app = create_app(config)
    http = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://huddle.test")
    async with app.router.lifespan_context(app), http:
        await wait_until_running(http)
        yield app, http


def refuse(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError("connection refused", request=request)


def assert_openai_error(response: httpx.Response, code: str) -> dict[str, Any]:
    body: dict[str, Any] = response.json()
    assert body["error"]["code"] == code, body
    assert body["error"]["type"] == "server_error"
    assert body["error"]["message"], body
    # Huddle's own web UI reads `detail`; both must carry the same words.
    assert body["detail"] == body["error"]["message"]
    return body


async def test_a_refused_connection_is_a_502_the_sdks_can_read(
    huddle_config: HuddleConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with running_app(huddle_config, monkeypatch, refuse) as (_, http):
        response = await http.post("/v1/chat/completions", json=CHAT)

    assert response.status_code == 502
    assert_openai_error(response, "upstream_unavailable")
    assert "retry-after" not in response.headers


async def test_a_cluster_that_is_restarting_asks_the_caller_to_retry(
    huddle_config: HuddleConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with running_app(huddle_config, monkeypatch, refuse) as (app, http):
        # Wanted, but not running, and not mid-start: the supervisor is on it.
        degraded = ClusterStatus(running=False, head_node="testnode", desired=True)
        monkeypatch.setattr(app.state.cluster, "status", lambda: degraded)

        response = await http.post("/v1/chat/completions", json=CHAT)

    assert response.status_code == 503
    assert response.headers["retry-after"] == "5"
    assert_openai_error(response, "cluster_recovering")


async def test_a_timeout_is_a_504(
    huddle_config: HuddleConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    def slow(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("no answer", request=request)

    async with running_app(huddle_config, monkeypatch, slow) as (_, http):
        response = await http.post("/v1/chat/completions", json=CHAT)

    assert response.status_code == 504
    assert_openai_error(response, "upstream_timeout")


@pytest.mark.parametrize(
    "error",
    [
        httpx.ConnectError("refused"),
        httpx.ConnectTimeout("slow"),
        httpx.ReadError("reset"),
        httpx.WriteError("broken pipe"),
        httpx.RemoteProtocolError("server disconnected"),
        httpx.PoolTimeout("no connection free"),
    ],
    ids=lambda e: type(e).__name__,
)
@pytest.mark.parametrize("path", ["/v1/chat/completions", "/v1/completions", "/v1/embeddings"])
async def test_no_transport_error_becomes_a_500(
    huddle_config: HuddleConfig,
    monkeypatch: pytest.MonkeyPatch,
    error: httpx.HTTPError,
    path: str,
) -> None:
    def fail(request: httpx.Request) -> httpx.Response:
        raise error

    async with running_app(huddle_config, monkeypatch, fail) as (_, http):
        response = await http.post(path, json=CHAT)

    assert response.status_code in (502, 504), response.text
    assert "error" in response.json()


async def test_listing_models_reports_a_dead_backend_too(
    huddle_config: HuddleConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with running_app(huddle_config, monkeypatch, refuse) as (_, http):
        response = await http.get("/v1/models")

    assert response.status_code == 502
    assert_openai_error(response, "upstream_unavailable")


async def test_a_stream_that_cannot_start_is_a_502_not_an_empty_200(
    huddle_config: HuddleConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with running_app(huddle_config, monkeypatch, refuse) as (_, http):
        response = await http.post("/v1/chat/completions", json={**CHAT, "stream": True})

    assert response.status_code == 502
    assert response.headers["content-type"].startswith("application/json")
    assert_openai_error(response, "upstream_unavailable")


class DiesMidStream(httpx.AsyncByteStream):
    """Sends one token, then the connection drops."""

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield b'data: {"choices":[{"delta":{"content":"hel"}}]}\n\n'
        raise httpx.ReadError("connection reset by peer")


async def test_a_stream_that_dies_ends_with_an_error_event_and_done(
    huddle_config: HuddleConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without this the reply just stops, and looks finished."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, stream=DiesMidStream()
        )

    events: list[str] = []
    async with (
        running_app(huddle_config, monkeypatch, handler) as (_, http),
        http.stream("POST", "/v1/chat/completions", json={**CHAT, "stream": True}) as response,
    ):
        assert response.status_code == 200
        async for line in response.aiter_lines():
            if line.startswith("data: "):
                events.append(line.removeprefix("data: "))

    assert json.loads(events[0])["choices"][0]["delta"]["content"] == "hel"
    assert events[-1] == "[DONE]"
    error = json.loads(events[-2])["error"]
    assert error["code"] == "upstream_lost"
    assert error["type"] == "server_error"


async def test_an_upstream_error_status_still_passes_through_untouched(
    huddle_config: HuddleConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """llama-server's own 4xx/5xx are the caller's to read; only transport failures are ours."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": {"message": "context too long"}})

    async with running_app(huddle_config, monkeypatch, handler) as (_, http):
        response = await http.post("/v1/chat/completions", json=CHAT)

    assert response.status_code == 400
    assert response.json() == {"error": {"message": "context too long"}}
