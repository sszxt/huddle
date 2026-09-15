"""The public OpenAI-compatible API.

llama-server already speaks OpenAI, so this is mostly a proxy. It exists rather
than exposing llama-server directly because Huddle owns auth, cluster status and
(later) model switching — a model change restarts the backend, and callers
should never see that.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import StreamingResponse

from huddle.agent.service import BackendService

# No read timeout: a long generation legitimately holds the connection open for
# minutes, and in a cluster the first request also waits on weight streaming.
PROXY_TIMEOUT = httpx.Timeout(connect=10.0, read=None, write=30.0, pool=10.0)

PROXIED_POST_PATHS = ("/v1/chat/completions", "/v1/completions", "/v1/embeddings")


def make_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=PROXY_TIMEOUT)


def build_router(service: BackendService, client: httpx.AsyncClient) -> APIRouter:
    router = APIRouter(tags=["openai"])

    def check_auth(request: Request) -> None:
        expected = service.config.api.api_key
        if not expected:
            return
        header = request.headers.get("authorization", "")
        token = header.removeprefix("Bearer ").strip()
        if token != expected:
            raise HTTPException(status_code=401, detail="invalid api key")

    def require_backend() -> None:
        if not service.running:
            raise HTTPException(
                status_code=503,
                detail="backend is not running; start it via POST /agent/backend/start",
            )

    @router.get("/health")
    async def health() -> dict[str, Any]:
        status = service.status()
        return {
            "status": "ok" if status.running else "backend_down",
            "node": service.config.node.name,
            "model": status.model,
        }

    @router.get("/v1/models")
    async def models(request: Request) -> Response:
        check_auth(request)
        require_backend()
        upstream = await client.get(f"{service.base_url}/v1/models")
        return Response(
            content=upstream.content,
            status_code=upstream.status_code,
            media_type=upstream.headers.get("content-type", "application/json"),
        )

    @router.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Response:
        return await _proxy(request, "/v1/chat/completions")

    @router.post("/v1/completions")
    async def completions(request: Request) -> Response:
        return await _proxy(request, "/v1/completions")

    @router.post("/v1/embeddings")
    async def embeddings(request: Request) -> Response:
        return await _proxy(request, "/v1/embeddings")

    async def _proxy(request: Request, path: str) -> Response:
        check_auth(request)
        require_backend()

        try:
            payload: dict[str, Any] = await request.json()
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="invalid JSON body") from exc

        url = f"{service.base_url}{path}"
        if not payload.get("stream"):
            upstream = await client.post(url, json=payload)
            return Response(
                content=upstream.content,
                status_code=upstream.status_code,
                media_type=upstream.headers.get("content-type", "application/json"),
            )

        # Open the stream before returning, so an upstream error still reaches
        # the caller as a real status code rather than as a 200 full of nothing.
        upstream_request = client.build_request("POST", url, json=payload)
        upstream_response = await client.send(upstream_request, stream=True)
        if upstream_response.status_code >= 400:
            body = await upstream_response.aread()
            await upstream_response.aclose()
            return Response(
                content=body,
                status_code=upstream_response.status_code,
                media_type="application/json",
            )

        return StreamingResponse(
            _forward_stream(upstream_response),
            status_code=upstream_response.status_code,
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return router


async def _forward_stream(upstream: httpx.Response) -> AsyncIterator[bytes]:
    """Relay SSE chunks as they arrive.

    ``aiter_raw`` yields whatever has arrived rather than waiting for whole
    lines, which is what keeps tokens flowing to the caller as generated.
    """
    try:
        async for chunk in upstream.aiter_raw():
            yield chunk
    finally:
        await upstream.aclose()
