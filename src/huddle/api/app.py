"""The public OpenAI-compatible API.

llama-server already speaks OpenAI, so this is mostly a proxy. It exists rather
than exposing llama-server directly because Huddle owns auth, cluster status and
model switching — a model change restarts the backend, and callers should never
see that.

Any node answers. One that is not serving passes the request to the node that
is, so the chat works from whichever PC the browser is on; ``aiter_raw`` on
both hops keeps a stream a stream.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import StreamingResponse

from huddle.agent.service import BackendService
from huddle.coordinator.service import FORWARDED_HEADER, ClusterService

# No read timeout: a long generation legitimately holds the connection open for
# minutes, and in a cluster the first request also waits on weight streaming.
PROXY_TIMEOUT = httpx.Timeout(connect=10.0, read=None, write=30.0, pool=10.0)

PROXIED_POST_PATHS = ("/v1/chat/completions", "/v1/completions", "/v1/embeddings")


def make_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=PROXY_TIMEOUT)


def build_router(
    service: BackendService,
    client: httpx.AsyncClient,
    cluster: ClusterService | None = None,
) -> APIRouter:
    router = APIRouter(tags=["openai"])

    def check_auth(request: Request) -> None:
        expected = service.config.api.api_key
        if not expected:
            return
        header = request.headers.get("authorization", "")
        token = header.removeprefix("Bearer ").strip()
        if token != expected:
            raise HTTPException(status_code=401, detail="invalid api key")

    def target(request: Request) -> tuple[str, dict[str, str]]:
        """Where to send a request: this node's backend, or the serving node.

        Returns the base URL and any headers to add. A request already passed
        on once is never passed on again, so two nodes cannot bounce it.
        """
        if service.running:
            return service.base_url, {}
        head = None
        if (
            cluster is not None
            and cluster.membership is not None
            and not cluster.is_head
            and not request.headers.get(FORWARDED_HEADER)
        ):
            head = cluster.membership.head()
        if head is not None and head.report is not None and not head.report.starting:
            headers = {FORWARDED_HEADER: "1"}
            if "authorization" in request.headers:
                headers["authorization"] = request.headers["authorization"]
            return f"http://{head.host}:{head.port}", headers
        if head is not None:
            raise HTTPException(
                status_code=503, detail=f"{head.display} is still loading the model"
            )
        raise HTTPException(
            status_code=503,
            detail="the backend is not running: no model is loaded; pick one in the web UI "
            "or POST /cluster/model",
        )

    def remote_health() -> dict[str, Any] | None:
        """/health for a node that is not serving, when another one is."""
        if cluster is None or cluster.membership is None or cluster.is_head:
            return None
        head = cluster.membership.head()
        if head is None or head.report is None:
            return None
        return {
            "status": "starting" if head.report.starting else "ok",
            "node": service.config.node.name,
            "model": head.report.model,
            "role": "worker" if service.rpc_running else "idle",
            "head": head.display,
            "head_id": head.id,
            "workers": head.report.workers,
        }

    @router.get("/health")
    async def health() -> dict[str, Any]:
        status = service.status()
        remote = None if status.running else remote_health()
        if remote is not None:
            return remote
        body: dict[str, Any] = {
            "status": "ok" if status.running else "backend_down",
            "node": service.config.node.name,
            "model": status.model,
        }
        if cluster is not None:
            if cluster.is_head:
                body["role"], body["head"] = "head", service.config.node.name
            else:
                body["role"] = "worker" if service.rpc_running else "idle"
            cluster_status = cluster.status()
            if cluster_status.starting:
                body["status"] = "starting"
                body["detail"] = "a start is in progress; large models take minutes to load"
            elif cluster_status.degraded:
                # The backend is meant to be up and is not. Saying "ok" here is
                # how a dead cluster keeps looking healthy to whatever is
                # watching it.
                body["status"] = "degraded"
                body["detail"] = cluster_status.last_failure or "head process is not running"
            body["restarts"] = cluster_status.restarts
            body["workers"] = cluster_status.workers
        return body

    @router.get("/v1/models")
    async def models(request: Request) -> Response:
        check_auth(request)
        base, headers = target(request)
        upstream = await client.get(f"{base}/v1/models", headers=headers)
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
        base, headers = target(request)

        try:
            payload: dict[str, Any] = await request.json()
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="invalid JSON body") from exc

        url = f"{base}{path}"
        if not payload.get("stream"):
            upstream = await client.post(url, json=payload, headers=headers)
            return Response(
                content=upstream.content,
                status_code=upstream.status_code,
                media_type=upstream.headers.get("content-type", "application/json"),
            )

        # Open the stream before returning, so an upstream error still reaches
        # the caller as a real status code rather than as a 200 full of nothing.
        upstream_request = client.build_request("POST", url, json=payload, headers=headers)
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
