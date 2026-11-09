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

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from huddle.accounts.app import Accounts, bearer_matches
from huddle.agent.service import BackendService
from huddle.coordinator.service import FORWARDED_HEADER, ClusterService

# No read timeout: a long generation legitimately holds the connection open for
# minutes, and in a cluster the first request also waits on weight streaming.
PROXY_TIMEOUT = httpx.Timeout(connect=10.0, read=None, write=30.0, pool=10.0)

PROXIED_POST_PATHS = ("/v1/chat/completions", "/v1/completions", "/v1/embeddings")

# How soon a caller should try again while the cluster is bringing a model back.
RETRY_AFTER_SECONDS = 5


class ProxyError(Exception):
    """The cluster cannot answer this request right now.

    Raised instead of letting the failure surface as a bare 500. The response
    carries OpenAI's error shape (``error.message``/``type``/``code``), which
    the official SDKs read, and ``detail``, which Huddle's own web UI reads.
    """

    def __init__(
        self,
        status: int,
        message: str,
        *,
        code: str,
        retry_after: int | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.message = message
        self.code = code
        self.retry_after = retry_after


async def proxy_error_response(request: Request, exc: Exception) -> JSONResponse:
    """Render a ``ProxyError``; register it with ``app.add_exception_handler``."""
    assert isinstance(exc, ProxyError)
    headers = {"Retry-After": str(exc.retry_after)} if exc.retry_after else None
    return JSONResponse(
        {
            "detail": exc.message,
            "error": {"message": exc.message, "type": "server_error", "code": exc.code},
        },
        status_code=exc.status,
        headers=headers,
    )


def _stream_error(exc: Exception) -> bytes:
    """The tail of an event stream whose upstream died after it had started.

    The 200 and some tokens have already reached the caller, so a status code
    is no longer possible. An error event and a terminator are: without them
    the stream just stops, and a client cannot tell a finished reply from a
    cut-off one.
    """
    message = f"the model server stopped responding mid-reply ({type(exc).__name__})"
    event = {"error": {"message": message, "type": "server_error", "code": "upstream_lost"}}
    # The blank line first closes any event the upstream was part-way through.
    return f"\n\ndata: {json.dumps(event)}\n\ndata: [DONE]\n\n".encode()


def make_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=PROXY_TIMEOUT)


def build_router(
    service: BackendService,
    client: httpx.AsyncClient,
    cluster: ClusterService | None = None,
    accounts: Accounts | None = None,
) -> APIRouter:
    router = APIRouter(tags=["openai"])

    async def check_auth(request: Request) -> None:
        """The API key, when one is set; a signed-in web UI user counts too."""
        expected = service.config.api.api_key
        if not expected or bearer_matches(request, expected):
            return
        if accounts is not None:
            user = await accounts.user(request)
            if user is not None and user.active:
                return
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
            # This node has already checked the caller. Pass the key on rather
            # than the caller's own credentials: a sign-in belongs to this node,
            # and the key is the same on every node.
            if service.config.api.api_key:
                headers["authorization"] = f"Bearer {service.config.api.api_key}"
            elif "authorization" in request.headers:
                headers["authorization"] = request.headers["authorization"]
            return f"http://{head.host}:{head.port}", headers
        if head is not None:
            raise ProxyError(
                503,
                f"{head.display} is still loading the model",
                code="model_loading",
                retry_after=RETRY_AFTER_SECONDS * 2,
            )
        raise ProxyError(
            503,
            "the backend is not running: no model is loaded; pick one in the web UI "
            "or POST /cluster/model",
            code="no_model_loaded",
        )

    def upstream_failure(exc: Exception) -> ProxyError:
        """Why a call to the model server failed, as the caller should hear it.

        A cluster that is meant to be up but is down (a worker died, the
        supervisor is restarting it) says so with a Retry-After: the next try
        a few seconds on usually works. Anything else is a bad gateway.
        """
        if isinstance(exc, httpx.TimeoutException):
            return ProxyError(
                504, "the model server did not answer in time", code="upstream_timeout"
            )
        if cluster is not None:
            status = cluster.status()
            if status.degraded or status.starting:
                return ProxyError(
                    503,
                    "the cluster is restarting its model server; try again shortly",
                    code="cluster_recovering",
                    retry_after=RETRY_AFTER_SECONDS,
                )
        return ProxyError(
            502,
            f"the model server could not be reached ({type(exc).__name__})",
            code="upstream_unavailable",
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
        await check_auth(request)
        base, headers = target(request)
        try:
            upstream = await client.get(f"{base}/v1/models", headers=headers)
        except httpx.HTTPError as exc:
            raise upstream_failure(exc) from exc
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
        await check_auth(request)
        base, headers = target(request)

        try:
            payload: dict[str, Any] = await request.json()
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="invalid JSON body") from exc

        url = f"{base}{path}"
        if not payload.get("stream"):
            try:
                upstream = await client.post(url, json=payload, headers=headers)
            except httpx.HTTPError as exc:
                raise upstream_failure(exc) from exc
            return Response(
                content=upstream.content,
                status_code=upstream.status_code,
                media_type=upstream.headers.get("content-type", "application/json"),
            )

        # Open the stream before returning, so an upstream error still reaches
        # the caller as a real status code rather than as a 200 full of nothing.
        upstream_request = client.build_request("POST", url, json=payload, headers=headers)
        try:
            upstream_response = await client.send(upstream_request, stream=True)
            if upstream_response.status_code >= 400:
                body = await upstream_response.aread()
                await upstream_response.aclose()
                return Response(
                    content=body,
                    status_code=upstream_response.status_code,
                    media_type="application/json",
                )
        except httpx.HTTPError as exc:
            raise upstream_failure(exc) from exc

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
    except httpx.HTTPError as exc:
        yield _stream_error(exc)
    finally:
        await upstream.aclose()
