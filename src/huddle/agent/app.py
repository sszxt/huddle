"""The node agent's internal control API.

Internal means internal: this exposes process control and has no authentication.
Bind it to localhost or a private interface, never to the open network.
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator

from fastapi import APIRouter, FastAPI, HTTPException
from pydantic import BaseModel

from huddle.agent.service import BackendService, BackendStatus, RpcStatus
from huddle.config import HuddleConfig
from huddle.discovery import Advertiser
from huddle.hardware import NodeHardware
from huddle.process import ProcessError


class StartRequest(BaseModel):
    model: str | None = None


class LogsResponse(BaseModel):
    lines: list[str]


class PlacementResponse(BaseModel):
    """Layer indices per device, as llama.cpp reported placing them."""

    layers: dict[str, list[int]]


def build_router(service: BackendService) -> APIRouter:
    router = APIRouter(prefix="/agent", tags=["agent"])

    @router.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "node": service.config.node.name}

    @router.get("/hardware")
    async def hardware() -> NodeHardware:
        return service.hardware()

    @router.get("/backend")
    async def backend_status() -> BackendStatus:
        return service.status()

    @router.post("/backend/start")
    async def backend_start(request: StartRequest | None = None) -> BackendStatus:
        try:
            return await service.start(request.model if request else None)
        except (ProcessError, ValueError) as exc:
            # The child's last log lines are usually the real explanation.
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @router.post("/backend/stop")
    async def backend_stop() -> BackendStatus:
        return await service.stop()

    @router.get("/backend/logs")
    async def backend_logs() -> LogsResponse:
        return LogsResponse(lines=service.logs())

    @router.get("/backend/placement")
    async def backend_placement() -> PlacementResponse:
        return PlacementResponse(layers=service.placement())

    @router.get("/rpc")
    async def rpc_status() -> RpcStatus:
        return await service.rpc_report()

    @router.post("/rpc/start")
    async def rpc_start() -> RpcStatus:
        try:
            return await service.start_rpc()
        except (ProcessError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @router.post("/rpc/stop")
    async def rpc_stop() -> RpcStatus:
        return await service.stop_rpc()

    @router.get("/rpc/logs")
    async def rpc_logs() -> LogsResponse:
        return LogsResponse(lines=service.rpc_logs())

    return router


def create_app(config: HuddleConfig) -> FastAPI:
    """Standalone agent app, for worker nodes that serve no public API."""
    service = BackendService(config)
    advertiser = Advertiser(config)

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Worker nodes are precisely the ones a coordinator needs to find.
        if config.discovery.enabled:
            with contextlib.suppress(Exception):
                advertiser.llamacpp_version = _version(config)
                await advertiser.start()
        try:
            yield
        finally:
            await advertiser.stop()
            await service.stop_rpc()

    app = FastAPI(title="Huddle agent", version="0.1.0", lifespan=lifespan)
    app.state.service = service
    app.include_router(build_router(service))
    return app


def _version(config: HuddleConfig) -> str | None:
    from huddle.llamacpp import LlamaCppError, binary_version

    try:
        return binary_version(config.binaries.llama_server)
    except LlamaCppError:
        return None
