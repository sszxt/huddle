"""The node agent's internal control API.

Internal means internal: this exposes process control and has no authentication.
Bind it to localhost or a private interface, never to the open network.
"""

from __future__ import annotations

from fastapi import APIRouter, FastAPI, HTTPException
from pydantic import BaseModel

from huddle.agent.service import BackendService, BackendStatus, RpcStatus
from huddle.config import HuddleConfig
from huddle.hardware import NodeHardware
from huddle.process import ProcessError


class StartRequest(BaseModel):
    model: str | None = None


class LogsResponse(BaseModel):
    lines: list[str]


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

    @router.get("/rpc")
    async def rpc_status() -> RpcStatus:
        return service.rpc_status()

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
    app = FastAPI(title="Huddle agent", version="0.1.0")
    app.state.service = service
    app.include_router(build_router(service))
    return app
