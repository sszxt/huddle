"""The node agent's internal control API.

Internal means internal: this exposes process control and has no authentication.
On a node that shares its GPUs it is reachable by the rest of the LAN, which is
the point — firewall it to the local subnet (`huddle setup` does), and never
expose it beyond that.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator

from fastapi import APIRouter, FastAPI, HTTPException, Request
from pydantic import BaseModel

from huddle import __version__
from huddle.agent.service import (
    OWNER_ID_HEADER,
    OWNER_NAME_HEADER,
    BackendService,
    BackendStatus,
    Owner,
    RpcStatus,
)
from huddle.config import HuddleConfig
from huddle.discovery import ROLE_WORKER, Advertiser, load_identity
from huddle.hardware import NodeHardware
from huddle.membership import NodeReport, ReportProvider
from huddle.process import ProcessError
from huddle.system import SystemInfo, probe_system

log = logging.getLogger("huddle")


class StartRequest(BaseModel):
    model: str | None = None


class LogsResponse(BaseModel):
    lines: list[str]


class PlacementResponse(BaseModel):
    """Layer indices per device, as llama.cpp reported placing them."""

    layers: dict[str, list[int]]


def owner_of(request: Request) -> Owner | None:
    """The head making this request, if it said who it is."""
    owner_id = request.headers.get(OWNER_ID_HEADER)
    if not owner_id:
        return None
    return Owner(id=owner_id, name=request.headers.get(OWNER_NAME_HEADER) or owner_id[:8])


def build_router(service: BackendService, report: ReportProvider | None = None) -> APIRouter:
    router = APIRouter(prefix="/agent", tags=["agent"])

    @router.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "node": service.config.node.name}

    @router.get("/state")
    async def state() -> NodeReport:
        """What this node is doing, for its peers' membership tables.

        Polled every few seconds by every other node, so it must stay cheap:
        bookkeeping only, never a hardware probe or a subprocess.
        """
        if report is None:
            raise HTTPException(status_code=404, detail="not a cluster member")
        return report()

    @router.get("/hardware")
    async def hardware() -> NodeHardware:
        # A fresh probe every time, as placement requires, but in a thread: it
        # runs llama-server and nvidia-smi, and a dashboard polling it would
        # otherwise stall every streaming reply on this node for the duration.
        return await asyncio.to_thread(service.hardware)

    @router.get("/system")
    async def system() -> SystemInfo:
        """OS, CPU and network details, for display only."""
        return await asyncio.to_thread(probe_system)

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
    async def rpc_status(request: Request) -> RpcStatus:
        # A head asking after its own worker renews the lease by asking.
        owner = owner_of(request)
        return await service.rpc_report(renew=owner.id if owner else None)

    @router.post("/rpc/start")
    async def rpc_start(request: Request) -> RpcStatus:
        try:
            return await service.start_rpc(owner_of(request))
        except (ProcessError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @router.post("/rpc/stop")
    async def rpc_stop(request: Request) -> RpcStatus:
        owner = owner_of(request)
        try:
            # A head may stop only a worker lent to it; a request that names
            # no head (a person, an older Huddle) is taken at its word.
            return await service.stop_rpc(owner_id=owner.id if owner else None, force=owner is None)
        except ProcessError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @router.get("/rpc/logs")
    async def rpc_logs() -> LogsResponse:
        return LogsResponse(lines=service.rpc_logs())

    return router


def worker_report(service: BackendService, identity_id: str, version: str | None) -> NodeReport:
    """A `huddle agent` node: it lends its GPUs and never runs a model."""
    owner = service.owner
    return NodeReport(
        id=identity_id,
        name=service.config.node.name,
        cluster=service.config.discovery.cluster,
        huddle=__version__,
        llamacpp=version,
        role="worker" if service.rpc_running else "idle",
        owner_id=owner.id if owner else None,
        owner_name=owner.name if owner else None,
        can_head=False,
    )


def create_app(config: HuddleConfig) -> FastAPI:
    """Standalone agent app, for worker nodes that serve no public API."""
    service = BackendService(config)
    identity = load_identity(config)
    version: dict[str, str | None] = {"llamacpp": None}
    advertiser = Advertiser(config, role=ROLE_WORKER, identity=identity)

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        version["llamacpp"] = await asyncio.to_thread(_version, config)
        # Worker nodes are precisely the ones a coordinator needs to find.
        if config.discovery.enabled:
            advertiser.llamacpp_version = version["llamacpp"]
            try:
                await advertiser.start()
            except Exception as exc:
                log.error("discovery: could not advertise: %s", exc)
        leases = asyncio.create_task(service.watch_leases())
        try:
            yield
        finally:
            leases.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await leases
            await advertiser.stop()
            await service.stop_rpc()

    app = FastAPI(title="Huddle agent", version=__version__, lifespan=lifespan)
    app.state.service = service
    app.include_router(
        build_router(service, lambda: worker_report(service, identity.id, version["llamacpp"]))
    )
    return app


def _version(config: HuddleConfig) -> str | None:
    from huddle.llamacpp import LlamaCppError, binary_version

    try:
        return binary_version(config.binaries.llama_server)
    except LlamaCppError:
        return None
