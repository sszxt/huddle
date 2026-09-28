"""Application factory.

One process per node serves the agent control API, the cluster API, the
OpenAI-compatible API and the web UI. They stay separate routers so a node can
run the agent alone (`huddle agent`), with no public surface.

Every `huddle serve` node is the same: it lends its GPU to whichever node is
serving a model, and becomes that node when someone loads a model on it. With
discovery on, it keeps a live table of the other nodes (``membership.py``);
with it off, it uses the static ``peers:`` list as before.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator, Callable

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from huddle import __version__, paths
from huddle.agent.app import build_router as build_agent_router
from huddle.agent.service import BackendService
from huddle.api.app import build_router as build_api_router
from huddle.api.app import make_client
from huddle.config import HuddleConfig
from huddle.coordinator.app import build_router as build_cluster_router
from huddle.coordinator.downloads import DownloadService
from huddle.coordinator.service import ClusterService
from huddle.discovery import ROLE_COORDINATOR, ROLE_NODE, Advertiser, Identity, ZeroconfLan
from huddle.membership import Lan, Membership, NodeReport, Role
from huddle.state import StateStore
from huddle.web import STATIC_DIR

log = logging.getLogger("huddle")

LOOPBACK = {"127.0.0.1", "localhost", "::1"}

LanFactory = Callable[[Advertiser], Lan]


def create_app(config: HuddleConfig, *, lan_factory: LanFactory | None = None) -> FastAPI:
    store = StateStore(config.node.state_dir or paths.state_dir())
    identity = Identity(id=config.node.id or store.load().node_id, name=config.node.name)
    service = BackendService(config)

    # Bound to loopback, the node's agent routes are not on the network: it can
    # borrow other nodes' GPUs but never lend its own, so it says so.
    on_lan = config.api.host not in LOOPBACK
    advertiser = Advertiser(
        config,
        role=ROLE_NODE if on_lan else ROLE_COORDINATOR,
        identity=identity,
        port=config.api.port,
    )
    membership: Membership | None = None
    if config.discovery.enabled:
        lan = lan_factory(advertiser) if lan_factory else ZeroconfLan(advertiser)
        membership = Membership(config, identity, lan, remembered=store.load().last_members)

    cluster = ClusterService(config, service, identity=identity, membership=membership, state=store)
    if membership is not None:
        # A node serving a model needs all of its own GPU, so while it is one
        # it lends nothing. (With a static peer list, which node borrows from
        # which is written down, and nothing needs refusing.)
        service.busy = cluster.busy_reason
    downloads = DownloadService(config.models.dir)
    client = make_client()

    def report() -> NodeReport:
        status = cluster.status()
        owner = service.owner
        role: Role
        if cluster.is_head:
            role = "head"
        elif service.rpc_running:
            role = "worker"
        else:
            role = "idle"
        return NodeReport(
            id=identity.id,
            name=identity.name,
            cluster=config.discovery.cluster,
            huddle=__version__,
            llamacpp=cluster.llamacpp_version(),
            role=role,
            model=status.model or status.requested_model,
            starting=status.starting,
            owner_id=owner.id if owner else None,
            owner_name=owner.name if owner else None,
            workers=status.workers,
            can_head=on_lan,
            resume_since=store.load().resume_since,
        )

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        with contextlib.suppress(OSError):
            config.models.dir.mkdir(parents=True, exist_ok=True)
        advertiser.llamacpp_version = await asyncio.to_thread(cluster.llamacpp_version)
        if membership is not None:
            # Advertise before starting any model: a peer that comes up while
            # we are still loading should still find us.
            await membership.start()
        leases = asyncio.create_task(service.watch_leases())
        # In the background, because loading a large model takes minutes and
        # the API — /health, /cluster, doctor — must answer meanwhile, which is
        # exactly when someone is looking.
        startup = asyncio.create_task(cluster.boot())
        try:
            yield
        finally:
            for task in (startup, leases):
                if not task.done():
                    task.cancel()
                    with contextlib.suppress(asyncio.CancelledError, Exception):
                        await task
            await cluster.shutdown()
            await service.stop_rpc()
            if membership is not None:
                await membership.stop()
            await downloads.cancel()
            await client.aclose()

    app = FastAPI(title="Huddle", version=__version__, lifespan=lifespan)
    app.state.service = service
    app.state.cluster = cluster
    app.state.membership = membership
    app.state.identity = identity
    app.include_router(build_agent_router(service, report))
    app.include_router(build_cluster_router(cluster, downloads))
    app.include_router(build_api_router(service, client, cluster))

    @app.get("/", include_in_schema=False)
    async def root() -> RedirectResponse:
        return RedirectResponse("/ui/")

    app.mount("/ui", StaticFiles(directory=STATIC_DIR, html=True), name="web-ui")
    return app
