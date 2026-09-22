"""Application factory.

For v0 one process serves both the agent control API and the public
OpenAI-compatible API. They stay separate routers so a v1 worker node can run
the agent alone, with no public surface.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from huddle.agent.app import build_router as build_agent_router
from huddle.agent.service import BackendService
from huddle.api.app import build_router as build_api_router
from huddle.api.app import make_client
from huddle.config import HuddleConfig
from huddle.coordinator.app import build_router as build_cluster_router
from huddle.coordinator.downloads import DownloadService
from huddle.coordinator.service import ClusterService
from huddle.discovery import ROLE_COORDINATOR, Advertiser
from huddle.web import STATIC_DIR

log = logging.getLogger("huddle")


def create_app(config: HuddleConfig) -> FastAPI:
    service = BackendService(config)
    cluster = ClusterService(config, service)
    downloads = DownloadService(config.models.dir)
    client = make_client()
    advertiser = Advertiser(config, role=ROLE_COORDINATOR)

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if config.discovery.enabled:
            # Advertise before starting the backend: a peer that comes up while
            # we are still loading should still find us.
            advertiser.llamacpp_version = _llamacpp_version(config)
            with contextlib.suppress(Exception):
                await advertiser.start()

        startup: asyncio.Task[object] | None = None
        if config.backend.autostart:
            # Always the cluster path, even with no peers: the planner skips
            # integrated GPUs and fits the model where `-ngl all` would not.
            # And in the background, because loading a large model takes
            # minutes and the API — /health, /cluster, doctor — must answer
            # meanwhile, which is exactly when someone is looking.
            startup = asyncio.create_task(cluster.start_with_retry())
        try:
            yield
        finally:
            if startup is not None and not startup.done():
                startup.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await startup
            await advertiser.stop()
            await cluster.stop()
            await service.stop_rpc()
            await downloads.cancel()
            await client.aclose()

    app = FastAPI(title="Huddle", version="0.1.0", lifespan=lifespan)
    app.state.service = service
    app.state.cluster = cluster
    app.include_router(build_agent_router(service))
    app.include_router(build_cluster_router(cluster, downloads))
    app.include_router(build_api_router(service, client, cluster))
    app.mount("/ui", StaticFiles(directory=STATIC_DIR, html=True), name="web-ui")
    return app


def _llamacpp_version(config: HuddleConfig) -> str | None:
    """Advertised so peers can refuse a version-mismatched cluster early."""
    from huddle.llamacpp import LlamaCppError, binary_version

    try:
        return binary_version(config.binaries.llama_server)
    except LlamaCppError:
        return None
