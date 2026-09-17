"""Application factory.

For v0 one process serves both the agent control API and the public
OpenAI-compatible API. They stay separate routers so a v1 worker node can run
the agent alone, with no public surface.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import AsyncIterator

from fastapi import FastAPI

from huddle.agent.app import build_router as build_agent_router
from huddle.agent.service import BackendService
from huddle.api.app import build_router as build_api_router
from huddle.api.app import make_client
from huddle.config import HuddleConfig
from huddle.coordinator.app import build_router as build_cluster_router
from huddle.coordinator.service import ClusterService
from huddle.discovery import Advertiser

log = logging.getLogger("huddle")


def create_app(config: HuddleConfig) -> FastAPI:
    service = BackendService(config)
    cluster = ClusterService(config, service)
    client = make_client()
    advertiser = Advertiser(config)

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if config.discovery.enabled:
            # Advertise before starting the backend: a peer that comes up while
            # we are still loading should still find us.
            advertiser.llamacpp_version = _llamacpp_version(config)
            with contextlib.suppress(Exception):
                await advertiser.start()

        if config.backend.autostart:
            try:
                if config.peers or config.discovery.enabled:
                    # Starting the backend alone here would quietly run
                    # single-node and leave every peer unused, which looks
                    # identical to a working cluster from the outside.
                    result = await cluster.start_with_retry()
                    if result is not None:
                        log.info(
                            "cluster ready: model=%s workers=%s",
                            result.model,
                            ",".join(result.workers) or "none",
                        )
                else:
                    status = await service.start()
                    log.info("backend ready: model=%s pid=%s", status.model, status.pid)
            except Exception as exc:
                log.error("autostart failed: %s", exc)
        try:
            yield
        finally:
            await advertiser.stop()
            await cluster.stop()
            await service.stop_rpc()
            await client.aclose()

    app = FastAPI(title="Huddle", version="0.1.0", lifespan=lifespan)
    app.state.service = service
    app.state.cluster = cluster
    app.include_router(build_agent_router(service))
    app.include_router(build_cluster_router(cluster))
    app.include_router(build_api_router(service, client, cluster))
    return app


def _llamacpp_version(config: HuddleConfig) -> str | None:
    """Advertised so peers can refuse a version-mismatched cluster early."""
    from huddle.llamacpp import LlamaCppError, binary_version

    try:
        return binary_version(config.binaries.llama_server)
    except LlamaCppError:
        return None
