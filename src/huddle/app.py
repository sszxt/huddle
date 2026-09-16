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

log = logging.getLogger("huddle")


def create_app(config: HuddleConfig) -> FastAPI:
    service = BackendService(config)
    cluster = ClusterService(config, service)
    client = make_client()

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if config.backend.autostart:
            try:
                if config.peers:
                    # Starting the backend alone here would quietly run
                    # single-node and leave every configured peer unused, which
                    # looks identical to a working cluster from the outside.
                    result = await cluster.start()
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
