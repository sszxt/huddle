"""Cluster control API."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from huddle.coordinator.service import ClusterPlan, ClusterService, ClusterStatus
from huddle.gguf import GGUFError
from huddle.process import ProcessError


class StartRequest(BaseModel):
    model: str | None = None


def build_router(service: ClusterService) -> APIRouter:
    router = APIRouter(prefix="/cluster", tags=["cluster"])

    @router.get("")
    async def status() -> ClusterStatus:
        return service.status()

    @router.get("/plan")
    async def plan(model: str | None = None, ctx: int | None = None) -> ClusterPlan:
        """Show the placement decision without starting anything."""
        try:
            return await service.build_plan(model, n_ctx=ctx)
        except (ProcessError, GGUFError, ValueError, OSError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @router.post("/start")
    async def start(request: StartRequest | None = None) -> ClusterStatus:
        try:
            return await service.start(request.model if request else None)
        except (ProcessError, GGUFError, ValueError, OSError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @router.get("/models")
    async def models() -> dict[str, list[str] | str | None]:
        """Which models this node could load, and which is loaded now."""
        return {"available": service.available_models(), "loaded": service.status().model}

    @router.post("/model")
    async def switch(request: StartRequest) -> ClusterStatus:
        """Switch to a different model, replanning the split for it."""
        if not request.model:
            raise HTTPException(status_code=422, detail="model is required")
        try:
            return await service.switch_model(request.model)
        except (ProcessError, GGUFError, ValueError, OSError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @router.post("/stop")
    async def stop() -> ClusterStatus:
        return await service.stop()

    return router
