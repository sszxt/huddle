"""Cluster control API."""

from __future__ import annotations

import asyncio
import shutil

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from huddle import hfhub
from huddle.coordinator.downloads import AlreadyDownloading, DownloadService, DownloadStatus
from huddle.coordinator.overview import ClusterOverview, OverviewService
from huddle.coordinator.service import ClusterPlan, ClusterService, ClusterStatus
from huddle.gguf import GGUFError
from huddle.process import ProcessError

# A 5% margin over the file's reported size before refusing a download for
# lack of disk space — the reported size is the LFS pointer's target size,
# not necessarily byte-exact once written.
_DISK_MARGIN = 1.05


class StartRequest(BaseModel):
    model: str | None = None


class SearchResponse(BaseModel):
    results: list[hfhub.HFModelSummary]


class RepoFilesResponse(BaseModel):
    repo_id: str
    files: list[hfhub.HFFile]


class DownloadRequest(BaseModel):
    repo_id: str
    filename: str


def build_router(service: ClusterService, downloads: DownloadService) -> APIRouter:
    """Cluster control, plus model search/download.

    Like every other route here, `/models/search` and `/models/download`
    have no authentication of their own — same posture as `/cluster/start`.
    Anyone who can reach this port can already start/stop the cluster; being
    able to also trigger a Hugging Face download is not a bigger exposure,
    just flagging it rather than leaving it implicit.
    """
    router = APIRouter(prefix="/cluster", tags=["cluster"])
    overview = OverviewService(service)

    @router.get("")
    async def status() -> ClusterStatus:
        return service.status()

    @router.get("/nodes")
    async def nodes() -> ClusterOverview:
        """Every node's hardware, system details and role, for the cluster page."""
        return await overview.overview()

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

    @router.get("/models/search")
    async def search_models(q: str) -> SearchResponse:
        if not q.strip():
            raise HTTPException(status_code=422, detail="q is required")
        try:
            results = await downloads.search(q)
        except hfhub.HFError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        return SearchResponse(results=results)

    @router.get("/models/repo-files")
    async def repo_files(repo_id: str) -> RepoFilesResponse:
        try:
            result = await downloads.repo_files(repo_id)
        except hfhub.HFError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return RepoFilesResponse(repo_id=result.repo_id, files=result.files)

    @router.get("/models/download")
    async def download_status() -> DownloadStatus:
        return downloads.status()

    @router.post("/models/download")
    async def start_download(request: DownloadRequest) -> DownloadStatus:
        try:
            info = await asyncio.to_thread(hfhub.preflight, request.repo_id, request.filename)
        except hfhub.HFError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        if info.file_size is not None:
            free = shutil.disk_usage(service.config.models.dir).free
            needed = int(info.file_size * _DISK_MARGIN)
            if free < needed:
                raise HTTPException(
                    status_code=507,
                    detail=f"not enough disk space: need ~{needed} bytes, {free} free",
                )
        try:
            downloads.start(request.repo_id, request.filename)
        except AlreadyDownloading as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return downloads.status()

    return router
