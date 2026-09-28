"""Cluster control API.

Every node answers for the whole cluster, not just itself: the web UI can be
opened on any of them. A node that is not serving passes cluster-wide
questions on to the one that is (the head), and a model load goes to the node
that holds the file, which then becomes the head. ``X-Huddle-Forwarded`` marks
a request that has already been passed on once, so none is passed on twice.
"""

from __future__ import annotations

import asyncio
import shutil
import time
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel

from huddle import catalog, hfhub
from huddle.coordinator.downloads import AlreadyDownloading, DownloadService, DownloadStatus
from huddle.coordinator.overview import OverviewService
from huddle.coordinator.service import (
    FORWARDED_HEADER,
    ClusterPlan,
    ClusterService,
    ClusterStatus,
)
from huddle.gguf import GGUFError
from huddle.membership import Member
from huddle.process import ProcessError

# A 5% margin over the file's reported size before refusing a download for
# lack of disk space — the reported size is the LFS pointer's target size,
# not necessarily byte-exact once written.
_DISK_MARGIN = 1.05
# Asking another node for its model list; the UI's model menu waits on this.
_LIST_TIMEOUT = httpx.Timeout(3.0, connect=1.0)
# A load streams weights to peers before answering: minutes, for a big model.
_LOAD_TIMEOUT = httpx.Timeout(900.0, connect=5.0)
_MODELS_CACHE_SECONDS = 5.0


class StartRequest(BaseModel):
    model: str | None = None
    # The node holding the file; this one when omitted.
    node_id: str | None = None


class SearchResponse(BaseModel):
    results: list[hfhub.HFModelSummary]


class RepoFilesResponse(BaseModel):
    repo_id: str
    files: list[hfhub.HFFile]


class DownloadRequest(BaseModel):
    repo_id: str
    filename: str


class ModelEntry(BaseModel):
    """One GGUF file on one node."""

    file: str
    node: str
    node_id: str | None = None
    size: int | None = None
    # This node's own file, as opposed to one on another node.
    local: bool = False


class CatalogResponse(BaseModel):
    capacity: catalog.Capacity
    models: list[catalog.CatalogEntry]


class ModelsResponse(BaseModel):
    # This node's files, by name: what older clients (the TUI) read.
    available: list[str]
    # Every node's files.
    entries: list[ModelEntry] = []
    loaded: str | None = None
    # A model still loading, when one is.
    loading: str | None = None
    head: str | None = None
    head_id: str | None = None


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
    remote_models: dict[str, Any] = {"at": 0.0, "entries": []}

    def remote_head(request: Request) -> Member | None:
        """The other node to pass this request to, if one is serving."""
        if request.headers.get(FORWARDED_HEADER) or service.membership is None:
            return None
        if service.is_head:
            return None
        return service.membership.head()

    async def forward(
        member: Member,
        request: Request,
        path: str,
        *,
        timeout: httpx.Timeout = _LIST_TIMEOUT,
        json: Any = None,
    ) -> httpx.Response:
        url = f"http://{member.host}:{member.port}{path}"
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                return await client.request(
                    request.method,
                    url,
                    params=request.query_params,
                    json=json,
                    headers={FORWARDED_HEADER: "1"},
                )
        except httpx.HTTPError as exc:
            raise HTTPException(
                status_code=502, detail=f"{member.display} did not answer: {exc}"
            ) from exc

    def relay(upstream: httpx.Response) -> Response:
        return Response(
            content=upstream.content,
            status_code=upstream.status_code,
            media_type=upstream.headers.get("content-type", "application/json"),
        )

    @router.get("", response_model=None)
    async def status(request: Request) -> ClusterStatus | Response:
        head = remote_head(request)
        if head is not None:
            return relay(await forward(head, request, "/cluster"))
        return service.status()

    @router.get("/nodes", response_model=None)
    async def nodes(request: Request) -> dict[str, Any] | Response:
        """Every node's hardware, system details and role, for the cluster page.

        Built by the head when there is one: it knows the plan, and round
        trips are measured from where the traffic flows.
        """
        head = remote_head(request)
        if head is not None:
            upstream = await forward(head, request, "/cluster/nodes")
            if upstream.status_code != 200:
                return relay(upstream)
            body: dict[str, Any] = upstream.json()
        else:
            body = (await overview.overview()).model_dump(mode="json")
        body["viewer_id"] = service.identity.id
        return body

    @router.get("/plan")
    async def plan(model: str | None = None, ctx: int | None = None) -> ClusterPlan:
        """Show the placement decision without starting anything.

        Always from this node's point of view, as the head it would become.
        """
        try:
            return await service.build_plan(model, n_ctx=ctx)
        except (ProcessError, GGUFError, ValueError, OSError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @router.post("/start")
    async def start(request: StartRequest | None = None) -> ClusterStatus:
        model = request.model if request else None
        try:
            if service.membership is None:
                return await service.start(model)
            chosen = model or service.config.models.default
            if chosen is None:
                raise ValueError("no model requested and no default configured")
            return await service.load_here(chosen)
        except (ProcessError, GGUFError, ValueError, OSError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @router.get("/models")
    async def models(scope: str = "cluster") -> ModelsResponse:
        """Which models the cluster could load, where each is, and what is loaded."""
        local = service.available_models()
        entries = [
            ModelEntry(
                file=name,
                node=service.identity.name,
                node_id=service.identity.id,
                size=_size(service.config.models.dir / name),
                local=True,
            )
            for name in local
        ]
        status = service.status()
        response = ModelsResponse(available=local, entries=entries)
        if status.running or status.desired:
            response.head, response.head_id = service.identity.name, service.identity.id
            response.loaded = status.model
            response.loading = status.requested_model if status.starting else None

        if scope == "local" or service.membership is None:
            return response

        response.entries += await _remote_entries()
        head = service.membership.head()
        if response.head is None and head is not None and head.report is not None:
            response.head, response.head_id = head.display, head.id
            if head.report.starting:
                response.loading = head.report.model
            else:
                response.loaded = head.report.model
        return response

    async def _remote_entries() -> list[ModelEntry]:
        """Every other node's files, cached briefly: the menu opens often."""
        if time.monotonic() - remote_models["at"] < _MODELS_CACHE_SECONDS:
            cached: list[ModelEntry] = remote_models["entries"]
            return cached
        membership = service.membership
        assert membership is not None
        members = [m for m in membership.known() if m.alive and m.report and m.report.can_head]

        async def ask(client: httpx.AsyncClient, member: Member) -> list[ModelEntry]:
            url = f"http://{member.host}:{member.port}/cluster/models"
            try:
                response = await client.get(
                    url, params={"scope": "local"}, headers={FORWARDED_HEADER: "1"}
                )
                response.raise_for_status()
                listed = ModelsResponse.model_validate(response.json())
            except (httpx.HTTPError, ValueError):
                return []
            return [
                ModelEntry(
                    file=entry.file,
                    node=member.display,
                    node_id=member.id,
                    size=entry.size,
                )
                for entry in listed.entries
            ]

        async with httpx.AsyncClient(timeout=_LIST_TIMEOUT) as client:
            found = await asyncio.gather(*(ask(client, member) for member in members))
        entries = [entry for group in found for entry in group]
        remote_models.update(at=time.monotonic(), entries=entries)
        return entries

    @router.get("/catalog")
    async def recommended() -> CatalogResponse:
        """Models to offer a cluster with none, each labelled with whether it fits.

        Sized for this node as the head, since a model downloaded from its
        page lands on its disk and it is the one that would serve it.
        """
        view = await overview.overview()
        live = [node for node in view.nodes if node.role != "offline" and node.hardware]
        me = next((node for node in live if node.id == service.identity.id), None)
        if me is None or me.hardware is None:
            raise HTTPException(status_code=503, detail="this node cannot describe its hardware")
        room = catalog.capacity(
            [node.hardware for node in live if node.hardware is not None],
            me.hardware,
            service.config.planner.headroom,
        )
        present = {name: service.identity.name for name in service.available_models()}
        if service.membership is not None:
            for entry in await _remote_entries():
                present.setdefault(entry.file, entry.node)
        return CatalogResponse(capacity=room, models=catalog.entries(room, present))

    @router.post("/model", response_model=None)
    async def switch(body: StartRequest, request: Request) -> ClusterStatus | Response:
        """Serve a model, from the node that holds it, replanning the split for it."""
        if not body.model:
            raise HTTPException(status_code=422, detail="model is required")
        target = body.node_id
        if target and target != service.identity.id and service.membership is not None:
            member = service.membership.by_id(target)
            if member is None or not member.alive:
                raise HTTPException(status_code=409, detail="that node is not reachable now")
            remote_models["at"] = 0.0
            upstream = await forward(
                member,
                request,
                "/cluster/model",
                timeout=_LOAD_TIMEOUT,
                json={"model": body.model},
            )
            return relay(upstream)
        try:
            if service.membership is None:
                return await service.switch_model(body.model)
            return await service.load_here(body.model)
        except (ProcessError, GGUFError, ValueError, OSError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @router.post("/stop", response_model=None)
    async def stop(request: Request) -> ClusterStatus | Response:
        head = remote_head(request)
        if head is not None:
            return relay(await forward(head, request, "/cluster/stop", timeout=_LOAD_TIMEOUT))
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
            # A fresh node has never had anything to put here yet.
            service.config.models.dir.mkdir(parents=True, exist_ok=True)
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


def _size(path: Any) -> int | None:
    try:
        return int(path.stat().st_size)
    except OSError:
        return None
