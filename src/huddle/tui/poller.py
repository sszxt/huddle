"""The TUI's data layer: poll the coordinator and its peers, reuse the answer.

Deliberately has no Textual import — this is independently testable against
ASGI-transport fakes, exactly like ``doctor.py``. It never recomputes
placement, peer resolution, or pipeline order; those come straight from
``ClusterStatus``/``ClusterPlan``, which already carry them correctly.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal

import httpx

from huddle import hfhub
from huddle.agent.service import BackendStatus, RpcStatus
from huddle.config import HuddleConfig
from huddle.coordinator.cluster import PeerReport, query_peers
from huddle.coordinator.downloads import DownloadStatus
from huddle.coordinator.service import ClusterStatus
from huddle.doctor import api_base_url
from huddle.hardware import NodeHardware

Role = Literal["head", "worker"]


@dataclass
class NodeSnapshot:
    """One node, as of the last poll."""

    name: str
    role: Role
    reachable: bool
    hardware: NodeHardware | None = None
    backend: BackendStatus | None = None  # head only
    rpc: RpcStatus | None = None  # worker only
    placement: dict[str, list[int]] = field(default_factory=dict)
    layers: int = 0
    error: str | None = None


@dataclass
class ClusterSnapshot:
    """Everything the dashboard needs to draw one frame."""

    fetched_at: float
    cluster: ClusterStatus | None
    nodes: list[NodeSnapshot] = field(default_factory=list)
    available_models: list[str] = field(default_factory=list)
    loaded_model: str | None = None
    # Set when this node has no /cluster route at all (an agent-only worker),
    # so the dashboard can say so instead of showing an empty cluster.
    coordinator_error: str | None = None
    log_lines: list[str] = field(default_factory=list)
    download: DownloadStatus | None = None


class ClusterPoller:
    """Fans out to this node's own API and, through it, its peers.

    Two client lifetimes, deliberately not shared: ``self._client`` talks to
    *this* node's own API and is injectable (e.g. an ``ASGITransport`` in
    tests). Peers are always real remote HTTP servers, so they are always
    reached over a fresh, real ``httpx.AsyncClient`` — the same split
    ``ClusterService`` itself uses (its own outbound peer calls never reuse
    whatever client a caller is using to talk to it).

    Controls (start/stop/switch model) are thin wrappers over the same
    ``/cluster/*`` endpoints ``ClusterService`` already exposes — no new
    orchestration logic lives here.
    """

    def __init__(self, config: HuddleConfig, *, client: httpx.AsyncClient | None = None) -> None:
        self.config = config
        self._client = client
        self._owns_client = client is None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient()
        return self._client

    async def aclose(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    async def poll(self) -> ClusterSnapshot:
        client = await self._get_client()
        base = api_base_url(self.config)

        try:
            response = await client.get(f"{base}/cluster", timeout=10.0)
        except httpx.HTTPError as exc:
            return ClusterSnapshot(
                fetched_at=time.time(),
                cluster=None,
                coordinator_error=f"cannot reach this node's own API: {exc}",
            )
        if response.status_code == 404:
            return await self._local_only_snapshot(client, base)

        cluster = ClusterStatus.model_validate(response.json())
        head = await self._head_snapshot(client, base, cluster)

        reports_by_name: dict[str, PeerReport] = {}
        if cluster.workers:
            reports_by_name = {r.peer.name: r for r in await query_peers(self.config)}

        nodes = [head]
        async with httpx.AsyncClient() as peer_client:
            for name in cluster.workers:
                nodes.append(
                    await self._worker_snapshot(
                        peer_client, name, cluster, reports_by_name.get(name)
                    )
                )

        available_models, loaded_model = await self._models(client, base)
        return ClusterSnapshot(
            fetched_at=time.time(),
            cluster=cluster,
            nodes=nodes,
            available_models=available_models,
            loaded_model=loaded_model,
            log_lines=await self._fetch_logs(client, base),
            download=await self._fetch_download_status(client, base),
        )

    async def _local_only_snapshot(self, client: httpx.AsyncClient, base: str) -> ClusterSnapshot:
        """This node exposes /agent/* only (a worker-only `huddle agent`).

        The TUI is designed to run against a coordinator; pointed at a
        worker, it shows what that worker knows about itself rather than
        crashing on a route that does not exist here.
        """
        hardware = await self._fetch_hardware(client, base)
        rpc = await self._fetch_rpc(client, base)
        node = NodeSnapshot(
            name=self.config.node.name,
            role="worker",
            reachable=hardware is not None,
            hardware=hardware,
            rpc=rpc,
            error=None if hardware is not None else "could not reach /agent/hardware",
        )
        return ClusterSnapshot(
            fetched_at=time.time(),
            cluster=None,
            nodes=[node],
            coordinator_error="not a coordinator: this node exposes /agent/* only",
            log_lines=await self._fetch_logs(client, base),
        )

    async def _head_snapshot(
        self, client: httpx.AsyncClient, base: str, cluster: ClusterStatus
    ) -> NodeSnapshot:
        hardware = await self._fetch_hardware(client, base)
        backend = await self._fetch_backend(client, base)
        placement = await self._fetch_placement(client, base)
        return NodeSnapshot(
            name=cluster.head_node,
            role="head",
            reachable=hardware is not None,
            hardware=hardware,
            backend=backend,
            placement=placement,
            layers=_layers_for(cluster, cluster.head_node),
            error=None
            if hardware is not None
            else "could not reach this node's own /agent/hardware",
        )

    async def _worker_snapshot(
        self,
        client: httpx.AsyncClient,
        name: str,
        cluster: ClusterStatus,
        report: PeerReport | None,
    ) -> NodeSnapshot:
        if report is None:
            return NodeSnapshot(
                name=name,
                role="worker",
                reachable=False,
                error="peer is not (or no longer) resolvable from this node's config/discovery",
            )
        peer_base = f"http://{report.peer.host}:{report.peer.agent_port}"
        rpc = await self._fetch_rpc(client, peer_base)
        return NodeSnapshot(
            name=name,
            role="worker",
            reachable=report.reachable,
            hardware=report.hardware,
            rpc=rpc,
            layers=_layers_for(cluster, name),
            error=report.error,
        )

    async def _models(self, client: httpx.AsyncClient, base: str) -> tuple[list[str], str | None]:
        try:
            response = await client.get(f"{base}/cluster/models", timeout=10.0)
            response.raise_for_status()
        except httpx.HTTPError:
            return [], None
        payload = response.json()
        available: list[str] = payload.get("available") or []
        loaded: str | None = payload.get("loaded")
        return available, loaded

    async def _fetch_hardware(self, client: httpx.AsyncClient, base: str) -> NodeHardware | None:
        try:
            response = await client.get(f"{base}/agent/hardware", timeout=10.0)
            response.raise_for_status()
        except httpx.HTTPError:
            return None
        return NodeHardware.model_validate(response.json())

    async def _fetch_backend(self, client: httpx.AsyncClient, base: str) -> BackendStatus | None:
        try:
            response = await client.get(f"{base}/agent/backend", timeout=10.0)
            response.raise_for_status()
        except httpx.HTTPError:
            return None
        return BackendStatus.model_validate(response.json())

    async def _fetch_rpc(self, client: httpx.AsyncClient, base: str) -> RpcStatus | None:
        try:
            response = await client.get(f"{base}/agent/rpc", timeout=10.0)
            response.raise_for_status()
        except httpx.HTTPError:
            return None
        return RpcStatus.model_validate(response.json())

    async def _fetch_placement(self, client: httpx.AsyncClient, base: str) -> dict[str, list[int]]:
        try:
            response = await client.get(f"{base}/agent/backend/placement", timeout=10.0)
            response.raise_for_status()
        except httpx.HTTPError:
            return {}
        layers: dict[str, list[int]] = response.json().get("layers", {})
        return layers

    async def _fetch_logs(self, client: httpx.AsyncClient, base: str) -> list[str]:
        """Recent output from this node's own backend, for a log pane."""
        try:
            response = await client.get(f"{base}/agent/backend/logs", timeout=10.0)
            response.raise_for_status()
        except httpx.HTTPError:
            return []
        lines: list[str] = response.json().get("lines", [])
        return lines

    async def _fetch_download_status(
        self, client: httpx.AsyncClient, base: str
    ) -> DownloadStatus | None:
        try:
            response = await client.get(f"{base}/cluster/models/download", timeout=10.0)
            response.raise_for_status()
        except httpx.HTTPError:
            return None
        return DownloadStatus.model_validate(response.json())

    # -- controls: thin wrappers over the existing /cluster/* endpoints ----

    async def start_cluster(self, model: str | None = None) -> ClusterStatus:
        client = await self._get_client()
        base = api_base_url(self.config)
        body = {"model": model} if model else None
        response = await client.post(f"{base}/cluster/start", json=body, timeout=900.0)
        response.raise_for_status()
        return ClusterStatus.model_validate(response.json())

    async def stop_cluster(self) -> ClusterStatus:
        client = await self._get_client()
        base = api_base_url(self.config)
        response = await client.post(f"{base}/cluster/stop", timeout=60.0)
        response.raise_for_status()
        return ClusterStatus.model_validate(response.json())

    async def switch_model(self, model: str) -> ClusterStatus:
        client = await self._get_client()
        base = api_base_url(self.config)
        response = await client.post(f"{base}/cluster/model", json={"model": model}, timeout=900.0)
        response.raise_for_status()
        return ClusterStatus.model_validate(response.json())

    async def search_models(self, query: str) -> list[hfhub.HFModelSummary]:
        client = await self._get_client()
        base = api_base_url(self.config)
        response = await client.get(
            f"{base}/cluster/models/search", params={"q": query}, timeout=30.0
        )
        response.raise_for_status()
        return [hfhub.HFModelSummary(**r) for r in response.json()["results"]]

    async def repo_files(self, repo_id: str) -> list[hfhub.HFFile]:
        client = await self._get_client()
        base = api_base_url(self.config)
        response = await client.get(
            f"{base}/cluster/models/repo-files", params={"repo_id": repo_id}, timeout=30.0
        )
        response.raise_for_status()
        return [hfhub.HFFile(**f) for f in response.json()["files"]]

    async def download_model(self, repo_id: str, filename: str) -> DownloadStatus:
        """Fire-and-forget: the server does not await the download itself."""
        client = await self._get_client()
        base = api_base_url(self.config)
        response = await client.post(
            f"{base}/cluster/models/download",
            json={"repo_id": repo_id, "filename": filename},
            timeout=30.0,
        )
        response.raise_for_status()
        return DownloadStatus.model_validate(response.json())


def _layers_for(cluster: ClusterStatus, node_name: str) -> int:
    if cluster.plan is None:
        return 0
    return sum(p.layers for p in cluster.plan.placement if p.node == node_name)


__all__ = ["ClusterPoller", "ClusterSnapshot", "NodeSnapshot"]
