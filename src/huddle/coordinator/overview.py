"""Every node in one response, for the web UI's cluster page.

A browser cannot reach peer agents itself — they bind private addresses and
send no CORS headers — so the coordinator gathers what each agent reports and
hands the page a single document. It reuses what already exists: the same
fresh ``/agent/hardware`` probe placement uses, ``/agent/rpc``, the plan in
``ClusterStatus``, and llama.cpp's own placement report. It never plans, never
browses for peers, and changes nothing.
"""

from __future__ import annotations

import asyncio
import socket
import time
from typing import Literal

import httpx
from pydantic import BaseModel, ValidationError

from huddle.agent.service import RpcStatus
from huddle.config import PeerConfig
from huddle.coordinator.service import ClusterPlan, ClusterService, ClusterStatus
from huddle.hardware import NodeHardware
from huddle.system import SystemInfo, probe_system

Role = Literal["head", "worker", "idle", "offline"]

# Per request to a peer. A peer that takes longer is not worth waiting for on
# a page that refreshes every few seconds; it shows as offline until it answers.
PEER_TIMEOUT = 3.0
# Connecting is the part a powered-off peer never finishes. On a LAN a live
# agent accepts in milliseconds, so waiting the full PEER_TIMEOUT for a dead one
# only makes every refresh slow (measured: 3.3 s per refresh with one offline).
PEER_CONNECT_TIMEOUT = 1.0
_TIMEOUT = httpx.Timeout(PEER_TIMEOUT, connect=PEER_CONNECT_TIMEOUT)
# However many tabs are open, probe at most this often.
CACHE_SECONDS = 1.5


class NodeView(BaseModel):
    """One node, as the cluster page draws it."""

    name: str
    role: Role
    # What the head dials to reach this node; None for the head itself.
    address: str | None = None
    agent_port: int | None = None
    # Network round trip from the head, in milliseconds: the time to open a TCP
    # connection to the agent, which is one SYN/SYN-ACK — what ping would say.
    rtt_ms: float | None = None
    hardware: NodeHardware | None = None
    system: SystemInfo | None = None
    rpc: RpcStatus | None = None
    layers: int = 0
    error: str | None = None


class ClusterOverview(BaseModel):
    """The whole cluster at one moment."""

    cluster: ClusterStatus
    tokens_per_sec: float | None = None
    backend_port: int
    nodes: list[NodeView]
    # Layer indices per device, as llama.cpp itself reported placing them.
    # Empty unless the head runs with -lv 5; the plan is the fallback.
    placement: dict[str, list[int]] = {}
    generated_at: float


def layers_by_node(plan: ClusterPlan | None) -> dict[str, int]:
    """Transformer layers each node holds under the current plan."""
    totals: dict[str, int] = {}
    for device in plan.placement if plan else []:
        totals[device.node] = totals.get(device.node, 0) + device.layers
    return totals


def _describe(exc: httpx.HTTPError) -> str:
    if isinstance(exc, httpx.ConnectTimeout):
        return f"no answer within {PEER_CONNECT_TIMEOUT:.0f}s"
    if isinstance(exc, httpx.TimeoutException):
        return f"no answer within {PEER_TIMEOUT:.0f}s"
    detail = str(exc) or type(exc).__name__
    return f"{type(exc).__name__}: {detail}"


def _connect_ms(host: str, port: int, attempts: int = 2) -> float | None:
    """Best of ``attempts`` TCP connects, in milliseconds, timed around the syscall."""
    best: float | None = None
    for _ in range(attempts):
        started = time.perf_counter()
        try:
            with socket.create_connection((host, port), timeout=PEER_CONNECT_TIMEOUT):
                elapsed = (time.perf_counter() - started) * 1000
        except OSError:
            break
        best = elapsed if best is None else min(best, elapsed)
    return round(best, 3) if best is not None else None


async def tcp_rtt(host: str, port: int) -> float | None:
    """Milliseconds to open a TCP connection: one network round trip.

    Timing an HTTP request instead measures mostly the client — a fresh httpx
    client's first request to a localhost agent took ~400 ms here, where curl
    took 1.4 ms. Timing a connect on the event loop is not much better: it
    includes however long the loop takes to get back to it, 50-100 ms while
    other replies are being parsed. So the connect is timed in a thread, and
    the best of two is kept, as ping would.
    """
    return await asyncio.to_thread(_connect_ms, host, port)


async def _fetch[M: BaseModel](client: httpx.AsyncClient, url: str, model: type[M]) -> M | None:
    """One optional part of a node's report; missing or malformed is just absent."""
    try:
        response = await client.get(url, timeout=_TIMEOUT)
        response.raise_for_status()
        return model.model_validate(response.json())
    except (httpx.HTTPError, ValidationError, ValueError):
        return None


async def peer_view(
    client: httpx.AsyncClient, peer: PeerConfig, workers: list[str], layers: dict[str, int]
) -> NodeView:
    """Ask one peer's agent about itself. Never raises: offline is an answer."""
    base = f"http://{peer.host}:{peer.agent_port}"
    view = NodeView(
        name=peer.name,
        role="offline",
        address=peer.host,
        agent_port=peer.agent_port,
        layers=layers.get(peer.name, 0),
    )
    try:
        response = await client.get(f"{base}/agent/health", timeout=_TIMEOUT)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        view.error = _describe(exc)
        return view

    rtt, hardware, system, rpc = await asyncio.gather(
        tcp_rtt(peer.host, peer.agent_port),
        _fetch(client, f"{base}/agent/hardware", NodeHardware),
        # Absent on a peer running older Huddle; the page says so.
        _fetch(client, f"{base}/agent/system", SystemInfo),
        _fetch(client, f"{base}/agent/rpc", RpcStatus),
    )
    view.role = "worker" if peer.name in workers else "idle"
    view.rtt_ms, view.hardware, view.system, view.rpc = rtt, hardware, system, rpc
    view.error = hardware.error if hardware else "the agent answered but sent no hardware report"
    return view


class OverviewService:
    """Builds ``ClusterOverview`` documents, at most one per ``CACHE_SECONDS``."""

    def __init__(self, cluster: ClusterService) -> None:
        self.cluster = cluster
        self._lock = asyncio.Lock()
        self._cached: ClusterOverview | None = None
        self._cached_at = 0.0

    def _fresh(self) -> ClusterOverview | None:
        if self._cached and time.monotonic() - self._cached_at < CACHE_SECONDS:
            return self._cached
        return None

    async def overview(self) -> ClusterOverview:
        cached = self._fresh()
        if cached is not None:
            return cached
        # Callers arriving while one is being built wait for it, then share it.
        async with self._lock:
            cached = self._fresh()
            if cached is not None:
                return cached
            self._cached = await self._build()
            self._cached_at = time.monotonic()
            return self._cached

    async def _build(self) -> ClusterOverview:
        cluster = self.cluster
        config = cluster.config
        backend = cluster.backend
        status = cluster.status()
        layers = layers_by_node(status.plan)

        hardware, system = await asyncio.gather(
            asyncio.to_thread(backend.hardware), asyncio.to_thread(probe_system)
        )
        head = NodeView(
            name=config.node.name,
            role="head",
            hardware=hardware,
            system=system,
            layers=layers.get(config.node.name, 0),
            error=hardware.error,
        )

        peers = cluster.known_peers
        async with httpx.AsyncClient() as client:
            views = await asyncio.gather(
                *(peer_view(client, peer, status.workers, layers) for peer in peers)
            )

        return ClusterOverview(
            cluster=status,
            tokens_per_sec=backend.status().tokens_per_sec,
            backend_port=config.backend.port,
            nodes=[head, *views],
            placement=backend.placement(),
            generated_at=time.time(),
        )


__all__ = [
    "ClusterOverview",
    "NodeView",
    "OverviewService",
    "layers_by_node",
    "peer_view",
    "tcp_rtt",
]
