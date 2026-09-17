"""Bringing the whole cluster up and down.

The sequence matters. Workers must be listening before the head starts, because
the head connects to every RPC endpoint while loading and streams each remote
layer's weights across on the way. And if the head fails to come up, workers we
started have to be torn down again, or the next attempt hits ports already held.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import re

import httpx
from pydantic import BaseModel

from huddle.agent.service import BackendService
from huddle.config import HuddleConfig, PeerConfig
from huddle.coordinator.cluster import PeerReport, build_device_list, query_peers
from huddle.coordinator.planner import PlacementDevice, Plan, plan_placement
from huddle.gguf import ModelInfo, read_gguf
from huddle.hardware import probe
from huddle.process import ProcessError

log = logging.getLogger("huddle.cluster")

# e.g. model-00002-of-00005.gguf
_SHARD = re.compile(r"-\d{5}-of-\d{5}\.gguf$")


class DevicePlacement(BaseModel):
    """One device and what it was given."""

    id: str
    name: str
    node: str
    layers: int
    free_mib: int
    skipped: str | None = None


class ClusterPlan(BaseModel):
    """A placement decision, in a form the API can return."""

    model: str
    n_layers: int
    n_gpu_layers: int
    per_layer_mib: float
    tensor_split: list[float]
    rpc_endpoints: list[str]
    placement: list[DevicePlacement]
    unreachable: list[str] = []
    warnings: list[str] = []


class ClusterStatus(BaseModel):
    """What the cluster is currently doing."""

    running: bool
    model: str | None = None
    head_node: str
    workers: list[str] = []
    plan: ClusterPlan | None = None
    # `desired` is what we were asked for; `running` is what is true. They differ
    # exactly when something died, which is the state worth alerting on.
    desired: bool = False
    restarts: int = 0
    last_failure: str | None = None

    @property
    def degraded(self) -> bool:
        return self.desired and not self.running


class ClusterService:
    """Plans a placement, then starts the processes that realise it."""

    def __init__(
        self,
        config: HuddleConfig,
        backend: BackendService,
        *,
        watch_interval: float = 5.0,
        max_restarts: int = 5,
    ) -> None:
        self.config = config
        self.backend = backend
        self.watch_interval = watch_interval
        self.max_restarts = max_restarts
        self._plan: ClusterPlan | None = None
        self._started_workers: list[PeerConfig] = []
        self._lock = asyncio.Lock()
        self._desired = False
        self._model_name: str | None = None
        self._restarts = 0
        self._last_failure: str | None = None
        self._watcher: asyncio.Task[None] | None = None
        # Peers as of the last plan: static config *plus* anything discovered.
        # Reading config.peers here instead would miss every discovered peer and
        # start no workers, while still passing their endpoints to --rpc.
        self._resolved_peers: list[PeerConfig] = []

    def status(self) -> ClusterStatus:
        return ClusterStatus(
            running=self.backend.running,
            model=self.backend.status().model,
            head_node=self.config.node.name,
            workers=[peer.name for peer in self._started_workers],
            plan=self._plan,
            desired=self._desired,
            restarts=self._restarts,
            last_failure=self._last_failure,
        )

    async def build_plan(
        self, model: str | None = None, *, n_ctx: int | None = None
    ) -> ClusterPlan:
        """Work out where layers should go, without starting anything."""
        info: ModelInfo = read_gguf(self.config.models.resolve(model))
        local = probe(self.config.node.name, self.config.binaries.llama_server)
        reports = await query_peers(self.config)

        ctx = n_ctx or self.config.backend.ctx_size
        reachable = [report for report in reports if report.reachable]

        devices, endpoints = build_device_list(local, reachable)
        if not devices:
            raise ProcessError("no usable devices found on this node or any peer")
        plan = self._plan_for(info, devices, ctx)

        # Drop peers that were given no layers and plan again. Every endpoint in
        # --rpc is one llama.cpp connects to at load and depends on afterwards,
        # so a peer holding nothing is pure liability: it can still take the head
        # down with it, and buys nothing. Re-planning is required rather than
        # cosmetic, because removing a peer removes its device slots and shifts
        # every later --tensor-split position.
        used = {
            device.node
            for device, layers in zip(plan.devices, plan.layers_per_device, strict=True)
            if layers > 0
        }
        active = [report for report in reachable if report.peer.name in used]
        if len(active) != len(reachable):
            devices, endpoints = build_device_list(local, active)
            plan = self._plan_for(info, devices, ctx)

        self._resolved_peers = [report.peer for report in reports]
        return _to_cluster_plan(info, plan, endpoints, reports)

    def _plan_for(self, info: ModelInfo, devices: list[PlacementDevice], n_ctx: int) -> Plan:
        return plan_placement(
            info,
            devices,
            n_ctx=n_ctx,
            headroom=self.config.planner.headroom,
            use_unified_memory=self.config.planner.use_unified_memory,
        )

    async def start(self, model: str | None = None) -> ClusterStatus:
        """Start workers, then the head, and roll back if the head fails."""
        async with self._lock:
            status = await self._start_locked(model)
        self._desired = True
        self._start_watching()
        return status

    async def start_with_retry(self, model: str | None = None) -> ClusterStatus | None:
        """Start, and keep trying in the background if the first attempt fails.

        Autostart failures are usually transient: a peer still booting, a network
        not yet carrying multicast. Marking the cluster as wanted *before*
        attempting means the supervisor keeps retrying instead of leaving a dead
        node that needs a human. Returns None if the first attempt failed.
        """
        self._desired = True
        self._model_name = model
        self._start_watching()
        try:
            async with self._lock:
                return await self._start_locked(model)
        except Exception as exc:
            self._last_failure = f"initial start failed: {exc}"
            log.error("%s; retrying in the background", self._last_failure)
            return None

    async def _start_locked(self, model: str | None = None) -> ClusterStatus:
        """Bring the cluster up. Caller holds the lock."""
        if self.backend.running:
            raise ProcessError("cluster is already running; stop it first")

        plan = await self.build_plan(model)
        for warning in plan.warnings:
            log.warning("%s", warning)

        # Only peers actually carrying layers are worth starting — and from the
        # resolved list, which includes discovered peers. config.peers is empty
        # when discovery is doing the work.
        wanted = _peers_with_layers(self._resolved_peers, plan)
        await self._start_workers(wanted)

        try:
            await self.backend.start(
                model,
                rpc_endpoints=plan.rpc_endpoints or None,
                tensor_split=plan.tensor_split or None,
                n_gpu_layers=plan.n_gpu_layers,
            )
        except Exception:
            # Leaving workers up would hold their ports and confuse the next
            # attempt into thinking a stale cluster is healthy.
            await self._stop_workers()
            raise

        self._plan = plan
        self._model_name = model
        return self.status()

    async def switch_model(self, model: str) -> ClusterStatus:
        """Load a different model, replanning the split for it.

        A restart is unavoidable — llama.cpp holds one model per process — but it
        must not be the caller's problem. Replanning is required rather than
        cosmetic: a different model has different layer sizes, so reusing the old
        split would misjudge every device.
        """
        async with self._lock:
            await self._teardown()
            self._restarts = 0
            self._last_failure = None
            status = await self._start_locked(model)
        self._desired = True
        self._start_watching()
        return status

    def available_models(self) -> list[str]:
        """GGUF files this node can load, newest first."""
        directory = self.config.models.dir
        if not directory.is_dir():
            return []
        files = sorted(directory.glob("*.gguf"), key=lambda f: f.stat().st_mtime, reverse=True)
        # A split model is loaded by naming its first shard; listing the others
        # would offer loads that fail.
        return [f.name for f in files if _SHARD.search(f.name) is None or "-00001-of-" in f.name]

    async def stop(self) -> ClusterStatus:
        """Stop the head first, then the workers it depends on."""
        self._desired = False
        await self._stop_watching()
        async with self._lock:
            await self._teardown()
            return self.status()

    async def _teardown(self) -> None:
        """Tear the cluster down. Caller holds the lock."""
        await self.backend.stop()
        await self._stop_workers()
        self._plan = None

    # -- supervision ------------------------------------------------------

    def _start_watching(self) -> None:
        if self._watcher is None or self._watcher.done():
            self._watcher = asyncio.create_task(self._watch())

    async def _stop_watching(self) -> None:
        if self._watcher is None:
            return
        self._watcher.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._watcher
        self._watcher = None

    async def _watch(self) -> None:
        """Restart the cluster when the head dies.

        Losing a worker takes the head down with it — llama.cpp core-dumps
        rather than failing gracefully — so head death is the single signal
        worth watching. Recovery has to rebuild the whole thing: the worker that
        died must come back before the head can reconnect to it.
        """
        while True:
            await asyncio.sleep(self.watch_interval)
            if not self._desired or self.backend.running or self._lock.locked():
                continue

            returncode = self.backend.status().returncode
            if self._restarts >= self.max_restarts:
                self._last_failure = (
                    f"head died (rc={returncode}) and the restart limit "
                    f"({self.max_restarts}) is exhausted; not retrying"
                )
                log.error("%s", self._last_failure)
                self._desired = False
                return

            self._restarts += 1
            log.error(
                "head process died (rc=%s); restart %d/%d",
                returncode,
                self._restarts,
                self.max_restarts,
            )
            try:
                async with self._lock:
                    await self._teardown()
                    await self._start_locked(self._model_name)
                self._last_failure = None
                log.info("cluster recovered after restart %d", self._restarts)
            except Exception as exc:
                self._last_failure = f"restart {self._restarts} failed: {exc}"
                log.error("%s", self._last_failure)

    async def _start_workers(self, peers: list[PeerConfig]) -> None:
        async with httpx.AsyncClient(timeout=60.0) as client:
            for peer in peers:
                url = f"http://{peer.host}:{peer.agent_port}/agent/rpc/start"
                try:
                    response = await client.post(url)
                    # 409 means it is already running, which is what we want.
                    if response.status_code not in (200, 409):
                        raise ProcessError(
                            f"peer {peer.name} refused to start its worker: "
                            f"{response.status_code} {response.text[:200]}"
                        )
                except httpx.HTTPError as exc:
                    await self._stop_workers()
                    raise ProcessError(f"peer {peer.name} unreachable: {exc}") from exc
                self._started_workers.append(peer)

    async def _stop_workers(self) -> None:
        async with httpx.AsyncClient(timeout=30.0) as client:
            for peer in self._started_workers:
                url = f"http://{peer.host}:{peer.agent_port}/agent/rpc/stop"
                try:
                    await client.post(url)
                except httpx.HTTPError as exc:
                    # Worth logging but not worth failing a shutdown over.
                    log.warning("could not stop worker on %s: %s", peer.name, exc)
        self._started_workers = []


def _peers_with_layers(peers: list[PeerConfig], plan: ClusterPlan) -> list[PeerConfig]:
    """Peers whose endpoints appear in ``--rpc``.

    Every one of them must have a live worker: llama.cpp connects to each
    endpoint while loading, and an endpoint with nothing behind it fails the
    whole start. The plan has already dropped peers holding no layers, so this
    is exactly the set that needs starting.
    """
    wanted = set(plan.rpc_endpoints)
    return [peer for peer in peers if peer.rpc_endpoint in wanted]


def _to_cluster_plan(
    info: ModelInfo, plan: Plan, endpoints: list[str], reports: list[PeerReport]
) -> ClusterPlan:
    skipped = dict(plan.excluded)
    placement = [
        DevicePlacement(
            id=device.id,
            name=device.name,
            node=device.node,
            layers=layers,
            free_mib=device.free_mib,
            skipped=skipped.get(device.id),
        )
        for device, layers in zip(plan.devices, plan.layers_per_device, strict=True)
    ]
    return ClusterPlan(
        model=info.path.name,
        n_layers=info.n_layers,
        n_gpu_layers=plan.n_gpu_layers,
        per_layer_mib=round(plan.per_layer_mib, 1),
        tensor_split=plan.tensor_split,
        rpc_endpoints=endpoints,
        placement=placement,
        unreachable=[r.peer.name for r in reports if not r.reachable],
        warnings=plan.warnings,
    )


__all__ = ["ClusterPlan", "ClusterService", "ClusterStatus", "DevicePlacement", "PlacementDevice"]
