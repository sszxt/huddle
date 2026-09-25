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
from collections.abc import Mapping

import httpx
from pydantic import BaseModel

from huddle.agent.service import BackendService
from huddle.config import HuddleConfig, PeerConfig
from huddle.coordinator.cluster import PeerReport, build_device_list, query_peers
from huddle.coordinator.planner import PlacementDevice, Plan, plan_placement
from huddle.gguf import ModelInfo, read_gguf
from huddle.hardware import NodeHardware, probe
from huddle.llamacpp import detect_out_of_memory
from huddle.process import ProcessError

log = logging.getLogger("huddle.cluster")

# e.g. model-00002-of-00005.gguf
_SHARD = re.compile(r"-\d{5}-of-\d{5}\.gguf$")


class PeersNotReady(ProcessError):
    """A known peer is unreachable while the cluster is still allowed to wait."""


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
    # What -ngl is set to: the GPU layers plus the output head.
    ngl: int = 0
    output_device: str | None = None
    per_layer_mib: float
    tensor_split: list[float]
    rpc_endpoints: list[str]
    placement: list[DevicePlacement]
    unreachable: list[str] = []
    warnings: list[str] = []
    # Margin added after real out-of-memory failures, by device id. Non-empty
    # means the first estimate was wrong and the plan was corrected.
    extra_reserve_mib: dict[str, float] = {}


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
    # False once the restart limit is exhausted: wanted, down, and no longer
    # being retried. That combination needs a human.
    supervising: bool = False
    # A start is in progress right now. Loading a large model takes minutes,
    # and that is not the same thing as being broken.
    starting: bool = False

    @property
    def degraded(self) -> bool:
        return self.desired and not self.running and not self.starting


class ClusterService:
    """Plans a placement, then starts the processes that realise it."""

    def __init__(
        self,
        config: HuddleConfig,
        backend: BackendService,
    ) -> None:
        self.config = config
        self.backend = backend
        supervisor = config.supervisor
        self.watch_interval = supervisor.watch_interval
        self.max_restarts = supervisor.max_restarts
        self.backoff_max = supervisor.backoff_max
        self.stable_after = supervisor.stable_after
        self.peer_wait = supervisor.peer_wait
        self.rejoin_interval = supervisor.rejoin_interval
        self._plan: ClusterPlan | None = None
        self._started_workers: list[PeerConfig] = []
        self._lock = asyncio.Lock()
        self._desired = False
        self._model_name: str | None = None
        self._restarts = 0
        self._failures_in_a_row = 0
        self._healthy_since: float | None = None
        self._last_failure: str | None = None
        self._watcher: asyncio.Task[None] | None = None
        # Peers as of the last plan: static config *plus* anything discovered.
        # Reading config.peers here instead would miss every discovered peer and
        # start no workers, while still passing their endpoints to --rpc.
        self._resolved_peers: list[PeerConfig] = []
        # Until when a plan missing a known peer is refused. Set when an outage
        # begins, cleared once the cluster is up.
        self._peer_wait_until: float | None = None
        self._last_rejoin_check = 0.0
        # True from an autostart until the cluster first comes up. Completing
        # that start is not a recovery, so it must not count as a restart.
        self._initial_pending = False
        # Consecutive checks that found a worker gone. Two in a row are needed,
        # so a single slow answer does not restart a healthy cluster.
        self._worker_strikes = 0

    @property
    def known_peers(self) -> list[PeerConfig]:
        """Configured peers, then any discovered ones the last plan saw.

        For showing the cluster, not planning it: no discovery browse happens
        here, because a browse takes seconds and this is read on every refresh.
        """
        peers = list(self.config.peers)
        names = {peer.name for peer in peers}
        return peers + [peer for peer in self._resolved_peers if peer.name not in names]

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
            supervising=self._watcher is not None and not self._watcher.done(),
            starting=self._desired and not self.backend.running and self._lock.locked(),
        )

    async def build_plan(
        self,
        model: str | None = None,
        *,
        n_ctx: int | None = None,
        extra_reserve: Mapping[str, float] | None = None,
    ) -> ClusterPlan:
        """Work out where layers should go, without starting anything."""
        info: ModelInfo = read_gguf(self.config.models.resolve(model))
        local = probe(self.config.node.name, self.config.binaries.llama_server)
        reports = await query_peers(self.config)
        self._resolved_peers = [report.peer for report in reports]
        return plan_cluster(
            self.config,
            info,
            local,
            reports,
            n_ctx=n_ctx or self.config.backend.ctx_size,
            extra_reserve=extra_reserve,
        )

    async def start(self, model: str | None = None) -> ClusterStatus:
        """Start workers, then the head, and roll back if the head fails."""
        async with self._lock:
            await self._start_locked(model)
        self._mark_started_fresh()
        # Report after the bookkeeping, not before: a status captured inside the
        # lock would still show the restarts and failure this start just cleared.
        return self.status()

    def _still_waiting(self) -> bool:
        return self._peer_wait_until is not None and self._wait_left() > 0

    def _wait_left(self) -> float:
        if self._peer_wait_until is None:
            return 0.0
        return self._peer_wait_until - asyncio.get_running_loop().time()

    def _mark_started_fresh(self) -> None:
        """Bookkeeping after a start someone asked for.

        An explicit start is a human (or autostart) deciding to try again, so
        earlier restarts are forgiven — otherwise a cluster that once exhausted
        its limit would give up on the very next failure after being revived.
        """
        self._desired = True
        self._restarts = 0
        self._failures_in_a_row = 0
        self._up(asyncio.get_running_loop().time())
        self._start_watching()

    async def start_with_retry(self, model: str | None = None) -> ClusterStatus | None:
        """Start, and keep trying in the background if the first attempt fails.

        Autostart failures are usually transient: a peer still booting, a network
        not yet carrying multicast. Marking the cluster as wanted *before*
        attempting means the supervisor keeps retrying instead of leaving a dead
        node that needs a human. Returns None if the first attempt failed.
        """
        self._desired = True
        self._model_name = model
        self._restarts = 0
        self._failures_in_a_row = 0
        loop = asyncio.get_running_loop()
        self._peer_wait_until = loop.time() + self.peer_wait
        self._initial_pending = True
        self._start_watching()
        try:
            async with self._lock:
                await self._start_locked(model, wait_for_peers=True)
        except PeersNotReady as exc:
            self._last_failure = str(exc)
            log.info("%s; retrying shortly", exc)
            return None
        except Exception as exc:
            self._last_failure = f"initial start failed: {exc}"
            log.error("%s; retrying in the background", self._last_failure)
            return None
        self._up(loop.time())
        status = self.status()
        log.info(
            "cluster ready: model=%s workers=%s",
            status.model,
            ",".join(status.workers) or "none",
        )
        return status

    def _up(self, now: float) -> None:
        """Bookkeeping for a cluster that has just come up."""
        self._healthy_since = now
        self._peer_wait_until = None
        self._last_failure = None
        self._last_rejoin_check = now
        self._initial_pending = False

    async def _start_locked(
        self, model: str | None = None, *, wait_for_peers: bool = False
    ) -> ClusterStatus:
        """Bring the cluster up. Caller holds the lock.

        With ``wait_for_peers``, a plan that leaves out an unreachable known
        peer is refused while the wait window is open: a peer that is merely
        slower to boot must not shrink the cluster for the rest of its life.
        Explicit starts do not wait — a person asking for a start wants it now.
        """
        if self.backend.running:
            raise ProcessError("cluster is already running; stop it first")

        planner = self.config.planner
        extra: dict[str, float] = {}
        attempts = planner.oom_retries + 1

        for attempt in range(1, attempts + 1):
            plan = await self.build_plan(model, extra_reserve=extra)
            if wait_for_peers and plan.unreachable and self._still_waiting():
                raise PeersNotReady(
                    f"waiting for peers to come up: {', '.join(plan.unreachable)} "
                    f"unreachable ({self._wait_left():.0f}s left before starting without them)"
                )
            for warning in plan.warnings:
                log.warning("%s", warning)

            # Only peers actually carrying layers are worth starting — and from
            # the resolved list, which includes discovered peers. config.peers
            # is empty when discovery is doing the work.
            wanted = _peers_with_layers(self._resolved_peers, plan)
            await self._start_workers(wanted)

            try:
                await self.backend.start(
                    model,
                    rpc_endpoints=plan.rpc_endpoints or None,
                    tensor_split=plan.tensor_split or None,
                    n_gpu_layers=plan.ngl,
                )
            except Exception as exc:
                # Leaving workers up would hold their ports and confuse the next
                # attempt into thinking a stale cluster is healthy.
                await self._stop_workers()
                oom = detect_out_of_memory(self.backend.logs())
                if not oom.detected:
                    raise
                if attempt == attempts:
                    raise ProcessError(
                        f"out of device memory on {_names(oom.devices)} after "
                        f"{attempts} attempts (extra margin tried: {extra or 'none'}); "
                        f"lower backend.ctx_size or raise planner.reserve_mib"
                    ) from exc
                # Back off where the estimate was wrong. If the log does not say
                # which device ran out, every device that held layers shares it.
                targets = oom.devices or {p.id for p in plan.placement if p.layers > 0}
                for device_id in targets:
                    extra[device_id] = extra.get(device_id, 0.0) + planner.oom_step_mib
                log.warning(
                    "load ran out of memory on %s; replanning with more margin %s (attempt %d/%d)",
                    _names(oom.devices),
                    extra,
                    attempt + 1,
                    attempts,
                )
                continue

            self._plan = plan
            self._model_name = model
            return self.status()

        raise AssertionError("unreachable: the loop returns or raises")

    async def switch_model(self, model: str) -> ClusterStatus:
        """Load a different model, replanning the split for it.

        A restart is unavoidable — llama.cpp holds one model per process — but it
        must not be the caller's problem. Replanning is required rather than
        cosmetic: a different model has different layer sizes, so reusing the old
        split would misjudge every device.
        """
        async with self._lock:
            await self._teardown()
            await self._start_locked(model)
        self._mark_started_fresh()
        return self.status()

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

        Failed restarts back off exponentially, and a long healthy run forgives
        earlier restarts, so the limit bounds a crash loop rather than a
        lifetime.
        """
        loop = asyncio.get_running_loop()
        delay = self.watch_interval
        while True:
            await asyncio.sleep(delay)
            delay = self.watch_interval
            if not self._desired or self._lock.locked():
                continue

            if self.backend.running:
                if await self._lost_workers():
                    delay = 0.0  # handle it on the next pass, as a head outage
                    continue
                self._forgive_if_stable(loop.time())
                await self._maybe_rejoin(loop.time())
                continue

            self._healthy_since = None
            if self._peer_wait_until is None:
                # A new outage: give slow peers the same grace as a fresh boot.
                self._peer_wait_until = loop.time() + self.peer_wait
            returncode = self.backend.status().returncode
            if self._restarts >= self.max_restarts:
                # Stay `desired`: the cluster is still wanted, so /health must
                # keep reporting it as degraded rather than deliberately down.
                self._last_failure = (
                    f"head is down (rc={returncode}) and the restart limit "
                    f"({self.max_restarts}) is exhausted; start the cluster again "
                    f"once the cause is fixed"
                )
                log.error("%s", self._last_failure)
                return

            # Every attempt counts once — except a wait for peers, and the
            # successful completion of an autostart, neither of which is a
            # recovery from anything.
            recovering = not self._initial_pending
            attempt = self._restarts + 1
            if recovering:
                log.error(
                    "head process is down (rc=%s); restart %d/%d",
                    returncode,
                    attempt,
                    self.max_restarts,
                )
            try:
                async with self._lock:
                    await self._teardown()
                    await self._start_locked(self._model_name, wait_for_peers=True)
            except PeersNotReady as exc:
                # Waiting is not failing: it spends no restart budget and does
                # not back off, but checks again soon.
                self._last_failure = str(exc)
                log.info("%s", exc)
                delay = min(self.watch_interval * 2, 15.0)
                continue
            except Exception as exc:
                self._restarts += 1
                self._failures_in_a_row += 1
                delay = min(self.watch_interval * 2**self._failures_in_a_row, self.backoff_max)
                self._last_failure = f"start attempt {attempt} failed: {exc}"
                log.error("%s; next attempt in %.0fs", self._last_failure, delay)
                continue

            if recovering:
                self._restarts += 1
            self._failures_in_a_row = 0
            self._up(loop.time())
            log.info("cluster up (%s)", f"restart {attempt}" if recovering else "startup complete")

    async def _lost_workers(self) -> bool:
        """Stop the head if a worker it depends on has gone.

        The head only notices a lost worker when it next touches a remote layer,
        and then the request fails: on the real cluster `/health` said "ok" while
        the next completion returned 500. Asking each worker's agent — rather
        than probing the RPC port, which the head is connected to and which
        must not be flooded with extra connections — catches it while idle.
        Returns True when the head was stopped so the restart path takes over.
        """
        if not self._started_workers:
            self._worker_strikes = 0
            return False

        async with httpx.AsyncClient(timeout=5.0) as client:
            states = [(peer, await _worker_state(client, peer)) for peer in self._started_workers]
        gone = [peer.name for peer, alive in states if alive is False]
        if not gone:
            self._worker_strikes = 0
            return False

        self._worker_strikes += 1
        if self._worker_strikes < 2:
            return False
        self._worker_strikes = 0

        self._last_failure = (
            f"worker lost on {', '.join(gone)}; the head cannot use its remote layers"
        )
        log.error("%s; restarting the cluster before a request fails", self._last_failure)
        async with self._lock:
            await self.backend.stop()
        return True

    async def _maybe_rejoin(self, now: float) -> None:
        """Grow the cluster when capacity it could use becomes reachable.

        Only while layers are running on CPU, and only for a reachable peer the
        plan is not already using that can hold at least one layer — so a model
        that fits already, or a peer that cannot help, never triggers a restart.
        A fresh plan cannot be used to decide this: the loaded model holds the
        memory it would measure.
        """
        plan = self._plan
        if self.rejoin_interval <= 0 or plan is None or plan.n_gpu_layers >= plan.n_layers:
            return
        if now - self._last_rejoin_check < self.rejoin_interval:
            return
        self._last_rejoin_check = now

        try:
            reports = await query_peers(self.config, timeout=5.0)
        except Exception as exc:
            log.debug("rejoin check failed: %s", exc)
            return

        in_use = set(plan.rpc_endpoints)
        headroom = self.config.planner.headroom
        helpful = [
            report.peer.name
            for report in reports
            if report.hardware is not None
            and report.peer.rpc_endpoint not in in_use
            and any(
                not d.is_rpc
                and not d.unified_memory
                and (d.free_mib or 0) * (1 - headroom) >= plan.per_layer_mib
                for d in report.hardware.devices
            )
        ]
        if not helpful:
            return

        log.info(
            "%d layer(s) are on CPU and %s can take some; replanning to include it",
            plan.n_layers - plan.n_gpu_layers,
            ", ".join(helpful),
        )
        try:
            async with self._lock:
                await self._teardown()
                await self._start_locked(self._model_name)
        except Exception as exc:
            self._last_failure = f"rejoin failed: {exc}"
            log.error("%s", self._last_failure)
            return
        self._up(now)

    def _forgive_if_stable(self, now: float) -> None:
        if self._restarts == 0 or self._healthy_since is None:
            return
        if now - self._healthy_since >= self.stable_after:
            log.info(
                "cluster stable for %.0fs; forgiving %d earlier restart(s)",
                now - self._healthy_since,
                self._restarts,
            )
            self._restarts = 0

    async def _start_workers(self, peers: list[PeerConfig]) -> None:
        async with httpx.AsyncClient(timeout=60.0) as client:
            for peer in peers:
                url = f"http://{peer.host}:{peer.agent_port}/agent/rpc/start"
                try:
                    response = await client.post(url)
                    if response.status_code == 409 and not await _worker_listening(client, peer):
                        # 409 covers "already running" but also "cannot start",
                        # e.g. no rpc_server binary. Only the first leaves
                        # anything for llama.cpp to connect to.
                        raise ProcessError(
                            f"peer {peer.name} could not start its worker: {response.text[:200]}"
                        )
                    if response.status_code not in (200, 409):
                        raise ProcessError(
                            f"peer {peer.name} refused to start its worker: "
                            f"{response.status_code} {response.text[:200]}"
                        )
                except httpx.HTTPError as exc:
                    await self._stop_workers()
                    raise ProcessError(f"peer {peer.name} unreachable: {exc}") from exc
                except ProcessError:
                    await self._stop_workers()
                    raise
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


async def _worker_state(client: httpx.AsyncClient, peer: PeerConfig) -> bool | None:
    """Whether a peer has a worker listening, whoever started it.

    None when its agent cannot be asked: an unanswered question is not evidence
    that the worker is gone.
    """
    try:
        response = await client.get(f"http://{peer.host}:{peer.agent_port}/agent/rpc")
        status = response.json()
    except (httpx.HTTPError, ValueError):
        return None
    return bool(status.get("running") or status.get("foreign"))


async def _worker_listening(client: httpx.AsyncClient, peer: PeerConfig) -> bool:
    return await _worker_state(client, peer) is True


def plan_cluster(
    config: HuddleConfig,
    info: ModelInfo,
    local: NodeHardware,
    reports: list[PeerReport],
    *,
    n_ctx: int,
    extra_reserve: Mapping[str, float] | None = None,
) -> ClusterPlan:
    """Plan a placement from hardware that has already been gathered.

    Separate from gathering so a caller that has the peer reports already —
    `huddle doctor` — can plan without browsing the network a second time.
    """
    reachable = [report for report in reports if report.reachable]
    devices, endpoints = build_device_list(local, reachable)
    if not devices:
        raise ProcessError("no usable devices found on this node or any peer")
    plan = _plan_for(config, info, devices, n_ctx, extra_reserve)

    # Drop peers that were given no layers and plan again. Every endpoint in
    # --rpc is one llama.cpp connects to at load and depends on afterwards, so a
    # peer holding nothing is pure liability: it can still take the head down
    # with it, and buys nothing. Re-planning is required rather than cosmetic,
    # because removing a peer removes its device slots and shifts every later
    # --tensor-split position.
    used = {
        device.node
        for device, layers in zip(plan.devices, plan.layers_per_device, strict=True)
        if layers > 0
    }
    active = [report for report in reachable if report.peer.name in used]
    if len(active) != len(reachable):
        devices, endpoints = build_device_list(local, active)
        plan = _plan_for(config, info, devices, n_ctx, extra_reserve)

    result = _to_cluster_plan(info, plan, endpoints, reports)
    result.extra_reserve_mib = dict(extra_reserve or {})
    return result


def _plan_for(
    config: HuddleConfig,
    info: ModelInfo,
    devices: list[PlacementDevice],
    n_ctx: int,
    extra_reserve: Mapping[str, float] | None = None,
) -> Plan:
    return plan_placement(
        info,
        devices,
        n_ctx=n_ctx,
        headroom=config.planner.headroom,
        use_unified_memory=config.planner.use_unified_memory,
        reserve_mib=config.planner.reserve_mib,
        extra_reserve_mib=extra_reserve,
    )


def _names(devices: frozenset[str]) -> str:
    return ", ".join(sorted(devices)) if devices else "an unnamed device"


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
        ngl=plan.ngl,
        output_device=plan.output_device,
        per_layer_mib=round(plan.per_layer_mib, 1),
        tensor_split=plan.tensor_split,
        rpc_endpoints=endpoints,
        placement=placement,
        unreachable=[r.peer.name for r in reports if not r.reachable],
        warnings=plan.warnings,
    )


__all__ = ["ClusterPlan", "ClusterService", "ClusterStatus", "DevicePlacement", "PlacementDevice"]
