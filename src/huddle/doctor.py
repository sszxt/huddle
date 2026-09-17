"""Explain why a cluster will not start, or is not doing what it claims.

Every check here corresponds to a failure this project has actually hit, and
each took far longer to find by hand than it takes to test for:

- a peer built from a different llama.cpp (the RPC handshake rejects it late)
- discovery finding nobody, so a large model is planned onto one GPU
- an --rpc endpoint with no worker behind it (llama.cpp aborts)
- a plan that does not fit (out of memory at load)
- a worker that outlived its agent and holds the port
- llama.cpp placing layers other than where the plan said

A doctor that crashes is useless exactly when it is needed, so every check is
isolated and turns its own exceptions into a failing result.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import httpx

from huddle.config import HuddleConfig
from huddle.coordinator.cluster import PeerReport, query_peer, resolve_peers
from huddle.coordinator.service import plan_cluster
from huddle.discovery import discover
from huddle.gguf import GGUFError, ModelInfo, read_gguf
from huddle.hardware import NodeHardware, probe
from huddle.llamacpp import LlamaCppError, binary_version, same_build, version_commit
from huddle.process import tcp_is_open

PIN_FILE = Path(__file__).resolve().parents[2] / "llamacpp.pin"
GIB = 1024**3


class Level(StrEnum):
    OK = "ok"
    WARN = "warn"
    FAIL = "fail"
    SKIP = "skip"


@dataclass
class Check:
    name: str
    level: Level
    summary: str
    details: list[str] = field(default_factory=list)
    hint: str | None = None


@dataclass
class Report:
    node: str
    checks: list[Check] = field(default_factory=list)

    @property
    def healthy(self) -> bool:
        return all(check.level is not Level.FAIL for check in self.checks)

    def as_dict(self) -> dict[str, Any]:
        return {
            "node": self.node,
            "healthy": self.healthy,
            "checks": [
                {
                    "name": c.name,
                    "level": c.level.value,
                    "summary": c.summary,
                    "details": c.details,
                    "hint": c.hint,
                }
                for c in self.checks
            ],
        }


def _size(num_bytes: int) -> str:
    if num_bytes >= GIB:
        return f"{num_bytes / GIB:.1f} GiB"
    return f"{num_bytes / 1024**2:.0f} MiB"


def read_pin(path: Path = PIN_FILE) -> str | None:
    """The llama.cpp commit every node is meant to be built from."""
    try:
        for line in path.read_text().splitlines():
            key, _, value = line.partition("=")
            if key.strip() == "LLAMACPP_REF" and value.strip():
                return value.strip()
    except OSError:
        return None
    return None


# -- local node ----------------------------------------------------------------


def check_binaries(config: HuddleConfig) -> Check:
    problems: list[str] = []
    notes: list[str] = []

    server = config.binaries.llama_server
    if not server.is_file():
        problems.append(f"llama-server not found: {server}")
    elif not os.access(server, os.X_OK):
        problems.append(f"llama-server is not executable: {server}")

    worker = config.binaries.rpc_server
    if worker is None:
        notes.append("binaries.rpc_server is not set, so this node cannot act as a worker")
    elif not worker.is_file():
        problems.append(f"rpc-server not found: {worker}")
    elif not os.access(worker, os.X_OK):
        problems.append(f"rpc-server is not executable: {worker}")

    if problems:
        return Check(
            "binaries",
            Level.FAIL,
            problems[0],
            details=problems[1:] + notes,
            hint="point binaries: at the build; scripts/bootstrap-node.sh prints the paths",
        )
    if notes:
        return Check("binaries", Level.WARN, notes[0])
    return Check("binaries", Level.OK, "llama-server and rpc-server found")


def check_build(config: HuddleConfig, pin: str | None) -> tuple[Check, str | None]:
    try:
        version = binary_version(config.binaries.llama_server)
    except LlamaCppError as exc:
        return Check("llama.cpp", Level.FAIL, str(exc)), None

    commit = version_commit(version)
    if commit is None:
        return (
            Check(
                "llama.cpp",
                Level.WARN,
                f"{version} (no commit, so builds cannot be compared across nodes)",
            ),
            version,
        )
    if pin and not (commit.startswith(pin) or pin.startswith(commit)):
        return (
            Check(
                "llama.cpp",
                Level.WARN,
                f"built from {commit}, but llamacpp.pin says {pin[:12]}",
                hint="rebuild with scripts/bootstrap-node.sh; every node must share one commit",
            ),
            version,
        )
    suffix = " (matches llamacpp.pin)" if pin else ""
    return Check("llama.cpp", Level.OK, f"{version}{suffix}"), version


def check_devices(hardware: NodeHardware) -> Check:
    if hardware.error:
        return Check("devices", Level.FAIL, hardware.error)
    if not hardware.devices:
        return Check("devices", Level.WARN, "llama.cpp reports no devices; everything runs on CPU")

    usable = [d for d in hardware.devices if not d.is_rpc and not d.unified_memory]
    details = [f"{d.id}: {d.name} ({d.free_mib} MiB free)" for d in usable]
    details += [
        f"{d.id}: {d.name} (skipped: unified memory, its figure is system RAM)"
        for d in hardware.devices
        if d.unified_memory
    ]
    if not usable:
        return Check(
            "devices",
            Level.WARN,
            "no discrete GPU; integrated devices are skipped by the planner",
            details=details,
            hint="set planner.use_unified_memory: true to use them anyway",
        )
    return Check("devices", Level.OK, f"{len(usable)} usable GPU(s)", details=details)


def check_model(config: HuddleConfig) -> tuple[Check, ModelInfo | None]:
    try:
        path = config.models.resolve()
    except ValueError:
        return Check("model", Level.SKIP, "no default model configured"), None
    if not path.exists():
        return (
            Check(
                "model",
                Level.FAIL,
                f"model not found: {path}",
                hint="check models.dir and models.default",
            ),
            None,
        )
    try:
        info = read_gguf(path)
    except (GGUFError, OSError) as exc:
        return Check("model", Level.FAIL, f"cannot read {path.name}: {exc}"), None
    return (
        Check(
            "model",
            Level.OK,
            f"{info.name or path.name} [{info.architecture}, {info.quantization}], "
            f"{info.n_layers} layers, {_size(info.file_size)}",
        ),
        info,
    )


# -- peers -----------------------------------------------------------------------


async def check_peers(
    config: HuddleConfig, local_version: str | None, *, timeout: float = 10.0
) -> tuple[list[Check], list[PeerReport]]:
    # Resolve without the version filter: a mismatched peer is exactly what the
    # doctor must report, and discovery would otherwise hide it.
    lenient = config.model_copy(deep=True)
    lenient.discovery.require_matching_version = False
    peers = await resolve_peers(lenient, local_version)

    if not peers:
        if config.discovery.enabled:
            return [
                Check(
                    "peers",
                    Level.WARN,
                    "discovery is on but found no peers",
                    hint=(
                        "peers must run the agent with discovery.enabled and an "
                        "advertise address; multicast must reach this node"
                    ),
                )
            ], []
        return [Check("peers", Level.SKIP, "none configured and discovery is off: single node")], []

    configured = {peer.name for peer in config.peers}
    summary = Check(
        "peers",
        Level.OK,
        f"{len(peers)} peer(s): {', '.join(p.name for p in peers)}",
        details=[
            f"{p.name} at {p.host} ({'configured' if p.name in configured else 'discovered'})"
            for p in peers
        ],
    )
    async with httpx.AsyncClient() as client:
        reports = list(
            await asyncio.gather(*(query_peer(client, peer, timeout=timeout) for peer in peers))
        )
        per_peer = [await check_peer(client, report, local_version) for report in reports]
    return [summary, *per_peer], reports


async def check_peer(
    client: httpx.AsyncClient, report: PeerReport, local_version: str | None
) -> Check:
    peer = report.peer
    name = f"peer {peer.name}"
    if not report.reachable or report.hardware is None:
        return Check(
            name,
            Level.FAIL,
            f"agent unreachable at {peer.host}:{peer.agent_port}",
            details=[report.error or "no response"],
            hint=(
                "is the agent running there? peers must be dialled on a stable "
                "address (Tailscale), not a LAN lease that moves"
            ),
        )

    hardware = report.hardware
    if hardware.error:
        return Check(name, Level.FAIL, hardware.error)

    if not same_build(local_version, hardware.llamacpp_version):
        return Check(
            name,
            Level.FAIL,
            f"llama.cpp build differs: {version_commit(local_version)} here, "
            f"{version_commit(hardware.llamacpp_version)} there",
            hint="the RPC handshake will reject it; rebuild every node from llamacpp.pin",
        )

    usable = [d for d in hardware.devices if not d.is_rpc and not d.unified_memory]
    details = [f"{d.id}: {d.name} ({d.free_mib} MiB free)" for d in usable]

    try:
        response = await client.get(f"http://{peer.host}:{peer.agent_port}/agent/rpc", timeout=10.0)
        worker = response.json() if response.status_code == 200 else {}
    except (httpx.HTTPError, ValueError):
        worker = {}
    if worker.get("foreign"):
        return Check(
            name,
            Level.WARN,
            f"port {worker.get('port')} is held by a worker its agent did not start",
            details=details,
            hint="a worker that outlived an agent restart; stop it before the next cluster start",
        )

    if not usable:
        return Check(name, Level.WARN, "reachable, but no usable GPU", details=details)
    return Check(
        name, Level.OK, f"reachable, same build, {len(usable)} usable GPU(s)", details=details
    )


# -- plan ------------------------------------------------------------------------


def check_plan(
    config: HuddleConfig, info: ModelInfo, local: NodeHardware, reports: list[PeerReport]
) -> Check:
    plan = plan_cluster(config, info, local, reports, n_ctx=config.backend.ctx_size)
    details = [
        f"{p.id:8} {p.node:10} {p.layers:3} layers"
        + (f"  (skipped: {p.skipped})" if p.skipped else "")
        for p in plan.placement
    ]
    details.append(
        f"-ngl {plan.ngl}  --tensor-split {','.join(f'{w:g}' for w in plan.tensor_split)}"
        + (f"  --rpc {','.join(plan.rpc_endpoints)}" if plan.rpc_endpoints else "")
    )
    details += [f"warning: {w}" for w in plan.warnings]

    if plan.n_gpu_layers == 0:
        return Check(
            "plan", Level.WARN, "nothing fits on a GPU; the model runs on CPU", details=details
        )
    if plan.n_gpu_layers < plan.n_layers:
        return Check(
            "plan",
            Level.WARN,
            f"{plan.n_gpu_layers}/{plan.n_layers} layers fit on GPUs; "
            f"{plan.n_layers - plan.n_gpu_layers} run on CPU",
            details=details,
            hint="more peers, a smaller quant, or a shorter backend.ctx_size would help",
        )
    devices = sum(1 for p in plan.placement if p.layers)
    return Check(
        "plan",
        Level.OK,
        f"all {plan.n_layers} layers on GPU across {devices} device(s)",
        details=details,
    )


# -- the running cluster ---------------------------------------------------------


async def check_cluster(api: httpx.AsyncClient) -> tuple[list[Check], dict[str, Any] | None]:
    try:
        response = await api.get("/cluster", timeout=10.0)
    except httpx.HTTPError:
        return [
            Check(
                "cluster",
                Level.SKIP,
                "huddle is not serving on this node",
                hint="start it (systemctl start huddle) to check the live cluster",
            )
        ], None
    if response.status_code == 404:
        return [Check("cluster", Level.SKIP, "agent-only node: no coordinator runs here")], None

    status: dict[str, Any] = response.json()
    if not status.get("desired"):
        return [Check("cluster", Level.SKIP, "the cluster has not been started")], status

    plan = status.get("plan") or {}
    workers = ", ".join(status.get("workers") or []) or "none"

    if not status.get("running"):
        if status.get("supervising"):
            hint = "the supervisor is still retrying"
        else:
            hint = "the restart limit is exhausted; fix the cause, then POST /cluster/start"
        return [
            Check(
                "cluster",
                Level.FAIL,
                "wanted but not running",
                details=[status.get("last_failure") or "no failure recorded"],
                hint=hint,
            )
        ], status

    checks = [Check("cluster", Level.OK, f"serving {status.get('model')}, workers: {workers}")]
    if status.get("restarts"):
        checks.append(
            Check(
                "recovery",
                Level.WARN,
                f"recovered from {status['restarts']} failure(s) since the last stable run",
            )
        )
    extra = plan.get("extra_reserve_mib") or {}
    if extra:
        checks.append(
            Check(
                "memory",
                Level.WARN,
                "the first plan ran out of device memory and was backed off",
                details=[f"{device}: +{mib:g} MiB margin" for device, mib in extra.items()],
                hint="raise planner.reserve_mib so the first plan fits",
            )
        )
    return checks, status


async def check_workers(status: dict[str, Any]) -> Check:
    endpoints: list[str] = (status.get("plan") or {}).get("rpc_endpoints") or []
    if not endpoints:
        return Check("workers", Level.SKIP, "single node: no workers in use")

    closed = []
    for endpoint in endpoints:
        host, _, port = endpoint.rpartition(":")
        if not await tcp_is_open(host, int(port), timeout=3.0):
            closed.append(endpoint)
    if closed:
        return Check(
            "workers",
            Level.FAIL,
            f"{len(closed)} of {len(endpoints)} --rpc endpoint(s) have nothing listening",
            details=closed,
            hint="llama.cpp aborts as soon as it needs a worker that is gone",
        )
    return Check("workers", Level.OK, f"all {len(endpoints)} worker endpoint(s) listening")


async def check_placement(api: httpx.AsyncClient, status: dict[str, Any]) -> Check:
    plan = status.get("plan") or {}
    try:
        response = await api.get("/agent/backend/placement", timeout=10.0)
        actual: dict[str, list[int]] = response.json().get("layers", {})
    except (httpx.HTTPError, ValueError):
        actual = {}
    if not actual:
        return Check(
            "placement",
            Level.SKIP,
            "llama.cpp did not report where layers went",
            hint="run the backend with backend.extra_args: ['-lv', '5'] to verify placement",
        )

    expected = {
        device["id"]: int(weight)
        for device, weight in zip(
            plan.get("placement", []), plan.get("tensor_split", []), strict=False
        )
        if weight
    }
    expected_cpu = plan.get("n_layers", 0) + 1 - plan.get("ngl", 0)
    if expected_cpu > 0:
        expected["CPU"] = expected_cpu
    got = {device: len(layers) for device, layers in actual.items()}

    if got != expected:
        return Check(
            "placement",
            Level.FAIL,
            "llama.cpp placed layers differently from the plan",
            details=[f"planned: {expected}", f"actual:  {got}"],
            hint=(
                "the --tensor-split order or weights do not match what llama.cpp "
                "enumerated; peers may be sharing layers they were not budgeted for"
            ),
        )
    return Check("placement", Level.OK, f"llama.cpp placed exactly what was planned: {got}")


# -- worker nodes ----------------------------------------------------------------


async def local_agent(config: HuddleConfig) -> dict[str, Any] | None:
    """This node's own agent, if it runs as a worker, with its worker state."""
    base = f"http://127.0.0.1:{config.node.agent_port}"
    try:
        async with httpx.AsyncClient(base_url=base, timeout=5.0) as client:
            health = await client.get("/agent/health")
            if health.status_code != 200:
                return None
            rpc = await client.get("/agent/rpc")
            return dict(rpc.json()) if rpc.status_code == 200 else {}
    except (httpx.HTTPError, ValueError):
        return None


def check_worker(config: HuddleConfig, rpc: dict[str, Any]) -> Check:
    port = config.node.agent_port
    if rpc.get("foreign"):
        return Check(
            "role",
            Level.WARN,
            f"worker agent on :{port}, but port {rpc.get('port')} is held by "
            f"a worker it did not start",
            hint="a worker that outlived an agent restart; stop it before the next cluster start",
        )
    state = "worker running" if rpc.get("running") else "no worker running (started on demand)"
    return Check("role", Level.OK, f"worker node: agent answering on :{port}, {state}")


async def check_visibility(config: HuddleConfig) -> Check:
    """Can coordinators find this worker, and which coordinators are around?"""
    if not config.discovery.enabled:
        return Check(
            "visibility",
            Level.SKIP,
            "discovery is off: a coordinator must list this node under peers",
        )
    found = await discover(config.discovery)
    me = next((p for p in found if p.name == config.node.name), None)
    heads = [p.name for p in found if p.is_coordinator]
    seen = f"coordinator(s) seen: {', '.join(heads)}" if heads else "no coordinator seen"
    if me is None:
        return Check(
            "visibility",
            Level.WARN,
            f"this node is not visible on the LAN; {seen}",
            hint="set discovery.advertise (or rpc.advertise); multicast must reach the LAN",
        )
    return Check(
        "visibility",
        Level.OK,
        f"advertising as {me.host}:{me.agent_port}; {seen}",
    )


# -- orchestration ---------------------------------------------------------------


async def _safely(name: str, run: Callable[[], Awaitable[Any]]) -> Any:
    """Run a check, turning any unexpected error into a failing result."""
    try:
        return await run()
    except Exception as exc:
        return Check(name, Level.FAIL, f"check crashed: {type(exc).__name__}: {exc}")


async def run_doctor(
    config: HuddleConfig, api: httpx.AsyncClient, *, pin: str | None = None
) -> Report:
    report = Report(node=config.node.name)
    add = report.checks.append

    add(check_binaries(config))

    local_version: str | None = None
    build, local_version = check_build(config, pin if pin is not None else read_pin())
    add(build)

    local = probe(config.node.name, config.binaries.llama_server)
    add(check_devices(local))

    model_check, info = check_model(config)
    add(model_check)

    cluster_checks, status = await check_cluster(api)
    running = bool(status and status.get("running"))

    worker = await local_agent(config) if status is None else None
    if worker is not None:
        # A worker has no peers and makes no plan; what matters is whether
        # coordinators can find and use it.
        add(check_worker(config, worker))

        async def visibility() -> Check:
            return await check_visibility(config)

        add(await _safely("visibility", visibility))
        return report

    async def peers() -> tuple[list[Check], list[PeerReport]]:
        return await check_peers(config, local_version)

    result = await _safely("peers", peers)
    if isinstance(result, Check):
        add(result)
        reports: list[PeerReport] = []
    else:
        peer_checks, reports = result
        report.checks.extend(peer_checks)

    if running:
        # A loaded model holds the memory a fresh plan would measure, so a plan
        # made now would be misleadingly pessimistic. The live plan is checked
        # against reality below instead.
        add(Check("plan", Level.SKIP, "the cluster is running; checking its live plan instead"))
    elif info is None or local.error:
        add(Check("plan", Level.SKIP, "needs a readable model and a working llama-server"))
    else:
        loaded_info = info

        async def plan() -> Check:
            return check_plan(config, loaded_info, local, reports)

        add(await _safely("plan", plan))

    report.checks.extend(cluster_checks)
    if running and status is not None:
        live = status

        async def workers() -> Check:
            return await check_workers(live)

        async def placement() -> Check:
            return await check_placement(api, live)

        add(await _safely("workers", workers))
        add(await _safely("placement", placement))

    return report


def api_base_url(config: HuddleConfig) -> str:
    """Where this node's own Huddle API answers."""
    host = config.api.host
    if host in ("0.0.0.0", "::", ""):
        host = "127.0.0.1"
    return f"http://{host}:{config.api.port}"


_MARKS = {Level.OK: "✓", Level.WARN: "!", Level.FAIL: "✗", Level.SKIP: "-"}


def render(report: Report) -> list[str]:
    """Plain-text report: one line per check, details and hints indented."""
    width = max((len(c.name) for c in report.checks), default=8)
    lines = [f"huddle doctor — {report.node}", ""]
    for check in report.checks:
        lines.append(f"  {_MARKS[check.level]} {check.name:<{width}}  {check.summary}")
        lines += [f"  {'':<{width + 4}}{detail}" for detail in check.details]
        if check.hint and check.level in (Level.WARN, Level.FAIL):
            lines.append(f"  {'':<{width + 4}}→ {check.hint}")
    failures = sum(c.level is Level.FAIL for c in report.checks)
    warnings = sum(c.level is Level.WARN for c in report.checks)
    lines += ["", f"  {failures} failing, {warnings} warning(s)"]
    return lines
