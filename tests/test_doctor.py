"""`huddle doctor`: each check against the failure it exists to catch."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

from huddle.app import create_app
from huddle.config import HuddleConfig, PeerConfig
from huddle.coordinator.cluster import PeerReport
from huddle.doctor import (
    Level,
    Report,
    check_peer,
    check_placement,
    check_workers,
    render,
    run_doctor,
)
from huddle.hardware import DeviceInfo, NodeHardware
from tests.test_cluster import with_peer

PIN = "4c9233c034fc450dcf34c7c0988aebe6da5cdf1a"
SAME = "version: 0.4.1-dev (build 10975, commit 4c9233c03)"
OTHER = "version: 0.4.1-dev (build 10975, commit deadbeef)"
CONSTRAINED = (
    "Available devices:\n"
    "  Vulkan0: NVIDIA GeForce RTX 5070 (12473 MiB, 20 MiB free)\n"
    "  Vulkan1: Intel(R) Graphics (RPL-S) (48045 MiB, 43241 MiB free)\n"
)


def by_name(report: Report) -> dict[str, Any]:
    return {check.name: check for check in report.checks}


@pytest.fixture
async def api(huddle_config: HuddleConfig) -> AsyncIterator[tuple[Any, httpx.AsyncClient]]:
    huddle_config.backend.autostart = False
    app = create_app(huddle_config)
    transport = httpx.ASGITransport(app=app)
    client = httpx.AsyncClient(transport=transport, base_url="http://huddle.test")
    async with app.router.lifespan_context(app), client:
        yield app, client


async def test_healthy_single_node(
    huddle_config: HuddleConfig, api: tuple[Any, httpx.AsyncClient]
) -> None:
    _, client = api
    report = await run_doctor(huddle_config, client, pin=PIN)
    checks = by_name(report)

    assert report.healthy
    assert checks["binaries"].level is Level.OK
    assert checks["llama.cpp"].level is Level.OK
    assert "matches llamacpp.pin" in checks["llama.cpp"].summary
    assert checks["devices"].level is Level.OK
    assert "Vulkan1" in "\n".join(checks["devices"].details), "skipped iGPU is still shown"
    assert checks["model"].level is Level.OK
    assert checks["peers"].level is Level.SKIP
    assert checks["plan"].level is Level.OK
    assert checks["cluster"].level is Level.SKIP


async def test_missing_binary_fails(
    huddle_config: HuddleConfig, api: tuple[Any, httpx.AsyncClient], tmp_path: Path
) -> None:
    _, client = api
    huddle_config.binaries.llama_server = tmp_path / "nope"
    report = await run_doctor(huddle_config, client, pin=PIN)
    assert not report.healthy
    assert by_name(report)["binaries"].level is Level.FAIL


async def test_build_off_the_pin_warns(
    huddle_config: HuddleConfig, api: tuple[Any, httpx.AsyncClient]
) -> None:
    _, client = api
    report = await run_doctor(huddle_config, client, pin="0123456789abcdef")
    check = by_name(report)["llama.cpp"]
    assert check.level is Level.WARN
    assert "llamacpp.pin" in check.summary


async def test_discovery_finding_nobody_warns(
    huddle_config: HuddleConfig,
    api: tuple[Any, httpx.AsyncClient],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The boot race: nobody found, so a large model gets planned onto one GPU."""

    async def nobody(*_args: object, **_kw: object) -> list[object]:
        return []

    monkeypatch.setattr("huddle.coordinator.cluster.discover", nobody)
    _, client = api
    huddle_config.discovery.enabled = True
    report = await run_doctor(huddle_config, client, pin=PIN)
    assert by_name(report)["peers"].level is Level.WARN


async def test_api_not_serving_skips_live_checks(
    huddle_config: HuddleConfig, free_port: int
) -> None:
    async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{free_port}") as client:
        report = await run_doctor(huddle_config, client, pin=PIN)
    checks = by_name(report)
    assert checks["cluster"].level is Level.SKIP
    assert "workers" not in checks
    assert report.healthy


async def test_live_two_node_cluster_checks_out(
    huddle_config: HuddleConfig,
    peer_agent: dict[str, int],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End to end: a running split cluster whose placement matches the plan."""
    monkeypatch.setenv("HUDDLE_FAKE_DEVICES", CONSTRAINED)
    huddle_config.backend.autostart = False
    huddle_config.backend.extra_args = ["-lv", "5"]
    with_peer(huddle_config, peer_agent)
    app = create_app(huddle_config)
    transport = httpx.ASGITransport(app=app)
    client = httpx.AsyncClient(transport=transport, base_url="http://huddle.test")

    async with app.router.lifespan_context(app), client:
        started = await client.post("/cluster/start")
        assert started.status_code == 200, started.text
        report = await run_doctor(huddle_config, client, pin=PIN)
        await client.post("/cluster/stop")

    checks = by_name(report)
    assert report.healthy, "\n".join(render(report))
    assert checks["peer peer1"].level is Level.OK
    assert checks["cluster"].level is Level.OK
    assert checks["plan"].level is Level.SKIP, "a live cluster is checked, not replanned"
    assert checks["workers"].level is Level.OK
    assert checks["placement"].level is Level.OK, checks["placement"].details


async def test_degraded_cluster_fails(
    huddle_config: HuddleConfig, api: tuple[Any, httpx.AsyncClient]
) -> None:
    app, client = api
    app.state.cluster.watch_interval = 60.0
    assert await app.state.cluster.start_with_retry("missing.gguf") is None

    report = await run_doctor(huddle_config, client, pin=PIN)
    check = by_name(report)["cluster"]
    assert check.level is Level.FAIL
    assert "retrying" in (check.hint or "")
    await app.state.cluster.stop()


# -- individual checks, with the failure constructed directly ------------------


def peer_report(version: str) -> PeerReport:
    return PeerReport(
        peer=PeerConfig(name="maksood", host="100.98.227.49"),
        hardware=NodeHardware(
            name="maksood",
            llamacpp_version=version,
            devices=[DeviceInfo(id="Vulkan0", name="RTX 5070", total_mib=12473, free_mib=11000)],
        ),
    )


def agent_answering(rpc: dict[str, Any]) -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=rpc)

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_peer_on_another_build_fails() -> None:
    async with agent_answering({"foreign": False}) as client:
        check = await check_peer(client, peer_report(OTHER), SAME)
    assert check.level is Level.FAIL
    assert "4c9233c03" in check.summary and "deadbeef" in check.summary


async def test_peer_on_a_shallow_clone_of_the_same_commit_passes() -> None:
    async with agent_answering({"foreign": False}) as client:
        check = await check_peer(
            client, peer_report(SAME), "version: 0.4.1-dev (build 1, commit 4c9233c)"
        )
    assert check.level is Level.OK


async def test_peer_with_an_orphaned_worker_warns() -> None:
    async with agent_answering({"foreign": True, "port": 50052}) as client:
        check = await check_peer(client, peer_report(SAME), SAME)
    assert check.level is Level.WARN
    assert "50052" in check.summary


async def test_unreachable_peer_fails() -> None:
    report = PeerReport(peer=PeerConfig(name="gone", host="100.64.0.9"), error="ConnectError")
    async with agent_answering({}) as client:
        check = await check_peer(client, report, SAME)
    assert check.level is Level.FAIL
    assert "Tailscale" in (check.hint or "")


async def test_endpoint_without_a_worker_fails(free_port: int) -> None:
    """The SIGABRT: an --rpc endpoint with nothing behind it."""
    status = {"plan": {"rpc_endpoints": [f"127.0.0.1:{free_port}"]}}
    check = await check_workers(status)
    assert check.level is Level.FAIL


def placement_api(layers: dict[str, list[int]]) -> Callable[[], httpx.AsyncClient]:
    def make() -> httpx.AsyncClient:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"layers": layers})

        return httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://h")

    return make


LIVE_PLAN = {
    "plan": {
        "n_layers": 64,
        "ngl": 61,
        "tensor_split": [30.0, 0.0, 31.0, 0.0],
        "placement": [{"id": "RPC0"}, {"id": "RPC1"}, {"id": "Vulkan0"}, {"id": "Vulkan1"}],
    }
}


async def test_placement_matching_the_plan_passes() -> None:
    layers = {"CPU": list(range(4)), "RPC0": list(range(4, 34)), "Vulkan0": list(range(34, 65))}
    async with placement_api(layers)() as client:
        check = await check_placement(client, LIVE_PLAN)
    assert check.level is Level.OK, check.details


async def test_placement_drifting_from_the_plan_fails() -> None:
    """A peer holding a layer it was never budgeted for."""
    layers = {"CPU": list(range(4)), "RPC0": list(range(4, 35)), "Vulkan0": list(range(35, 65))}
    async with placement_api(layers)() as client:
        check = await check_placement(client, LIVE_PLAN)
    assert check.level is Level.FAIL


async def test_placement_unknown_without_verbose_logs() -> None:
    async with placement_api({})() as client:
        check = await check_placement(client, LIVE_PLAN)
    assert check.level is Level.SKIP
    assert "-lv" in (check.hint or "")


def test_render_shows_hints_only_for_problems() -> None:
    from huddle.doctor import Check

    report = Report(
        node="omarchy",
        checks=[
            Check("devices", Level.OK, "1 usable GPU(s)", hint="not shown"),
            Check("workers", Level.FAIL, "nothing listening", details=["x:1"], hint="shown"),
        ],
    )
    text = "\n".join(render(report))
    assert "✓ devices" in text and "✗ workers" in text
    assert "→ shown" in text and "not shown" not in text
    assert "1 failing" in text
    assert not report.healthy


async def test_worker_node_is_checked_as_a_worker(
    fake_llama_server: Path,
    fake_rpc_server: Path,
    tmp_path: Path,
    free_ports: list[int],
) -> None:
    """No peers or plan on a worker: its agent and its visibility matter instead."""
    from tests.conftest import serve_agent, worker_config

    config = worker_config(
        fake_llama_server, fake_rpc_server, tmp_path / "w", free_ports[0], free_ports[1]
    )
    # Nothing serves the coordinator API on this node.
    client = httpx.AsyncClient(base_url=f"http://127.0.0.1:{free_ports[2]}")
    async with serve_agent(config), client:
        report = await run_doctor(config, client, pin=PIN)

    checks = by_name(report)
    assert checks["role"].level is Level.OK
    assert "worker node" in checks["role"].summary
    assert checks["visibility"].level is Level.SKIP, "discovery is off in this config"
    assert "peers" not in checks and "plan" not in checks
    assert report.healthy


async def test_worker_visible_on_the_lan(
    fake_llama_server: Path,
    fake_rpc_server: Path,
    tmp_path: Path,
    free_ports: list[int],
) -> None:
    from huddle.discovery import Advertiser
    from tests.conftest import worker_config

    config = worker_config(
        fake_llama_server, fake_rpc_server, tmp_path / "w", free_ports[0], free_ports[1]
    )
    config.node.name = "visible-worker"
    config.discovery.enabled = True
    config.discovery.timeout = 3.0
    config.discovery.advertise = "100.64.7.7"

    from huddle.doctor import check_visibility

    advertiser = Advertiser(config)
    await advertiser.start()
    try:
        check = await check_visibility(config)
    finally:
        await advertiser.stop()
    if check.level is Level.WARN:
        pytest.skip("no multicast on this host")
    assert check.level is Level.OK
    assert "100.64.7.7" in check.summary
