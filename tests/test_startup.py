"""Starting up well: waiting for slow peers, rejoining late ones, and staying
observable while a large model loads.

The motivating incident, on the real nodes: omarchy rebooted, its discovery
found maksood, but maksood's agent did not answer in time — so the 32B came up
with 34 of 64 layers on CPU, stayed that way, and the API was unreachable for
the whole two-and-a-half-minute load.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from huddle.app import create_app
from huddle.config import HuddleConfig, PeerConfig
from tests.conftest import serve_agent, wait_until_running, worker_config

CONSTRAINED = (
    "Available devices:\n"
    "  Vulkan0: NVIDIA GeForce RTX 5070 (12473 MiB, 20 MiB free)\n"
    "  Vulkan1: Intel(R) Graphics (RPL-S) (48045 MiB, 43241 MiB free)\n"
)


@pytest.fixture
def ports(free_ports: list[int]) -> dict[str, int]:
    return {"agent": free_ports[2], "rpc": free_ports[3]}


@pytest.fixture
def late_peer(
    fake_llama_server: Path, fake_rpc_server: Path, tmp_path: Path, ports: dict[str, int]
) -> HuddleConfig:
    """A worker that is configured but not yet running."""
    return worker_config(
        fake_llama_server, fake_rpc_server, tmp_path / "peer", ports["agent"], ports["rpc"]
    )


def pointing_at(config: HuddleConfig, ports: dict[str, int]) -> HuddleConfig:
    config.peers = [
        PeerConfig(name="peer1", host="127.0.0.1", agent_port=ports["agent"], rpc_port=ports["rpc"])
    ]
    return config


@pytest.fixture
async def head(
    huddle_config: HuddleConfig, ports: dict[str, int], monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[tuple[Any, httpx.AsyncClient]]:
    """A head whose GPU cannot hold the whole model, with a configured peer."""
    monkeypatch.setenv("HUDDLE_FAKE_DEVICES", CONSTRAINED)
    huddle_config.backend.autostart = False
    pointing_at(huddle_config, ports)
    app = create_app(huddle_config)
    cluster = app.state.cluster
    cluster.watch_interval = 0.1
    transport = httpx.ASGITransport(app=app)
    client = httpx.AsyncClient(transport=transport, base_url="http://huddle.test")
    async with app.router.lifespan_context(app), client:
        yield app, client
        await cluster.stop()


async def eventually(check: Any, timeout: float = 15.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not check():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError("condition never became true")
        await asyncio.sleep(0.1)


async def test_waits_for_a_peer_that_is_still_booting(
    head: tuple[Any, httpx.AsyncClient], late_peer: HuddleConfig
) -> None:
    app, _ = head
    cluster = app.state.cluster
    cluster.peer_wait = 30.0

    assert await cluster.start_with_retry() is None, "must not start without the peer"
    status = cluster.status()
    assert not status.running
    assert "waiting for peers" in status.last_failure
    assert status.restarts == 0, "waiting is not a failed restart"

    async with serve_agent(late_peer):
        await eventually(lambda: cluster.status().running)
        status = cluster.status()
        assert status.workers == ["peer1"], "the late peer was included"
        assert status.plan.n_gpu_layers == status.plan.n_layers
        assert status.restarts == 0
        await cluster.stop()


async def test_gives_up_waiting_and_starts_smaller(head: tuple[Any, httpx.AsyncClient]) -> None:
    """A smaller cluster that says so beats no cluster."""
    app, _ = head
    cluster = app.state.cluster
    # Short, but not so short that probing local hardware (a real subprocess
    # spawn) can eat the whole window on a slow filesystem/runner and make
    # the "still waiting" check never actually run.
    cluster.peer_wait = 3.0

    assert await cluster.start_with_retry() is None
    await eventually(lambda: cluster.status().running)
    plan = cluster.status().plan
    assert plan.unreachable == ["peer1"]
    assert plan.n_gpu_layers < plan.n_layers


async def test_explicit_start_does_not_wait(head: tuple[Any, httpx.AsyncClient]) -> None:
    """A person asking for a start wants it now."""
    _, http = head
    response = await http.post("/cluster/start")
    assert response.status_code == 200, response.text
    assert response.json()["plan"]["unreachable"] == ["peer1"]


async def test_rejoins_a_peer_that_appears_later(
    head: tuple[Any, httpx.AsyncClient], late_peer: HuddleConfig
) -> None:
    app, http = head
    cluster = app.state.cluster
    cluster.rejoin_interval = 0.2

    await http.post("/cluster/start")
    before = cluster.status().plan
    assert before.n_gpu_layers < before.n_layers, "layers on CPU: worth rejoining"

    async with serve_agent(late_peer):
        await eventually(lambda: cluster.status().workers == ["peer1"] and cluster.status().running)
        after = cluster.status()
        assert after.running
        assert after.plan.n_gpu_layers == after.plan.n_layers
        await cluster.stop()


async def test_no_rejoin_when_the_model_already_fits(
    huddle_config: HuddleConfig,
    peer_agent: dict[str, int],
) -> None:
    """A peer that could not help must never cause a restart loop."""
    huddle_config.backend.autostart = False
    huddle_config.peers = [
        PeerConfig(
            name="peer1",
            host="127.0.0.1",
            agent_port=peer_agent["agent_port"],
            rpc_port=peer_agent["rpc_port"],
        )
    ]
    app = create_app(huddle_config)
    cluster = app.state.cluster
    cluster.watch_interval = 0.1
    cluster.rejoin_interval = 0.1
    transport = httpx.ASGITransport(app=app)
    client = httpx.AsyncClient(transport=transport, base_url="http://huddle.test")
    async with app.router.lifespan_context(app), client:
        await client.post("/cluster/start")
        pid = (await client.get("/agent/backend")).json()["pid"]
        await asyncio.sleep(1.0)
        assert (await client.get("/agent/backend")).json()["pid"] == pid, "no restart"
        assert cluster.status().workers == []
        await cluster.stop()


async def test_a_peer_that_cannot_start_its_worker_fails_the_start(
    huddle_config: HuddleConfig,
    fake_llama_server: Path,
    tmp_path: Path,
    ports: dict[str, int],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 409 is only "already running" if something is actually listening."""
    monkeypatch.setenv("HUDDLE_FAKE_DEVICES", CONSTRAINED)
    broken = worker_config(fake_llama_server, None, tmp_path / "p", ports["agent"], ports["rpc"])
    huddle_config.backend.autostart = False
    pointing_at(huddle_config, ports)
    app = create_app(huddle_config)
    transport = httpx.ASGITransport(app=app)
    client = httpx.AsyncClient(transport=transport, base_url="http://huddle.test")
    async with serve_agent(broken), app.router.lifespan_context(app), client:
        response = await client.post("/cluster/start")
    assert response.status_code == 409
    assert "could not start its worker" in response.json()["detail"]


async def test_api_answers_while_a_slow_model_loads(
    huddle_config: HuddleConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HUDDLE_FAKE_DELAY", "2")
    app = create_app(huddle_config)  # autostart on
    transport = httpx.ASGITransport(app=app)
    client = httpx.AsyncClient(transport=transport, base_url="http://huddle.test")
    async with app.router.lifespan_context(app), client:
        await asyncio.sleep(0.5)
        health = (await client.get("/health")).json()
        assert health["status"] == "starting", health
        await wait_until_running(client)
        assert (await client.get("/health")).json()["status"] == "ok"


async def test_shutdown_during_a_load_leaves_no_process(
    huddle_config: HuddleConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cancelling a start must not leave llama-server holding the GPU and port."""
    monkeypatch.setenv("HUDDLE_FAKE_DELAY", "30")
    marker = str(huddle_config.models.dir)
    app = create_app(huddle_config)
    async with app.router.lifespan_context(app):
        await asyncio.sleep(1.0)
        assert _processes_mentioning(marker), "the load should be under way"
    await asyncio.sleep(0.5)
    assert not _processes_mentioning(marker), "the cancelled load leaked its process"


def _processes_mentioning(text: str) -> list[int]:
    found = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            cmdline = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode()
        except OSError:
            continue
        if text in cmdline and "fake_llama_server" in cmdline:
            found.append(int(entry.name))
    return found


def test_coordinators_are_not_taken_as_workers() -> None:
    from huddle.discovery import DiscoveredPeer

    head = DiscoveredPeer(
        name="omarchy", host="100.94.234.89", agent_port=8081, rpc_port=50052, role="coordinator"
    )
    legacy = DiscoveredPeer(name="old", host="100.64.0.1", agent_port=8081, rpc_port=50052)
    assert head.is_coordinator
    assert not legacy.is_coordinator, "records without a role are workers"


async def test_resolve_skips_discovered_coordinators(
    huddle_config: HuddleConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    from huddle.coordinator.cluster import resolve_peers
    from huddle.discovery import DiscoveredPeer

    async def found(*_a: object, **_k: object) -> list[DiscoveredPeer]:
        return [
            DiscoveredPeer(name="omarchy", host="100.94.234.89", agent_port=8081,
                           rpc_port=50052, role="coordinator"),
            DiscoveredPeer(name="maksood", host="100.98.227.49", agent_port=8081,
                           rpc_port=50052, role="worker"),
        ]  # fmt: skip

    monkeypatch.setattr("huddle.coordinator.cluster.discover", found)
    huddle_config.discovery.enabled = True
    assert [p.name for p in await resolve_peers(huddle_config)] == ["maksood"]
