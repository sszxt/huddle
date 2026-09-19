"""ClusterPoller: aggregating a coordinator plus its peers into one snapshot.

The coordinator is driven through an ASGI transport, like `doctor.py`'s own
tests; a peer is a real second agent process (`peer_agent`), since
`ClusterPoller` reaches peers the same way `ClusterService` does — over a
real network client, not whatever transport is used to reach the coordinator
itself. See `poller.py`'s own docstring for why those two clients are kept
separate.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from huddle.agent.app import create_app as create_agent_app
from huddle.app import create_app
from huddle.config import HuddleConfig
from huddle.tui.poller import ClusterPoller
from tests.conftest import wait_until_running, worker_config
from tests.test_cluster import CONSTRAINED_DEVICES, with_peer


@pytest.fixture
async def api(huddle_config: HuddleConfig) -> AsyncIterator[tuple[Any, httpx.AsyncClient]]:
    app = create_app(huddle_config)
    transport = httpx.ASGITransport(app=app)
    client = httpx.AsyncClient(
        transport=transport, base_url=f"http://127.0.0.1:{huddle_config.api.port}"
    )
    async with app.router.lifespan_context(app), client:
        await wait_until_running(client)
        yield app, client


async def test_single_node_snapshot(
    huddle_config: HuddleConfig, api: tuple[Any, httpx.AsyncClient]
) -> None:
    _, client = api
    poller = ClusterPoller(huddle_config, client=client)

    snapshot = await poller.poll()

    assert snapshot.coordinator_error is None
    assert [n.name for n in snapshot.nodes] == [huddle_config.node.name]
    head = snapshot.nodes[0]
    assert head.role == "head"
    assert head.reachable is True
    assert head.hardware is not None
    assert head.backend is not None
    assert snapshot.loaded_model == "tiny.gguf"
    assert snapshot.available_models == ["tiny.gguf"]


async def test_head_then_workers_ordering(
    huddle_config: HuddleConfig,
    api: tuple[Any, httpx.AsyncClient],
    peer_agent: dict[str, int],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A GPU too small to hold tiny.gguf alone, so the peer is genuinely needed
    # rather than silently dropped by the planner for holding nothing.
    monkeypatch.setenv("HUDDLE_FAKE_DEVICES", CONSTRAINED_DEVICES)
    _, client = api
    with_peer(huddle_config, peer_agent)
    poller = ClusterPoller(huddle_config, client=client)

    # Reload picks up the peer that was added after autostart already ran
    # with no peers configured.
    await poller.stop_cluster()
    status = await poller.start_cluster()
    assert status.running
    assert status.workers == ["peer1"]

    snapshot = await poller.poll()

    assert [n.name for n in snapshot.nodes] == [huddle_config.node.name, "peer1"]
    assert snapshot.nodes[0].role == "head"
    assert snapshot.nodes[1].role == "worker"
    assert snapshot.nodes[1].reachable is True
    assert snapshot.nodes[1].hardware is not None
    assert snapshot.nodes[1].rpc is not None


async def test_a_peer_that_cannot_be_resolved_is_reported_unreachable(
    huddle_config: HuddleConfig, api: tuple[Any, httpx.AsyncClient]
) -> None:
    """A worker named in ClusterStatus.workers but no longer configured/discovered.

    Should not happen in practice (the plan and the peer list come from the
    same resolution), but the poller must not crash if it ever does.
    """
    from huddle.coordinator.service import ClusterStatus

    _, client = api
    poller = ClusterPoller(huddle_config, client=client)
    fake_cluster = ClusterStatus(running=True, head_node=huddle_config.node.name, workers=["ghost"])
    node = await poller._worker_snapshot(httpx.AsyncClient(), "ghost", fake_cluster, None)
    assert node.reachable is False
    assert node.error is not None


async def test_agent_only_node_reports_a_coordinator_error(
    fake_llama_server: Path,
    fake_rpc_server: Path,
    tmp_path: Path,
    free_ports: list[int],
) -> None:
    """Pointed at a worker-only `huddle agent`, /cluster is 404, not a crash."""
    config = worker_config(
        fake_llama_server, fake_rpc_server, tmp_path / "models", free_ports[0], free_ports[1]
    )
    app = create_agent_app(config)
    transport = httpx.ASGITransport(app=app)
    client = httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1")

    async with app.router.lifespan_context(app), client:
        poller = ClusterPoller(config, client=client)
        snapshot = await poller.poll()

    assert snapshot.coordinator_error is not None
    assert snapshot.cluster is None
    assert len(snapshot.nodes) == 1
    assert snapshot.nodes[0].role == "worker"
    assert snapshot.nodes[0].reachable is True
