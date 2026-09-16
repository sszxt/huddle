"""Cluster orchestration, driven against a second Huddle agent on localhost.

The coordinator's job is talking to peers over HTTP and sequencing processes, so
the peer here is a real agent rather than a mock.
"""

from __future__ import annotations

import asyncio
import os
import signal

import httpx
import pytest

from huddle.app import create_app
from huddle.config import HuddleConfig

# A GPU too small to hold the whole test model, so a peer is genuinely needed.
# Without this the head fits everything locally and correctly drops the peer.
CONSTRAINED_DEVICES = (
    "Available devices:\n"
    "  Vulkan0: NVIDIA GeForce RTX 5070 (12473 MiB, 20 MiB free)\n"
    "  Vulkan1: Intel(R) Graphics (RPL-S) (48045 MiB, 43241 MiB free)\n"
)


@pytest.fixture
def constrained_gpus(monkeypatch: pytest.MonkeyPatch) -> None:
    """Shrink every node's GPU so the model cannot fit on one of them."""
    monkeypatch.setenv("HUDDLE_FAKE_DEVICES", CONSTRAINED_DEVICES)


def with_peer(config: HuddleConfig, peer: dict[str, int]) -> HuddleConfig:
    config.peers = [
        type(config)
        .model_validate(
            {
                "binaries": {"llama_server": str(config.binaries.llama_server)},
                "models": {"dir": str(config.models.dir)},
                "peers": [
                    {
                        "name": "peer1",
                        "host": "127.0.0.1",
                        "agent_port": peer["agent_port"],
                        "rpc_port": peer["rpc_port"],
                    }
                ],
            }
        )
        .peers[0]
    ]
    return config


async def cluster_client(config: HuddleConfig) -> tuple[object, httpx.AsyncClient]:
    app = create_app(config)
    return app, httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://huddle.test"
    )


async def test_plan_includes_peer_devices_first(
    huddle_config: HuddleConfig, peer_agent: dict[str, int], constrained_gpus: None
) -> None:
    """A peer's GPUs must appear as RPC devices, ahead of local ones."""
    huddle_config.backend.autostart = False
    with_peer(huddle_config, peer_agent)
    app, http = await cluster_client(huddle_config)

    async with app.router.lifespan_context(app), http:  # type: ignore[attr-defined]
        plan = (await http.get("/cluster/plan")).json()

    ids = [p["id"] for p in plan["placement"]]
    assert ids[:2] == ["RPC0", "RPC1"], "peer devices are enumerated first"
    assert ids[2:] == ["Vulkan0", "Vulkan1"]
    assert plan["rpc_endpoints"] == [f"127.0.0.1:{peer_agent['rpc_port']}"]
    assert plan["unreachable"] == []
    assert plan["n_gpu_layers"] > 0


async def test_plan_skips_unified_memory_devices(
    huddle_config: HuddleConfig, peer_agent: dict[str, int], constrained_gpus: None
) -> None:
    huddle_config.backend.autostart = False
    with_peer(huddle_config, peer_agent)
    app, http = await cluster_client(huddle_config)

    async with app.router.lifespan_context(app), http:  # type: ignore[attr-defined]
        plan = (await http.get("/cluster/plan")).json()

    by_id = {p["id"]: p for p in plan["placement"]}
    assert by_id["RPC1"]["skipped"] is not None
    assert by_id["Vulkan1"]["skipped"] is not None
    assert by_id["RPC1"]["layers"] == 0


async def test_start_brings_up_peer_worker_then_head(
    huddle_config: HuddleConfig, peer_agent: dict[str, int]
) -> None:
    """Workers must be listening before the head connects to them."""
    huddle_config.backend.autostart = False
    with_peer(huddle_config, peer_agent)
    app, http = await cluster_client(huddle_config)

    async with app.router.lifespan_context(app), http:  # type: ignore[attr-defined]
        status = (await http.post("/cluster/start")).json()
        assert status["running"] is True
        assert status["model"] == "tiny.gguf"

        argv = (await http.get("/agent/backend")).json()["argv"]
        assert "--tensor-split" in argv, "head must be told the planned split"

        stopped = (await http.post("/cluster/stop")).json()
        assert stopped["running"] is False
        assert stopped["workers"] == []


async def test_unreachable_peer_is_reported_not_fatal(huddle_config: HuddleConfig) -> None:
    """A cluster that starts smaller and says so beats one that refuses to start."""
    huddle_config.backend.autostart = False
    huddle_config.peers = [
        HuddleConfig.model_validate(
            {
                "binaries": {"llama_server": str(huddle_config.binaries.llama_server)},
                "models": {"dir": str(huddle_config.models.dir)},
                "peers": [{"name": "ghost", "host": "127.0.0.1", "agent_port": 1}],
            }
        ).peers[0]
    ]
    app, http = await cluster_client(huddle_config)

    async with app.router.lifespan_context(app), http:  # type: ignore[attr-defined]
        plan = (await http.get("/cluster/plan")).json()
        assert plan["unreachable"] == ["ghost"]
        assert plan["rpc_endpoints"] == []
        assert [p["id"] for p in plan["placement"]] == ["Vulkan0", "Vulkan1"]


async def test_start_rejects_when_already_running(
    huddle_config: HuddleConfig, peer_agent: dict[str, int]
) -> None:
    huddle_config.backend.autostart = False
    with_peer(huddle_config, peer_agent)
    app, http = await cluster_client(huddle_config)

    async with app.router.lifespan_context(app), http:  # type: ignore[attr-defined]
        await http.post("/cluster/start")
        again = await http.post("/cluster/start")
        assert again.status_code == 409
        assert "already running" in again.json()["detail"]
        await http.post("/cluster/stop")


async def test_autostart_uses_the_cluster_when_peers_are_configured(
    huddle_config: HuddleConfig, peer_agent: dict[str, int], constrained_gpus: None
) -> None:
    """Autostart must not quietly run single-node with peers configured.

    That failure is invisible from outside: the API answers, completions work,
    and the peers simply never get used.
    """
    huddle_config.backend.autostart = True
    with_peer(huddle_config, peer_agent)
    app, http = await cluster_client(huddle_config)

    async with app.router.lifespan_context(app), http:  # type: ignore[attr-defined]
        argv = (await http.get("/agent/backend")).json()["argv"]
        assert "--rpc" in argv, "autostart ignored the configured peer"

        status = (await http.get("/cluster")).json()
        assert status["running"] is True
        assert status["plan"] is not None
        # Every endpoint passed to --rpc must have a worker we started behind it.
        assert status["workers"] == [p.name for p in huddle_config.peers]


async def test_autostart_stays_single_node_without_peers(huddle_config: HuddleConfig) -> None:
    huddle_config.backend.autostart = True
    huddle_config.peers = []
    app, http = await cluster_client(huddle_config)

    async with app.router.lifespan_context(app), http:  # type: ignore[attr-defined]
        argv = (await http.get("/agent/backend")).json()["argv"]
        assert "--rpc" not in argv


async def test_peer_with_no_layers_is_dropped_entirely(
    huddle_config: HuddleConfig, peer_agent: dict[str, int]
) -> None:
    """A peer holding nothing must not appear in --rpc at all.

    llama.cpp connects to every endpoint at load and depends on it afterwards,
    so an unused peer is a liability that can still take the head down.
    """
    huddle_config.backend.autostart = False
    # Generous local headroom, so the whole model fits on the head node.
    huddle_config.planner.headroom = 0.0
    with_peer(huddle_config, peer_agent)
    app, http = await cluster_client(huddle_config)

    async with app.router.lifespan_context(app), http:  # type: ignore[attr-defined]
        plan = (await http.get("/cluster/plan")).json()
        assert plan["rpc_endpoints"] == [], "unused peer should be dropped"
        ids = [p["id"] for p in plan["placement"]]
        assert ids == ["Vulkan0", "Vulkan1"], "only local devices remain"

        status = (await http.post("/cluster/start")).json()
        assert status["workers"] == [], "no worker started for an unused peer"
        await http.post("/cluster/stop")


async def test_split_positions_survive_dropping_a_peer(
    huddle_config: HuddleConfig, peer_agent: dict[str, int]
) -> None:
    """Dropping a peer removes its device slots, so the split must be re-planned."""
    huddle_config.backend.autostart = False
    huddle_config.planner.headroom = 0.0
    with_peer(huddle_config, peer_agent)
    app, http = await cluster_client(huddle_config)

    async with app.router.lifespan_context(app), http:  # type: ignore[attr-defined]
        plan = (await http.get("/cluster/plan")).json()

    assert len(plan["tensor_split"]) == len(plan["placement"]), (
        "one split weight per enumerated device, or every later position shifts"
    )


async def test_supervisor_restarts_the_head_when_it_dies(
    huddle_config: HuddleConfig, peer_agent: dict[str, int], constrained_gpus: None
) -> None:
    """A dead head must come back, not sit there looking healthy.

    llama.cpp core-dumps when a worker disappears, so this is the failure that
    actually happens in production rather than a hypothetical one.
    """
    huddle_config.backend.autostart = False
    with_peer(huddle_config, peer_agent)
    app = create_app(huddle_config)
    cluster = app.state.cluster
    cluster.watch_interval = 0.2

    http = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://huddle.test")
    async with app.router.lifespan_context(app), http:
        await http.post("/cluster/start")
        first_pid = (await http.get("/agent/backend")).json()["pid"]
        assert first_pid is not None

        # Kill the head the way a dying worker would.
        os.kill(first_pid, signal.SIGKILL)

        for _ in range(100):
            await asyncio.sleep(0.2)
            status = (await http.get("/cluster")).json()
            if status["running"] and status["restarts"] > 0:
                break
        else:
            raise AssertionError("supervisor never restarted the head")

        assert status["restarts"] == 1
        assert status["last_failure"] is None
        second_pid = (await http.get("/agent/backend")).json()["pid"]
        assert second_pid != first_pid, "a new process should be running"


async def test_status_reports_degraded_between_death_and_recovery(
    huddle_config: HuddleConfig, peer_agent: dict[str, int], constrained_gpus: None
) -> None:
    """`desired` vs `running` is what makes a dead cluster visible."""
    huddle_config.backend.autostart = False
    with_peer(huddle_config, peer_agent)
    app = create_app(huddle_config)
    app.state.cluster.watch_interval = 60.0  # do not recover during this test

    http = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://huddle.test")
    async with app.router.lifespan_context(app), http:
        await http.post("/cluster/start")
        assert (await http.get("/cluster")).json()["desired"] is True

        pid = (await http.get("/agent/backend")).json()["pid"]
        os.kill(pid, signal.SIGKILL)
        await asyncio.sleep(1.0)

        status = (await http.get("/cluster")).json()
        assert status["desired"] is True and status["running"] is False, "should read degraded"


async def test_stop_disables_supervision(
    huddle_config: HuddleConfig, peer_agent: dict[str, int], constrained_gpus: None
) -> None:
    """An explicit stop must not be undone by the supervisor."""
    huddle_config.backend.autostart = False
    with_peer(huddle_config, peer_agent)
    app = create_app(huddle_config)
    app.state.cluster.watch_interval = 0.2

    http = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://huddle.test")
    async with app.router.lifespan_context(app), http:
        await http.post("/cluster/start")
        await http.post("/cluster/stop")
        await asyncio.sleep(1.0)

        status = (await http.get("/cluster")).json()
        assert status["running"] is False
        assert status["desired"] is False
        assert status["restarts"] == 0, "stopping is not a failure to recover from"
