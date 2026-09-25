"""The cluster page's /cluster/nodes, against a real second agent and the fakes."""

from __future__ import annotations

import asyncio

import httpx
import pytest

from huddle.agent.service import BackendService
from huddle.config import HuddleConfig
from huddle.hardware import NodeHardware
from tests.test_cluster import CONSTRAINED_DEVICES, cluster_client, with_peer


@pytest.fixture
def constrained_gpus(monkeypatch: pytest.MonkeyPatch) -> None:
    """GPUs too small for the test model, so a peer is genuinely given layers."""
    monkeypatch.setenv("HUDDLE_FAKE_DEVICES", CONSTRAINED_DEVICES)


def ghost_peer(config: HuddleConfig) -> HuddleConfig:
    """A configured peer nothing answers for."""
    config.peers = [
        HuddleConfig.model_validate(
            {
                "binaries": {"llama_server": str(config.binaries.llama_server)},
                "models": {"dir": str(config.models.dir)},
                "peers": [{"name": "ghost", "host": "127.0.0.1", "agent_port": 1}],
            }
        ).peers[0]
    ]
    return config


async def test_a_lone_head_lists_itself(huddle_config: HuddleConfig) -> None:
    huddle_config.backend.autostart = False
    app, http = await cluster_client(huddle_config)

    async with app.router.lifespan_context(app), http:  # type: ignore[attr-defined]
        body = (await http.get("/cluster/nodes")).json()

    assert [node["name"] for node in body["nodes"]] == ["testnode"]
    head = body["nodes"][0]
    assert head["role"] == "head"
    assert head["address"] is None
    assert [d["id"] for d in head["hardware"]["devices"]] == ["Vulkan0", "Vulkan1"]
    assert head["system"]["cpu_threads"] > 0
    assert body["cluster"]["running"] is False
    assert body["backend_port"] == huddle_config.backend.port


async def test_a_reachable_unused_peer_is_idle(
    huddle_config: HuddleConfig, peer_agent: dict[str, int]
) -> None:
    huddle_config.backend.autostart = False
    with_peer(huddle_config, peer_agent)
    app, http = await cluster_client(huddle_config)

    async with app.router.lifespan_context(app), http:  # type: ignore[attr-defined]
        body = (await http.get("/cluster/nodes")).json()

    peer = body["nodes"][1]
    assert peer["name"] == "peer1"
    assert peer["role"] == "idle"
    assert peer["address"] == "127.0.0.1"
    assert peer["agent_port"] == peer_agent["agent_port"]
    assert peer["rtt_ms"] is not None and peer["rtt_ms"] >= 0
    assert peer["hardware"]["devices"], "the peer's own device report comes through"
    assert peer["system"]["hostname"]
    assert peer["rpc"]["running"] is False
    assert peer["error"] is None


async def test_a_peer_carrying_layers_is_a_worker(
    huddle_config: HuddleConfig, peer_agent: dict[str, int], constrained_gpus: None
) -> None:
    huddle_config.backend.autostart = False
    with_peer(huddle_config, peer_agent)
    app, http = await cluster_client(huddle_config)

    async with app.router.lifespan_context(app), http:  # type: ignore[attr-defined]
        assert (await http.post("/cluster/start")).json()["running"] is True
        body = (await http.get("/cluster/nodes")).json()
        await http.post("/cluster/stop")

    head, peer = body["nodes"]
    assert peer["role"] == "worker"
    assert peer["layers"] > 0
    assert peer["rpc"]["running"] is True
    plan = body["cluster"]["plan"]
    assert head["layers"] + peer["layers"] == plan["n_gpu_layers"]


async def test_an_unreachable_peer_is_offline_with_a_reason(huddle_config: HuddleConfig) -> None:
    huddle_config.backend.autostart = False
    ghost_peer(huddle_config)
    app, http = await cluster_client(huddle_config)

    async with app.router.lifespan_context(app), http:  # type: ignore[attr-defined]
        body = (await http.get("/cluster/nodes")).json()

    ghost = body["nodes"][1]
    assert ghost["role"] == "offline"
    assert ghost["hardware"] is None and ghost["rtt_ms"] is None
    assert ghost["error"], "the page needs to say why"
    assert ghost["address"] == "127.0.0.1" and ghost["agent_port"] == 1


async def test_concurrent_viewers_share_one_probe(
    huddle_config: HuddleConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """However many tabs poll, the node is probed once per refresh window."""
    huddle_config.backend.autostart = False
    calls = 0
    real_hardware = BackendService.hardware

    def counting(self: BackendService) -> NodeHardware:
        nonlocal calls
        calls += 1
        return real_hardware(self)

    monkeypatch.setattr(BackendService, "hardware", counting)
    app, http = await cluster_client(huddle_config)

    async with app.router.lifespan_context(app), http:  # type: ignore[attr-defined]
        responses = await asyncio.gather(*(http.get("/cluster/nodes") for _ in range(5)))

    assert all(r.status_code == 200 for r in responses)
    assert calls == 1


async def test_the_agent_reports_system_details(client: httpx.AsyncClient) -> None:
    body = (await client.get("/agent/system")).json()
    assert body["hostname"]
    assert body["cpu_threads"] > 0
    assert isinstance(body["addresses"], list)
