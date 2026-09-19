"""Node agent control-API tests."""

from __future__ import annotations

import httpx
import pytest

from huddle.app import create_app
from huddle.config import HuddleConfig
from tests.conftest import wait_until_running


async def test_agent_health(client: httpx.AsyncClient) -> None:
    response = await client.get("/agent/health")
    assert response.status_code == 200
    assert response.json()["node"] == "testnode"


async def test_hardware_reports_devices_and_memory(client: httpx.AsyncClient) -> None:
    response = await client.get("/agent/hardware")
    assert response.status_code == 200
    hardware = response.json()
    assert hardware["name"] == "testnode"
    assert hardware["cpu_count"] > 0
    assert hardware["ram_total_mib"] > 0
    assert [d["id"] for d in hardware["devices"]] == ["Vulkan0", "Vulkan1"]


async def test_hardware_flags_unified_memory_device(client: httpx.AsyncClient) -> None:
    """The Intel iGPU advertises system RAM; placement must not trust it."""
    devices = (await client.get("/agent/hardware")).json()["devices"]
    discrete, integrated = devices[0], devices[1]
    assert discrete["unified_memory"] is False
    assert integrated["unified_memory"] is True


async def test_backend_status_and_logs(client: httpx.AsyncClient) -> None:
    status = (await client.get("/agent/backend")).json()
    assert status["running"] is True
    assert status["model"] == "tiny.gguf"
    assert status["pid"] is not None
    assert "-ngl" in status["argv"]

    logs = (await client.get("/agent/backend/logs")).json()["lines"]
    assert any("listening on" in line for line in logs)


async def test_stop_then_start(client: httpx.AsyncClient) -> None:
    stopped = (await client.post("/agent/backend/stop")).json()
    assert stopped["running"] is False
    assert (await client.get("/agent/backend")).json()["running"] is False

    started = (await client.post("/agent/backend/start")).json()
    assert started["running"] is True
    assert started["model"] == "tiny.gguf"


async def test_start_rejects_when_already_running(client: httpx.AsyncClient) -> None:
    response = await client.post("/agent/backend/start")
    assert response.status_code == 409
    assert "already running" in response.json()["detail"]


async def test_start_rejects_missing_model(huddle_config: HuddleConfig) -> None:
    huddle_config.backend.autostart = False
    app = create_app(huddle_config)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://huddle.test") as http:
            response = await http.post("/agent/backend/start", json={"model": "absent.gguf"})
            assert response.status_code == 409
            assert "model not found" in response.json()["detail"]


async def test_logs_survive_a_failed_start(huddle_config: HuddleConfig, tmp_path: object) -> None:
    """A crashed load is when its output matters most; it used to be discarded."""
    from pathlib import Path

    crash = Path(str(tmp_path)) / "crash-server"
    crash.write_text("#!/bin/sh\necho 'E llama_model_load: something specific broke'\nexit 1\n")
    crash.chmod(0o755)

    huddle_config.backend.autostart = False
    huddle_config.binaries.llama_server = crash
    app = create_app(huddle_config)

    transport = httpx.ASGITransport(app=app)
    http = httpx.AsyncClient(transport=transport, base_url="http://huddle.test")
    async with app.router.lifespan_context(app), http:
        assert (await http.post("/agent/backend/start")).status_code == 409
        lines = (await http.get("/agent/backend/logs")).json()["lines"]
        assert any("something specific broke" in line for line in lines)


async def test_placement_survives_a_verbose_flood(
    huddle_config: HuddleConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real failure: -lv 5 output evicted layer placement within seconds."""
    monkeypatch.setenv("HUDDLE_FAKE_LOG_FLOOD", "3000")
    huddle_config.backend.autostart = False
    huddle_config.backend.extra_args = ["-lv", "5"]
    app = create_app(huddle_config)

    transport = httpx.ASGITransport(app=app)
    http = httpx.AsyncClient(transport=transport, base_url="http://huddle.test")
    async with app.router.lifespan_context(app), http:
        await http.post("/agent/backend/start")
        placement = (await http.get("/agent/backend/placement")).json()["layers"]
        await http.post("/agent/backend/stop")

    # tiny.gguf has 12 layers, so 13 entries with the output head; -ngl all
    # offloads every one of them.
    assert "CPU" not in placement
    assert placement["Vulkan0"] == list(range(13))


async def test_placement_is_empty_without_verbose_logging(client: httpx.AsyncClient) -> None:
    """At default verbosity llama.cpp prints nothing about placement."""
    assert (await client.get("/agent/backend/placement")).json()["layers"] == {}


async def test_tokens_per_sec_is_none_before_any_completion(client: httpx.AsyncClient) -> None:
    status = (await client.get("/agent/backend")).json()
    assert status["tokens_per_sec"] is None


async def test_tokens_per_sec_is_parsed_from_the_backend_log(
    huddle_config: HuddleConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HUDDLE_FAKE_TOKENS_PER_SEC", "27.7")
    app = create_app(huddle_config)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://huddle.test") as http:
            await wait_until_running(http)
            status = (await http.get("/agent/backend")).json()
            assert status["tokens_per_sec"] == 27.7
