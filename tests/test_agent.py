"""Node agent control-API tests."""

from __future__ import annotations

import httpx

from huddle.app import create_app
from huddle.config import HuddleConfig


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
