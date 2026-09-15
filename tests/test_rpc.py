"""Worker-role lifecycle, driven against the fake ggml-rpc-server.

The bind address is the thing to guard: llama.cpp defaults to 127.0.0.1, which
makes a worker unreachable from other machines in a way that looks exactly like
a firewall problem.
"""

from __future__ import annotations

import httpx

from huddle.app import create_app
from huddle.config import HuddleConfig


async def test_rpc_not_running_initially(client: httpx.AsyncClient) -> None:
    status = (await client.get("/agent/rpc")).json()
    assert status["running"] is False
    assert status["bind"] == "0.0.0.0"
    assert status["cache"] is True


async def test_rpc_start_stop_cycle(client: httpx.AsyncClient) -> None:
    started = (await client.post("/agent/rpc/start")).json()
    assert started["running"] is True
    assert started["pid"] is not None

    argv = started["argv"]
    assert argv[argv.index("-H") + 1] == "0.0.0.0", "worker must not bind localhost"
    assert "-c" in argv, "tensor cache must be on; the LAN transfer is expensive"

    assert (await client.get("/agent/rpc")).json()["running"] is True

    stopped = (await client.post("/agent/rpc/stop")).json()
    assert stopped["running"] is False


async def test_rpc_reports_advertised_endpoint(huddle_config: HuddleConfig) -> None:
    """Peers dial the advertised address, since 0.0.0.0 is not connectable."""
    huddle_config.rpc.advertise = "100.98.227.49"
    app = create_app(huddle_config)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://huddle.test") as http:
            started = (await http.post("/agent/rpc/start")).json()
            assert started["endpoint"] == f"100.98.227.49:{huddle_config.rpc.port}"
            await http.post("/agent/rpc/stop")


async def test_rpc_rejects_double_start(client: httpx.AsyncClient) -> None:
    await client.post("/agent/rpc/start")
    response = await client.post("/agent/rpc/start")
    assert response.status_code == 409
    assert "already running" in response.json()["detail"]
    await client.post("/agent/rpc/stop")


async def test_rpc_requires_configured_binary(huddle_config: HuddleConfig) -> None:
    huddle_config.binaries.rpc_server = None
    app = create_app(huddle_config)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://huddle.test") as http:
            response = await http.post("/agent/rpc/start")
            assert response.status_code == 409
            assert "no rpc_server binary" in response.json()["detail"]


async def test_rpc_logs_captured(client: httpx.AsyncClient) -> None:
    await client.post("/agent/rpc/start")
    lines = (await client.get("/agent/rpc/logs")).json()["lines"]
    assert any("listening on" in line for line in lines)
    await client.post("/agent/rpc/stop")
