from __future__ import annotations

import asyncio
import contextlib
import socket
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from huddle.app import create_app
from huddle.config import HuddleConfig
from tests.fakes.gguf_builder import write_gguf

FAKES_DIR = Path(__file__).parent / "fakes"


@pytest.fixture
def fake_llama_server() -> Path:
    return FAKES_DIR / "fake_llama_server.py"


@pytest.fixture
def fake_rpc_server() -> Path:
    return FAKES_DIR / "fake_rpc_server.py"


@pytest.fixture
def free_port() -> int:
    """Claim a port from the ephemeral range and release it immediately."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.fixture
def models_dir(tmp_path: Path) -> Path:
    """A model directory holding one synthetic but *valid* GGUF.

    It must be real: the fake llama-server never reads it, but the planner parses
    its header to work out layer count and per-layer cost.
    """
    directory = tmp_path / "models"
    directory.mkdir()
    write_gguf(directory / "tiny.gguf", n_layers=12, n_embd=256)
    return directory


@pytest.fixture
def huddle_config(
    fake_llama_server: Path, fake_rpc_server: Path, models_dir: Path, free_port: int
) -> HuddleConfig:
    return HuddleConfig.model_validate(
        {
            "node": {"name": "testnode"},
            "binaries": {
                "llama_server": str(fake_llama_server),
                "rpc_server": str(fake_rpc_server),
            },
            "models": {"dir": str(models_dir), "default": "tiny.gguf"},
            "backend": {"host": "127.0.0.1", "port": free_port, "autostart": True},
            "api": {"host": "127.0.0.1", "port": 8000},
            "rpc": {"port": free_port + 1},
        }
    )


async def wait_until_running(http: httpx.AsyncClient, timeout: float = 30.0) -> dict[str, Any]:
    """Wait for autostart, which runs in the background so the API answers early."""
    deadline = asyncio.get_running_loop().time() + timeout
    while True:
        status: dict[str, Any] = (await http.get("/cluster")).json()
        if status["running"]:
            return status
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError(f"cluster never came up: {status}")
        await asyncio.sleep(0.1)


@pytest.fixture
async def client(huddle_config: HuddleConfig) -> AsyncIterator[httpx.AsyncClient]:
    """The full app, with lifespan run and the backend actually started."""
    app = create_app(huddle_config)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://huddle.test") as http:
            await wait_until_running(http)
            yield http


@pytest.fixture
def free_ports() -> list[int]:
    """Several distinct free ports, held open together to avoid collisions."""
    socks = [socket.socket(socket.AF_INET, socket.SOCK_STREAM) for _ in range(4)]
    try:
        ports = []
        for sock in socks:
            sock.bind(("127.0.0.1", 0))
            ports.append(int(sock.getsockname()[1]))
        return ports
    finally:
        for sock in socks:
            sock.close()


@contextlib.asynccontextmanager
async def serve_agent(config: HuddleConfig) -> AsyncIterator[None]:
    """Run a worker agent on its configured port for the duration of the block."""
    import uvicorn

    from huddle.agent.app import create_app as create_agent_app

    server = uvicorn.Server(
        uvicorn.Config(
            create_agent_app(config),
            host="127.0.0.1",
            port=config.node.agent_port,
            log_level="error",
        )
    )
    task = asyncio.create_task(server.serve())
    deadline = asyncio.get_running_loop().time() + 15
    while not server.started:
        if asyncio.get_running_loop().time() > deadline:
            raise RuntimeError("agent did not start")
        await asyncio.sleep(0.05)
    try:
        yield
    finally:
        server.should_exit = True
        await task


def worker_config(
    fake_llama_server: Path,
    fake_rpc_server: Path | None,
    models: Path,
    agent_port: int,
    rpc_port: int,
) -> HuddleConfig:
    models.mkdir(exist_ok=True)
    return HuddleConfig.model_validate(
        {
            "node": {"name": "peer1", "agent_port": agent_port},
            "binaries": {
                "llama_server": str(fake_llama_server),
                "rpc_server": str(fake_rpc_server) if fake_rpc_server else None,
            },
            "models": {"dir": str(models)},
            "backend": {"autostart": False},
            "rpc": {"port": rpc_port, "advertise": "127.0.0.1"},
        }
    )


@pytest.fixture
async def peer_agent(
    fake_llama_server: Path, fake_rpc_server: Path, tmp_path: Path, free_ports: list[int]
) -> AsyncIterator[dict[str, int]]:
    """A second Huddle agent on localhost, standing in for a cluster peer.

    A real HTTP server rather than a mock: the coordinator's job is talking to
    peers over HTTP, so that is the part worth exercising.
    """
    import uvicorn

    from huddle.agent.app import create_app as create_agent_app

    agent_port, rpc_port = free_ports[0], free_ports[1]
    models = tmp_path / "peer-models"
    models.mkdir(exist_ok=True)

    config = HuddleConfig.model_validate(
        {
            "node": {"name": "peer1", "agent_port": agent_port},
            "binaries": {
                "llama_server": str(fake_llama_server),
                "rpc_server": str(fake_rpc_server),
            },
            "models": {"dir": str(models)},
            "backend": {"autostart": False},
            "rpc": {"port": rpc_port, "advertise": "127.0.0.1"},
        }
    )
    server = uvicorn.Server(
        uvicorn.Config(
            create_agent_app(config), host="127.0.0.1", port=agent_port, log_level="error"
        )
    )
    task = asyncio.create_task(server.serve())

    deadline = asyncio.get_running_loop().time() + 15
    while not server.started:
        if asyncio.get_running_loop().time() > deadline:
            raise RuntimeError("peer agent did not start")
        await asyncio.sleep(0.05)

    yield {"agent_port": agent_port, "rpc_port": rpc_port}

    server.should_exit = True
    await task
