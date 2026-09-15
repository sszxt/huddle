from __future__ import annotations

import socket
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest

from huddle.app import create_app
from huddle.config import HuddleConfig

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
    """A model directory holding one fake GGUF.

    The fake llama-server never reads it; only existence is checked.
    """
    directory = tmp_path / "models"
    directory.mkdir()
    (directory / "tiny.gguf").write_bytes(b"GGUF\x00fake")
    return directory


@pytest.fixture
def huddle_config(fake_llama_server: Path, models_dir: Path, free_port: int) -> HuddleConfig:
    return HuddleConfig.model_validate(
        {
            "node": {"name": "testnode"},
            "binaries": {"llama_server": str(fake_llama_server)},
            "models": {"dir": str(models_dir), "default": "tiny.gguf"},
            "backend": {"host": "127.0.0.1", "port": free_port, "autostart": True},
            "api": {"host": "127.0.0.1", "port": 8000},
        }
    )


@pytest.fixture
async def client(huddle_config: HuddleConfig) -> AsyncIterator[httpx.AsyncClient]:
    """The full app, with lifespan run so the backend actually starts."""
    app = create_app(huddle_config)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://huddle.test") as http:
            yield http
