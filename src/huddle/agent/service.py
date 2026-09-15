"""Ownership of this node's llama.cpp processes.

v0 manages the head ``llama-server`` only. The worker ``ggml-rpc-server`` joins
in v1, which is why lifecycle and status are shaped generically here rather than
around a single process.
"""

from __future__ import annotations

import asyncio

import httpx
from pydantic import BaseModel

from huddle.config import HuddleConfig
from huddle.hardware import NodeHardware, probe
from huddle.llamacpp import build_llama_server_argv, require_binary
from huddle.process import ManagedProcess, ProcessError, ReadyCheck


class BackendStatus(BaseModel):
    """What the head ``llama-server`` is currently doing."""

    running: bool
    model: str | None = None
    pid: int | None = None
    base_url: str
    argv: list[str] = []
    returncode: int | None = None


class BackendService:
    """Starts, stops and reports on this node's ``llama-server``."""

    def __init__(self, config: HuddleConfig) -> None:
        self.config = config
        self._process: ManagedProcess | None = None
        self._model: str | None = None
        self._lock = asyncio.Lock()

    @property
    def base_url(self) -> str:
        """Where the backend listens. Not the public API address."""
        return f"http://{self.config.backend.host}:{self.config.backend.port}"

    @property
    def running(self) -> bool:
        return self._process is not None and self._process.running

    def status(self) -> BackendStatus:
        return BackendStatus(
            running=self.running,
            model=self._model,
            pid=self._process.pid if self._process else None,
            base_url=self.base_url,
            argv=self._process.argv if self._process else [],
            returncode=self._process.returncode if self._process else None,
        )

    def logs(self) -> list[str]:
        return self._process.logs() if self._process else []

    def hardware(self) -> NodeHardware:
        return probe(self.config.node.name, self.config.binaries.llama_server)

    async def start(self, model: str | None = None, *, timeout: float = 300.0) -> BackendStatus:
        """Launch ``llama-server`` and wait until it reports healthy.

        The timeout is generous: a large GGUF takes real time to load, and in a
        cluster the head node also streams weights to every peer first.
        """
        async with self._lock:
            if self.running:
                raise ProcessError("backend is already running; stop it first")

            binary = require_binary(self.config.binaries.llama_server, role="llama-server")
            model_path = self.config.models.resolve(model)
            if not model_path.exists():
                raise ProcessError(f"model not found: {model_path}")

            argv = build_llama_server_argv(
                binary,
                self.config.backend,
                model_path,
                alias=model_path.stem,
            )
            process = ManagedProcess("llama-server", argv)
            await process.start(ready=self._health_probe(), timeout=timeout)

            self._process = process
            self._model = model_path.name
            return self.status()

    async def stop(self, *, timeout: float = 30.0) -> BackendStatus:
        async with self._lock:
            if self._process is not None:
                await self._process.stop(timeout=timeout)
            status = self.status()
            self._process = None
            self._model = None
            return status

    def _health_probe(self) -> ReadyCheck:
        """Poll llama-server's own /health, which 503s until the model loads."""
        url = f"{self.base_url}/health"

        async def ready() -> bool:
            async with httpx.AsyncClient(timeout=2.0) as client:
                response = await client.get(url)
                return response.status_code == 200

        return ready
