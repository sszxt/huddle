"""Ownership of this node's llama.cpp processes.

Two roles, either or both of which a node may play: the head ``llama-server``
that serves requests, and the worker ``ggml-rpc-server`` that holds layers for
some other node's head process.
"""

from __future__ import annotations

import asyncio

import httpx
from pydantic import BaseModel

from huddle.config import HuddleConfig
from huddle.hardware import NodeHardware, probe
from huddle.llamacpp import (
    DIAGNOSTIC_LINES,
    build_llama_server_argv,
    build_rpc_server_argv,
    parse_layer_assignments,
    require_binary,
)
from huddle.process import ManagedProcess, ProcessError, ReadyCheck, tcp_is_open


class BackendStatus(BaseModel):
    """What the head ``llama-server`` is currently doing."""

    running: bool
    model: str | None = None
    pid: int | None = None
    base_url: str
    argv: list[str] = []
    returncode: int | None = None


class RpcStatus(BaseModel):
    """What this node's worker ``ggml-rpc-server`` is doing."""

    running: bool
    # True when the RPC port is held by a process this agent did not start —
    # typically a worker that outlived an agent restart. It is still serving a
    # head node somewhere, so it must be reported rather than silently ignored
    # or killed.
    foreign: bool = False
    endpoint: str | None = None
    bind: str
    port: int
    cache: bool
    pid: int | None = None
    argv: list[str] = []
    returncode: int | None = None


class BackendService:
    """Starts, stops and reports on this node's llama.cpp processes."""

    def __init__(self, config: HuddleConfig) -> None:
        self.config = config
        self._process: ManagedProcess | None = None
        # The most recent launch, kept even when it failed. A failed start is
        # exactly when its output matters, and it used to be discarded with the
        # process — /agent/backend/logs returned nothing after a crashed load.
        self._last_attempt: ManagedProcess | None = None
        self._model: str | None = None
        self._lock = asyncio.Lock()
        self._rpc: ManagedProcess | None = None
        self._rpc_lock = asyncio.Lock()

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
        """Output of the running backend, or of the last attempt if none runs."""
        source = self._process or self._last_attempt
        return source.logs() if source else []

    def placement(self) -> dict[str, list[int]]:
        """Where llama.cpp says each layer went, from its own verbose output.

        Empty unless the backend runs with ``-lv 5``: at default verbosity
        llama.cpp prints nothing about placement.
        """
        source = self._process or self._last_attempt
        if source is None:
            return {}
        return parse_layer_assignments("\n".join(source.pinned_logs()))

    def hardware(self) -> NodeHardware:
        return probe(self.config.node.name, self.config.binaries.llama_server)

    async def start(
        self,
        model: str | None = None,
        *,
        rpc_endpoints: list[str] | None = None,
        tensor_split: list[float] | None = None,
        n_gpu_layers: int | str | None = None,
        timeout: float = 900.0,
    ) -> BackendStatus:
        """Launch ``llama-server`` and wait until it reports healthy.

        The timeout is generous because a clustered start is slow for a real
        reason: the head streams every remote layer's weights across the network
        before it will answer, which on 1 GbE is minutes for a large model. The
        peers' tensor caches make that a one-off.
        """
        async with self._lock:
            if self.running:
                raise ProcessError("backend is already running; stop it first")

            # Forget the previous attempt before validating anything. Otherwise a
            # start that fails before launching (a missing model, say) would
            # still show the old attempt's output, and a caller reading it for
            # an out-of-memory signature would react to a failure that is not
            # this one.
            self._last_attempt = None
            binary = require_binary(self.config.binaries.llama_server, role="llama-server")
            model_path = self.config.models.resolve(model)
            if not model_path.exists():
                raise ProcessError(f"model not found: {model_path}")

            backend = self.config.backend
            if n_gpu_layers is not None:
                backend = backend.model_copy(update={"n_gpu_layers": n_gpu_layers})

            argv = build_llama_server_argv(
                binary,
                backend,
                model_path,
                rpc_endpoints=rpc_endpoints,
                tensor_split=tensor_split,
                alias=model_path.stem,
            )
            process = ManagedProcess("llama-server", argv, keep=DIAGNOSTIC_LINES)
            self._last_attempt = process
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

    # -- worker role: ggml-rpc-server ------------------------------------

    @property
    def rpc_running(self) -> bool:
        return self._rpc is not None and self._rpc.running

    async def rpc_report(self) -> RpcStatus:
        """Status reconciled against the port, not just our own bookkeeping.

        An agent restart loses track of a worker it started, because the child
        outlives it. Reporting `running=False` while something is serving on the
        port is worse than useless: the coordinator would try to start another.
        """
        status = self.rpc_status()
        if not status.running and await tcp_is_open("127.0.0.1", self.config.rpc.port, timeout=1.0):
            status.foreign = True
        return status

    def rpc_status(self) -> RpcStatus:
        rpc = self.config.rpc
        endpoint = f"{rpc.advertise}:{rpc.port}" if rpc.advertise else None
        return RpcStatus(
            running=self.rpc_running,
            endpoint=endpoint if self.rpc_running else None,
            bind=rpc.bind,
            port=rpc.port,
            cache=rpc.cache,
            pid=self._rpc.pid if self._rpc else None,
            argv=self._rpc.argv if self._rpc else [],
            returncode=self._rpc.returncode if self._rpc else None,
        )

    def rpc_logs(self) -> list[str]:
        return self._rpc.logs() if self._rpc else []

    async def start_rpc(self, *, timeout: float = 60.0) -> RpcStatus:
        """Launch this node's worker so a remote head can place layers here."""
        async with self._rpc_lock:
            if self.rpc_running:
                raise ProcessError("rpc-server is already running; stop it first")

            port = self.config.rpc.port
            if await tcp_is_open("127.0.0.1", port, timeout=1.0):
                # Something is already serving here that we did not start. It is
                # very likely a worker that outlived an agent restart and is
                # still holding layers for a live head, so refuse loudly rather
                # than fail obscurely on a port bind.
                raise ProcessError(
                    f"port {port} is already in use by a worker this agent did not start; "
                    f"stop it manually before starting a new one"
                )

            binary = self.config.binaries.rpc_server
            if binary is None:
                raise ProcessError("no rpc_server binary configured for this node")
            # Upstream builds it as ggml-rpc-server; the config names the path.
            require_binary(binary, role="rpc-server")

            argv = build_rpc_server_argv(binary, self.config.rpc)
            process = ManagedProcess("rpc-server", argv)
            await process.start(ready=self._rpc_probe(), timeout=timeout)

            self._rpc = process
            return self.rpc_status()

    async def stop_rpc(self, *, timeout: float = 30.0) -> RpcStatus:
        async with self._rpc_lock:
            if self._rpc is not None:
                await self._rpc.stop(timeout=timeout)
            status = self.rpc_status()
            self._rpc = None
            return status

    def _rpc_probe(self) -> ReadyCheck:
        """Readiness is a successful TCP connect: there is no health endpoint.

        Probes 127.0.0.1 rather than the bind address, because 0.0.0.0 is not
        a connectable destination.
        """
        port = self.config.rpc.port

        async def ready() -> bool:
            return await tcp_is_open("127.0.0.1", port, timeout=1.0)

        return ready
