"""Supervision for the llama.cpp child processes.

All subprocess handling goes through here. Scattering ``Popen`` calls across
modules is how you end up with orphaned ``rpc-server`` processes holding ports
after a failed start, and with logs that vanish exactly when you need them.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
from collections import deque
from collections.abc import Awaitable, Callable

ReadyCheck = Callable[[], Awaitable[bool]]


class ProcessError(RuntimeError):
    """A managed process failed to start, or died while being supervised."""


class ManagedProcess:
    """One supervised child process with captured output.

    Output is drained continuously into a bounded ring buffer. If the child
    dies, its last lines are what explain why, and they are the only diagnostic
    that survives to the API layer.
    """

    def __init__(
        self,
        name: str,
        argv: list[str],
        *,
        env: dict[str, str] | None = None,
        log_capacity: int = 400,
        startup_capacity: int = 300,
    ) -> None:
        self.name = name
        self.argv = argv
        self.env = env
        self._process: asyncio.subprocess.Process | None = None
        # Two buffers, not one. Everything worth knowing about a llama.cpp
        # start — device registration, layer placement, load errors — is printed
        # in the first few hundred lines, and under `-lv 5` a single rolling
        # buffer evicts all of it within seconds of serving traffic.
        self._startup: list[str] = []
        self._startup_capacity = startup_capacity
        self._recent: deque[str] = deque(maxlen=log_capacity)
        self._seen = 0
        self._drain_task: asyncio.Task[None] | None = None

    @property
    def running(self) -> bool:
        return self._process is not None and self._process.returncode is None

    @property
    def pid(self) -> int | None:
        return self._process.pid if self._process else None

    @property
    def returncode(self) -> int | None:
        return self._process.returncode if self._process else None

    def startup_logs(self) -> list[str]:
        """The process's first lines, which never get evicted."""
        return list(self._startup)

    def logs(self) -> list[str]:
        """Startup output followed by recent output, oldest first.

        A marker is inserted where lines were dropped, so a gap is visible
        rather than being mistaken for contiguous output.
        """
        if not self._recent:
            return list(self._startup)
        omitted = self._seen - len(self._startup) - len(self._recent)
        gap = [f"... {omitted} lines omitted ..."] if omitted > 0 else []
        return [*self._startup, *gap, *self._recent]

    async def start(self, *, ready: ReadyCheck | None = None, timeout: float = 120.0) -> None:
        """Launch the process and wait until it reports ready.

        Raises ``ProcessError`` if the child exits during startup or fails to
        become ready within ``timeout``.
        """
        if self.running:
            raise ProcessError(f"{self.name} is already running")

        environment = {**os.environ, **(self.env or {})}
        self._process = await asyncio.create_subprocess_exec(
            *self.argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=environment,
            # Own process group, so stop() can signal the whole tree rather than
            # leaving grandchildren holding the port.
            start_new_session=True,
        )
        self._drain_task = asyncio.create_task(self._drain())

        if ready is None:
            return

        try:
            await self._await_ready(ready, timeout)
        except Exception:
            await self.stop()
            raise

    async def _await_ready(self, ready: ReadyCheck, timeout: float) -> None:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while loop.time() < deadline:
            if not self.running:
                raise ProcessError(
                    f"{self.name} exited with code {self.returncode} during startup\n"
                    + "\n".join(self.logs()[-20:])
                )
            with contextlib.suppress(Exception):
                if await ready():
                    return
            await asyncio.sleep(0.25)

        tail = "\n".join(self.logs()[-20:])
        raise ProcessError(f"{self.name} did not become ready within {timeout:g}s\n{tail}")

    async def stop(self, *, timeout: float = 10.0) -> None:
        """Stop the process, escalating from SIGTERM to SIGKILL."""
        process = self._process
        if process is None or process.returncode is not None:
            await self._cancel_drain()
            return

        self._signal_group(process, signal.SIGTERM)
        try:
            await asyncio.wait_for(process.wait(), timeout=timeout)
        except TimeoutError:
            self._signal_group(process, signal.SIGKILL)
            with contextlib.suppress(Exception):
                await asyncio.wait_for(process.wait(), timeout=5.0)

        await self._cancel_drain()

    @staticmethod
    def _signal_group(process: asyncio.subprocess.Process, sig: signal.Signals) -> None:
        """Signal the child's whole process group, ignoring races with exit."""
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(os.getpgid(process.pid), sig)

    async def _drain(self) -> None:
        """Continuously copy child output into the ring buffer."""
        assert self._process is not None
        stream = self._process.stdout
        if stream is None:
            return
        while True:
            line = await stream.readline()
            if not line:
                break
            self._record(line.decode(errors="replace").rstrip("\n"))

    def _record(self, line: str) -> None:
        """Keep the first lines forever, then roll the rest."""
        self._seen += 1
        if len(self._startup) < self._startup_capacity:
            self._startup.append(line)
        else:
            self._recent.append(line)

    async def _cancel_drain(self) -> None:
        if self._drain_task is None:
            return
        self._drain_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._drain_task
        self._drain_task = None


async def tcp_is_open(host: str, port: int, *, timeout: float = 2.0) -> bool:
    """Whether something accepts TCP connections at ``host:port``.

    Readiness for ``ggml-rpc-server``, which speaks its own binary protocol
    rather than HTTP, so there is no health endpoint to poll.
    """
    try:
        _reader, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port), timeout=timeout
        )
    except (OSError, TimeoutError):
        return False
    writer.close()
    with contextlib.suppress(Exception):
        await writer.wait_closed()
    return True
