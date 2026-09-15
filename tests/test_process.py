"""Supervision tests, driven against the fake llama-server binary."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx
import pytest

from huddle.process import ManagedProcess, ProcessError, ReadyCheck


def health_check(port: int) -> ReadyCheck:
    async def ready() -> bool:
        async with httpx.AsyncClient(timeout=1.0) as client:
            response = await client.get(f"http://127.0.0.1:{port}/health")
            return response.status_code == 200

    return ready


async def test_starts_and_reports_ready(fake_llama_server: Path, free_port: int) -> None:
    process = ManagedProcess(
        "fake", [str(fake_llama_server), "--host", "127.0.0.1", "--port", str(free_port)]
    )
    try:
        await process.start(ready=health_check(free_port), timeout=20.0)
        assert process.running
        assert process.pid is not None
    finally:
        await process.stop()
    assert not process.running


async def test_stop_is_idempotent(fake_llama_server: Path, free_port: int) -> None:
    process = ManagedProcess(
        "fake", [str(fake_llama_server), "--host", "127.0.0.1", "--port", str(free_port)]
    )
    await process.start(ready=health_check(free_port), timeout=20.0)
    await process.stop()
    await process.stop()
    assert not process.running


async def test_start_fails_when_child_exits_immediately(tmp_path: Path) -> None:
    """A child that dies during startup must raise, not hang until timeout."""
    script = tmp_path / "die.py"
    script.write_text("import sys; print('boom'); sys.exit(3)")
    process = ManagedProcess("dying", [sys.executable, str(script)])

    with pytest.raises(ProcessError, match="exited with code 3"):
        await process.start(ready=health_check(1), timeout=15.0)


async def test_readiness_timeout_stops_the_child(fake_llama_server: Path, free_port: int) -> None:
    """If readiness never arrives we must not leak the process."""
    process = ManagedProcess(
        "fake", [str(fake_llama_server), "--host", "127.0.0.1", "--port", str(free_port)]
    )

    async def never_ready() -> bool:
        return False

    with pytest.raises(ProcessError, match="did not become ready"):
        await process.start(ready=never_ready, timeout=1.5)
    assert not process.running


async def test_logs_are_captured(fake_llama_server: Path, free_port: int) -> None:
    process = ManagedProcess(
        "fake", [str(fake_llama_server), "--host", "127.0.0.1", "--port", str(free_port)]
    )
    try:
        await process.start(ready=health_check(free_port), timeout=20.0)
        assert any("listening on" in line for line in process.logs())
    finally:
        await process.stop()


async def test_fake_records_argv(fake_llama_server: Path, free_port: int, tmp_path: Path) -> None:
    """The fake records argv; this guards the mechanism the argv tests rely on."""
    argv_file = tmp_path / "argv.json"
    process = ManagedProcess(
        "fake",
        [str(fake_llama_server), "--host", "127.0.0.1", "--port", str(free_port), "-ngl", "all"],
        env={"HUDDLE_FAKE_ARGV": str(argv_file)},
    )
    try:
        await process.start(ready=health_check(free_port), timeout=20.0)
    finally:
        await process.stop()

    recorded = json.loads(argv_file.read_text())
    assert "-ngl" in recorded
    assert recorded[recorded.index("-ngl") + 1] == "all"
