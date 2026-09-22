"""DownloadService in isolation: no HTTP, no real Hugging Face calls."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from huddle import hfhub
from huddle.coordinator.downloads import AlreadyDownloading, DownloadService


@pytest.fixture
def service(tmp_path: Path) -> DownloadService:
    return DownloadService(tmp_path)


async def test_idle_status(service: DownloadService) -> None:
    status = service.status()
    assert status.active is False
    assert status.repo_id is None
    assert status.done is False


async def test_second_download_while_active_raises(
    service: DownloadService, monkeypatch: pytest.MonkeyPatch
) -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    async def slow_run(self: DownloadService) -> None:
        started.set()
        await release.wait()
        self._done = True

    monkeypatch.setattr(DownloadService, "_run", slow_run)

    service.start("someone/repo", "model.gguf")
    await started.wait()

    with pytest.raises(AlreadyDownloading):
        service.start("someone/other", "other.gguf")

    release.set()
    assert service._task is not None
    await service._task


async def test_status_reflects_progress(service: DownloadService) -> None:
    service.start("someone/repo", "model.gguf")
    service._on_progress(42, 100)

    status = service.status()
    assert status.repo_id == "someone/repo"
    assert status.filename == "model.gguf"
    assert status.bytes_done == 42
    assert status.total_bytes == 100

    if service._task is not None:
        service._task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await service._task


async def test_download_failure_is_captured_not_raised(
    service: DownloadService, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*args: object, **kwargs: object) -> Path:
        raise hfhub.HFError("network is down")

    monkeypatch.setattr(hfhub, "download", boom)

    service.start("someone/repo", "model.gguf")
    assert service._task is not None
    await service._task

    status = service.status()
    assert status.active is False
    assert status.done is False
    assert status.error == "network is down"


async def test_successful_download_marks_done(
    service: DownloadService, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_download(repo_id: str, filename: str, *, local_dir: Path, on_progress: object) -> Path:
        target = local_dir / filename
        target.write_bytes(b"fake gguf content")
        return target

    monkeypatch.setattr(hfhub, "download", fake_download)

    service.start("someone/repo", "model.gguf")
    assert service._task is not None
    await service._task

    status = service.status()
    assert status.done is True
    assert status.error is None
    assert (tmp_path / "model.gguf").exists()


async def test_cancel_stops_a_running_download(
    service: DownloadService, monkeypatch: pytest.MonkeyPatch
) -> None:
    started = asyncio.Event()

    async def hang_forever(self: DownloadService) -> None:
        started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(DownloadService, "_run", hang_forever)

    service.start("someone/repo", "model.gguf")
    await started.wait()

    await service.cancel()

    assert service.status().active is False
