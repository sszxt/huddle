"""Downloading a model onto this node, in the background.

Fully independent of ``ClusterService`` — they only meet inside ``app.py``.
``ClusterService.available_models()`` globs ``models.dir`` for ``*.gguf``
files, so a completed download needs no code there to become selectable.

Progress is read without a lock. ``_on_progress`` runs on the worker thread
``asyncio.to_thread`` uses for the blocking Hugging Face call; the read side
is an async HTTP handler reading plain ``int``/``None`` attributes off this
service. Attribute assignment and reads are atomic under the GIL, and this is
purely informational — a stale-by-one-chunk progress number is harmless. Same
pragmatic style as ``ManagedProcess``'s own unlocked log ring buffer.
"""

from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path

from pydantic import BaseModel

from huddle import hfhub


class DownloadStatus(BaseModel):
    """What the current (or most recent) download is doing."""

    active: bool
    repo_id: str | None = None
    filename: str | None = None
    bytes_done: int = 0
    total_bytes: int | None = None
    done: bool = False
    error: str | None = None


class AlreadyDownloading(RuntimeError):
    """A download is already in progress; only one runs at a time."""


class DownloadService:
    """Starts, tracks, and can cancel one download at a time."""

    def __init__(self, models_dir: Path) -> None:
        self._models_dir = models_dir
        self._task: asyncio.Task[None] | None = None
        self._repo_id: str | None = None
        self._filename: str | None = None
        self._bytes_done = 0
        self._total_bytes: int | None = None
        self._done = False
        self._error: str | None = None

    def status(self) -> DownloadStatus:
        return DownloadStatus(
            active=self._task is not None and not self._task.done(),
            repo_id=self._repo_id,
            filename=self._filename,
            bytes_done=self._bytes_done,
            total_bytes=self._total_bytes,
            done=self._done,
            error=self._error,
        )

    async def search(self, query: str) -> list[hfhub.HFModelSummary]:
        return await asyncio.to_thread(hfhub.search_models, query)

    async def repo_files(self, repo_id: str) -> hfhub.HFRepoFiles:
        return await asyncio.to_thread(hfhub.repo_files, repo_id)

    def start(self, repo_id: str, filename: str) -> None:
        """Fire-and-forget: the caller does not await the download itself.

        Disk-space and existence checks happen before this is called (in the
        HTTP layer, so they can fail the *triggering* request synchronously
        instead of being buried in ``DownloadStatus.error`` on a later poll).
        """
        if self._task is not None and not self._task.done():
            raise AlreadyDownloading(f"already downloading {self._repo_id}/{self._filename}")
        self._repo_id, self._filename = repo_id, filename
        self._bytes_done, self._total_bytes = 0, None
        self._done, self._error = False, None
        self._task = asyncio.create_task(self._run())

    async def _run(self) -> None:
        assert self._repo_id is not None
        assert self._filename is not None
        try:
            await asyncio.to_thread(
                hfhub.download,
                self._repo_id,
                self._filename,
                local_dir=self._models_dir,
                on_progress=self._on_progress,
            )
        except Exception as exc:  # a download failure must not crash the coordinator
            self._error = str(exc)
        else:
            self._done = True

    def _on_progress(self, done: int, total: int | None) -> None:
        self._bytes_done, self._total_bytes = done, total

    async def cancel(self) -> None:
        if self._task is not None and not self._task.done():
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        self._task = None


__all__ = ["AlreadyDownloading", "DownloadService", "DownloadStatus"]
