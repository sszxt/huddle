"""Searching and downloading GGUF models from Hugging Face.

Every function here does blocking I/O (``huggingface_hub`` is built on
``requests``) — callers in async code must run them through
``asyncio.to_thread``, never await them directly.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import DryRunFileInfo, HfApi, hf_hub_download
from huggingface_hub.errors import EntryNotFoundError, HfHubHTTPError
from tqdm.std import tqdm as tqdm_std


class HFError(RuntimeError):
    """A Hugging Face Hub call failed or returned something unusable."""


@dataclass
class HFFile:
    """One file in a repo. ``size`` is None when the Hub doesn't report it —
    that must never fail the caller, only leave the size unknown to display."""

    filename: str
    size: int | None


@dataclass
class HFModelSummary:
    """One search hit: enough to list it, not enough to download it yet."""

    repo_id: str
    gguf_files: list[str]


@dataclass
class HFRepoFiles:
    """A repo's GGUF files, with sizes — fetched only once a repo is picked."""

    repo_id: str
    files: list[HFFile]


def _wrap(exc: Exception, *, what: str) -> HFError:
    return HFError(f"{what}: {exc}")


def search_models(query: str, *, limit: int = 20) -> list[HFModelSummary]:
    """Repos matching ``query`` that have at least one GGUF file.

    Two-step search by design: this call gives filenames only, not sizes —
    ``HfApi.list_models`` doesn't populate per-file size, only
    ``HfApi.model_info(..., files_metadata=True)`` does, and calling that for
    every hit here would be an HTTP request storm. Call `repo_files` once a
    repo is actually picked.
    """
    try:
        hits = HfApi().list_models(search=query, filter="gguf", limit=limit)
        results: list[HFModelSummary] = []
        for hit in hits:
            siblings = hit.siblings or []
            gguf_files = [s.rfilename for s in siblings if s.rfilename.endswith(".gguf")]
            if gguf_files:
                results.append(HFModelSummary(repo_id=hit.id, gguf_files=gguf_files))
        return results
    except (HfHubHTTPError, EntryNotFoundError, OSError) as exc:
        raise _wrap(exc, what=f"search failed for {query!r}") from exc


def repo_files(repo_id: str) -> HFRepoFiles:
    """A repo's GGUF files with sizes, fetched with ``files_metadata=True``."""
    try:
        info = HfApi().model_info(repo_id, files_metadata=True)
        siblings = info.siblings or []
        files = [
            HFFile(filename=s.rfilename, size=s.size)
            for s in siblings
            if s.rfilename.endswith(".gguf")
        ]
        return HFRepoFiles(repo_id=repo_id, files=files)
    except (HfHubHTTPError, EntryNotFoundError, OSError) as exc:
        raise _wrap(exc, what=f"could not list files for {repo_id!r}") from exc


def preflight(repo_id: str, filename: str) -> DryRunFileInfo:
    """Metadata only (a HEAD request) — no content downloaded.

    Used for the size shown before a download is confirmed, and for the
    disk-space check the caller does before actually starting one.
    """
    try:
        info = hf_hub_download(repo_id=repo_id, filename=filename, dry_run=True)
    except (HfHubHTTPError, EntryNotFoundError, OSError) as exc:
        raise _wrap(exc, what=f"could not check {repo_id!r}/{filename!r}") from exc
    if not isinstance(info, DryRunFileInfo):
        # dry_run=True always returns DryRunFileInfo; this satisfies mypy's
        # narrowing of hf_hub_download's `str | DryRunFileInfo` return type.
        raise HFError(f"unexpected response checking {repo_id!r}/{filename!r}")
    return info


def download(
    repo_id: str,
    filename: str,
    *,
    local_dir: Path,
    on_progress: Callable[[int, int | None], None],
) -> Path:
    """Blocking. Downloads flat to ``local_dir/filename`` — matches exactly
    how ``ClusterService.available_models()`` globs ``models.dir``.

    There is no ``progress_callback`` parameter on ``hf_hub_download`` —
    verified against upstream source. The only hook is ``tqdm_class``: the
    chunked-download loop calls ``.update(len(chunk))`` on whatever instance
    it constructs, so a ``tqdm`` subclass forwards progress from there. It is
    defined inside this function (closing over ``on_progress`` directly)
    rather than as a module-level class taking a constructor argument,
    because ``hf_hub_download``'s ``tqdm_class`` parameter is typed as
    ``type[Any]`` — an actual class, not a factory function returning one.
    """

    # The type stub declares tqdm generic (hence the ignore below), but the
    # real runtime class does not support subscripting — `tqdm_std[Any]`
    # raises `TypeError: type 'tqdm' is not subscriptable` at import time.
    class _ProgressTqdm(tqdm_std):  # type: ignore[type-arg]
        def update(self, n: float | None = 1) -> bool | None:
            result = super().update(n)
            total = int(self.total) if self.total is not None else None
            on_progress(int(self.n), total)
            return result

    try:
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=local_dir,
            tqdm_class=_ProgressTqdm,
        )
    except (HfHubHTTPError, EntryNotFoundError, OSError) as exc:
        raise _wrap(exc, what=f"download failed for {repo_id!r}/{filename!r}") from exc
    return Path(path)


__all__ = [
    "HFError",
    "HFFile",
    "HFModelSummary",
    "HFRepoFiles",
    "download",
    "preflight",
    "repo_files",
    "search_models",
]
