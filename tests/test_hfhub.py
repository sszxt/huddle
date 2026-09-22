"""Searching/downloading from Hugging Face: never hits the real network here."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from huggingface_hub import HfApi
from huggingface_hub.errors import EntryNotFoundError

from huddle import hfhub


def _sibling(rfilename: str, size: int | None = None) -> SimpleNamespace:
    return SimpleNamespace(rfilename=rfilename, size=size)


def _model_info(model_id: str, *siblings: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(id=model_id, siblings=list(siblings))


def test_search_keeps_only_repos_with_gguf_files(monkeypatch: pytest.MonkeyPatch) -> None:
    hits = [
        _model_info("someone/has-gguf", _sibling("model.gguf"), _sibling("README.md")),
        _model_info("someone/no-gguf", _sibling("README.md")),
    ]
    monkeypatch.setattr(HfApi, "list_models", lambda self, **kw: iter(hits))

    results = hfhub.search_models("qwen")

    assert [r.repo_id for r in results] == ["someone/has-gguf"]
    assert results[0].gguf_files == ["model.gguf"]


def test_search_wraps_hub_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(self: HfApi, **kw: object) -> None:
        raise EntryNotFoundError("nope")

    monkeypatch.setattr(HfApi, "list_models", boom)

    with pytest.raises(hfhub.HFError):
        hfhub.search_models("qwen")


def test_repo_files_maps_sizes_and_filters_to_gguf(monkeypatch: pytest.MonkeyPatch) -> None:
    info = _model_info(
        "bartowski/Qwen2.5-32B-Instruct-GGUF",
        _sibling("Qwen2.5-32B-Instruct-Q4_K_M.gguf", size=19_851_336_576),
        _sibling("README.md"),
    )
    monkeypatch.setattr(HfApi, "model_info", lambda self, repo_id, **kw: info)

    result = hfhub.repo_files("bartowski/Qwen2.5-32B-Instruct-GGUF")

    assert result.repo_id == "bartowski/Qwen2.5-32B-Instruct-GGUF"
    assert len(result.files) == 1
    assert result.files[0].filename == "Qwen2.5-32B-Instruct-Q4_K_M.gguf"
    assert result.files[0].size == 19_851_336_576


def test_repo_files_tolerates_missing_size(monkeypatch: pytest.MonkeyPatch) -> None:
    """The Hub not reporting a size must never fail the caller."""
    info = _model_info("someone/repo", _sibling("model.gguf", size=None))
    monkeypatch.setattr(HfApi, "model_info", lambda self, repo_id, **kw: info)

    result = hfhub.repo_files("someone/repo")

    assert result.files[0].size is None


def test_download_reports_progress_through_tqdm_class(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`hf_hub_download` has no progress_callback; the only hook is tqdm_class."""
    progress_calls: list[tuple[int, int | None]] = []

    def fake_hf_hub_download(
        *, repo_id: str, filename: str, local_dir: Path, tqdm_class: type
    ) -> str:
        # Drive the real tqdm_class the way http_get actually does: construct
        # it, then call .update() per chunk.
        bar = tqdm_class(total=100)
        bar.update(40)
        bar.update(60)
        return str(local_dir / filename)

    monkeypatch.setattr(hfhub, "hf_hub_download", fake_hf_hub_download)

    path = hfhub.download(
        "someone/repo",
        "model.gguf",
        local_dir=tmp_path,
        on_progress=lambda done, total: progress_calls.append((done, total)),
    )

    assert path == tmp_path / "model.gguf"
    assert progress_calls == [(40, 100), (100, 100)]


def test_download_wraps_hub_errors(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def boom(**kw: object) -> str:
        raise EntryNotFoundError("nope")

    monkeypatch.setattr(hfhub, "hf_hub_download", boom)

    with pytest.raises(hfhub.HFError):
        hfhub.download(
            "someone/repo", "model.gguf", local_dir=tmp_path, on_progress=lambda a, b: None
        )
