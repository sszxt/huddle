"""The /cluster/models/* download endpoints, driven over a real ASGI transport.

Every Hugging Face call is monkeypatched at the `huddle.hfhub` module level —
never real network, matching every other test in this suite.
"""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from huddle import hfhub
from huddle.config import HuddleConfig
from tests.fakes.gguf_builder import write_gguf


async def test_search_endpoint_returns_results(
    client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_results = [hfhub.HFModelSummary(repo_id="someone/repo", gguf_files=["model.gguf"])]
    monkeypatch.setattr(hfhub, "search_models", lambda q, **kw: fake_results)

    response = await client.get("/cluster/models/search", params={"q": "qwen"})

    assert response.status_code == 200
    assert response.json()["results"] == [{"repo_id": "someone/repo", "gguf_files": ["model.gguf"]}]


async def test_search_requires_a_query(client: httpx.AsyncClient) -> None:
    response = await client.get("/cluster/models/search", params={"q": "  "})
    assert response.status_code == 422


async def test_search_failure_is_a_502(
    client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(q: str, **kw: object) -> None:
        raise hfhub.HFError("hub unreachable")

    monkeypatch.setattr(hfhub, "search_models", boom)

    response = await client.get("/cluster/models/search", params={"q": "qwen"})
    assert response.status_code == 502


async def test_repo_files_endpoint(
    client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = hfhub.HFRepoFiles(
        repo_id="someone/repo", files=[hfhub.HFFile(filename="model.gguf", size=123)]
    )
    monkeypatch.setattr(hfhub, "repo_files", lambda repo_id: fake)

    response = await client.get("/cluster/models/repo-files", params={"repo_id": "someone/repo"})

    assert response.status_code == 200
    assert response.json() == {
        "repo_id": "someone/repo",
        "files": [{"filename": "model.gguf", "size": 123}],
    }


async def test_repo_files_unknown_repo_is_404(
    client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(repo_id: str) -> None:
        raise hfhub.HFError("not found")

    monkeypatch.setattr(hfhub, "repo_files", boom)

    response = await client.get("/cluster/models/repo-files", params={"repo_id": "nope/nope"})
    assert response.status_code == 404


async def test_download_rejects_insufficient_disk_space(
    client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    huge = SimpleNamespace(file_size=10**18)  # an absurd size no test machine has free
    monkeypatch.setattr(hfhub, "preflight", lambda repo_id, filename: huge)

    response = await client.post(
        "/cluster/models/download", json={"repo_id": "someone/repo", "filename": "model.gguf"}
    )

    assert response.status_code == 507


async def test_download_unknown_repo_is_404(
    client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(repo_id: str, filename: str) -> None:
        raise hfhub.HFError("not found")

    monkeypatch.setattr(hfhub, "preflight", boom)

    response = await client.post(
        "/cluster/models/download", json={"repo_id": "nope/nope", "filename": "x.gguf"}
    )
    assert response.status_code == 404


async def test_download_rejects_a_second_concurrent_download(
    client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        hfhub, "preflight", lambda repo_id, filename: SimpleNamespace(file_size=None)
    )
    # asyncio.to_thread runs `download` in a real worker thread, so a
    # threading.Event (not an asyncio.Event) is what actually blocks it.
    started = threading.Event()
    release = threading.Event()

    def slow_download(repo_id: str, filename: str, *, local_dir: Path, on_progress: object) -> Path:
        started.set()
        release.wait(timeout=5)
        return local_dir / filename

    monkeypatch.setattr(hfhub, "download", slow_download)

    try:
        first = await client.post(
            "/cluster/models/download",
            json={"repo_id": "someone/repo", "filename": "model.gguf"},
        )
        assert first.status_code == 200

        for _ in range(50):
            if started.is_set():
                break
            await asyncio.sleep(0.02)
        assert started.is_set(), "download never actually started"

        second = await client.post(
            "/cluster/models/download",
            json={"repo_id": "someone/other", "filename": "other.gguf"},
        )
        assert second.status_code == 409
    finally:
        release.set()


async def test_download_completes_and_available_models_picks_it_up(
    client: httpx.AsyncClient, huddle_config: HuddleConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercises the real, unmodified `ClusterService.available_models()`."""
    monkeypatch.setattr(
        hfhub, "preflight", lambda repo_id, filename: SimpleNamespace(file_size=None)
    )

    def fake_download(repo_id: str, filename: str, *, local_dir: Path, on_progress: object) -> Path:
        target = local_dir / filename
        write_gguf(target, n_layers=4, n_embd=64)
        return target

    monkeypatch.setattr(hfhub, "download", fake_download)

    response = await client.post(
        "/cluster/models/download",
        json={"repo_id": "someone/repo", "filename": "downloaded.gguf"},
    )
    assert response.status_code == 200

    status = {"active": True}
    for _ in range(50):
        status = (await client.get("/cluster/models/download")).json()
        if not status["active"]:
            break
        await asyncio.sleep(0.05)

    assert status["done"] is True, status
    assert status["error"] is None

    models = (await client.get("/cluster/models")).json()
    assert "downloaded.gguf" in models["available"]
