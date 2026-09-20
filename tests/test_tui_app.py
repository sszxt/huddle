"""HuddleTUI's action wiring, driven directly (no real terminal involved).

`_stop_cluster_confirmed`/`_switch_model_prompt` own the interactive
`input()` prompts; the underlying `_stop_cluster`/`_switch_model`/
`_start_cluster` methods they delegate to are what's tested here, exactly
the same split `poller.py`'s controls already have (ask vs. act).
"""

from __future__ import annotations

import httpx
import pytest

from huddle.config import HuddleConfig
from huddle.coordinator.service import ClusterStatus
from huddle.tui.app import HuddleTUI
from huddle.tui.poller import ClusterPoller, ClusterSnapshot, NodeSnapshot


def _config() -> HuddleConfig:
    return HuddleConfig.model_validate(
        {
            "node": {"name": "head1"},
            "binaries": {"llama_server": "/opt/llama-server"},
            "models": {"dir": "/models"},
            "api": {"host": "127.0.0.1", "port": 8000},
        }
    )


def _snapshot(**overrides: object) -> ClusterSnapshot:
    defaults: dict[str, object] = dict(
        fetched_at=0.0,
        cluster=ClusterStatus(running=True, head_node="head1", model="tiny.gguf"),
        nodes=[NodeSnapshot(name="head1", role="head", reachable=True, layers=13)],
        available_models=["tiny.gguf", "other.gguf"],
        loaded_model="tiny.gguf",
    )
    defaults.update(overrides)
    return ClusterSnapshot(**defaults)  # type: ignore[arg-type]


async def test_refresh_populates_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_poll(self: ClusterPoller) -> ClusterSnapshot:
        return _snapshot()

    monkeypatch.setattr(ClusterPoller, "poll", fake_poll)

    app = HuddleTUI(_config())
    await app._refresh()

    assert app.snapshot is not None
    assert app.snapshot.nodes[0].name == "head1"
    assert app.message is None


async def test_refresh_records_a_poll_failure_without_raising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_poll(self: ClusterPoller) -> ClusterSnapshot:
        raise RuntimeError("connection refused")

    monkeypatch.setattr(ClusterPoller, "poll", fake_poll)

    app = HuddleTUI(_config())
    await app._refresh()  # must not raise

    assert app.message is not None
    assert "connection refused" in app.message


async def test_start_cluster_calls_the_poller(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    async def fake_start(self: ClusterPoller, model: str | None = None) -> ClusterStatus:
        calls.append("start")
        return ClusterStatus(running=True, head_node="head1")

    async def fake_poll(self: ClusterPoller) -> ClusterSnapshot:
        return _snapshot()

    monkeypatch.setattr(ClusterPoller, "start_cluster", fake_start)
    monkeypatch.setattr(ClusterPoller, "poll", fake_poll)

    app = HuddleTUI(_config())
    await app._start_cluster()

    assert calls == ["start"]
    assert app.snapshot is not None  # _start_cluster refreshes afterward


async def test_stop_cluster_calls_the_poller(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    async def fake_stop(self: ClusterPoller) -> ClusterStatus:
        calls.append("stop")
        return ClusterStatus(running=False, head_node="head1")

    async def fake_poll(self: ClusterPoller) -> ClusterSnapshot:
        return _snapshot(cluster=ClusterStatus(running=False, head_node="head1"))

    monkeypatch.setattr(ClusterPoller, "stop_cluster", fake_stop)
    monkeypatch.setattr(ClusterPoller, "poll", fake_poll)

    app = HuddleTUI(_config())
    await app._stop_cluster()

    assert calls == ["stop"]


async def test_switch_model_calls_the_poller_with_the_chosen_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    async def fake_switch(self: ClusterPoller, model: str) -> ClusterStatus:
        calls.append(model)
        return ClusterStatus(running=True, head_node="head1", model=model)

    async def fake_poll(self: ClusterPoller) -> ClusterSnapshot:
        return _snapshot()

    monkeypatch.setattr(ClusterPoller, "switch_model", fake_switch)
    monkeypatch.setattr(ClusterPoller, "poll", fake_poll)

    app = HuddleTUI(_config())
    await app._switch_model("other.gguf")

    assert calls == ["other.gguf"]


async def test_start_failure_sets_a_message_instead_of_raising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_start(self: ClusterPoller, model: str | None = None) -> ClusterStatus:
        raise httpx.ConnectError("boom")

    async def fake_poll(self: ClusterPoller) -> ClusterSnapshot:
        return _snapshot()

    monkeypatch.setattr(ClusterPoller, "start_cluster", fake_start)
    monkeypatch.setattr(ClusterPoller, "poll", fake_poll)

    app = HuddleTUI(_config())
    await app._start_cluster()  # must not raise

    assert app.message is not None
    assert "boom" in app.message
