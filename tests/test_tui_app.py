"""Smoke test: the TUI mounts, renders a snapshot, and its actions call the
poller — driven entirely against a stubbed `ClusterPoller`, never real
network I/O. Nothing here proves the *data* is right; `test_tui_poller.py`
covers that. This only proves the widget tree comes up and wiring works.
"""

from __future__ import annotations

import pytest

from huddle.config import HuddleConfig
from huddle.coordinator.service import ClusterStatus
from huddle.hardware import DeviceInfo, NodeHardware
from huddle.tui.app import HuddleTUI
from huddle.tui.poller import ClusterPoller, ClusterSnapshot, NodeSnapshot


def _canned_snapshot() -> ClusterSnapshot:
    head = NodeSnapshot(
        name="head1",
        role="head",
        reachable=True,
        hardware=NodeHardware(
            name="head1",
            devices=[
                DeviceInfo(
                    id="Vulkan0", name="NVIDIA GeForce RTX 5070", total_mib=12227, free_mib=9000
                )
            ],
            cpu_count=8,
            ram_total_mib=32768,
            ram_available_mib=20000,
        ),
        layers=13,
    )
    return ClusterSnapshot(
        fetched_at=0.0,
        cluster=ClusterStatus(running=True, model="tiny.gguf", head_node="head1"),
        nodes=[head],
        available_models=["tiny.gguf"],
        loaded_model="tiny.gguf",
    )


@pytest.fixture(autouse=True)
def _stub_poller(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_poll(self: ClusterPoller) -> ClusterSnapshot:
        return _canned_snapshot()

    async def fake_head_logs(self: ClusterPoller) -> list[str]:
        return ["fake llama-server listening on 127.0.0.1:8080"]

    async def fake_aclose(self: ClusterPoller) -> None:
        return None

    monkeypatch.setattr(ClusterPoller, "poll", fake_poll)
    monkeypatch.setattr(ClusterPoller, "head_logs", fake_head_logs)
    monkeypatch.setattr(ClusterPoller, "aclose", fake_aclose)


async def test_mounts_one_node_box_per_snapshot_node(huddle_config: HuddleConfig) -> None:
    app = HuddleTUI(huddle_config)
    async with app.run_test() as pilot:
        await pilot.pause()
        assert len(app.query(".node-box")) == 1
        assert app.snapshot is not None
        assert app.snapshot.nodes[0].name == "head1"


async def test_q_quits_cleanly(huddle_config: HuddleConfig) -> None:
    app = HuddleTUI(huddle_config)
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("q")


async def test_start_action_calls_the_poller(
    huddle_config: HuddleConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    async def fake_start(self: ClusterPoller, model: str | None = None) -> ClusterStatus:
        calls.append("start")
        return ClusterStatus(running=True, head_node="head1")

    monkeypatch.setattr(ClusterPoller, "start_cluster", fake_start)

    app = HuddleTUI(huddle_config)
    async with app.run_test() as pilot:
        await pilot.pause()
        app.action_start_cluster()
        await pilot.pause()

    assert calls == ["start"]


async def test_stop_action_requires_confirmation(
    huddle_config: HuddleConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pressing 'x' opens a confirmation modal rather than stopping immediately."""
    calls: list[str] = []

    async def fake_stop(self: ClusterPoller) -> ClusterStatus:
        calls.append("stop")
        return ClusterStatus(running=False, head_node="head1")

    monkeypatch.setattr(ClusterPoller, "stop_cluster", fake_stop)

    app = HuddleTUI(huddle_config)
    async with app.run_test() as pilot:
        await pilot.pause()
        app.action_stop_cluster()
        await pilot.pause()
        assert calls == [], "must wait for confirmation before stopping"
