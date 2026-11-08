"""`huddle plan` must show the placement a start would use.

It once planned by itself: no ``planner.reserve_mib``, and a peer that was given
no layers still listed in ``--rpc``. These tests pin both, and the free-memory
reading they both depend on.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from huddle.cli import app
from huddle.config import HuddleConfig, PeerConfig
from huddle.coordinator.cluster import PeerReport, build_device_list
from huddle.coordinator.service import place
from huddle.hardware import DeviceInfo, NodeHardware
from tests.test_planner import make_model


def node(name: str, *devices: DeviceInfo) -> NodeHardware:
    return NodeHardware(name=name, devices=list(devices))


def card(id_: str, total: int | None, free: int | None) -> DeviceInfo:
    return DeviceInfo(id=id_, name=f"card-{id_}", total_mib=total, free_mib=free)


def peer_report(*devices: DeviceInfo) -> PeerReport:
    return PeerReport(
        peer=PeerConfig(name="peer1", host="127.0.0.1"), hardware=node("peer1", *devices)
    )


# -- the free-memory reading ----------------------------------------------------------


def test_a_device_with_nothing_free_is_planned_as_full() -> None:
    """0 MiB free is a reading, not a missing one: it must not become the total."""
    devices, _ = build_device_list(
        node("head", card("Vulkan0", 12000, 0)), [peer_report(card("Vulkan0", 12000, 0))]
    )
    assert [d.free_mib for d in devices] == [0, 0]


def test_an_unknown_free_figure_falls_back_to_the_total() -> None:
    devices, _ = build_device_list(node("head", card("Vulkan0", 12000, None)), [])
    assert devices[0].free_mib == 12000


def test_a_device_reporting_nothing_at_all_has_no_memory() -> None:
    devices, _ = build_device_list(node("head", card("Vulkan0", None, None)), [])
    assert devices[0].free_mib == 0


# -- the placement the CLI shares with a start ----------------------------------------


def test_the_configured_reserve_is_applied() -> None:
    local = node("head", card("Vulkan0", 6000, 6000))
    model = make_model(n_layers=24, layer_mib=500.0)

    plain, _ = place(HuddleConfig(), model, local, [], n_ctx=4096)
    reserved, _ = place(
        HuddleConfig.model_validate({"planner": {"reserve_mib": 2000}}),
        model,
        local,
        [],
        n_ctx=4096,
    )

    assert reserved.n_gpu_layers < plain.n_gpu_layers, "reserve_mib must shrink the budget"


def test_a_peer_given_no_layers_is_left_out_of_rpc() -> None:
    """The model fits on the head, so the peer is dropped rather than left in --rpc."""
    local = node("head", card("Vulkan0", 11000, 11000))
    peer = peer_report(card("Vulkan0", 11000, 11000))

    plan, endpoints = place(HuddleConfig(), make_model(n_layers=12), local, [peer], n_ctx=4096)

    assert endpoints == []
    assert len(plan.tensor_split) == 1, "a dropped peer takes its device slots with it"
    assert plan.layers_per_device == [plan.n_gpu_layers]


def test_a_peer_that_is_needed_stays() -> None:
    local = node("head", card("Vulkan0", 6000, 6000))
    peer = peer_report(card("Vulkan0", 6000, 6000))

    plan, endpoints = place(HuddleConfig(), make_model(n_layers=24), local, [peer], n_ctx=4096)

    assert endpoints == [peer.peer.rpc_endpoint]
    assert plan.layers_per_device[0] > 0, "the peer's device should hold layers"


# -- the command ----------------------------------------------------------------------


@pytest.fixture
def config_file(tmp_path: Path, fake_llama_server: Path, models_dir: Path) -> Path:
    path = tmp_path / "huddle.yaml"
    path.write_text(
        "node:\n  name: testnode\n"
        f"binaries:\n  llama_server: {fake_llama_server}\n"
        f"models:\n  dir: {models_dir}\n  default: tiny.gguf\n"
        "discovery:\n  enabled: false\n"
        "planner:\n  use_unified_memory: false\n"
    )
    return path


def test_the_command_prints_a_plan(config_file: Path) -> None:
    result = CliRunner().invoke(app, ["plan", "--config", str(config_file)])

    assert result.exit_code == 0, result.output
    assert "12 layers" in result.output
    assert "--tensor-split" in result.output


def test_the_command_plans_a_full_gpu_as_full(
    config_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(
        "HUDDLE_FAKE_DEVICES",
        "Available devices:\n  Vulkan0: NVIDIA GeForce RTX 5070 (12227 MiB, 0 MiB free)\n",
    )
    result = CliRunner().invoke(app, ["plan", "--config", str(config_file)])

    assert result.exit_code == 0, result.output
    line = next(line for line in result.output.splitlines() if "Vulkan0" in line)
    assert " 0 layers" in line, f"nothing is free, so nothing should be placed: {line!r}"


def test_the_command_fails_clearly_with_no_devices(
    config_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HUDDLE_FAKE_DEVICES", "Available devices:\n")
    result = CliRunner().invoke(app, ["plan", "--config", str(config_file)])

    assert result.exit_code == 1
    assert "no usable devices" in result.output
