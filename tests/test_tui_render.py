"""Pure rendering functions: no I/O, so these just call and inspect output."""

from __future__ import annotations

from rich.console import Console

from huddle.agent.service import BackendStatus
from huddle.config import HuddleConfig
from huddle.coordinator.service import ClusterStatus
from huddle.hardware import DeviceInfo, NodeHardware
from huddle.tui.poller import ClusterSnapshot, NodeSnapshot
from huddle.tui.render import (
    banner,
    cluster_state,
    frame,
    meter,
    node_panel,
    pipeline_line,
)


def render_to_text(renderable: object) -> str:
    console = Console(width=100, file=None, record=True, force_terminal=False)
    console.print(renderable)
    return console.export_text()


def _config() -> HuddleConfig:
    return HuddleConfig.model_validate(
        {
            "node": {"name": "omarchy"},
            "binaries": {"llama_server": "/opt/llama-server"},
            "models": {"dir": "/models"},
            "api": {"host": "0.0.0.0", "port": 8000},
        }
    )


def _reachable_node(**overrides: object) -> NodeSnapshot:
    defaults: dict[str, object] = dict(
        name="omarchy",
        role="head",
        reachable=True,
        hardware=NodeHardware(
            name="omarchy",
            devices=[
                DeviceInfo(
                    id="Vulkan0",
                    name="NVIDIA GeForce RTX 5070",
                    total_mib=12473,
                    free_mib=1822,
                    util_pct=42,
                    temp_c=55,
                    power_w=123.4,
                ),
                DeviceInfo(id="Vulkan1", name="Intel(R) Graphics (RPL-S)", unified_memory=True),
            ],
            cpu_count=20,
            ram_total_mib=64061,
            ram_available_mib=57000,
        ),
        layers=25,
    )
    defaults.update(overrides)
    return NodeSnapshot(**defaults)  # type: ignore[arg-type]


def test_meter_shows_na_without_data() -> None:
    assert "n/a" in render_to_text(meter(None))


def test_meter_shows_a_percentage() -> None:
    assert "42%" in render_to_text(meter(0.42))


def test_node_panel_shows_reachable_device_and_meters() -> None:
    text = render_to_text(node_panel(_reachable_node()))
    assert "omarchy" in text
    assert "head" in text
    assert "RTX 5070" in text
    assert "55C" in text
    assert "123W" in text
    assert "layers: 25" in text


def test_node_panel_shows_unreachable_with_error() -> None:
    node = NodeSnapshot(name="maksood", role="worker", reachable=False, error="connection refused")
    text = render_to_text(node_panel(node))
    assert "unreachable" in text
    assert "connection refused" in text


def test_node_panel_omits_gpu_meter_without_util_data() -> None:
    """A non-NVIDIA device must not show a fabricated gpu meter."""
    node = _reachable_node()
    text = render_to_text(node_panel(node))
    # Only one "gpu" row should appear (Vulkan0's), none for Vulkan1.
    assert text.count("gpu") == 1


def test_cluster_state_reflects_status_flags() -> None:
    running = ClusterSnapshot(fetched_at=0.0, cluster=ClusterStatus(running=True, head_node="h"))
    starting = ClusterSnapshot(
        fetched_at=0.0,
        cluster=ClusterStatus(running=False, head_node="h", desired=True, starting=True),
    )
    degraded = ClusterSnapshot(
        fetched_at=0.0, cluster=ClusterStatus(running=False, head_node="h", desired=True)
    )
    stopped = ClusterSnapshot(fetched_at=0.0, cluster=None)

    assert cluster_state(running) == "running"
    assert cluster_state(starting) == "starting"
    assert cluster_state(degraded) == "degraded"
    assert cluster_state(stopped) == "stopped"


def test_pipeline_line_shows_coordinator_error() -> None:
    snapshot = ClusterSnapshot(fetched_at=0.0, cluster=None, coordinator_error="not a coordinator")
    assert "not a coordinator" in render_to_text(pipeline_line(snapshot))


def test_pipeline_line_chains_nodes_in_order() -> None:
    snapshot = ClusterSnapshot(
        fetched_at=0.0,
        cluster=ClusterStatus(running=True, head_node="omarchy", workers=["maksood"]),
        nodes=[_reachable_node(), _reachable_node(name="maksood", role="worker", layers=34)],
    )
    text = render_to_text(pipeline_line(snapshot))
    assert text.index("omarchy") < text.index("maksood")
    assert "rpc" in text


def test_banner_shows_node_count_and_endpoint() -> None:
    config = _config()
    snapshot = ClusterSnapshot(
        fetched_at=0.0,
        cluster=ClusterStatus(running=True, head_node="omarchy"),
        nodes=[_reachable_node()],
    )
    text = render_to_text(banner(snapshot, config))
    assert "HUDDLE" in text.replace(" ", "").upper()
    assert "1 node" in text
    assert "/v1/chat/completions" in text


def test_frame_includes_tokens_per_sec_from_head_backend() -> None:
    config = _config()
    head = _reachable_node(
        backend=BackendStatus(running=True, base_url="http://x", tokens_per_sec=27.7)
    )
    snapshot = ClusterSnapshot(
        fetched_at=0.0,
        cluster=ClusterStatus(running=True, head_node="omarchy", model="tiny.gguf"),
        nodes=[head],
        available_models=["tiny.gguf"],
        loaded_model="tiny.gguf",
    )
    text = render_to_text(frame(snapshot, config))
    assert "27.7 tok/s" in text
    assert "tiny.gguf" in text


def test_frame_before_any_poll_shows_connecting() -> None:
    text = render_to_text(frame(None, _config()))
    assert "connecting" in text
