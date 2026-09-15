"""Placement planning.

Two failures here are silent rather than loud, so both get explicit regression
tests: mis-ordered devices (the cluster runs with peers idle) and trusting an
integrated GPU's advertised memory (layers land somewhere slower than the CPU
path they displaced).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from huddle.coordinator.planner import (
    MIB,
    PlacementDevice,
    order_devices,
    plan_placement,
)
from huddle.gguf import ModelInfo


def make_model(
    *, n_layers: int = 24, layer_mib: float = 500.0, overhead_mib: float = 200.0
) -> ModelInfo:
    layer_bytes = int(layer_mib * MIB)
    return ModelInfo(
        path=Path("/models/test.gguf"),
        architecture="llama",
        name="test",
        n_layers=n_layers,
        n_ctx_train=8192,
        n_embd=4096,
        n_head=32,
        n_head_kv=8,
        file_size=0,
        tensor_bytes=layer_bytes * n_layers + int(overhead_mib * MIB),
        layer_bytes=[layer_bytes] * n_layers,
        overhead_bytes=int(overhead_mib * MIB),
    )


def gpu(id_: str, node: str, free_mib: int, **kw: object) -> PlacementDevice:
    return PlacementDevice(id=id_, name=f"card-{id_}", node=node, free_mib=free_mib, **kw)  # type: ignore[arg-type]


def test_rpc_devices_are_ordered_first() -> None:
    """llama.cpp enumerates RPC devices before local ones."""
    ordered = order_devices(
        [
            gpu("Vulkan0", "head", 12000),
            gpu("RPC0", "peer", 12000, is_rpc=True),
            gpu("RPC1", "peer", 47000, is_rpc=True, unified_memory=True),
        ]
    )
    assert [d.id for d in ordered] == ["RPC0", "RPC1", "Vulkan0"]


def test_unified_memory_device_is_skipped_but_keeps_its_slot() -> None:
    """The iGPU trap: 47 GiB advertised is system RAM, not VRAM.

    It must be skipped *and* still occupy its position, because --tensor-split
    is positional — dropping it would shift every later weight onto the wrong
    device.
    """
    devices = [
        gpu("RPC0", "peer", 11000, is_rpc=True),
        gpu("RPC1", "peer", 47000, is_rpc=True, unified_memory=True),
        gpu("Vulkan0", "head", 11000),
    ]
    plan = plan_placement(make_model(), devices, n_ctx=4096)

    assert len(plan.tensor_split) == 3, "every enumerated device needs a slot"
    assert plan.tensor_split[1] == 0.0, "the iGPU must get no layers"
    assert plan.layers_per_device[0] > 0 and plan.layers_per_device[2] > 0
    assert any("unified memory" in reason for _, reason in plan.excluded)


def test_unified_memory_can_be_opted_into() -> None:
    devices = [gpu("Vulkan1", "head", 47000, unified_memory=True)]
    assert plan_placement(make_model(), devices, n_ctx=4096).n_gpu_layers == 0

    opted_in = plan_placement(make_model(), devices, n_ctx=4096, use_unified_memory=True)
    assert opted_in.n_gpu_layers > 0


def test_split_positions_match_device_positions() -> None:
    devices = [gpu(f"D{i}", "n", 11000, is_rpc=i < 2) for i in range(4)]
    plan = plan_placement(make_model(), devices, n_ctx=4096)
    assert len(plan.tensor_split) == len(devices) == len(plan.layers_per_device)


def test_partial_offload_is_reported() -> None:
    """A model too big for the cluster should warn, not silently truncate."""
    devices = [gpu("Vulkan0", "head", 4000)]
    plan = plan_placement(make_model(n_layers=24, layer_mib=500), devices, n_ctx=4096)
    assert plan.n_gpu_layers < 24
    assert any("run on CPU" in w for w in plan.warnings)


def test_no_capacity_falls_back_to_cpu() -> None:
    devices = [gpu("Vulkan0", "head", 100)]
    plan = plan_placement(make_model(), devices, n_ctx=4096)
    assert plan.n_gpu_layers == 0
    assert plan.layers_per_device == [0]
    assert any("entirely on CPU" in w for w in plan.warnings)


def test_headroom_is_respected() -> None:
    """Filling a device to its advertised free memory means OOM at load."""
    devices = [gpu("Vulkan0", "head", 10_000)]
    model = make_model(n_layers=100, layer_mib=1000, overhead_mib=0)
    generous = plan_placement(model, devices, n_ctx=1, headroom=0.5)
    tight = plan_placement(model, devices, n_ctx=1, headroom=0.0)
    assert generous.n_gpu_layers < tight.n_gpu_layers


def test_kv_cache_reduces_capacity_at_longer_context() -> None:
    devices = [gpu("Vulkan0", "head", 12_000)]
    model = make_model(n_layers=40, layer_mib=200)
    short = plan_placement(model, devices, n_ctx=2048)
    long = plan_placement(model, devices, n_ctx=65536)
    assert long.n_gpu_layers < short.n_gpu_layers, "KV cache must count against capacity"


def test_overhead_charged_to_first_device() -> None:
    devices = [gpu("Vulkan0", "head", 6000), gpu("Vulkan1", "head", 6000)]
    model = make_model(n_layers=40, layer_mib=200, overhead_mib=3000)
    plan = plan_placement(model, devices, n_ctx=1)
    assert plan.layers_per_device[0] < plan.layers_per_device[1]


def test_rejects_model_with_no_layers() -> None:
    with pytest.raises(ValueError, match="no layers"):
        plan_placement(make_model(n_layers=0), [gpu("V0", "n", 8000)], n_ctx=4096)


def test_real_cluster_shape() -> None:
    """The actual cluster: two RTX 5070s and two iGPUs across two nodes."""
    devices = order_devices(
        [
            gpu("Vulkan0", "omarchy", 11060),
            gpu("Vulkan1", "omarchy", 48045, unified_memory=True),
            gpu("RPC0", "maksood", 10712, is_rpc=True),
            gpu("RPC1", "maksood", 42377, is_rpc=True, unified_memory=True),
        ]
    )
    assert [d.id for d in devices] == ["RPC0", "RPC1", "Vulkan0", "Vulkan1"]

    plan = plan_placement(make_model(n_layers=80, layer_mib=250), devices, n_ctx=4096)

    # Both iGPUs skipped; both discrete cards used.
    assert plan.layers_per_device[1] == 0 and plan.layers_per_device[3] == 0
    assert plan.layers_per_device[0] > 0 and plan.layers_per_device[2] > 0
    assert len(plan.excluded) == 2
    assert plan.total_layers == plan.n_gpu_layers


def test_prefers_local_devices_over_peers() -> None:
    """A model that fits on the head node must not be shipped to a peer.

    Enumeration puts RPC devices first, but every node boundary costs a network
    round trip per token, so local capacity is used first.
    """
    devices = [
        gpu("RPC0", "peer", 11000, is_rpc=True),
        gpu("Vulkan0", "head", 11000),
    ]
    plan = plan_placement(make_model(n_layers=10, layer_mib=200), devices, n_ctx=4096)

    assert plan.layers_per_device[1] == 10, "local device should take the whole model"
    assert plan.layers_per_device[0] == 0, "peer should be untouched when not needed"
    assert plan.tensor_split == [0.0, 10.0]


def test_spills_to_peers_only_when_local_is_full() -> None:
    devices = [
        gpu("RPC0", "peer", 11000, is_rpc=True),
        gpu("Vulkan0", "head", 2500),
    ]
    plan = plan_placement(make_model(n_layers=40, layer_mib=200), devices, n_ctx=4096)
    assert plan.layers_per_device[1] > 0, "local filled first"
    assert plan.layers_per_device[0] > 0, "overflow went to the peer"


def test_summary_lists_every_device_including_unused() -> None:
    """A device missing from the summary looks like one we forgot to consider."""
    devices = [
        gpu("RPC0", "peer", 11000, is_rpc=True),
        gpu("RPC1", "peer", 47000, is_rpc=True, unified_memory=True),
        gpu("Vulkan0", "head", 11000),
    ]
    plan = plan_placement(make_model(n_layers=8, layer_mib=200), devices, n_ctx=4096)
    summary = "\n".join(plan.summary())
    for device in devices:
        assert device.id in summary
    assert "unused" in summary and "skipped" in summary
