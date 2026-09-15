"""Decide which layers go where.

This is the part that has to beat llama.cpp's own automatic fitting, and there
is one specific case where it must: a node's integrated GPU advertises system
RAM as if it were VRAM, so auto-fit piles layers onto a slow iGPU claiming
47 GiB in preference to the 12 GiB discrete card beside it. Measured on this
cluster, not hypothetical.

Two properties of ``--tensor-split`` drive the whole design:

1. It is **positional over every device llama.cpp enumerates**. A device we want
   to skip still needs a ``0`` in its slot, or every later weight shifts onto the
   wrong device.
2. **RPC devices are enumerated first**, then local ones. Getting this backwards
   fails silently: the cluster runs, correctly, with the peers idle.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from huddle.gguf import ModelInfo

MIB = 1024 * 1024

# Left for compute buffers, fragmentation and whatever else the backend wants.
# Overshooting here is not a slow cluster, it is an out-of-memory failure at
# load time, so the default is deliberately generous.
DEFAULT_HEADROOM = 0.15


@dataclass(frozen=True)
class PlacementDevice:
    """One device llama.cpp will enumerate, in enumeration order."""

    id: str
    name: str
    node: str
    free_mib: int
    unified_memory: bool = False
    is_rpc: bool = False


@dataclass
class Plan:
    """Where each layer goes, and the flags that express it."""

    n_gpu_layers: int
    tensor_split: list[float]
    layers_per_device: list[int]
    devices: list[PlacementDevice]
    excluded: list[tuple[str, str]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    per_layer_mib: float = 0.0

    @property
    def total_layers(self) -> int:
        return sum(self.layers_per_device)

    def summary(self) -> list[str]:
        """Human-readable placement, for logs and the CLI.

        Lists every enumerated device, including ones given no layers: a device
        silently missing from this output is indistinguishable from a device we
        forgot to consider.
        """
        reasons = dict(self.excluded)
        lines = []
        for device, layers in zip(self.devices, self.layers_per_device, strict=True):
            note = device.name
            if device.id in reasons:
                note = f"skipped: {reasons[device.id]}"
            elif layers == 0:
                note = f"unused: {device.name}"
            lines.append(f"{device.id:8} {device.node:10} {layers:3} layers  ({note})")
        return lines


def order_devices(devices: list[PlacementDevice]) -> list[PlacementDevice]:
    """Put devices in llama.cpp's enumeration order: RPC first, then local.

    Relative order within each group is preserved, because that is the order the
    endpoints were passed to ``--rpc``.
    """
    return [d for d in devices if d.is_rpc] + [d for d in devices if not d.is_rpc]


def _fill_order(devices: list[PlacementDevice]) -> list[int]:
    """Indices in the order we would rather use devices.

    Local before remote. Within each group, enumeration order is preserved so
    the resulting layer ranges stay contiguous.
    """
    local = [i for i, d in enumerate(devices) if not d.is_rpc]
    remote = [i for i, d in enumerate(devices) if d.is_rpc]
    return local + remote


def plan_placement(
    model: ModelInfo,
    devices: list[PlacementDevice],
    *,
    n_ctx: int,
    headroom: float = DEFAULT_HEADROOM,
    use_unified_memory: bool = False,
    reserve_overhead_on_first: bool = True,
) -> Plan:
    """Work out ``-ngl`` and ``--tensor-split`` for this model on these devices.

    ``devices`` must already be in llama.cpp enumeration order; use
    ``order_devices`` if unsure. Devices that are skipped keep their slot with a
    weight of zero.
    """
    if model.n_layers <= 0:
        raise ValueError("model reports no layers; cannot plan a split")

    per_layer_bytes = model.mean_layer_bytes + model.kv_cache_bytes(n_ctx) / model.n_layers
    per_layer_mib = per_layer_bytes / MIB
    if per_layer_mib <= 0:
        raise ValueError("model reports zero-sized layers; cannot plan a split")

    excluded: list[tuple[str, str]] = []
    warnings: list[str] = []
    capacities: list[int] = []

    for index, device in enumerate(devices):
        if device.unified_memory and not use_unified_memory:
            # Its advertised memory is system RAM, and its bandwidth is shared
            # with the CPU. Trusting the number is how layers end up somewhere
            # slower than the CPU path they displaced.
            excluded.append((device.id, "unified memory (shares system RAM)"))
            capacities.append(0)
            continue

        budget_mib = device.free_mib * (1.0 - headroom)
        if reserve_overhead_on_first and index == 0:
            # Embeddings and the output head are not part of any layer and have
            # to live somewhere; they land with the first device.
            budget_mib -= model.overhead_bytes / MIB

        fits = int(budget_mib // per_layer_mib)
        if fits <= 0:
            excluded.append((device.id, "too little free memory for one layer"))
            capacities.append(0)
            continue
        capacities.append(fits)

    total_capacity = sum(capacities)
    if total_capacity == 0:
        warnings.append("no device can hold a layer; running entirely on CPU")
        return Plan(
            n_gpu_layers=0,
            tensor_split=[],
            layers_per_device=[0] * len(devices),
            devices=devices,
            excluded=excluded,
            warnings=warnings,
            per_layer_mib=per_layer_mib,
        )

    n_gpu_layers = min(model.n_layers, total_capacity)
    if n_gpu_layers < model.n_layers:
        warnings.append(
            f"only {n_gpu_layers}/{model.n_layers} layers fit on GPUs; "
            f"the remaining {model.n_layers - n_gpu_layers} run on CPU"
        )

    # Enumeration order is llama.cpp's and fixed; *fill* order is ours. Prefer
    # local devices, because every node boundary costs a network round trip per
    # token — a model that fits on the head node should never be shipped to a
    # peer just because that peer is enumerated first.
    layers_per_device = [0] * len(devices)
    remaining = n_gpu_layers
    for index in _fill_order(devices):
        take = min(capacities[index], remaining)
        layers_per_device[index] = take
        remaining -= take
        if remaining == 0:
            break

    head_idle = all(d.is_rpc or n == 0 for d, n in zip(devices, layers_per_device, strict=True))
    if head_idle and any(not d.is_rpc for d in devices):
        warnings.append("all layers placed on peers; the head node's GPUs are idle")

    # Layer counts double as proportions: llama.cpp normalises them, and the raw
    # counts make the flag self-documenting in logs.
    tensor_split = [float(count) for count in layers_per_device]

    return Plan(
        n_gpu_layers=n_gpu_layers,
        tensor_split=tensor_split,
        layers_per_device=layers_per_device,
        devices=devices,
        excluded=excluded,
        warnings=warnings,
        per_layer_mib=per_layer_mib,
    )
