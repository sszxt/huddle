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

And one property of ``-ngl``: llama.cpp offloads *entries* ``0..n_layer``, where
the last entry is the output head, taking them from the end. Asking for exactly
``n_layer`` leaves a real transformer layer on CPU — measured: ``-ngl 24`` on a
24-layer model put entry 0 on CPU, ``-ngl 60`` on a 64-layer model put five
there. So the flag is one more than the layers planned on GPU, and the split
weights are entry counts, which llama.cpp then assigns exactly.
"""

from __future__ import annotations

from collections.abc import Mapping
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

    # Transformer layers placed on GPUs.
    n_gpu_layers: int
    # Weights for --tensor-split: *entries* per device, including the output
    # head on `output_device`. Integer counts summing to `ngl`, which llama.cpp
    # assigns exactly rather than by rounding proportions.
    tensor_split: list[float]
    layers_per_device: list[int]
    devices: list[PlacementDevice]
    excluded: list[tuple[str, str]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    per_layer_mib: float = 0.0
    # The value for -ngl: layers on GPU plus the output head.
    ngl: int = 0
    # Which device holds the output head; it is charged for it.
    output_device: str | None = None

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
            elif device.id == self.output_device:
                note = f"{device.name}, + output head"
            lines.append(f"{device.id:8} {device.node:10} {layers:3} layers  ({note})")
        return lines


def order_devices(devices: list[PlacementDevice]) -> list[PlacementDevice]:
    """Put devices in llama.cpp's enumeration order: RPC first, then local.

    Relative order within each group is preserved, because that is the order the
    endpoints were passed to ``--rpc``.
    """
    return [d for d in devices if d.is_rpc] + [d for d in devices if not d.is_rpc]


def _fill_order(devices: list[PlacementDevice]) -> list[int]:
    """Indices in the order we would rather use devices: reverse enumeration.

    Local devices are enumerated last, so filling from the end is local-first —
    every node boundary costs a network round trip per token. It also makes the
    first device to receive layers the *last* enumerated one to hold any, which
    is where llama.cpp puts the output head, so the device charged for the head
    is the device that really gets it.
    """
    return list(range(len(devices) - 1, -1, -1))


def plan_placement(
    model: ModelInfo,
    devices: list[PlacementDevice],
    *,
    n_ctx: int,
    headroom: float = DEFAULT_HEADROOM,
    use_unified_memory: bool = False,
    reserve_overhead_on_first: bool = True,
    reserve_mib: float = 0.0,
    extra_reserve_mib: Mapping[str, float] | None = None,
) -> Plan:
    """Work out ``-ngl`` and ``--tensor-split`` for this model on these devices.

    ``devices`` must already be in llama.cpp enumeration order; use
    ``order_devices`` if unsure. Devices that are skipped keep their slot with a
    weight of zero.

    ``reserve_mib`` is a fixed margin taken from every device, on top of the
    proportional ``headroom``. Compute buffers do not scale with a card's free
    memory, so a percentage alone under-provisions small cards.
    ``extra_reserve_mib`` adds more to specific devices by id — the coordinator
    raises it for a device that ran out of memory on a real load, so the next
    plan backs off exactly where the estimate was wrong.

    ``reserve_overhead_on_first`` offloads the output head with the layers and
    charges its memory to the device that receives it. Without it the head
    stays on CPU and ``-ngl`` equals the layer count.
    """
    if model.n_layers <= 0:
        raise ValueError("model reports no layers; cannot plan a split")

    per_layer_bytes = model.mean_layer_bytes + model.kv_cache_bytes(n_ctx) / model.n_layers
    per_layer_mib = per_layer_bytes / MIB
    if per_layer_mib <= 0:
        raise ValueError("model reports zero-sized layers; cannot plan a split")

    excluded: list[tuple[str, str]] = []
    warnings: list[str] = []
    capacities = [0] * len(devices)
    fill_order = _fill_order(devices)
    head_mib = model.overhead_bytes / MIB if reserve_overhead_on_first else 0.0
    output_index: int | None = None

    for index in fill_order:
        device = devices[index]
        if device.unified_memory and not use_unified_memory:
            # Its advertised memory is system RAM, and its bandwidth is shared
            # with the CPU. Trusting the number is how layers end up somewhere
            # slower than the CPU path they displaced.
            excluded.append((device.id, "unified memory (shares system RAM)"))
            continue

        budget_mib = device.free_mib * (1.0 - headroom) - reserve_mib
        budget_mib -= (extra_reserve_mib or {}).get(device.id, 0.0)

        if output_index is None:
            # Nothing placed yet, so this device would be the last enumerated to
            # hold layers — the one llama.cpp gives the output head. It only
            # qualifies if it can hold the head *and* a layer; otherwise it must
            # stay empty, or it would take the head without being charged for it.
            fits = int((budget_mib - head_mib) // per_layer_mib)
            if fits >= 1:
                output_index = index
                capacities[index] = fits
            else:
                excluded.append(
                    (device.id, "too little free memory for a layer and the output head")
                )
            continue

        fits = int(budget_mib // per_layer_mib)
        if fits <= 0:
            excluded.append((device.id, "too little free memory for one layer"))
            continue
        capacities[index] = fits

    # Report exclusions in enumeration order, the order the split is read in.
    position = {device.id: i for i, device in enumerate(devices)}
    excluded.sort(key=lambda item: position[item[0]])

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

    layers_per_device = [0] * len(devices)
    remaining = n_gpu_layers
    for index in fill_order:
        take = min(capacities[index], remaining)
        layers_per_device[index] = take
        remaining -= take
        if remaining == 0:
            break

    head_idle = all(d.is_rpc or n == 0 for d, n in zip(devices, layers_per_device, strict=True))
    if head_idle and any(not d.is_rpc for d in devices):
        warnings.append("all layers placed on peers; the head node's GPUs are idle")

    # Entry counts, not proportions: integer weights summing to -ngl make
    # llama.cpp's assignment exact, where rounded proportions can push a layer
    # onto a neighbour — onto a peer that was never budgeted for it.
    tensor_split = [float(count) for count in layers_per_device]
    ngl = n_gpu_layers
    output_device = None
    if reserve_overhead_on_first and output_index is not None:
        tensor_split[output_index] += 1
        ngl += 1
        output_device = devices[output_index].id

    return Plan(
        n_gpu_layers=n_gpu_layers,
        tensor_split=tensor_split,
        layers_per_device=layers_per_device,
        devices=devices,
        excluded=excluded,
        warnings=warnings,
        per_layer_mib=per_layer_mib,
        ngl=ngl,
        output_device=output_device,
    )
