"""Properties the placement planner must hold for *any* model and *any* devices.

The example tests in ``test_planner.py`` pin the cases we have met on real
hardware. These ask a different question: is there any input at all, however odd,
for which the plan breaks a rule that real hardware punishes? Hypothesis tries a
few hundred shapes a run (zero-memory cards, one layer, seven devices, peers
bigger than the head, an iGPU claiming 60 GiB) and shrinks any failure to the
smallest case.

The failures these rules guard against are silent on a real cluster. A weight
that shifts onto the wrong device runs fine with a peer idle; a layer budgeted to
the wrong card runs fine until it runs out of memory at load.

No GPU or llama.cpp is involved: the planner is arithmetic, and llama.cpp's layer
assignment is modelled by the fake in ``tests/fakes`` (same operations, in the
same order and precision, as ``llama_model::load_tensors`` at the pinned build).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from hypothesis import assume, given
from hypothesis import strategies as st

from huddle.coordinator.planner import (
    MIB,
    PlacementDevice,
    Plan,
    order_devices,
    plan_placement,
)
from huddle.gguf import ModelInfo
from tests.fakes.fake_llama_server import device_for_entry
from tests.test_planner import make_model

# -- generators -----------------------------------------------------------------------


@st.composite
def models(draw: st.DrawFn) -> ModelInfo:
    return make_model(
        n_layers=draw(st.integers(1, 130)),
        layer_mib=draw(st.integers(1, 3000)),
        overhead_mib=draw(st.integers(0, 2500)),
    )


@st.composite
def device_lists(draw: st.DrawFn, *, allow_empty: bool = True) -> list[PlacementDevice]:
    """Devices in llama.cpp's enumeration order, with ids that say where they sit."""
    specs = draw(
        st.lists(
            st.tuples(
                st.booleans(),  # a peer's device?
                st.integers(0, 70_000),  # free MiB: 0 is a real thing a driver reports
                st.booleans(),  # unified memory?
            ),
            min_size=0 if allow_empty else 1,
            max_size=7,
        )
    )
    peers = [s for s in specs if s[0]]
    locals_ = [s for s in specs if not s[0]]
    devices = [
        PlacementDevice(
            id=f"RPC{i}",
            name="peer gpu",
            node="peer",
            free_mib=free,
            unified_memory=uma,
            is_rpc=True,
        )
        for i, (_, free, uma) in enumerate(peers)
    ] + [
        PlacementDevice(
            id=f"Vulkan{i}", name="local gpu", node="head", free_mib=free, unified_memory=uma
        )
        for i, (_, free, uma) in enumerate(locals_)
    ]
    assert devices == order_devices(devices)
    return devices


@dataclass(frozen=True)
class Scenario:
    model: ModelInfo
    devices: list[PlacementDevice]
    n_ctx: int
    headroom: float
    reserve_mib: float
    extra_reserve_mib: dict[str, float]
    use_unified_memory: bool

    def plan(self, *, devices: list[PlacementDevice] | None = None, **overrides: Any) -> Plan:
        """Plan this scenario, optionally with other devices or settings swapped in."""
        arguments: dict[str, Any] = {
            "n_ctx": self.n_ctx,
            "headroom": self.headroom,
            "reserve_mib": self.reserve_mib,
            "extra_reserve_mib": self.extra_reserve_mib,
            "use_unified_memory": self.use_unified_memory,
        }
        arguments.update(overrides)
        return plan_placement(self.model, self.devices if devices is None else devices, **arguments)


@st.composite
def scenarios(draw: st.DrawFn, *, allow_empty: bool = True) -> Scenario:
    devices = draw(device_lists(allow_empty=allow_empty))
    extra = {
        d.id: float(draw(st.integers(0, 4000)))
        for d in devices
        if draw(st.booleans())  # most devices get none
    }
    return Scenario(
        model=draw(models()),
        devices=devices,
        n_ctx=draw(st.sampled_from([1, 512, 2048, 8192, 32768])),
        headroom=draw(st.sampled_from([0.0, 0.05, 0.15, 0.3, 0.5])),
        reserve_mib=float(draw(st.sampled_from([0, 256, 1024, 2048]))),
        extra_reserve_mib=extra,
        use_unified_memory=draw(st.booleans()),
    )


def budget_mib(scenario: Scenario, device: PlacementDevice) -> float:
    return (
        device.free_mib * (1.0 - scenario.headroom)
        - scenario.reserve_mib
        - scenario.extra_reserve_mib.get(device.id, 0.0)
    )


# -- the flags are self-consistent ----------------------------------------------------


@given(scenarios())
def test_the_plan_agrees_with_itself(s: Scenario) -> None:
    plan = s.plan()

    assert len(plan.layers_per_device) == len(s.devices)
    assert all(n >= 0 for n in plan.layers_per_device)
    assert plan.total_layers == plan.n_gpu_layers <= s.model.n_layers

    if plan.n_gpu_layers == 0:
        # Nothing on GPU: no split, no -ngl, so llama.cpp runs on CPU.
        assert plan.tensor_split == []
        assert plan.ngl == 0
        assert plan.output_device is None
        return

    # One weight per enumerated device, in order; skipped devices keep a 0.
    assert len(plan.tensor_split) == len(s.devices)
    assert all(w >= 0 and w == int(w) for w in plan.tensor_split), "entry counts, not shares"
    assert sum(plan.tensor_split) == plan.ngl, "weights are entry counts that add up to -ngl"
    assert plan.ngl == plan.n_gpu_layers + (1 if plan.output_device else 0)
    assert plan.ngl <= s.model.n_layers + 1


@given(scenarios())
def test_planning_is_deterministic_and_leaves_its_input_alone(s: Scenario) -> None:
    before = list(s.devices)
    first, second = s.plan(), s.plan()
    assert first == second
    assert s.devices == before


# -- nothing is asked of a device that cannot give it ---------------------------------


@given(scenarios())
def test_no_device_is_planned_past_its_budget(s: Scenario) -> None:
    """The out-of-memory rule: layers (and the head, where it lands) fit the budget."""
    plan = s.plan()
    head = s.model.overhead_bytes / MIB
    for device, layers in zip(s.devices, plan.layers_per_device, strict=True):
        if layers == 0:
            continue
        cost = layers * plan.per_layer_mib + (head if device.id == plan.output_device else 0.0)
        assert cost <= budget_mib(s, device) + 1e-6, (device, layers, cost)


@given(scenarios())
def test_a_device_with_no_free_memory_gets_nothing(s: Scenario) -> None:
    drained = [
        PlacementDevice(d.id, d.name, d.node, 0, d.unified_memory, d.is_rpc) for d in s.devices
    ]
    plan = s.plan(devices=drained)
    assert plan.n_gpu_layers == 0
    assert plan.tensor_split == []


@given(scenarios())
def test_unified_memory_never_gets_layers_unless_asked(s: Scenario) -> None:
    """The iGPU trap: its advertised 47 GiB is system RAM."""
    plan = s.plan(use_unified_memory=False)
    for device, layers in zip(s.devices, plan.layers_per_device, strict=True):
        if device.unified_memory:
            assert layers == 0
            assert device.id in dict(plan.excluded)


# -- more resources never make things worse -------------------------------------------


@given(scenarios(allow_empty=False), st.data(), st.integers(1, 40_000))
def test_more_free_memory_never_offloads_fewer_layers(
    s: Scenario, data: st.DataObject, extra_mib: int
) -> None:
    index = data.draw(st.integers(0, len(s.devices) - 1))
    roomier = [
        PlacementDevice(d.id, d.name, d.node, d.free_mib + extra_mib, d.unified_memory, d.is_rpc)
        if i == index
        else d
        for i, d in enumerate(s.devices)
    ]
    assert s.plan(devices=roomier).n_gpu_layers >= s.plan().n_gpu_layers


@given(scenarios(), st.integers(1, 3000))
def test_a_bigger_reserve_never_offloads_more_layers(s: Scenario, more: int) -> None:
    assert s.plan(reserve_mib=s.reserve_mib + more).n_gpu_layers <= s.plan().n_gpu_layers


@given(scenarios(), st.sampled_from([512, 2048, 8192, 32768]))
def test_a_longer_context_never_offloads_more_layers(s: Scenario, longer: int) -> None:
    assume(longer >= s.n_ctx)
    assert s.plan(n_ctx=longer).n_gpu_layers <= s.plan().n_gpu_layers


# -- peers are a last resort ----------------------------------------------------------


@given(scenarios())
def test_peers_only_add_capacity_and_leave_local_placement_alone(s: Scenario) -> None:
    """Every node boundary costs a round trip per token, so a peer never displaces
    the head's own GPUs, and adding one never offloads less."""
    local_only = [d for d in s.devices if not d.is_rpc]
    alone, together = s.plan(devices=local_only), s.plan()

    assert together.n_gpu_layers >= alone.n_gpu_layers
    on_peers = sum(
        n for d, n in zip(s.devices, together.layers_per_device, strict=True) if d.is_rpc
    )
    if alone.n_gpu_layers == s.model.n_layers:
        assert on_peers == 0, "the model fit locally; nothing should cross the network"
    local_layers = [
        n for d, n in zip(s.devices, together.layers_per_device, strict=True) if not d.is_rpc
    ]
    assert local_layers == alone.layers_per_device, "peers must not take layers from local GPUs"


# -- the output head ------------------------------------------------------------------


@given(scenarios())
def test_the_output_head_is_charged_to_the_last_enumerated_device_that_has_layers(
    s: Scenario,
) -> None:
    """llama.cpp puts the head where the last offloaded entry lands, so that device
    must be the one the planner charged for it."""
    plan = s.plan()
    if plan.output_device is None:
        return
    holders = [d.id for d, n in zip(s.devices, plan.layers_per_device, strict=True) if n > 0]
    assert plan.output_device == holders[-1]


# -- the flags produce the placement that was planned ---------------------------------


@given(scenarios())
def test_llama_cpps_assignment_reproduces_the_plan_exactly(s: Scenario) -> None:
    """Hand the emitted flags to llama.cpp's own layer-assignment rule.

    Every device must end up with exactly the layers it was budgeted (plus the
    head, where charged), and the CPU with exactly the layers that did not fit.
    This is the property the entry-count weights exist for: with shares that
    round, one layer lands on a neighbour that was never budgeted for it.
    """
    plan = s.plan()
    assume(plan.ngl > 0)

    held = [0] * len(s.devices)
    for entry in range(plan.ngl):
        held[device_for_entry(plan.tensor_split, entry, plan.ngl)] += 1

    expected = [
        n + (1 if d.id == plan.output_device else 0)
        for d, n in zip(s.devices, plan.layers_per_device, strict=True)
    ]
    assert held == expected

    # llama.cpp counts n_layers + 1 entries (layers plus head); the rest are on CPU.
    assert s.model.n_layers + 1 - plan.ngl == s.model.n_layers - plan.n_gpu_layers

    # The head is the last entry, so it is on the device the planner charged.
    assert s.devices[device_for_entry(plan.tensor_split, plan.ngl - 1, plan.ngl)].id == (
        plan.output_device
    )


# -- device ordering ------------------------------------------------------------------


@given(
    st.lists(
        st.builds(
            PlacementDevice,
            id=st.uuids().map(str),
            name=st.just("gpu"),
            node=st.just("n"),
            free_mib=st.integers(0, 1000),
            is_rpc=st.booleans(),
        ),
        max_size=10,
        unique_by=lambda d: d.id,
    )
)
def test_order_devices_puts_peers_first_and_changes_nothing_else(
    devices: list[PlacementDevice],
) -> None:
    ordered = order_devices(devices)

    assert sorted(d.id for d in ordered) == sorted(d.id for d in devices)
    flags = [d.is_rpc for d in ordered]
    assert flags == sorted(flags, reverse=True), "all RPC devices before all local ones"
    for group in (True, False):
        assert [d.id for d in ordered if d.is_rpc is group] == [
            d.id for d in devices if d.is_rpc is group
        ], "order inside a group is the order the endpoints were given"
