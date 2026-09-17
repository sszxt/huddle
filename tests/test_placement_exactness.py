"""The flags the planner emits produce exactly the placement it planned.

Checked against the fake, whose assignment rule reproduces llama.cpp's on the
placements we measured on the real cluster. The last test shows the failure the
entry-count weights exist to prevent.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from huddle.llamacpp import parse_layer_assignments
from tests.fakes.gguf_builder import write_gguf

FAKE = Path(__file__).parent / "fakes" / "fake_llama_server.py"
TWO_LOCAL = (
    "Available devices:\n"
    "  Vulkan0: NVIDIA GeForce RTX 5070 (12473 MiB, 11060 MiB free)\n"
    "  Vulkan1: Intel(R) Graphics (RPL-S) (48045 MiB, 43241 MiB free)\n"
)


def placement(model: Path, ngl: int, split: list[float], rpc: bool) -> dict[str, int]:
    argv = [str(FAKE), "-m", str(model), "-ngl", str(ngl), "-lv", "5", "--port", "0"]
    argv += ["--tensor-split", ",".join(f"{w:g}" for w in split)]
    if rpc:
        argv += ["--rpc", "100.98.227.49:50052"]
    env = {**os.environ, "HUDDLE_FAKE_DEVICES": TWO_LOCAL}
    # The fake serves forever after printing; placement is printed first.
    process = subprocess.Popen(argv, stdout=subprocess.PIPE, text=True, env=env)
    lines = []
    assert process.stdout is not None
    for line in process.stdout:
        lines.append(line)
        if "listening on" in line:
            break
    process.kill()
    process.wait()
    return {k: len(v) for k, v in parse_layer_assignments("".join(lines)).items()}


@pytest.fixture
def model64(tmp_path: Path) -> Path:
    return write_gguf(tmp_path / "m64.gguf", n_layers=64, n_embd=256)


def test_measured_32b_split_is_reproduced(model64: Path) -> None:
    """The real two-node run: -ngl 60, split 30,0,30,0 -> five entries on CPU."""
    assert placement(model64, 60, [30, 0, 30, 0], rpc=True) == {
        "CPU": 5, "RPC0": 30, "Vulkan0": 30
    }  # fmt: skip


def test_planned_entries_are_placed_exactly(model64: Path) -> None:
    """With the head counted, 30 layers each means 30 and 31 entries."""
    assert placement(model64, 61, [30, 0, 31, 0], rpc=True) == {
        "CPU": 4, "RPC0": 30, "Vulkan0": 31
    }  # fmt: skip


def test_proportional_weights_push_a_layer_onto_the_peer(model64: Path) -> None:
    """Why weights are entry counts: 30,30 over 61 entries rounds a layer remote."""
    got = placement(model64, 61, [30, 0, 30, 0], rpc=True)
    assert got["RPC0"] == 31, "the peer gets a layer it was never budgeted for"
