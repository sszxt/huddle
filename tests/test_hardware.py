"""GPU sample matching: nvidia-smi is a second, unreliable source of truth.

Device *placement* never depends on this (hardware.py's own docstring: it
trusts llama.cpp's own --list-devices for that), so these tests only cover
whether util/temp/power get attached to the right DeviceInfo, and that an
ambiguous match is left alone rather than guessed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from huddle import hardware
from huddle.gpu import GpuSample
from huddle.hardware import probe


def test_gpu_samples_attach_to_the_matching_nvidia_device(
    fake_llama_server: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The fixture's default devices are a real RTX 5070 plus an Intel iGPU."""
    monkeypatch.setattr(
        hardware,
        "probe_gpu_samples",
        lambda: [
            GpuSample(
                index=0,
                name="NVIDIA GeForce RTX 5070",
                used_mib=1000,
                total_mib=12227,
                util_pct=42,
                temp_c=55,
                power_w=123.4,
            )
        ],
    )
    result = probe("testnode", fake_llama_server)
    nvidia, igpu = result.devices[0], result.devices[1]

    assert nvidia.id == "Vulkan0"
    assert nvidia.util_pct == 42
    assert nvidia.temp_c == 55
    assert nvidia.power_w == 123.4
    # The Intel device must not pick up the NVIDIA sample.
    assert igpu.util_pct is None
    assert igpu.temp_c is None
    assert igpu.power_w is None


def test_no_samples_leaves_devices_unchanged(
    fake_llama_server: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(hardware, "probe_gpu_samples", lambda: [])
    result = probe("testnode", fake_llama_server)
    assert all(d.util_pct is None for d in result.devices)


def test_ambiguous_matches_are_left_unmatched(
    fake_llama_server: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two NVIDIA-looking devices and two unnamed samples: too ambiguous to guess."""
    monkeypatch.setenv(
        "HUDDLE_FAKE_DEVICES",
        "Available devices:\n"
        "  Vulkan0: NVIDIA GeForce RTX 5070 (12227 MiB, 11041 MiB free)\n"
        "  Vulkan1: NVIDIA GeForce RTX 5070 (12227 MiB, 11041 MiB free)\n",
    )
    monkeypatch.setattr(
        hardware,
        "probe_gpu_samples",
        lambda: [
            GpuSample(0, "GPU 0", 1000, 12227, 10, 40, 100.0),
            GpuSample(1, "GPU 1", 1000, 12227, 20, 50, 110.0),
        ],
    )
    result = probe("testnode", fake_llama_server)
    assert all(d.util_pct is None for d in result.devices)


def test_one_candidate_and_one_sample_pair_positionally(
    fake_llama_server: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A single NVIDIA device and a single sample that doesn't name-match."""
    monkeypatch.setenv(
        "HUDDLE_FAKE_DEVICES",
        "Available devices:\n  Vulkan0: NVIDIA GeForce RTX 5070 (12227 MiB, 11041 MiB free)\n",
    )
    monkeypatch.setattr(
        hardware,
        "probe_gpu_samples",
        lambda: [GpuSample(0, "GPU 0", 1000, 12227, 33, 60, 150.0)],
    )
    result = probe("testnode", fake_llama_server)
    assert result.devices[0].util_pct == 33
