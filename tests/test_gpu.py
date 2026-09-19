"""Parsing nvidia-smi's CSV output. Informational only, never used for placement."""

from __future__ import annotations

import subprocess

import pytest

from huddle import gpu
from huddle.gpu import parse_nvidia_smi_csv, probe_gpu_samples


def test_parses_a_normal_multi_gpu_line() -> None:
    output = (
        "0, NVIDIA GeForce RTX 5070, 1234, 12227, 42, 55, 123.45\n"
        "1, Intel(R) Graphics (RPL-S), 500, 8192, 5, 40, 15.00\n"
    )
    samples = parse_nvidia_smi_csv(output)
    assert len(samples) == 2
    assert samples[0].name == "NVIDIA GeForce RTX 5070"
    assert samples[0].used_mib == 1234
    assert samples[0].total_mib == 12227
    assert samples[0].util_pct == 42
    assert samples[0].temp_c == 55
    assert samples[0].power_w == 123.45


def test_power_draw_not_available_becomes_none() -> None:
    """Some cards/drivers report power.draw as "[N/A]"."""
    samples = parse_nvidia_smi_csv("0, NVIDIA GeForce RTX 5070, 1234, 12227, 42, 55, [N/A]\n")
    assert samples[0].power_w is None


def test_empty_output_is_no_samples() -> None:
    assert parse_nvidia_smi_csv("") == []
    assert parse_nvidia_smi_csv("\n\n") == []


def test_malformed_line_is_skipped_not_raised() -> None:
    assert parse_nvidia_smi_csv("garbage, not enough fields\n") == []


def test_probe_returns_nothing_without_nvidia_smi(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gpu, "nvidia_smi_available", lambda: False)
    assert probe_gpu_samples() == []


def test_probe_returns_nothing_on_nonzero_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gpu, "nvidia_smi_available", lambda: True)

    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="error")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert probe_gpu_samples() == []


def test_probe_returns_nothing_when_the_call_times_out(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gpu, "nvidia_smi_available", lambda: True)

    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=5.0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert probe_gpu_samples() == []
