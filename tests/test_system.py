"""OS details for the cluster page: the parsers, and a probe that never raises."""

from __future__ import annotations

from pathlib import Path

import pytest

from huddle import system
from huddle.system import (
    NetAddress,
    cpu_percent,
    link_speed,
    parse_cpu_model,
    parse_cpu_times,
    parse_ip_addresses,
    parse_loadavg,
    parse_os_release,
    parse_uptime,
    probe_system,
)

OS_RELEASE = 'NAME="Ubuntu"\nVERSION_ID="26.04"\nPRETTY_NAME="Ubuntu 26.04.1 LTS"\nID=ubuntu\n'

CPUINFO = (
    "processor\t: 0\nvendor_id\t: GenuineIntel\n"
    "model name\t: Intel(R) Core(TM) i7-14700\ncpu MHz\t\t: 800.000\n\n"
    "processor\t: 1\nmodel name\t: Intel(R) Core(TM) i7-14700\n"
)

# user nice system idle iowait irq softirq steal guest guest_nice
STAT_BEFORE = "cpu  100 0 50 800 50 0 0 0 7 0\ncpu0 50 0 25 400 25 0 0 0 0 0\n"
STAT_AFTER = "cpu  160 0 70 880 50 0 0 0 9 0\ncpu0 80 0 35 440 25 0 0 0 0 0\n"

# The shape `ip -j -4 addr show` prints (trimmed), from a real Ubuntu node.
IP_JSON = """[
  {"ifindex": 1, "ifname": "lo", "flags": ["LOOPBACK", "UP", "LOWER_UP"],
   "addr_info": [{"family": "inet", "local": "127.0.0.1", "prefixlen": 8}]},
  {"ifindex": 2, "ifname": "enp4s0", "flags": ["BROADCAST", "MULTICAST", "UP", "LOWER_UP"],
   "addr_info": [{"family": "inet", "local": "192.168.0.72", "prefixlen": 24}]},
  {"ifindex": 5, "ifname": "tailscale0", "flags": ["POINTOPOINT", "UP", "LOWER_UP"],
   "addr_info": [{"family": "inet", "local": "100.94.234.89", "prefixlen": 32}]}
]"""


def test_os_release_pretty_name() -> None:
    assert parse_os_release(OS_RELEASE) == "Ubuntu 26.04.1 LTS"
    assert parse_os_release("NAME=Arch\n") is None


def test_cpu_model_is_the_first_model_name() -> None:
    assert parse_cpu_model(CPUINFO) == "Intel(R) Core(TM) i7-14700"
    assert parse_cpu_model("processor : 0\n") is None


def test_cpu_times_count_iowait_as_idle_and_ignore_guest() -> None:
    assert parse_cpu_times(STAT_BEFORE) == (850, 1000)
    assert parse_cpu_times("intr 12 34\n") is None


def test_cpu_percent_is_the_busy_share_between_samples() -> None:
    before, after = parse_cpu_times(STAT_BEFORE), parse_cpu_times(STAT_AFTER)
    assert before is not None and after is not None
    assert cpu_percent(before, after) == 50.0
    assert cpu_percent(after, after) is None


def test_uptime_and_load() -> None:
    assert parse_uptime("7796.08 180422.51\n") == 7796.08
    assert parse_loadavg("0.28 0.43 0.38 1/1874 39900\n") == 0.28
    assert parse_uptime("") is None
    assert parse_loadavg("garbage") is None


def test_ip_addresses_leave_out_loopback() -> None:
    assert parse_ip_addresses(IP_JSON) == [
        NetAddress(interface="enp4s0", address="192.168.0.72", prefix=24),
        NetAddress(interface="tailscale0", address="100.94.234.89", prefix=32),
    ]
    assert parse_ip_addresses("not json") == []


def test_link_speed_reports_only_real_speeds(tmp_path: Path) -> None:
    for name, speed in (("enp4s0", "1000\n"), ("wlan0", "-1\n")):
        (tmp_path / name).mkdir()
        (tmp_path / name / "speed").write_text(speed)
    assert link_speed("enp4s0", tmp_path) == 1000
    assert link_speed("wlan0", tmp_path) is None
    assert link_speed("missing0", tmp_path) is None


def test_probe_never_raises_when_every_source_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("huddle.system.shutil.which", lambda name: None)
    missing = tmp_path / "missing"
    info = probe_system(sample_interval=0, proc=missing, os_release=missing, sys_net=missing)
    assert info.hostname
    assert info.os is None and info.cpu_model is None and info.cpu_pct is None
    assert info.uptime_s is None and info.addresses == []


def test_probe_reads_fixture_sources(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    proc = tmp_path / "proc"
    proc.mkdir()
    (proc / "cpuinfo").write_text(CPUINFO)
    (proc / "uptime").write_text("120.5 900.0\n")
    (proc / "loadavg").write_text("1.50 1.00 0.50 2/300 1234\n")
    samples = iter([STAT_BEFORE, STAT_AFTER])
    real_read = system._read

    def fake_read(path: Path) -> str | None:
        return next(samples) if path == proc / "stat" else real_read(path)

    monkeypatch.setattr(system, "_read", fake_read)
    release = tmp_path / "os-release"
    release.write_text(OS_RELEASE)

    info = probe_system(sample_interval=0, proc=proc, os_release=release, sys_net=tmp_path)
    assert info.os == "Ubuntu 26.04.1 LTS"
    assert info.cpu_model == "Intel(R) Core(TM) i7-14700"
    assert info.cpu_pct == 50.0
    assert info.load_1m == 1.5
    assert info.uptime_s == 120.5
