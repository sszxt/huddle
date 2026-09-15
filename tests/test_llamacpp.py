"""The argv builders are our real contract with llama.cpp.

We cannot run inference here, so these assertions are the thing standing between
us and a broken command line discovered on the target box.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from huddle.config import BackendConfig, RpcConfig
from huddle.llamacpp import (
    LlamaCppError,
    build_llama_server_argv,
    build_rpc_server_argv,
    parse_devices,
    parse_layer_assignments,
    require_binary,
)

RPC_BINARY = Path("/opt/llama.cpp/rpc-server")
SERVER_BINARY = Path("/opt/llama.cpp/llama-server")
MODEL = Path("/models/qwen.gguf")


def test_rpc_argv_always_sets_bind_host() -> None:
    """llama.cpp binds 127.0.0.1 by default, which breaks cross-machine use."""
    argv = build_rpc_server_argv(RPC_BINARY, RpcConfig())
    assert "-H" in argv
    assert argv[argv.index("-H") + 1] == "0.0.0.0"


def test_rpc_argv_enables_cache_by_default() -> None:
    """Without -c, every load re-streams the weights over the network."""
    assert "-c" in build_rpc_server_argv(RPC_BINARY, RpcConfig())


def test_rpc_argv_omits_cache_when_disabled() -> None:
    assert "-c" not in build_rpc_server_argv(RPC_BINARY, RpcConfig(cache=False))


def test_rpc_argv_full() -> None:
    config = RpcConfig(bind="10.0.0.5", port=50100, devices=["Vulkan0", "Vulkan1"], threads=8)
    assert build_rpc_server_argv(RPC_BINARY, config) == [
        str(RPC_BINARY),
        "-H", "10.0.0.5",
        "-p", "50100",
        "-c",
        "-d", "Vulkan0,Vulkan1",
        "-t", "8",
    ]  # fmt: skip


def test_llama_server_argv_single_node() -> None:
    argv = build_llama_server_argv(SERVER_BINARY, BackendConfig(), MODEL)
    assert argv == [
        str(SERVER_BINARY),
        "-m", str(MODEL),
        "--host", "127.0.0.1",
        "--port", "8080",
        "-c", "4096",
        "-np", "1",
        "-ngl", "all",
    ]  # fmt: skip
    assert "--rpc" not in argv, "a single-node launch must not pass --rpc"


def test_llama_server_argv_with_peers() -> None:
    argv = build_llama_server_argv(
        SERVER_BINARY,
        BackendConfig(),
        MODEL,
        rpc_endpoints=["192.168.0.54:50052", "192.168.0.55:50052"],
        tensor_split=[0.7, 0.3],
        alias="qwen",
    )
    assert argv[argv.index("--rpc") + 1] == "192.168.0.54:50052,192.168.0.55:50052"
    assert argv[argv.index("--tensor-split") + 1] == "0.7,0.3"
    assert argv[argv.index("--alias") + 1] == "qwen"


def test_llama_server_argv_appends_extra_args_last() -> None:
    backend = BackendConfig(extra_args=["--flash-attn"])
    argv = build_llama_server_argv(SERVER_BINARY, backend, MODEL)
    assert argv[-1] == "--flash-attn"


def test_require_binary_rejects_missing(tmp_path: Path) -> None:
    with pytest.raises(LlamaCppError, match="not found"):
        require_binary(tmp_path / "nope", role="llama-server")


def test_parse_devices_reads_id_name_and_memory() -> None:
    output = (
        "Available devices:\n"
        "  Vulkan0: NVIDIA GeForce RTX 5070 (12227 MiB, 11041 MiB free)\n"
        "  Vulkan1: Intel(R) Graphics (RPL-S) (8192 MiB, 7900 MiB free)\n"
    )
    devices = parse_devices(output)
    assert [d.id for d in devices] == ["Vulkan0", "Vulkan1"]
    assert devices[0].name == "NVIDIA GeForce RTX 5070"
    assert devices[0].total_mib == 12227
    assert devices[0].free_mib == 11041


def test_parse_devices_tolerates_missing_memory() -> None:
    devices = parse_devices("  RPC0: 192.168.0.55:50052\n")
    assert len(devices) == 1
    assert devices[0].total_mib is None


def test_parse_devices_flags_rpc_devices() -> None:
    devices = parse_devices("  RPC0: remote (4096 MiB, 4000 MiB free)\n")
    assert devices[0].is_rpc


def test_parse_devices_ignores_noise() -> None:
    assert parse_devices("Available devices:\n\n") == []


def test_parse_devices_against_real_output() -> None:
    """Golden file captured from `llama-server --list-devices` on omarchy.

    Guards the parser against the actual format, not our guess at it. Recapture
    this file if the llama.cpp pin moves and the format changes.
    """
    golden = Path(__file__).parent / "golden" / "list_devices_omarchy.txt"
    devices = parse_devices(golden.read_text())

    assert [d.id for d in devices] == ["Vulkan0", "Vulkan1"]
    assert devices[0].name == "NVIDIA GeForce RTX 5070"
    assert (devices[0].total_mib, devices[0].free_mib) == (12473, 11447)
    assert devices[1].name == "Intel(R) Graphics (RPL-S)"
    # The iGPU advertises 48 GiB because it carves out system RAM. Placement
    # must not read this as dedicated VRAM.
    assert devices[1].total_mib == 48045


def test_parse_layer_assignments_uses_the_final_pass() -> None:
    """The log holds a fitting dry run and the real load; only the last counts.

    Format captured from a real two-node run on omarchy with `-lv 5`.
    """
    golden = Path(__file__).parent / "golden" / "layer_assignment_omarchy.txt"
    placement = parse_layer_assignments(golden.read_text())

    assert placement == {"CPU": [0], "RPC0": [1, 2, 3], "Vulkan0": [4, 5]}
    # The dry run put layer 3 on Vulkan0; the real load put it on RPC0.
    assert 3 in placement["RPC0"]


def test_parse_layer_assignments_empty_without_verbosity() -> None:
    """At default verbosity llama.cpp prints nothing about placement."""
    assert parse_layer_assignments("model loaded\nlistening on http://...\n") == {}
