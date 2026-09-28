"""A short list of models to offer someone who has none yet.

Not a model zoo: a handful of well-known single-file GGUFs spanning sizes, so
a fresh cluster can be put to use in one click, with a label saying whether
each will fit. Anything else is a search away in the Workspace page. Every
file name and size here was checked against the Hugging Face Hub when added;
``verified`` marks the ones actually run on the real two-node cluster.

The fit is an estimate from file size, like the planner's own first guess,
and labelled as one. The planner decides the real placement at load.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from huddle.hardware import NodeHardware

MIB = 1024 * 1024
# Beyond the weights: KV cache at the default context, compute buffers and
# the output head. Roughly what the 32B needed on the real cluster at ctx 4096.
OVERHEAD = 1.10
OVERHEAD_MIB = 1024

Fit = Literal["gpu", "partial", "too_large"]


class CatalogModel(BaseModel):
    repo_id: str
    file: str
    name: str
    size: int
    summary: str
    verified: bool = False


CATALOG: tuple[CatalogModel, ...] = (
    CatalogModel(
        repo_id="Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        file="qwen2.5-0.5b-instruct-q4_k_m.gguf",
        name="Qwen2.5 0.5B",
        size=491_400_032,
        summary="Tiny and quick to download: check that everything works.",
        verified=True,
    ),
    CatalogModel(
        repo_id="bartowski/Qwen_Qwen3-8B-GGUF",
        file="Qwen_Qwen3-8B-Q4_K_M.gguf",
        name="Qwen3 8B",
        size=5_027_784_224,
        summary="Fast everyday assistant that fits on one mid-range GPU.",
    ),
    CatalogModel(
        repo_id="bartowski/Qwen_Qwen3-14B-GGUF",
        file="Qwen_Qwen3-14B-Q4_K_M.gguf",
        name="Qwen3 14B",
        size=9_001_753_632,
        summary="Stronger reasoning and writing; wants a 12 GB GPU or two smaller ones.",
    ),
    CatalogModel(
        repo_id="bartowski/google_gemma-3-27b-it-GGUF",
        file="google_gemma-3-27b-it-Q4_K_M.gguf",
        name="Gemma 3 27B",
        size=16_546_404_992,
        summary="Google's large open model: more than one consumer GPU holds.",
    ),
    CatalogModel(
        repo_id="bartowski/Qwen2.5-32B-Instruct-GGUF",
        file="Qwen2.5-32B-Instruct-Q4_K_M.gguf",
        name="Qwen2.5 32B",
        size=19_851_336_576,
        summary="Run across two 12 GB GPUs on the real cluster: neither holds it alone.",
        verified=True,
    ),
    CatalogModel(
        repo_id="bartowski/Llama-3.3-70B-Instruct-GGUF",
        file="Llama-3.3-70B-Instruct-Q4_K_M.gguf",
        name="Llama 3.3 70B",
        size=42_520_398_816,
        summary="Meta's 70B: needs about 45 GB between GPUs and system memory.",
    ),
)


class Capacity(BaseModel):
    """What a cluster could hold, for the estimate."""

    nodes: int
    # Free memory on discrete GPUs, after the planner's headroom.
    gpu_mib: int
    # Memory on the node that would load the model, for layers left on CPU:
    # only the head's CPU runs layers, workers lend their GPUs alone.
    head_ram_mib: int


class CatalogEntry(CatalogModel):
    fit: Fit
    need_mib: int
    # Already on some node, so it can be started without downloading.
    on_node: str | None = None


def capacity(hardware: list[NodeHardware], head: NodeHardware, headroom: float) -> Capacity:
    gpu = 0
    for node in hardware:
        for device in node.devices:
            if device.is_rpc or device.unified_memory:
                continue  # an integrated GPU's figure is system RAM
            gpu += int((device.free_mib or 0) * (1 - headroom))
    return Capacity(nodes=len(hardware), gpu_mib=gpu, head_ram_mib=head.ram_available_mib)


def need_mib(size: int) -> int:
    return int(size / MIB * OVERHEAD) + OVERHEAD_MIB


def fit_for(size: int, room: Capacity) -> Fit:
    need = need_mib(size)
    if need <= room.gpu_mib:
        return "gpu"
    # Layers the GPUs cannot take run on the head's CPU, from its RAM. Leave
    # it a fifth for everything else the machine is doing.
    if need <= room.gpu_mib + int(room.head_ram_mib * 0.8):
        return "partial"
    return "too_large"


def entries(room: Capacity, present: dict[str, str]) -> list[CatalogEntry]:
    """The catalog with fits, given which files already sit on which node."""
    return [
        CatalogEntry(
            **model.model_dump(),
            fit=fit_for(model.size, room),
            need_mib=need_mib(model.size),
            on_node=present.get(model.file),
        )
        for model in CATALOG
    ]


__all__ = ["CATALOG", "Capacity", "CatalogEntry", "CatalogModel", "capacity", "entries", "fit_for"]
