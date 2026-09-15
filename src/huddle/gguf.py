"""Read GGUF metadata without loading the model.

The planner has to know how many layers a model has, and what each one costs,
*before* anything is loaded — llama-server's ``/props`` only answers after the
model is in memory, which is far too late to decide a split. So we read the
header ourselves.

Format reference: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
"""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import BinaryIO

GGUF_MAGIC = b"GGUF"
DEFAULT_ALIGNMENT = 32

# Arrays longer than this are payload, not configuration (tokenizer vocabularies
# run to six figures). We consume them to stay in sync with the stream, but do
# not keep them.
MAX_KEPT_ARRAY = 1024

# ggml block and type sizes, dumped from the pinned llama.cpp build via
# ggml_blck_size()/ggml_type_size() rather than transcribed from memory.
# id: (name, block_size, type_size)
GGML_TYPES: dict[int, tuple[str, int, int]] = {
    0: ("f32", 1, 4),
    1: ("f16", 1, 2),
    2: ("q4_0", 32, 18),
    3: ("q4_1", 32, 20),
    6: ("q5_0", 32, 22),
    7: ("q5_1", 32, 24),
    8: ("q8_0", 32, 34),
    9: ("q8_1", 32, 36),
    10: ("q2_K", 256, 84),
    11: ("q3_K", 256, 110),
    12: ("q4_K", 256, 144),
    13: ("q5_K", 256, 176),
    14: ("q6_K", 256, 210),
    15: ("q8_K", 256, 292),
    16: ("iq2_xxs", 256, 66),
    17: ("iq2_xs", 256, 74),
    18: ("iq3_xxs", 256, 98),
    19: ("iq1_s", 256, 50),
    20: ("iq4_nl", 32, 18),
    21: ("iq3_s", 256, 110),
    22: ("iq2_s", 256, 82),
    23: ("iq4_xs", 256, 136),
    24: ("i8", 1, 1),
    25: ("i16", 1, 2),
    26: ("i32", 1, 4),
    27: ("i64", 1, 8),
    28: ("f64", 1, 8),
    29: ("iq1_m", 256, 56),
    30: ("bf16", 1, 2),
    34: ("tq1_0", 256, 54),
    35: ("tq2_0", 256, 66),
    39: ("mxfp4", 32, 17),
    40: ("nvfp4", 64, 36),
    41: ("q1_0", 128, 18),
    42: ("q2_0", 64, 18),
}

# llama_ftype, from include/llama.h in the pinned build. A K-quant file is a
# *mixture* of tensor types, so the dominant tensor type is not the model's
# quantization name — general.file_type is. Gaps are types upstream removed.
LLAMA_FTYPES: dict[int, str] = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0", 8: "Q5_0", 9: "Q5_1",
    10: "Q2_K", 11: "Q3_K_S", 12: "Q3_K_M", 13: "Q3_K_L", 14: "Q4_K_S", 15: "Q4_K_M",
    16: "Q5_K_S", 17: "Q5_K_M", 18: "Q6_K", 19: "IQ2_XXS", 20: "IQ2_XS", 21: "Q2_K_S",
    22: "IQ3_XS", 23: "IQ3_XXS", 24: "IQ1_S", 25: "IQ4_NL", 26: "IQ3_S", 27: "IQ3_M",
    28: "IQ2_S", 29: "IQ2_M", 30: "IQ4_XS", 31: "IQ1_M", 32: "BF16",
    36: "TQ1_0", 37: "TQ2_0", 38: "MXFP4_MOE", 39: "NVFP4",
}

_LAYER_TENSOR = re.compile(r"^blk\.(\d+)\.")


class GGUFError(ValueError):
    """The file is not a GGUF we can read."""


class ValueType(IntEnum):
    UINT8 = 0
    INT8 = 1
    UINT16 = 2
    INT16 = 3
    UINT32 = 4
    INT32 = 5
    FLOAT32 = 6
    BOOL = 7
    STRING = 8
    ARRAY = 9
    UINT64 = 10
    INT64 = 11
    FLOAT64 = 12


_SCALAR_FORMATS: dict[int, str] = {
    ValueType.UINT8: "<B",
    ValueType.INT8: "<b",
    ValueType.UINT16: "<H",
    ValueType.INT16: "<h",
    ValueType.UINT32: "<I",
    ValueType.INT32: "<i",
    ValueType.FLOAT32: "<f",
    ValueType.BOOL: "<?",
    ValueType.UINT64: "<Q",
    ValueType.INT64: "<q",
    ValueType.FLOAT64: "<d",
}


@dataclass(frozen=True)
class TensorInfo:
    """One tensor's shape and storage cost."""

    name: str
    dimensions: tuple[int, ...]
    ggml_type: int
    offset: int

    @property
    def element_count(self) -> int:
        count = 1
        for dimension in self.dimensions:
            count *= dimension
        return count

    @property
    def nbytes(self) -> int:
        """Storage size, accounting for block quantization."""
        traits = GGML_TYPES.get(self.ggml_type)
        if traits is None:
            raise GGUFError(f"unknown ggml type {self.ggml_type} for tensor {self.name!r}")
        _, block_size, type_size = traits
        if block_size == 0:
            raise GGUFError(f"tensor {self.name!r} uses a removed ggml type {self.ggml_type}")
        return self.element_count // block_size * type_size

    @property
    def layer_index(self) -> int | None:
        """Which transformer block this belongs to, if any."""
        match = _LAYER_TENSOR.match(self.name)
        return int(match.group(1)) if match else None


@dataclass
class ModelInfo:
    """What the planner needs to know about a GGUF before loading it."""

    path: Path
    architecture: str
    name: str | None
    n_layers: int
    n_ctx_train: int | None
    n_embd: int | None
    n_head: int | None
    n_head_kv: int | None
    file_size: int
    tensor_bytes: int
    file_type: int | None = None
    layer_bytes: list[int] = field(default_factory=list)
    overhead_bytes: int = 0
    quantization: str | None = None

    @property
    def mean_layer_bytes(self) -> int:
        """Average cost of one transformer block.

        Blocks are near-identical in size, so the mean is a sound unit for
        deciding how many fit on a device.
        """
        return sum(self.layer_bytes) // len(self.layer_bytes) if self.layer_bytes else 0

    def kv_cache_bytes(self, n_ctx: int, *, bytes_per_element: int = 2) -> int:
        """Approximate KV cache cost at a given context length.

        Easy to forget when planning, and large enough to cause an out-of-memory
        failure at load time if it is. Defaults to f16 cache entries.
        """
        if not self.n_embd or not self.n_head:
            return 0
        n_embd_head = self.n_embd // self.n_head
        n_head_kv = self.n_head_kv or self.n_head
        # One K and one V entry per token, per layer.
        return 2 * self.n_layers * n_ctx * n_embd_head * n_head_kv * bytes_per_element


class _Reader:
    """Little-endian primitive reads over a binary stream."""

    def __init__(self, stream: BinaryIO) -> None:
        self.stream = stream

    def read(self, size: int) -> bytes:
        data = self.stream.read(size)
        if len(data) != size:
            raise GGUFError("unexpected end of file while reading header")
        return data

    def scalar(self, value_type: int) -> int | float | bool:
        fmt = _SCALAR_FORMATS.get(value_type)
        if fmt is None:
            raise GGUFError(f"unsupported metadata value type {value_type}")
        size = struct.calcsize(fmt)
        return struct.unpack(fmt, self.read(size))[0]  # type: ignore[no-any-return]

    def u32(self) -> int:
        return int(struct.unpack("<I", self.read(4))[0])

    def u64(self) -> int:
        return int(struct.unpack("<Q", self.read(8))[0])

    def string(self) -> str:
        return self.read(self.u64()).decode("utf-8", errors="replace")

    def skip(self, size: int) -> None:
        self.stream.seek(size, 1)


def _read_value(reader: _Reader, value_type: int) -> object:
    if value_type == ValueType.STRING:
        return reader.string()
    if value_type == ValueType.ARRAY:
        return _read_array(reader)
    return reader.scalar(value_type)


def _read_array(reader: _Reader) -> list[object] | int:
    """Read an array, discarding oversized ones but staying in sync.

    Returns the element count instead of the values when the array is too large
    to be configuration, so callers can tell the difference.
    """
    element_type = reader.u32()
    count = reader.u64()

    if count > MAX_KEPT_ARRAY:
        if element_type == ValueType.STRING:
            # Variable-length: the only way past is to walk it.
            for _ in range(count):
                reader.skip(reader.u64())
        elif element_type == ValueType.ARRAY:
            for _ in range(count):
                _read_array(reader)
        else:
            fmt = _SCALAR_FORMATS.get(element_type)
            if fmt is None:
                raise GGUFError(f"unsupported array element type {element_type}")
            reader.skip(struct.calcsize(fmt) * count)
        return count

    return [_read_value(reader, element_type) for _ in range(count)]


def _as_int(value: object) -> int | None:
    return int(value) if isinstance(value, int | float) and not isinstance(value, bool) else None


def read_gguf(path: str | Path) -> ModelInfo:
    """Parse a GGUF header and summarise what the planner needs.

    Reads only the header, so cost is independent of model size.
    """
    path = Path(path)
    with path.open("rb") as handle:
        reader = _Reader(handle)

        if reader.read(4) != GGUF_MAGIC:
            raise GGUFError(f"not a GGUF file: {path}")
        version = reader.u32()
        if version < 2:
            raise GGUFError(f"unsupported GGUF version {version} in {path}")

        tensor_count = reader.u64()
        kv_count = reader.u64()

        metadata: dict[str, object] = {}
        for _ in range(kv_count):
            key = reader.string()
            metadata[key] = _read_value(reader, reader.u32())

        tensors = []
        for _ in range(tensor_count):
            name = reader.string()
            n_dims = reader.u32()
            dimensions = tuple(reader.u64() for _ in range(n_dims))
            tensors.append(
                TensorInfo(
                    name=name,
                    dimensions=dimensions,
                    ggml_type=reader.u32(),
                    offset=reader.u64(),
                )
            )

    architecture = str(metadata.get("general.architecture", "unknown"))

    def arch_key(suffix: str) -> int | None:
        return _as_int(metadata.get(f"{architecture}.{suffix}"))

    n_layers = arch_key("block_count") or 0
    layer_bytes = [0] * n_layers
    overhead = 0
    total = 0
    for tensor in tensors:
        size = tensor.nbytes
        total += size
        index = tensor.layer_index
        if index is not None and index < n_layers:
            layer_bytes[index] += size
        else:
            overhead += size

    file_type = _as_int(metadata.get("general.file_type"))
    # Fall back to the dominant block tensor type, which is at least honest
    # about what is actually stored.
    quantization = LLAMA_FTYPES.get(file_type) if file_type is not None else None
    if quantization is None:
        quantization = _dominant_tensor_type(tensors)

    model_name = metadata.get("general.name")
    return ModelInfo(
        path=path,
        architecture=architecture,
        name=str(model_name) if model_name is not None else None,
        n_layers=n_layers,
        n_ctx_train=arch_key("context_length"),
        n_embd=arch_key("embedding_length"),
        n_head=arch_key("attention.head_count"),
        n_head_kv=arch_key("attention.head_count_kv"),
        file_size=path.stat().st_size,
        tensor_bytes=total,
        file_type=file_type,
        layer_bytes=layer_bytes,
        overhead_bytes=overhead,
        quantization=quantization,
    )


def _dominant_tensor_type(tensors: list[TensorInfo]) -> str | None:
    """The most common tensor type across transformer blocks."""
    counts: dict[int, int] = {}
    for tensor in tensors:
        if tensor.layer_index is not None:
            counts[tensor.ggml_type] = counts.get(tensor.ggml_type, 0) + 1
    if not counts:
        return None
    dominant = max(counts, key=lambda key: counts[key])
    traits = GGML_TYPES.get(dominant)
    return traits[0] if traits else None
