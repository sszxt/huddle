"""Build minimal but valid GGUF files, so the parser is testable without models.

Only the header is written — the parser never reads tensor data, and a real
model on the dev box is not an option.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

MAGIC = b"GGUF"
VERSION = 3

UINT32, FLOAT32, STRING, ARRAY, UINT64 = 4, 6, 8, 9, 10

Q4_K = 12
F32 = 0


def _string(value: str) -> bytes:
    raw = value.encode()
    return struct.pack("<Q", len(raw)) + raw


def _kv(key: str, value_type: int, payload: bytes) -> bytes:
    return _string(key) + struct.pack("<I", value_type) + payload


def _u32_kv(key: str, value: int) -> bytes:
    return _kv(key, UINT32, struct.pack("<I", value))


def _str_kv(key: str, value: str) -> bytes:
    return _kv(key, STRING, _string(value))


def _str_array_kv(key: str, values: list[str]) -> bytes:
    body = struct.pack("<I", STRING) + struct.pack("<Q", len(values))
    body += b"".join(_string(v) for v in values)
    return _kv(key, ARRAY, body)


def _tensor(name: str, dimensions: tuple[int, ...], ggml_type: int, offset: int) -> bytes:
    out = _string(name) + struct.pack("<I", len(dimensions))
    out += b"".join(struct.pack("<Q", d) for d in dimensions)
    out += struct.pack("<I", ggml_type) + struct.pack("<Q", offset)
    return out


def write_gguf(
    path: Path,
    *,
    architecture: str = "llama",
    name: str = "test-model",
    n_layers: int = 4,
    n_embd: int = 256,
    n_head: int = 4,
    n_head_kv: int = 2,
    n_ctx_train: int = 2048,
    tensors_per_layer: int = 3,
    vocab_size: int = 0,
    extra_metadata: dict[str, Any] | None = None,
) -> Path:
    """Write a GGUF whose header describes a plausible transformer.

    ``vocab_size`` adds a large string array, to exercise the oversized-array
    path the way a real tokenizer vocabulary does.
    """
    entries = [
        _str_kv("general.architecture", architecture),
        _str_kv("general.name", name),
        _u32_kv(f"{architecture}.block_count", n_layers),
        _u32_kv(f"{architecture}.embedding_length", n_embd),
        _u32_kv(f"{architecture}.attention.head_count", n_head),
        _u32_kv(f"{architecture}.attention.head_count_kv", n_head_kv),
        _u32_kv(f"{architecture}.context_length", n_ctx_train),
    ]
    if vocab_size:
        entries.append(_str_array_kv("tokenizer.ggml.tokens", [f"t{i}" for i in range(vocab_size)]))
    entries += [_u32_kv(key, int(value)) for key, value in (extra_metadata or {}).items()]

    metadata = b"".join(entries)
    count = len(entries)

    tensor_blob = b""
    tensor_count = 0
    offset = 0

    # Embedding and output head: real, sizeable, and not part of any layer.
    for special in ("token_embd.weight", "output.weight"):
        tensor_blob += _tensor(special, (n_embd, n_embd * 2), Q4_K, offset)
        offset += n_embd * n_embd * 2 // 256 * 144
        tensor_count += 1
    tensor_blob += _tensor("output_norm.weight", (n_embd,), F32, offset)
    offset += n_embd * 4
    tensor_count += 1

    for layer in range(n_layers):
        for part in range(tensors_per_layer):
            tensor_blob += _tensor(f"blk.{layer}.w{part}.weight", (n_embd, n_embd), Q4_K, offset)
            offset += n_embd * n_embd // 256 * 144
            tensor_count += 1

    header = MAGIC + struct.pack("<I", VERSION)
    header += struct.pack("<Q", tensor_count) + struct.pack("<Q", count)
    path.write_bytes(header + metadata + tensor_blob)
    return path
