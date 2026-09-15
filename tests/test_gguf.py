"""GGUF header parsing.

The planner depends on these numbers being right, and a wrong layer count
produces a split that is silently wrong rather than an error.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from huddle.gguf import GGML_TYPES, GGUFError, TensorInfo, read_gguf
from tests.fakes.gguf_builder import write_gguf


def test_reads_architecture_and_shape(tmp_path: Path) -> None:
    info = read_gguf(write_gguf(tmp_path / "m.gguf", architecture="qwen2", n_layers=6))
    assert info.architecture == "qwen2"
    assert info.name == "test-model"
    assert info.n_layers == 6
    assert info.n_embd == 256
    assert info.n_head == 4
    assert info.n_head_kv == 2
    assert info.n_ctx_train == 2048


def test_layer_bytes_are_per_block_and_uniform(tmp_path: Path) -> None:
    info = read_gguf(write_gguf(tmp_path / "m.gguf", n_layers=4, tensors_per_layer=3))
    assert len(info.layer_bytes) == 4
    # 3 tensors of 256x256 q4_K per layer: 65536/256*144 each.
    expected = 3 * (256 * 256 // 256 * 144)
    assert info.layer_bytes == [expected] * 4
    assert info.mean_layer_bytes == expected


def test_non_layer_tensors_counted_as_overhead(tmp_path: Path) -> None:
    """Embeddings and the output head live on one device regardless of split."""
    info = read_gguf(write_gguf(tmp_path / "m.gguf", n_layers=2))
    assert info.overhead_bytes > 0
    assert info.tensor_bytes == sum(info.layer_bytes) + info.overhead_bytes


def test_quantization_name_comes_from_file_type(tmp_path: Path) -> None:
    """A K-quant file mixes tensor types; general.file_type is the real name."""
    path = write_gguf(tmp_path / "m.gguf", extra_metadata={"general.file_type": 15})
    info = read_gguf(path)
    assert info.file_type == 15
    assert info.quantization == "Q4_K_M"


def test_quantization_falls_back_to_dominant_tensor_type(tmp_path: Path) -> None:
    """Without general.file_type, report what is actually stored."""
    info = read_gguf(write_gguf(tmp_path / "m.gguf"))
    assert info.file_type is None
    assert info.quantization == "q4_K"


def test_skips_oversized_arrays(tmp_path: Path) -> None:
    """A tokenizer vocabulary must not derail the parse or be held in memory."""
    info = read_gguf(write_gguf(tmp_path / "m.gguf", n_layers=3, vocab_size=5000))
    assert info.n_layers == 3
    assert info.architecture == "llama"


def test_kv_cache_scales_with_context(tmp_path: Path) -> None:
    info = read_gguf(write_gguf(tmp_path / "m.gguf", n_layers=4, n_embd=256, n_head=4, n_head_kv=2))
    # 2 (K+V) * layers * ctx * head_dim(64) * kv_heads(2) * 2 bytes
    assert info.kv_cache_bytes(1024) == 2 * 4 * 1024 * 64 * 2 * 2
    assert info.kv_cache_bytes(2048) == 2 * info.kv_cache_bytes(1024)


def test_rejects_non_gguf(tmp_path: Path) -> None:
    bogus = tmp_path / "not.gguf"
    bogus.write_bytes(b"XXXX" + b"\x00" * 64)
    with pytest.raises(GGUFError, match="not a GGUF file"):
        read_gguf(bogus)


def test_rejects_truncated_file(tmp_path: Path) -> None:
    truncated = tmp_path / "cut.gguf"
    truncated.write_bytes(b"GGUF" + b"\x03\x00\x00\x00" + b"\x01")
    with pytest.raises(GGUFError, match="unexpected end of file"):
        read_gguf(truncated)


def test_unknown_tensor_type_is_reported(tmp_path: Path) -> None:
    tensor = TensorInfo(name="blk.0.w.weight", dimensions=(256, 256), ggml_type=999, offset=0)
    with pytest.raises(GGUFError, match="unknown ggml type"):
        _ = tensor.nbytes


@pytest.mark.parametrize(
    ("type_id", "name", "block", "size"),
    [
        (0, "f32", 1, 4),
        (1, "f16", 1, 2),
        (12, "q4_K", 256, 144),
        (14, "q6_K", 256, 210),
        (29, "iq1_m", 256, 56),
        (39, "mxfp4", 32, 17),
    ],
)
def test_type_table_matches_ggml(type_id: int, name: str, block: int, size: int) -> None:
    """Values dumped from the pinned build's ggml_blck_size/ggml_type_size."""
    assert GGML_TYPES[type_id] == (name, block, size)
