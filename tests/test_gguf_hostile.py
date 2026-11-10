"""A damaged or hostile GGUF header must fail cleanly, never badly.

``read_gguf`` runs on files nobody has vetted: a download that stopped half way,
or something dropped into the models directory. Every caller expects a
``GGUFError`` (``huddle doctor`` catches nothing else), so the contract is that
the parser either returns a ``ModelInfo`` or raises ``GGUFError``: no other
exception, no hang, and no allocation sized by a number the file merely claims.

The named cases below are headers that used to break that contract (found by
reading the parser, then confirmed against it): ``OverflowError`` and
``MemoryError`` from lengths, ``RecursionError`` from nesting, a layer count
that asked for a ten-billion-slot list. The Hypothesis tests then look for more.

The fuzz tests run under an address-space cap, so a regression that allocates
gigabytes fails the test instead of the machine.
"""

from __future__ import annotations

import struct
import tempfile
from collections.abc import Iterator
from datetime import timedelta
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from huddle.gguf import MAX_LAYERS, GGUFError, ModelInfo, read_gguf
from tests.fakes.gguf_builder import write_gguf

U8, I32, U32, F32, STRING, ARRAY, U64, F64 = 0, 5, 4, 6, 8, 9, 10, 12

CAP_BYTES = 1536 * 1024 * 1024


@pytest.fixture(autouse=True, scope="module")
def memory_cap() -> Iterator[None]:
    """Fail with MemoryError, in this process only, long before the PC would swap."""
    resource = pytest.importorskip("resource")
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    try:
        status = Path("/proc/self/status").read_text()
        used = next(
            int(line.split()[1]) * 1024
            for line in status.splitlines()
            if line.startswith("VmSize:")
        )
    except (OSError, StopIteration):
        yield
        return
    limit = used + CAP_BYTES
    if hard != resource.RLIM_INFINITY:
        limit = min(limit, hard)
    resource.setrlimit(resource.RLIMIT_AS, (limit, hard))
    try:
        yield
    finally:
        resource.setrlimit(resource.RLIMIT_AS, (soft, hard))


# -- building headers by hand ---------------------------------------------------------


def string(raw: bytes) -> bytes:
    return struct.pack("<Q", len(raw)) + raw


def entry(key: str, value_type: int, payload: bytes) -> bytes:
    return string(key.encode()) + struct.pack("<I", value_type) + payload


def header(entries: list[bytes], tensors: bytes = b"", tensor_count: int = 0) -> bytes:
    return (
        b"GGUF"
        + struct.pack("<I", 3)
        + struct.pack("<QQ", tensor_count, len(entries))
        + b"".join(entries)
        + tensors
    )


def tensor(name: bytes, dimensions: tuple[int, ...], ggml_type: int = 0) -> bytes:
    packed = b"".join(struct.pack("<Q", d) for d in dimensions)
    return (
        string(name)
        + struct.pack("<I", len(dimensions))
        + packed
        + struct.pack("<IQ", ggml_type, 0)
    )


ARCHITECTURE = entry("general.architecture", STRING, string(b"llama"))


def parse(blob: bytes, tmp_path: Path) -> ModelInfo:
    path = tmp_path / "hostile.gguf"
    path.write_bytes(blob)
    return read_gguf(path)


# -- named cases: each one used to escape as something other than GGUFError -----------

REFUSED = {
    "a key claiming 2**63-1 bytes": b"GGUF"
    + struct.pack("<IQQ", 3, 0, 1)
    + struct.pack("<Q", 2**63 - 1),
    "a key claiming 2**64-1 bytes": b"GGUF"
    + struct.pack("<IQQ", 3, 0, 1)
    + struct.pack("<Q", 2**64 - 1),
    "a key claiming a terabyte": b"GGUF" + struct.pack("<IQQ", 3, 0, 1) + struct.pack("<Q", 2**40),
    "a byte array of 2**64-1 elements": header(
        [ARCHITECTURE, entry("x", ARRAY, struct.pack("<IQ", U8, 2**64 - 1))]
    ),
    "a string array of 2**40 elements": header(
        [ARCHITECTURE, entry("x", ARRAY, struct.pack("<IQ", STRING, 2**40))]
    ),
    "a block count of 2**40": header(
        [ARCHITECTURE, entry("llama.block_count", U64, struct.pack("<Q", 2**40))]
    ),
    "a block count of a billion": header(
        [ARCHITECTURE, entry("llama.block_count", U32, struct.pack("<I", 10**9))]
    ),
    "a negative block count": header(
        [ARCHITECTURE, entry("llama.block_count", I32, struct.pack("<i", -5))]
    ),
    "arrays nested 20000 deep": header(
        [
            ARCHITECTURE,
            entry("x", ARRAY, struct.pack("<IQ", ARRAY, 1) * 20000 + struct.pack("<IQ", U32, 0)),
        ]
    ),
    "a tensor with 2**32-1 dimensions": header(
        [ARCHITECTURE], string(b"t") + struct.pack("<I", 2**32 - 1) + bytes(64), 1
    ),
    "2**63 tensors": b"GGUF" + struct.pack("<IQQ", 3, 2**63, 0),
}


@pytest.mark.parametrize("blob", REFUSED.values(), ids=REFUSED.keys())
def test_a_lying_header_is_refused_as_a_gguf_error(blob: bytes, tmp_path: Path) -> None:
    with pytest.raises(GGUFError):
        parse(blob, tmp_path)


def test_the_largest_layer_count_we_accept_still_loads(tmp_path: Path) -> None:
    blob = header([ARCHITECTURE, entry("llama.block_count", U32, struct.pack("<I", MAX_LAYERS))])
    assert parse(blob, tmp_path).n_layers == MAX_LAYERS


@pytest.mark.parametrize(
    ("value_type", "payload"),
    [(F32, struct.pack("<f", float("inf"))), (F64, struct.pack("<d", float("nan")))],
    ids=["infinity", "nan"],
)
def test_a_nonsense_float_block_count_reads_as_missing(
    value_type: int, payload: bytes, tmp_path: Path
) -> None:
    """Not a number, so there is no layer count: the same as the key being absent."""
    info = parse(header([ARCHITECTURE, entry("llama.block_count", value_type, payload)]), tmp_path)
    assert info.n_layers == 0


def test_a_five_thousand_digit_layer_number_does_not_crash(tmp_path: Path) -> None:
    """Python refuses to parse integers past 4300 digits; a tensor name can ask it to."""
    blob = header([ARCHITECTURE], tensor(b"blk." + b"9" * 5000 + b".w", (4,)), 1)
    assert parse(blob, tmp_path).overhead_bytes > 0, "counted as a non-layer tensor"


def test_astronomical_tensor_dimensions_are_just_big_numbers(tmp_path: Path) -> None:
    info = parse(header([ARCHITECTURE], tensor(b"blk.0.w", (2**63,) * 4), 1), tmp_path)
    assert info.tensor_bytes > 0


def test_a_header_cut_short_anywhere_is_a_gguf_error(tmp_path: Path) -> None:
    whole = write_gguf(tmp_path / "whole.gguf", n_layers=3, vocab_size=5).read_bytes()
    for cut in range(len(whole)):
        # The header states how many tensors follow, so any cut leaves it short.
        with pytest.raises(GGUFError):
            parse(whole[:cut], tmp_path)


# -- fuzzing --------------------------------------------------------------------------


def valid_file() -> bytes:
    with tempfile.TemporaryDirectory() as directory:
        return write_gguf(Path(directory) / "base.gguf", n_layers=3, vocab_size=5).read_bytes()


VALID = valid_file()

# Numbers that sit on the edges of the fields a header uses for sizes.
EDGE_VALUES = [0, 1, 2, 7, 8, 255, 256, 1023, 1024, 1025, 2**16, 2**31 - 1, 2**31]
EDGE_VALUES += [2**32 - 1, 2**32, 2**40, 2**63 - 1, 2**63, 2**64 - 1]
EDGE = st.sampled_from(EDGE_VALUES)


def check(blob: bytes) -> None:
    """The contract: a ModelInfo, or a GGUFError, and nothing else."""
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "fuzz.gguf"
        path.write_bytes(blob)
        try:
            info = read_gguf(path)
        except GGUFError:
            return
    assert info.n_layers >= 0
    assert len(info.layer_bytes) == info.n_layers


FUZZ = settings(deadline=timedelta(seconds=2), max_examples=400)


@FUZZ
@given(st.binary(max_size=400))
def test_random_bytes_after_a_good_signature(tail: bytes) -> None:
    check(b"GGUF" + struct.pack("<I", 3) + tail)


@FUZZ
@given(
    st.lists(st.tuples(st.integers(0, len(VALID) - 1), st.integers(0, 255)), max_size=12),
    st.one_of(st.none(), st.integers(0, len(VALID))),
)
def test_a_good_file_with_bytes_changed_and_maybe_cut_short(
    changes: list[tuple[int, int]], cut: int | None
) -> None:
    damaged = bytearray(VALID)
    for position, value in changes:
        damaged[position] = value
    check(bytes(damaged if cut is None else damaged[:cut]))


@st.composite
def edge_soup(draw: st.DrawFn) -> bytes:
    """A plausible header whose sizes, counts and types are drawn from the edges."""
    pieces: list[bytes] = []
    for _ in range(draw(st.integers(0, 24))):
        kind = draw(st.integers(0, 4))
        if kind == 0:
            pieces.append(struct.pack("<Q", draw(EDGE)))
        elif kind == 1:
            pieces.append(struct.pack("<I", draw(EDGE) % 2**32))
        elif kind == 2:
            pieces.append(string(draw(st.binary(max_size=24))))
        elif kind == 3:
            pieces.append(string(b"llama.block_count"))
        else:
            pieces.append(draw(st.binary(max_size=16)))
    counts = struct.pack("<QQ", draw(EDGE), draw(st.integers(0, 6)))
    return b"GGUF" + struct.pack("<I", 3) + counts + b"".join(pieces)


@FUZZ
@given(edge_soup())
def test_headers_whose_sizes_are_all_extreme(blob: bytes) -> None:
    check(blob)
