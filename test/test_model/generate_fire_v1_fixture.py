"""Generate the independent 260-byte Python/C++ .fire v1 wire fixture.

The implementation will use only Python's stdlib ``struct`` module and must
not import FireWriter. Keeping this oracle independent lets it catch matching
encoder/decoder mistakes later.
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path
from typing import Sequence


FIXTURE_SIZE = 260
DATA_OFFSET = 224
NORM_OFFSET = 224
ATTENTION_WQ_OFFSET = 236

_HEADER = struct.Struct("<8sIIQQ")
_TENSOR_INFO = struct.Struct("<64sQQIIBB6s")
_ATTENTION_WQ_PAYLOAD = struct.Struct("<6f")


def _name_field(name: str) -> bytes:
    encoded = name.encode("ascii")
    if not encoded or len(encoded) > 63:
        raise ValueError(f"fixture tensor name has invalid length: {name!r}")
    return encoded + bytes(64 - len(encoded))


def _tensor_info(
    name: str,
    byte_offset: int,
    byte_size: int,
    shape: tuple[int, int],
    ndim: int,
) -> bytes:
    return _TENSOR_INFO.pack(
        _name_field(name),
        byte_offset,
        byte_size,
        shape[0],
        shape[1],
        1,  # Fire v1 wire dtype: FP32
        ndim,
        bytes(6),
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(f"invalid generated .fire fixture: {message}")


def _validate_fixture(fixture: bytes) -> None:
    """Validate literals at their specified wire offsets before writing."""
    _require(_HEADER.size == 32, "header encoder is not 32 bytes")
    _require(_TENSOR_INFO.size == 96, "TensorInfo encoder is not 96 bytes")
    _require(len(fixture) == FIXTURE_SIZE, "file size is not 260 bytes")

    _require(fixture[0:8] == b"FIRECKPT", "magic bytes differ")
    _require(struct.unpack_from("<I", fixture, 8)[0] == 1, "format version differs")
    _require(struct.unpack_from("<I", fixture, 12)[0] == 2, "tensor count differs")
    _require(struct.unpack_from("<Q", fixture, 16)[0] == 32, "directory offset differs")
    _require(
        struct.unpack_from("<Q", fixture, 24)[0] == DATA_OFFSET,
        "data offset differs",
    )

    _require(fixture[32:44] == b"norm.weight\0", "first tensor name differs")
    _require(
        struct.unpack_from("<QQII", fixture, 96) == (NORM_OFFSET, 12, 3, 0),
        "first tensor location, size, or shape differs",
    )
    _require(fixture[120:122] == b"\x01\x01", "first tensor dtype/rank differs")

    second_info = 32 + _TENSOR_INFO.size
    _require(
        fixture[second_info : second_info + 29]
        == b"layers.0.attention.wq.weight\0",
        "second tensor name differs",
    )
    _require(
        struct.unpack_from("<QQII", fixture, second_info + 64)
        == (ATTENTION_WQ_OFFSET, 24, 2, 3),
        "second tensor location, size, or shape differs",
    )
    _require(
        fixture[second_info + 88 : second_info + 90] == b"\x01\x02",
        "second tensor dtype/rank differs",
    )

    # These are the exact little-endian FP32 bit patterns for 1.0, -2.0,
    # and 1.5. 1.5 is exactly representable as BF16 (0x3fc0).
    _require(
        struct.unpack_from("<III", fixture, NORM_OFFSET)
        == (0x3F800000, 0xC0000000, 0x3FC00000),
        "rank-1 payload bit patterns differ",
    )
    _require(
        _ATTENTION_WQ_PAYLOAD.unpack_from(fixture, ATTENTION_WQ_OFFSET)
        == (1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
        "rank-2 payload is not the expected row-major matrix",
    )


def write_fixture(output_path: Path) -> None:
    """Write the two-tensor fixture described in model_export_v0_1.md."""
    header = _HEADER.pack(b"FIRECKPT", 1, 2, 32, DATA_OFFSET)
    directory = b"".join(
        (
            _tensor_info("norm.weight", NORM_OFFSET, 12, (3, 0), 1),
            _tensor_info(
                "layers.0.attention.wq.weight",
                ATTENTION_WQ_OFFSET,
                24,
                (2, 3),
                2,
            ),
        )
    )
    norm_payload = struct.pack("<III", 0x3F800000, 0xC0000000, 0x3FC00000)
    attention_wq_payload = _ATTENTION_WQ_PAYLOAD.pack(1, 2, 3, 4, 5, 6)
    fixture = header + directory + norm_payload + attention_wq_payload

    _validate_fixture(fixture)
    output_path.write_bytes(fixture)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    write_fixture(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
