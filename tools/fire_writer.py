"""Minimal wire-only writer surface for a Fire v1 tensor container.

This module deliberately knows nothing about Hugging Face, safetensors, or a
model architecture. The encoding and payload-writing bodies are left for the
next implementation step.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import BinaryIO, Sequence


MAGIC = b"FIRECKPT"
FORMAT_VERSION = 1
HEADER_SIZE = 32
TENSOR_INFO_SIZE = 96


class WireDType(IntEnum):
    """Dtype values written to the .fire wire format."""

    FP32 = 1


@dataclass(frozen=True)
class TensorInfo:
    """One normalized tensor directory entry.

    ``byte_offset`` and ``byte_size`` are absolute byte quantities. ``shape``
    has one or two dimensions in format v1.
    """

    name: str
    dtype: WireDType
    shape: tuple[int, ...]
    byte_offset: int
    byte_size: int


class FireWriter:
    """Writes already-normalized tensor metadata and payloads.

    The TinyLlama exporter owns source validation, exclusive-create cleanup,
    and the safetensors loop. This class only owns .fire wire encoding.
    """

    def write_header_and_directory(
        self,
        destination: BinaryIO,
        tensors: Sequence[TensorInfo],
    ) -> None:
        """Write the fixed header and complete tensor directory."""
        raise NotImplementedError("FireWriter wire encoding is not implemented yet")

    def write_tensor(
        self,
        destination: BinaryIO,
        info: TensorInfo,
        tensor: object,
    ) -> None:
        """Append one normalized FP32, C-contiguous tensor payload."""
        raise NotImplementedError("FireWriter payload writing is not implemented yet")
