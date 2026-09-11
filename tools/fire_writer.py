"""Minimal wire-only writer surface for a Fire v1 tensor container.

This module deliberately knows nothing about Hugging Face, safetensors, or a
model architecture. The encoding and payload-writing bodies are left for the
next implementation step.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import math
import struct
from typing import BinaryIO, Sequence

import numpy as np
from numpy.typing import NDArray

MAGIC = b"FIRECKPT"
FORMAT_VERSION = 1
HEADER_SIZE = 32
TENSOR_INFO_SIZE = 96
_UINT32_MAX = (1 << 32) - 1
_UINT64_MAX = (1 << 64) - 1

_HEADER_STRUCT = struct.Struct("<8sIIQQ")
assert _HEADER_STRUCT.size == HEADER_SIZE
_TENSOR_INFO_STRUCT = struct.Struct("<64sQQIIBB6s")
assert _TENSOR_INFO_STRUCT.size == TENSOR_INFO_SIZE


class WireDType(IntEnum):
    """Dtype values written to the .fire wire format."""

    FP32 = 1


def _encode_name(name: str) -> bytes:
    try:
        raw = name.encode("ascii")
    except UnicodeEncodeError as error:
        raise ValueError(f"tensor name must be ASCII: {name!r}") from error

    if not raw:
        raise ValueError("tensor name must not be empty")
    if b"\0" in raw:
        raise ValueError(f"tensor name must not contain NUL: {name!r}")
    if len(raw) >= 64:
        raise ValueError(f"tensor name too long: {name}")

    return raw + b"\0" * (64 - len(raw))

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
        if destination.tell() != 0:
            raise ValueError(
                "header must be written at the beginning of the file"
            )

        tensor_count = len(tensors)
        if tensor_count > _UINT32_MAX:
            raise ValueError(f"too many tensors for Fire v1: {tensor_count}")

        data_offset = HEADER_SIZE + tensor_count * TENSOR_INFO_SIZE
        if data_offset > _UINT64_MAX:
            raise ValueError(f"tensor directory exceeds uint64 range: {data_offset}")
        next_offset = data_offset

        seen_names: set[str] = set()
        encoded_names: list[bytes] = []

        # 1. 先校验所有 TensorInfo
        for info in tensors:
            if info.name in seen_names:
                raise ValueError(
                    f"duplicate tensor name: {info.name}"
                )
            seen_names.add(info.name)
            encoded_names.append(_encode_name(info.name))

            if info.dtype != WireDType.FP32:
                raise ValueError(
                    f"unsupported wire dtype for {info.name}: {info.dtype}"
                )

            if len(info.shape) not in (1, 2):
                raise ValueError(
                    f"invalid rank for {info.name}: {len(info.shape)}"
                )

            if any(dim <= 0 or dim > _UINT32_MAX for dim in info.shape):
                raise ValueError(
                    f"invalid shape for {info.name}: {info.shape}"
                )

            expected_size = math.prod(info.shape) * 4
            if expected_size > _UINT64_MAX:
                raise ValueError(
                    f"tensor byte_size exceeds uint64 range for {info.name}: "
                    f"{expected_size}"
                )

            if info.byte_size != expected_size:
                raise ValueError(
                    f"byte_size mismatch for {info.name}: "
                    f"expected {expected_size}, got {info.byte_size}"
                )

            if info.byte_offset != next_offset:
                raise ValueError(
                    f"byte_offset mismatch for {info.name}: "
                    f"expected {next_offset}, got {info.byte_offset}"
                )

            if next_offset > _UINT64_MAX - info.byte_size:
                raise ValueError(
                    f"tensor payload range exceeds uint64 for {info.name}"
                )
            next_offset += info.byte_size

        # 2. 写 Header
        header = _HEADER_STRUCT.pack(
            MAGIC,
            FORMAT_VERSION,
            tensor_count,
            HEADER_SIZE,
            data_offset,
        )

        written = destination.write(header)
        if written != len(header):
            raise OSError("failed to write complete Fire header")

        # 3. 写 Tensor Directory
        for info, encoded_name in zip(tensors, encoded_names):
            ndim = len(info.shape)

            shape0 = info.shape[0]
            shape1 = info.shape[1] if ndim == 2 else 0

            encoded = _TENSOR_INFO_STRUCT.pack(
                encoded_name,
                info.byte_offset,
                info.byte_size,
                shape0,
                shape1,
                int(info.dtype),
                ndim,
                b"\0" * 6,
            )

            written = destination.write(encoded)
            if written != len(encoded):
                raise OSError(
                    f"failed to write directory entry for {info.name}"
                )

        # 4. 确认现在刚好位于 payload 起点
        if destination.tell() != data_offset:
            raise OSError(
                f"directory ended at unexpected offset: "
                f"expected {data_offset}, got {destination.tell()}"
            )   
    
    def write_tensor(
        self,
        destination: BinaryIO,
        info: TensorInfo,
        tensor: NDArray[np.float32],
    ) -> None:
    # 1. Fire v1 当前只支持 FP32
        if info.dtype != WireDType.FP32:
            raise ValueError(
                f"unsupported wire dtype for {info.name}: {info.dtype}"
            )

        # 2. 检查实际 tensor dtype
        if tensor.dtype != np.dtype(np.float32):
            raise ValueError(
                f"dtype mismatch for {info.name}: "
                f"expected float32, got {tensor.dtype}"
            )

        # 3. 检查 shape
        if tuple(tensor.shape) != info.shape:
            raise ValueError(
                f"shape mismatch for {info.name}: "
                f"expected {info.shape}, got {tuple(tensor.shape)}"
            )

        # 4. 必须是连续内存
        if not tensor.flags.c_contiguous:
            raise ValueError(
                f"tensor is not C-contiguous: {info.name}"
            )

        # 5. 实际字节数必须和目录里记录的一样
        if tensor.nbytes != info.byte_size:
            raise ValueError(
                f"payload size mismatch for {info.name}: "
                f"expected {info.byte_size}, got {tensor.nbytes}"
            )

        # 6. 当前文件位置必须正好等于这个 tensor 的 offset
        current_offset = destination.tell()

        if current_offset != info.byte_offset:
            raise ValueError(
                f"write offset mismatch for {info.name}: "
                f"expected {info.byte_offset}, got {current_offset}"
            )

        # 7. 直接把 NumPy 的原始内存当作 bytes 写入
        payload = memoryview(tensor).cast("B")

        written = destination.write(payload)

        if written != len(payload):
            raise OSError(
                f"failed to write complete tensor payload for {info.name}: "
                f"expected {len(payload)} bytes, wrote {written}"
            )
