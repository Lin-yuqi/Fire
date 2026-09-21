"""Model-independent INT4 group-wise quantization primitives.

This module owns quantization math and UInt4 packing only. Qwen3 tensor
selection and ``.fire`` serialization belong to their respective exporters.
The encoding details must be fixed by contract tests before implementation.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def quantize_int4_groupwise(
    weights: NDArray[np.float32], *, group_size: int
) -> tuple[NDArray[np.uint8], NDArray[np.float32], NDArray[np.uint8]]:
    """Quantize a FP32 ``[out_features, in_features]`` weight matrix.

    Each output row is grouped along ``in_features``. ``group_size`` must
    divide that dimension. Return unpacked UInt4 values (stored as UInt8,
    with the same shape as ``weights``), FP32 scales, and UInt8 zero-points.
    Both metadata arrays have shape ``[out_features, in_features/group_size]``.

    The intended rule is asymmetric, calibration-free RTN with values in
    ``[0, 15]`` and ``weight ~= scale * (value - zero_point)``. Rounding uses
    NumPy ties-to-even. An all-zero group is encoded with
    q=0, scale=1, and zero-point=0.
    """
    # 1. 参数检查
    if not isinstance(weights, np.ndarray):
        raise TypeError("weights must be a numpy.ndarray")

    if weights.dtype != np.float32:
        raise TypeError(
            f"weights must have dtype float32, got {weights.dtype}"
        )

    if weights.ndim != 2:
        raise ValueError(
            f"weights must be 2-D, got rank {weights.ndim}"
        )

    if group_size <= 0:
        raise ValueError(
            f"group_size must be positive, got {group_size}"
        )

    out_features, in_features = weights.shape

    if in_features % group_size != 0:
        raise ValueError(
            f"in_features ({in_features}) must be divisible "
            f"by group_size ({group_size})"
        )

    if not np.all(np.isfinite(weights)):
        raise ValueError("weights must contain only finite values")
    # 2. [O, K] -> [O, G, group_size]
    group_count =  in_features // group_size
    groups = weights.reshape(out_features,group_count,group_size)
    
    # 3. 每个 group 求范围，并保证 0 可表示
    group_min = np.min(groups,axis=2)
    group_max = np.max(groups,axis=2)
    
    rmin = np.minimum(group_min, np.float32(0.0))
    rmax = np.maximum(group_max, np.float32(0.0))
    
    # 4. scale = (max - min) / 15
    scales = (rmax - rmin) / np.float32(15.0)
    
    # 特殊情况：整个 group 全是 0
    zero_range = scales == 0.0
    scales = scales.astype(np.float32, copy=False)
    scales[zero_range] = np.float32(1.0)
    
    # 5. zero_point = round(-rmin / scale)
    zero_points_fp = np.rint(-rmin / scales)
    zero_points_fp = np.clip(zero_points_fp, 0, 15)

    zero_points = zero_points_fp.astype(np.uint8)
    # 6. q = round(w / scale + zero_point)
    q = np.rint(
        groups / scales[..., None]
        + zero_points[..., None]
    )
    q = np.clip(q, 0, 15).astype(np.uint8)
    
    # 全 0 group 明确编码成 q=0, zp=0, scale=1
    if np.any(zero_range):
        q[zero_range] = 0
        zero_points[zero_range] = 0

    # 7. 恢复成逻辑 [O, K]
    q = q.reshape(out_features, in_features)

    return (
        np.ascontiguousarray(q),
        np.ascontiguousarray(scales, dtype=np.float32),
        np.ascontiguousarray(zero_points, dtype=np.uint8),
    )


def pack_uint4(values: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Pack two UInt4 values per byte along the last axis of a 2-D array.

    Input values must be in ``[0, 15]`` and the last dimension must be even.
    The result has shape ``[out_features, in_features/2]``. Nibble order is
    fixed as even column in the low nibble, odd column in the high nibble.
    """
    if not isinstance(values, np.ndarray):
        raise TypeError("values must be a numpy.ndarray")

    if values.dtype != np.uint8:
        raise TypeError(
            f"values must have dtype uint8, got {values.dtype}"
        )

    if values.ndim != 2:
        raise ValueError(
            f"values must be 2-D, got rank {values.ndim}"
        )

    if values.shape[1] % 2 != 0:
        raise ValueError(
            f"last dimension must be even, got {values.shape[1]}"
        )

    if np.any(values > 15):
        raise ValueError("UInt4 values must be in [0, 15]")

    low = values[:, 0::2]
    high = values[:, 1::2]

    packed = low | (high << 4)

    return np.ascontiguousarray(packed, dtype=np.uint8)



def dequantize_int4_groupwise(
    packed_weights: NDArray[np.uint8],
    scales: NDArray[np.float32],
    zero_points: NDArray[np.uint8],
    *,
    group_size: int,
) -> NDArray[np.float32]:
    """Recover FP32 weights from the canonical packed group-wise layout.

    ``packed_weights`` has shape ``[out_features, in_features/2]``; scales and
    zero-points have shape ``[out_features, in_features/group_size]``. The
    result has shape ``[out_features, in_features]`` and follows
    ``scale[o, g] * (q[o, k] - zero_point[o, g])`` for ``g = k // group_size``.
    """
        # 1. 基本类型检查
    if not isinstance(packed_weights, np.ndarray):
        raise TypeError("packed_weights must be a numpy.ndarray")
    if not isinstance(scales, np.ndarray):
        raise TypeError("scales must be a numpy.ndarray")
    if not isinstance(zero_points, np.ndarray):
        raise TypeError("zero_points must be a numpy.ndarray")

    if packed_weights.dtype != np.uint8:
        raise TypeError(
            f"packed_weights must have dtype uint8, got {packed_weights.dtype}"
        )
    if scales.dtype != np.float32:
        raise TypeError(
            f"scales must have dtype float32, got {scales.dtype}"
        )
    if zero_points.dtype != np.uint8:
        raise TypeError(
            f"zero_points must have dtype uint8, got {zero_points.dtype}"
        )

    # 2. rank 检查
    if packed_weights.ndim != 2:
        raise ValueError(
            f"packed_weights must be 2-D, got rank {packed_weights.ndim}"
        )
    if scales.ndim != 2:
        raise ValueError(
            f"scales must be 2-D, got rank {scales.ndim}"
        )
    if zero_points.ndim != 2:
        raise ValueError(
            f"zero_points must be 2-D, got rank {zero_points.ndim}"
        )

    if group_size <= 0:
        raise ValueError(
            f"group_size must be positive, got {group_size}"
        )

    out_features = packed_weights.shape[0]
    in_features = packed_weights.shape[1] * 2

    if in_features % group_size != 0:
        raise ValueError(
            f"in_features ({in_features}) must be divisible "
            f"by group_size ({group_size})"
        )

    group_count = in_features // group_size
    expected_metadata_shape = (out_features, group_count)

    if scales.shape != expected_metadata_shape:
        raise ValueError(
            f"scales shape must be {expected_metadata_shape}, "
            f"got {scales.shape}"
        )

    if zero_points.shape != expected_metadata_shape:
        raise ValueError(
            f"zero_points shape must be {expected_metadata_shape}, "
            f"got {zero_points.shape}"
        )

    if not np.all(np.isfinite(scales)):
        raise ValueError("scales must contain only finite values")

    if np.any(scales <= 0):
        raise ValueError("scales must be positive")

    if np.any(zero_points > 15):
        raise ValueError("zero_points must be in [0, 15]")
    low = packed_weights&np.uint8(0X0F)
    high = (packed_weights >> np.uint8(4)) & np.uint8(0x0F)
    
    q = np.empty(
        (out_features, in_features),
        dtype=np.uint8,
    )
    
    q[:, 0::2] = low
    q[:, 1::2] = high
    
    # 4. [O, K] -> [O, G, group_size]
    q_groups = q.reshape(
        out_features,
        group_count,
        group_size,
    )
    
    # 5. w = scale * (q - zero_point)
    #
    # 注意一定要先转 float32。
    # uint8 做 q - zero_point 会发生无符号下溢。
    dequantized = (
        scales[..., None]
        * (
            q_groups.astype(np.float32)
            - zero_points[..., None].astype(np.float32)
        )
    )

    # 6. [O, G, group_size] -> [O, K]
    return np.ascontiguousarray(
        dequantized.reshape(out_features, in_features),
        dtype=np.float32,
    )
    
