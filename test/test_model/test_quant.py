"""Public numerical contract for Fire group-wise UInt4 weights."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))

from quant import (  # noqa: E402
    dequantize_int4_groupwise,
    pack_uint4,
    quantize_int4_groupwise,
)


class QuantContractTest(unittest.TestCase):
    def test_even_column_occupies_low_nibble(self) -> None:
        values = np.array([[1, 2, 15, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(
            pack_uint4(values), np.array([[0x21, 0x0F]], dtype=np.uint8)
        )

    def test_zero_group_has_stable_encoding_and_round_trip(self) -> None:
        weights = np.zeros((1, 128), dtype=np.float32)
        q, scales, zero_points = quantize_int4_groupwise(weights, group_size=128)
        self.assertEqual(q.dtype, np.uint8)
        self.assertEqual(scales.tolist(), [[1.0]])
        self.assertEqual(zero_points.tolist(), [[0]])
        self.assertEqual(pack_uint4(q).tolist(), [[0] * 64])
        np.testing.assert_array_equal(
            dequantize_int4_groupwise(
                pack_uint4(q), scales, zero_points, group_size=128
            ),
            weights,
        )


if __name__ == "__main__":
    unittest.main()
