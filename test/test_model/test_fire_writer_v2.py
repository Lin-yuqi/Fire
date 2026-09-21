"""Independent byte-level contract for the public FireWriter v2 API."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
import struct
import sys
import unittest

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))

from fire_writer import FireWriter, QuantizationInfo, TensorInfo, WireDType  # noqa: E402


class FireWriterV2Test(unittest.TestCase):
    def test_writes_int4_triplet_with_zero_alignment_padding(self) -> None:
        records = [
            TensorInfo(
                "linear.qweight", WireDType.UINT8, (1, 1), 320, 1,
                quantization=QuantizationInfo(group_size=2),
            ),
            TensorInfo("linear.scales", WireDType.FP32, (1, 1), 324, 4),
            TensorInfo("linear.zero_points", WireDType.UINT8, (1, 1), 328, 1),
        ]
        output = BytesIO()
        writer = FireWriter(version=2)
        writer.write_header_and_directory(output, records)
        writer.write_tensor(output, records[0], np.array([[0x21]], dtype=np.uint8))
        writer.write_tensor(output, records[1], np.array([[0.5]], dtype=np.float32))
        writer.write_tensor(output, records[2], np.array([[1]], dtype=np.uint8))
        wire = output.getvalue()

        self.assertEqual(len(wire), 329)
        self.assertEqual(struct.unpack_from("<8sIIQQ", wire), (b"FIRECKPT", 2, 3, 32, 320))
        self.assertEqual(struct.unpack_from("<QQII", wire, 32 + 64), (320, 1, 1, 1))
        self.assertEqual(wire[32 + 88:32 + 96], b"\x02\x02\x01\x02\x00\x00\x00\x00")
        self.assertEqual(wire[320:329], b"\x21\x00\x00\x00\x00\x00\x00\x3f\x01")

    def test_v1_rejects_quantized_metadata(self) -> None:
        record = TensorInfo(
            "linear.qweight", WireDType.UINT8, (1, 1), 128, 1,
            quantization=QuantizationInfo(group_size=2),
        )
        with self.assertRaises(ValueError):
            FireWriter().write_header_and_directory(BytesIO(), [record])

    def test_rejects_group_larger_than_logical_input_width(self) -> None:
        record = TensorInfo(
            "linear.qweight", WireDType.UINT8, (1, 1), 128, 1,
            quantization=QuantizationInfo(group_size=4),
        )
        with self.assertRaisesRegex(ValueError, "group_size"):
            FireWriter(version=2).write_header_and_directory(BytesIO(), [record])


if __name__ == "__main__":
    unittest.main()
