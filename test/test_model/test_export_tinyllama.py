"""Contract tests for the TinyLlama exporter and FireWriter."""

from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
import struct
import sys
import tempfile
from types import TracebackType
from types import SimpleNamespace
from typing import Any, BinaryIO, Mapping, cast
import unittest
from unittest import mock

import numpy as np


TOOLS_DIR = Path(__file__).resolve().parents[2] / "tools"
sys.path.insert(0, str(TOOLS_DIR))

import export_tinyllama  # noqa: E402,F401
import fire_writer  # noqa: E402,F401


_VALID_CONFIG = {
    "architectures": ["LlamaForCausalLM"],
    "attention_bias": False,
    "hidden_act": "silu",
    "hidden_size": 2048,
    "intermediate_size": 5632,
    "max_position_embeddings": 2048,
    "model_type": "llama",
    "num_attention_heads": 32,
    "num_hidden_layers": 22,
    "num_key_value_heads": 4,
    "rms_norm_eps": 1e-5,
    "rope_scaling": None,
    "rope_theta": 10_000.0,
    "tie_word_embeddings": False,
    "torch_dtype": "bfloat16",
    "vocab_size": 32_000,
}

_LAYER_PROFILE = (
    (
        "model.layers.{layer}.input_layernorm.weight",
        "layers.{layer}.attention_norm.weight",
        (2_048,),
    ),
    (
        "model.layers.{layer}.self_attn.q_proj.weight",
        "layers.{layer}.attention.wq.weight",
        (2_048, 2_048),
    ),
    (
        "model.layers.{layer}.self_attn.k_proj.weight",
        "layers.{layer}.attention.wk.weight",
        (256, 2_048),
    ),
    (
        "model.layers.{layer}.self_attn.v_proj.weight",
        "layers.{layer}.attention.wv.weight",
        (256, 2_048),
    ),
    (
        "model.layers.{layer}.self_attn.o_proj.weight",
        "layers.{layer}.attention.wo.weight",
        (2_048, 2_048),
    ),
    (
        "model.layers.{layer}.post_attention_layernorm.weight",
        "layers.{layer}.ffn_norm.weight",
        (2_048,),
    ),
    (
        "model.layers.{layer}.mlp.gate_proj.weight",
        "layers.{layer}.feed_forward.w1.weight",
        (5_632, 2_048),
    ),
    (
        "model.layers.{layer}.mlp.down_proj.weight",
        "layers.{layer}.feed_forward.w2.weight",
        (2_048, 5_632),
    ),
    (
        "model.layers.{layer}.mlp.up_proj.weight",
        "layers.{layer}.feed_forward.w3.weight",
        (5_632, 2_048),
    ),
)


def _expected_profile() -> tuple[tuple[str, str, tuple[int, ...]], ...]:
    expected: list[tuple[str, str, tuple[int, ...]]] = [
        (
            "model.embed_tokens.weight",
            "tok_embeddings.weight",
            (32_000, 2_048),
        )
    ]
    for layer in range(22):
        expected.extend(
            (source.format(layer=layer), target.format(layer=layer), shape)
            for source, target, shape in _LAYER_PROFILE
        )
    expected.extend(
        (
            ("model.norm.weight", "norm.weight", (2_048,)),
            ("lm_head.weight", "output.weight", (32_000, 2_048)),
        )
    )
    return tuple(expected)


class _MetadataSlice:
    def __init__(self, dtype: str, shape: tuple[int, ...]) -> None:
        self._dtype = dtype
        self._shape = shape

    def get_dtype(self) -> str:
        return self._dtype

    def get_shape(self) -> tuple[int, ...]:
        return self._shape


class _MetadataHandle:
    def __init__(self, metadata: dict[str, tuple[str, tuple[int, ...]]]) -> None:
        self._metadata = metadata

    def __enter__(self) -> _MetadataHandle:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def keys(self) -> list[str]:
        return list(self._metadata)

    def get_slice(self, name: str) -> _MetadataSlice:
        dtype, shape = self._metadata[name]
        return _MetadataSlice(dtype, shape)


class _PayloadTensor:
    def __init__(
        self,
        value: np.ndarray,
        *,
        dtype: str = "torch.bfloat16",
        shape: tuple[int, ...] | None = None,
        element_size: int = 2,
    ) -> None:
        self._value = value
        self.dtype = dtype
        self.shape = tuple(value.shape) if shape is None else shape
        self._element_size = element_size
        self.float_calls = 0
        self.contiguous_calls = 0

    def float(self) -> _ConvertedPayloadTensor:
        self.float_calls += 1
        return _ConvertedPayloadTensor(self)

    def numel(self) -> int:
        return int(np.prod(self.shape))

    def element_size(self) -> int:
        return self._element_size


class _ConvertedPayloadTensor:
    def __init__(self, source: _PayloadTensor) -> None:
        self._source = source
        self._is_contiguous = False

    def contiguous(self) -> _ConvertedPayloadTensor:
        self._source.contiguous_calls += 1
        self._is_contiguous = True
        return self

    def numpy(self) -> np.ndarray:
        if not self._is_contiguous:
            raise AssertionError("FP32 payload must be made contiguous before NumPy view")
        return self._source._value


class _PayloadHandle:
    def __init__(
        self,
        tensors: Mapping[str, np.ndarray | _PayloadTensor],
    ) -> None:
        self._tensors = tensors

    def __enter__(self) -> _PayloadHandle:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def get_tensor(self, name: str) -> _PayloadTensor:
        tensor = self._tensors[name]
        if isinstance(tensor, _PayloadTensor):
            return tensor
        return _PayloadTensor(tensor)


class _OutputFile:
    """File-system boundary fake that can fail exactly at payload write."""

    def __init__(self, delegate: BinaryIO, failure: BaseException | None) -> None:
        self._delegate = delegate
        self._failure = failure

    def __enter__(self) -> _OutputFile:
        self._delegate.__enter__()
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        return self._delegate.__exit__(exception_type, exception, traceback)

    def tell(self) -> int:
        return self._delegate.tell()

    def write(self, data: bytes | bytearray | memoryview) -> int:
        if self._failure is not None and self.tell() >= 224:
            failure = self._failure
            self._failure = None
            raise failure
        return self._delegate.write(data)


class _OutputPath:
    def __init__(self, path: Path, failure: BaseException | None = None) -> None:
        self.path = path
        self.failure = failure
        self.open_calls: list[tuple[str, int | None]] = []
        self.last_delegate: BinaryIO | None = None

    def __str__(self) -> str:
        return str(self.path)

    def open(self, mode: str, buffering: int | None = None) -> _OutputFile:
        self.open_calls.append((mode, buffering))
        if buffering is None:
            delegate = self.path.open(mode)
        else:
            delegate = self.path.open(mode, buffering=buffering)
        self.last_delegate = cast(BinaryIO, delegate)
        return _OutputFile(self.last_delegate, self.failure)

    def unlink(self) -> None:
        if self.last_delegate is not None and not self.last_delegate.closed:
            raise AssertionError("partial output must be closed before unlink")
        self.path.unlink()


def _small_export_entries() -> list[SimpleNamespace]:
    return [
        SimpleNamespace(
            source_name="source.norm",
            tensor_info=fire_writer.TensorInfo(
                name="norm.weight",
                dtype=fire_writer.WireDType.FP32,
                shape=(3,),
                byte_offset=224,
                byte_size=12,
            ),
        ),
        SimpleNamespace(
            source_name="source.wq",
            tensor_info=fire_writer.TensorInfo(
                name="layers.0.attention.wq.weight",
                dtype=fire_writer.WireDType.FP32,
                shape=(2, 3),
                byte_offset=236,
                byte_size=24,
            ),
        ),
    ]


def _write_valid_config(source_dir: Path) -> None:
    source_dir.mkdir()
    (source_dir / "config.json").write_text(
        json.dumps(_VALID_CONFIG),
        encoding="utf-8",
    )


class FireWriterWireContractTest(unittest.TestCase):
    def test_writes_documented_header_directory_and_payload_layout(self) -> None:
        destination = BytesIO()
        entries = [
            fire_writer.TensorInfo(
                name="norm.weight",
                dtype=fire_writer.WireDType.FP32,
                shape=(3,),
                byte_offset=224,
                byte_size=12,
            ),
            fire_writer.TensorInfo(
                name="layers.0.attention.wq.weight",
                dtype=fire_writer.WireDType.FP32,
                shape=(2, 3),
                byte_offset=236,
                byte_size=24,
            ),
        ]

        fire_writer.FireWriter().write_header_and_directory(destination, entries)

        expected_header = struct.pack(
            "<8sIIQQ",
            b"FIRECKPT",
            1,
            2,
            32,
            224,
        )
        expected_directory = b"".join(
            (
                struct.pack(
                    "<64sQQIIBB6s",
                    b"norm.weight",
                    224,
                    12,
                    3,
                    0,
                    1,
                    1,
                    b"\0" * 6,
                ),
                struct.pack(
                    "<64sQQIIBB6s",
                    b"layers.0.attention.wq.weight",
                    236,
                    24,
                    2,
                    3,
                    1,
                    2,
                    b"\0" * 6,
                ),
            )
        )

        self.assertEqual(destination.tell(), 224)
        self.assertEqual(destination.getvalue(), expected_header + expected_directory)

        first = np.array([1.0, -2.0, 3.5], dtype=np.float32)
        second = np.array([[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float32)
        writer = fire_writer.FireWriter()
        writer.write_tensor(destination, entries[0], first)
        writer.write_tensor(destination, entries[1], second)

        self.assertEqual(destination.tell(), 260)
        self.assertEqual(
            destination.getvalue()[224:],
            struct.pack("<9f", 1.0, -2.0, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0),
        )

    def test_name_encoding_follows_the_wire_contract(self) -> None:
        destination = BytesIO()
        info = fire_writer.TensorInfo(
            name="a" * 63,
            dtype=fire_writer.WireDType.FP32,
            shape=(1,),
            byte_offset=128,
            byte_size=4,
        )

        fire_writer.FireWriter().write_header_and_directory(destination, [info])

        self.assertEqual(destination.getvalue()[32:96], b"a" * 63 + b"\0")

        for invalid_name in ("", "contains\0nul", "非ASCII", "a" * 64):
            with self.subTest(name=invalid_name):
                invalid = fire_writer.TensorInfo(
                    name=invalid_name,
                    dtype=fire_writer.WireDType.FP32,
                    shape=(1,),
                    byte_offset=128,
                    byte_size=4,
                )
                rejected = BytesIO()
                with self.assertRaises(ValueError):
                    fire_writer.FireWriter().write_header_and_directory(
                        rejected,
                        [invalid],
                    )
                self.assertEqual(rejected.getvalue(), b"")

    def test_directory_preflight_rejects_invalid_metadata_before_writing(self) -> None:
        valid = fire_writer.TensorInfo(
            name="norm.weight",
            dtype=fire_writer.WireDType.FP32,
            shape=(3,),
            byte_offset=128,
            byte_size=12,
        )
        invalid_cases = {
            "unsupported dtype": fire_writer.TensorInfo(
                "norm.weight",
                cast(fire_writer.WireDType, 2),
                (3,),
                128,
                12,
            ),
            "rank zero": fire_writer.TensorInfo(
                "norm.weight", fire_writer.WireDType.FP32, (), 128, 4
            ),
            "rank three": fire_writer.TensorInfo(
                "norm.weight", fire_writer.WireDType.FP32, (1, 1, 1), 128, 4
            ),
            "zero dimension": fire_writer.TensorInfo(
                "norm.weight", fire_writer.WireDType.FP32, (0,), 128, 0
            ),
            "byte size mismatch": fire_writer.TensorInfo(
                "norm.weight", fire_writer.WireDType.FP32, (3,), 128, 8
            ),
            "byte offset mismatch": fire_writer.TensorInfo(
                "norm.weight", fire_writer.WireDType.FP32, (3,), 132, 12
            ),
            "shape outside uint32": fire_writer.TensorInfo(
                "norm.weight",
                fire_writer.WireDType.FP32,
                (1 << 32,),
                128,
                (1 << 32) * 4,
            ),
        }

        for label, invalid in invalid_cases.items():
            with self.subTest(case=label):
                destination = BytesIO()
                with self.assertRaises(ValueError):
                    fire_writer.FireWriter().write_header_and_directory(
                        destination,
                        [invalid],
                    )
                self.assertEqual(destination.getvalue(), b"")

        duplicate_destination = BytesIO()
        first = fire_writer.TensorInfo(
            name=valid.name,
            dtype=valid.dtype,
            shape=valid.shape,
            byte_offset=224,
            byte_size=valid.byte_size,
        )
        duplicate = fire_writer.TensorInfo(
            name=valid.name,
            dtype=valid.dtype,
            shape=valid.shape,
            byte_offset=236,
            byte_size=valid.byte_size,
        )
        with self.assertRaises(ValueError):
            fire_writer.FireWriter().write_header_and_directory(
                duplicate_destination,
                [first, duplicate],
            )
        self.assertEqual(duplicate_destination.getvalue(), b"")

    def test_tensor_payload_validation_happens_before_writing(self) -> None:
        info = fire_writer.TensorInfo(
            name="norm.weight",
            dtype=fire_writer.WireDType.FP32,
            shape=(3,),
            byte_offset=128,
            byte_size=12,
        )
        non_contiguous = np.arange(6, dtype=np.float32).reshape(3, 2)[:, 0]
        invalid_cases: dict[
            str,
            tuple[fire_writer.TensorInfo, Any, int],
        ] = {
            "source dtype": (info, np.ones(3, dtype=np.float64), 128),
            "source shape": (info, np.ones(2, dtype=np.float32), 128),
            "source layout": (info, non_contiguous, 128),
            "file position": (info, np.ones(3, dtype=np.float32), 0),
            "wire dtype": (
                fire_writer.TensorInfo(
                    "norm.weight",
                    cast(fire_writer.WireDType, 2),
                    (3,),
                    128,
                    12,
                ),
                np.ones(3, dtype=np.float32),
                128,
            ),
        }

        for label, (case_info, tensor, position) in invalid_cases.items():
            with self.subTest(case=label):
                destination = BytesIO(b"\0" * position)
                before = destination.getvalue()
                with self.assertRaises(ValueError):
                    fire_writer.FireWriter().write_tensor(
                        destination,
                        case_info,
                        tensor,
                    )
                self.assertEqual(destination.getvalue(), before)


class TinyLlamaExportContractTest(unittest.TestCase):
    def test_descriptor_contains_the_exact_201_tensor_profile(self) -> None:
        actual = export_tinyllama._build_tinyllama_descriptor()
        expected = _expected_profile()

        self.assertEqual(len(actual), 201)
        self.assertEqual(
            tuple((item.source_name, item.fire_name, item.shape) for item in actual),
            expected,
        )
        self.assertEqual(len({item.fire_name for item in actual}), 201)
        self.assertTrue(
            all(
                item.fire_name.isascii()
                and "\0" not in item.fire_name
                and 1 <= len(item.fire_name.encode("ascii")) <= 63
                for item in actual
            )
        )
        payload_bytes = sum(
            4 * int(np.prod(shape, dtype=np.int64)) for _, _, shape in expected
        )
        self.assertEqual(payload_bytes, 4_400_193_536)
        self.assertEqual(32 + 201 * 96 + payload_bytes, 4_400_212_864)

    def test_preflight_failure_does_not_create_output(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            missing_config_output = root / "missing-config.fire"
            with self.assertRaises(export_tinyllama.ExportError):
                export_tinyllama.export_tinyllama(
                    root / "missing-source",
                    missing_config_output,
                )
            self.assertFalse(missing_config_output.exists())

            invalid_config_source = root / "invalid-config-source"
            invalid_config_source.mkdir()
            invalid_config = dict(_VALID_CONFIG)
            invalid_config["hidden_size"] = 4096
            (invalid_config_source / "config.json").write_text(
                json.dumps(invalid_config),
                encoding="utf-8",
            )
            invalid_config_output = root / "invalid-config.fire"
            with self.assertRaisesRegex(
                export_tinyllama.ExportError,
                "hidden_size",
            ):
                export_tinyllama.export_tinyllama(
                    invalid_config_source,
                    invalid_config_output,
                )
            self.assertFalse(invalid_config_output.exists())

            expected = _expected_profile()
            base_metadata = {
                source_name: ("BF16", shape)
                for source_name, _, shape in expected
            }
            mutations = {}

            missing = dict(base_metadata)
            missing.pop("model.norm.weight")
            mutations["missing source tensor"] = missing

            extra = dict(base_metadata)
            extra["unexpected.weight"] = ("BF16", (1,))
            mutations["unexpected source tensor"] = extra

            wrong_dtype = dict(base_metadata)
            wrong_dtype["model.norm.weight"] = ("F32", (2_048,))
            mutations["dtype mismatch"] = wrong_dtype

            wrong_shape = dict(base_metadata)
            wrong_shape["model.norm.weight"] = ("BF16", (2_049,))
            mutations["shape mismatch"] = wrong_shape

            for index, (label, metadata) in enumerate(mutations.items()):
                with self.subTest(case=label):
                    source = root / f"metadata-source-{index}"
                    _write_valid_config(source)
                    (source / "model.safetensors").touch()
                    output = root / f"metadata-{index}.fire"
                    with mock.patch.object(
                        export_tinyllama,
                        "safe_open",
                        return_value=_MetadataHandle(metadata),
                    ):
                        with self.assertRaisesRegex(
                            export_tinyllama.ExportError,
                            label,
                        ):
                            export_tinyllama.export_tinyllama(source, output)
                    self.assertFalse(output.exists())

    def test_metadata_preflight_builds_fp32_entries_and_absolute_offsets(self) -> None:
        expected = _expected_profile()
        metadata = {
            source_name: ("BF16", shape)
            for source_name, _, shape in expected
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir)
            (source / "model.safetensors").touch()
            with mock.patch.object(
                export_tinyllama,
                "safe_open",
                return_value=_MetadataHandle(metadata),
            ):
                entries = export_tinyllama._build_export_entries(source)

        self.assertEqual(len(entries), 201)
        self.assertTrue(
            all(
                entry.tensor_info.dtype == fire_writer.WireDType.FP32
                for entry in entries
            )
        )
        self.assertEqual(entries[0].tensor_info.byte_offset, 19_328)
        self.assertEqual(
            entries[-1].tensor_info.byte_offset
            + entries[-1].tensor_info.byte_size,
            4_400_212_864,
        )

    def test_canonical_metadata_failure_precedes_output_creation(self) -> None:
        valid_entries = _small_export_entries()
        invalid_entries = {
            "name too long": [
                SimpleNamespace(
                    source_name=valid_entries[0].source_name,
                    tensor_info=fire_writer.TensorInfo(
                        name="a" * 64,
                        dtype=fire_writer.WireDType.FP32,
                        shape=(3,),
                        byte_offset=224,
                        byte_size=12,
                    ),
                ),
                valid_entries[1],
            ],
            "duplicate name": [
                valid_entries[0],
                SimpleNamespace(
                    source_name=valid_entries[1].source_name,
                    tensor_info=fire_writer.TensorInfo(
                        name=valid_entries[0].tensor_info.name,
                        dtype=fire_writer.WireDType.FP32,
                        shape=(2, 3),
                        byte_offset=236,
                        byte_size=24,
                    ),
                ),
            ],
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source"
            _write_valid_config(source)
            for index, (label, entries) in enumerate(invalid_entries.items()):
                with self.subTest(case=label):
                    output = root / f"invalid-canonical-{index}.fire"
                    with mock.patch.object(
                        export_tinyllama,
                        "_build_export_entries",
                        return_value=entries,
                    ):
                        with self.assertRaises(ValueError):
                            export_tinyllama.export_tinyllama(source, output)
                    self.assertFalse(output.exists())

    def test_existing_output_is_left_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source"
            _write_valid_config(source)
            output = root / "model.fire"
            output.write_bytes(b"keep this exact content")

            with mock.patch.object(
                export_tinyllama,
                "_build_export_entries",
                return_value=_small_export_entries(),
            ):
                with self.assertRaises(FileExistsError):
                    export_tinyllama.export_tinyllama(source, output)

            self.assertEqual(output.read_bytes(), b"keep this exact content")

    def test_payload_failure_removes_this_export_partial(self) -> None:
        self._assert_payload_failure_removes_partial(OSError("injected short write"))

    def test_keyboard_interrupt_removes_this_export_partial(self) -> None:
        self._assert_payload_failure_removes_partial(KeyboardInterrupt())

    def test_payload_pass_revalidates_source_dtype_shape_and_size(self) -> None:
        invalid_tensors = {
            "dtype": _PayloadTensor(
                np.ones(3, dtype=np.float32),
                dtype="torch.float32",
            ),
            "shape": _PayloadTensor(
                np.ones(3, dtype=np.float32),
                shape=(4,),
            ),
            "byte size": _PayloadTensor(
                np.ones(3, dtype=np.float32),
                element_size=4,
            ),
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source"
            _write_valid_config(source)
            for index, (label, invalid_tensor) in enumerate(invalid_tensors.items()):
                with self.subTest(case=label):
                    output = root / f"invalid-payload-{index}.fire"
                    tensors: dict[str, np.ndarray | _PayloadTensor] = {
                        "source.norm": invalid_tensor,
                        "source.wq": np.arange(6, dtype=np.float32).reshape(2, 3),
                    }
                    with mock.patch.object(
                        export_tinyllama,
                        "_build_export_entries",
                        return_value=_small_export_entries(),
                    ), mock.patch.object(
                        export_tinyllama,
                        "safe_open",
                        return_value=_PayloadHandle(tensors),
                    ):
                        with self.assertRaisesRegex(
                            export_tinyllama.ExportError,
                            label,
                        ):
                            export_tinyllama.export_tinyllama(source, output)
                    self.assertFalse(output.exists())

    def test_success_has_the_exact_expected_file_size(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source"
            _write_valid_config(source)
            physical_output = root / "model.fire"
            output = _OutputPath(physical_output)
            norm_source = _PayloadTensor(
                np.array([1.0, -2.0, 1.5], dtype=np.float32)
            )
            wq_source = _PayloadTensor(
                np.array(
                    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                    dtype=np.float32,
                )
            )
            tensors = {
                "source.norm": norm_source,
                "source.wq": wq_source,
            }

            with mock.patch.object(
                export_tinyllama,
                "_build_export_entries",
                return_value=_small_export_entries(),
            ), mock.patch.object(
                export_tinyllama,
                "safe_open",
                return_value=_PayloadHandle(tensors),
            ):
                export_tinyllama.export_tinyllama(source, cast(Path, output))

            self.assertEqual(output.open_calls, [("xb", 0)])
            self.assertEqual(norm_source.float_calls, 1)
            self.assertEqual(norm_source.contiguous_calls, 1)
            self.assertEqual(wq_source.float_calls, 1)
            self.assertEqual(wq_source.contiguous_calls, 1)
            self.assertEqual(physical_output.stat().st_size, 260)
            self.assertEqual(
                physical_output.read_bytes()[224:],
                struct.pack("<9f", 1.0, -2.0, 1.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
            )

    def _assert_payload_failure_removes_partial(
        self,
        failure: BaseException,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source"
            _write_valid_config(source)
            physical_output = root / "model.fire"
            output = _OutputPath(physical_output, failure)
            tensors = {
                "source.norm": np.array([1.0, -2.0, 1.5], dtype=np.float32),
                "source.wq": np.arange(6, dtype=np.float32).reshape(2, 3),
            }

            with mock.patch.object(
                export_tinyllama,
                "_build_export_entries",
                return_value=_small_export_entries(),
            ), mock.patch.object(
                export_tinyllama,
                "safe_open",
                return_value=_PayloadHandle(tensors),
            ):
                with self.assertRaises(type(failure)):
                    export_tinyllama.export_tinyllama(source, cast(Path, output))

            self.assertFalse(physical_output.exists())


if __name__ == "__main__":
    unittest.main()
