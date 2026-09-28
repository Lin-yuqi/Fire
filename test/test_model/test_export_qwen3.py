"""Contract tests for the dense Qwen3 Fire v1 exporter."""

from __future__ import annotations

from io import BytesIO, StringIO
import json
from pathlib import Path
import struct
import sys
import tempfile
from typing import Mapping
import unittest
from unittest import mock

import numpy as np
from safetensors.torch import save_file
import torch


TOOLS_DIR = Path(__file__).resolve().parents[2] / "tools"
sys.path.insert(0, str(TOOLS_DIR))

import export_qwen3  # noqa: E402
import fire_writer  # noqa: E402


class _MetadataSlice:
    def __init__(self, dtype: str, shape: tuple[int, ...]) -> None:
        self._dtype = dtype
        self._shape = shape

    def get_dtype(self) -> str:
        return self._dtype

    def get_shape(self) -> tuple[int, ...]:
        return self._shape


class _MetadataHandle:
    def __init__(self, metadata: Mapping[str, tuple[str, tuple[int, ...]]]) -> None:
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
    def __init__(self, value: np.ndarray) -> None:
        self._value = value
        self.dtype = "torch.bfloat16"
        self.shape = value.shape

    def numel(self) -> int:
        return int(np.prod(self.shape))

    def element_size(self) -> int:
        return 2

    def float(self) -> _ConvertedPayloadTensor:
        return _ConvertedPayloadTensor(self._value)


class _ConvertedPayloadTensor:
    def __init__(self, value: np.ndarray) -> None:
        self._value = value
        self._contiguous = False

    def contiguous(self) -> _ConvertedPayloadTensor:
        self._contiguous = True
        return self

    def numpy(self) -> np.ndarray:
        if not self._contiguous:
            raise AssertionError("payload must be contiguous")
        return self._value


class _PayloadHandle:
    def __init__(self, tensors: Mapping[str, np.ndarray]) -> None:
        self._tensors = tensors

    def __enter__(self) -> _PayloadHandle:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def get_tensor(self, name: str) -> _PayloadTensor:
        return _PayloadTensor(self._tensors[name])


def _metadata_for(
    patterns: tuple[export_qwen3._TensorPattern, ...],
) -> dict[str, tuple[str, tuple[int, ...]]]:
    return {
        pattern.source_name: (export_qwen3.EXPECTED_SOURCE_DTYPE, pattern.shape)
        for pattern in patterns
    }


def _write_config(source_dir: Path, profile: export_qwen3._Qwen3Profile) -> None:
    source_dir.mkdir()
    (source_dir / "config.json").write_text(
        json.dumps(dict(profile.expected_config)),
        encoding="utf-8",
    )


class Qwen3ProfileContractTest(unittest.TestCase):
    def test_selects_supported_profiles_from_semantic_config(self) -> None:
        for profile in (export_qwen3.QWEN3_0_6B, export_qwen3.QWEN3_8B):
            with self.subTest(profile=profile.model_id):
                config = dict(profile.expected_config)
                config["transformers_version"] = "incidental-field"
                self.assertIs(export_qwen3._select_profile(config), profile)

        invalid = dict(export_qwen3.QWEN3_0_6B.expected_config)
        invalid["head_dim"] = 64
        with self.assertRaisesRegex(export_qwen3.ExportError, "unsupported Qwen3 config"):
            export_qwen3._select_profile(invalid)

        missing_null_field = dict(export_qwen3.QWEN3_0_6B.expected_config)
        missing_null_field.pop("rope_scaling")
        with self.assertRaisesRegex(export_qwen3.ExportError, "unsupported Qwen3 config"):
            export_qwen3._select_profile(missing_null_field)

    def test_0_6b_descriptor_matches_cpp_loader_contract(self) -> None:
        profile = export_qwen3.QWEN3_0_6B
        descriptor = export_qwen3._build_descriptor(profile)
        by_source = {pattern.source_name: pattern for pattern in descriptor}

        self.assertEqual(len(descriptor), 311)
        self.assertEqual(len({pattern.fire_name for pattern in descriptor}), 311)
        self.assertEqual(profile.q_dim, 2_048)
        self.assertEqual(profile.kv_dim, 1_024)
        layer_zero = [
            (pattern.source_name, pattern.fire_name, pattern.shape)
            for pattern in descriptor
            if pattern.source_name.startswith("model.layers.0.")
        ]
        self.assertEqual(
            layer_zero,
            [
                (
                    "model.layers.0.input_layernorm.weight",
                    "layers.0.attention_norm.weight",
                    (1_024,),
                ),
                (
                    "model.layers.0.self_attn.q_proj.weight",
                    "layers.0.attention.wq.weight",
                    (2_048, 1_024),
                ),
                (
                    "model.layers.0.self_attn.k_proj.weight",
                    "layers.0.attention.wk.weight",
                    (1_024, 1_024),
                ),
                (
                    "model.layers.0.self_attn.v_proj.weight",
                    "layers.0.attention.wv.weight",
                    (1_024, 1_024),
                ),
                (
                    "model.layers.0.self_attn.o_proj.weight",
                    "layers.0.attention.wo.weight",
                    (1_024, 2_048),
                ),
                (
                    "model.layers.0.self_attn.q_norm.weight",
                    "layers.0.attention.q_norm.weight",
                    (128,),
                ),
                (
                    "model.layers.0.self_attn.k_norm.weight",
                    "layers.0.attention.k_norm.weight",
                    (128,),
                ),
                (
                    "model.layers.0.post_attention_layernorm.weight",
                    "layers.0.ffn_norm.weight",
                    (1_024,),
                ),
                (
                    "model.layers.0.mlp.gate_proj.weight",
                    "layers.0.feed_forward.w1.weight",
                    (3_072, 1_024),
                ),
                (
                    "model.layers.0.mlp.down_proj.weight",
                    "layers.0.feed_forward.w2.weight",
                    (1_024, 3_072),
                ),
                (
                    "model.layers.0.mlp.up_proj.weight",
                    "layers.0.feed_forward.w3.weight",
                    (3_072, 1_024),
                ),
            ],
        )
        self.assertEqual(
            by_source["model.layers.0.self_attn.q_proj.weight"],
            export_qwen3._TensorPattern(
                "model.layers.0.self_attn.q_proj.weight",
                "layers.0.attention.wq.weight",
                (2_048, 1_024),
            ),
        )
        self.assertEqual(
            by_source["model.layers.27.self_attn.o_proj.weight"].shape,
            (1_024, 2_048),
        )
        self.assertEqual(
            by_source["model.layers.0.self_attn.q_norm.weight"].shape,
            (128,),
        )
        self.assertEqual(
            by_source["model.layers.0.self_attn.k_norm.weight"].fire_name,
            "layers.0.attention.k_norm.weight",
        )
        self.assertEqual(profile.source_payload_bytes, 1_503_264_768)
        self.assertEqual(profile.fp32_payload_bytes, 3_006_529_536)
        self.assertEqual(profile.data_offset, 29_888)
        self.assertEqual(profile.fire_file_bytes, 3_006_559_424)

    def test_8b_descriptor_size_is_derived_from_the_same_profile_type(self) -> None:
        profile = export_qwen3.QWEN3_8B
        descriptor = export_qwen3._build_descriptor(profile)

        self.assertEqual(len(descriptor), 399)
        self.assertEqual(profile.source_payload_bytes, 16_381_470_720)
        self.assertEqual(profile.fp32_payload_bytes, 32_762_941_440)
        self.assertEqual(profile.data_offset, 38_336)
        self.assertEqual(profile.fire_file_bytes, 32_762_979_776)


class Qwen3SourceContractTest(unittest.TestCase):
    def test_single_file_0_6b_preflight_builds_exact_fp32_layout(self) -> None:
        profile = export_qwen3.QWEN3_0_6B
        descriptor = export_qwen3._build_descriptor(profile)
        metadata = _metadata_for(descriptor)

        with tempfile.TemporaryDirectory() as temp_dir:
            source_dir = Path(temp_dir)
            source_path = source_dir / "model.safetensors"
            source_path.touch()
            source_index = export_qwen3._build_source_index(source_dir, descriptor)

            with mock.patch.object(
                export_qwen3,
                "safe_open",
                return_value=_MetadataHandle(metadata),
            ) as safe_open:
                entries = export_qwen3._build_export_entries(
                    source_index,
                    descriptor,
                    profile,
                )

        safe_open.assert_called_once_with(source_path, framework="numpy")
        self.assertEqual(len(entries), 311)
        self.assertTrue(all(entry.shard_path == source_path for entry in entries))
        self.assertTrue(
            all(entry.tensor_info.dtype == fire_writer.WireDType.FP32 for entry in entries)
        )
        self.assertEqual(entries[0].tensor_info.byte_offset, 29_888)
        self.assertEqual(
            entries[-1].tensor_info.byte_offset + entries[-1].tensor_info.byte_size,
            3_006_559_424,
        )

    def test_sharded_preflight_opens_each_shard_once_and_groups_output(self) -> None:
        profile = export_qwen3.QWEN3_0_6B
        descriptor = export_qwen3._build_descriptor(profile)

        with tempfile.TemporaryDirectory() as temp_dir:
            source_dir = Path(temp_dir)
            shard_names = ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors")
            for shard_name in shard_names:
                (source_dir / shard_name).touch()

            weight_map = {
                pattern.source_name: shard_names[index % 2]
                for index, pattern in enumerate(descriptor)
            }
            (source_dir / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "metadata": {"total_size": profile.source_payload_bytes},
                        "weight_map": weight_map,
                    }
                ),
                encoding="utf-8",
            )
            source_index = export_qwen3._build_source_index(source_dir, descriptor)
            metadata_by_path = {
                source_dir / shard_name: {
                    pattern.source_name: ("BF16", pattern.shape)
                    for pattern in descriptor
                    if weight_map[pattern.source_name] == shard_name
                }
                for shard_name in shard_names
            }
            calls: list[Path] = []

            def fake_safe_open(path: Path, *, framework: str) -> _MetadataHandle:
                self.assertEqual(framework, "numpy")
                calls.append(path)
                return _MetadataHandle(metadata_by_path[path])

            with mock.patch.object(export_qwen3, "safe_open", side_effect=fake_safe_open):
                entries = export_qwen3._build_export_entries(
                    source_index,
                    descriptor,
                    profile,
                )

        first_path = source_dir / shard_names[0]
        second_path = source_dir / shard_names[1]
        self.assertEqual(calls, [first_path, second_path])
        transitions = sum(
            left.shard_path != right.shard_path
            for left, right in zip(entries, entries[1:])
        )
        self.assertEqual(transitions, 1)
        self.assertTrue(all(entry.shard_path == first_path for entry in entries[:156]))
        self.assertTrue(all(entry.shard_path == second_path for entry in entries[156:]))

    def test_index_rejects_tensor_set_path_and_size_errors(self) -> None:
        profile = export_qwen3.QWEN3_0_6B
        descriptor = export_qwen3._build_descriptor(profile)
        valid_map = {pattern.source_name: "model-00001.safetensors" for pattern in descriptor}

        cases: dict[str, tuple[dict[str, object], str]] = {}
        missing_map = dict(valid_map)
        missing_map.pop("model.norm.weight")
        cases["missing"] = ({"weight_map": missing_map}, "missing source tensors")
        extra_map = dict(valid_map)
        extra_map["unexpected.weight"] = "model-00001.safetensors"
        cases["extra"] = ({"weight_map": extra_map}, "unexpected source tensors")
        unsafe_map = dict(valid_map)
        unsafe_map["model.norm.weight"] = "../outside.safetensors"
        cases["unsafe"] = ({"weight_map": unsafe_map}, "unsafe shard path")
        cases["bad size"] = (
            {"metadata": {"total_size": True}, "weight_map": valid_map},
            "invalid metadata.total_size",
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            source_dir = Path(temp_dir)
            (source_dir / "model-00001.safetensors").touch()
            for label, (index, message) in cases.items():
                with self.subTest(case=label):
                    (source_dir / "model.safetensors.index.json").write_text(
                        json.dumps(index),
                        encoding="utf-8",
                    )
                    with self.assertRaisesRegex(export_qwen3.ExportError, message):
                        export_qwen3._build_source_index(source_dir, descriptor)

    def test_0_6b_metadata_failure_precedes_output_creation(self) -> None:
        profile = export_qwen3.QWEN3_0_6B
        descriptor = export_qwen3._build_descriptor(profile)
        metadata = _metadata_for(descriptor)
        metadata["model.layers.0.self_attn.q_proj.weight"] = ("BF16", (1_024, 1_024))

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_dir = root / "source"
            _write_config(source_dir, profile)
            (source_dir / "model.safetensors").touch()
            output_path = root / "qwen3.fire"

            with mock.patch.object(
                export_qwen3,
                "safe_open",
                return_value=_MetadataHandle(metadata),
            ):
                with self.assertRaisesRegex(export_qwen3.ExportError, "shape mismatch"):
                    export_qwen3.export_qwen3(source_dir, output_path)

            self.assertFalse(output_path.exists())


class Qwen3PayloadContractTest(unittest.TestCase):
    def test_payload_opens_each_shard_once_and_writes_in_directory_order(self) -> None:
        shard_a = Path("a.safetensors")
        shard_b = Path("b.safetensors")
        entries = [
            export_qwen3._ExportEntry(
                source_name="source.a",
                shard_path=shard_a,
                tensor_info=fire_writer.TensorInfo(
                    "tensor.a", fire_writer.WireDType.FP32, (2,), 224, 8
                ),
            ),
            export_qwen3._ExportEntry(
                source_name="source.b",
                shard_path=shard_b,
                tensor_info=fire_writer.TensorInfo(
                    "tensor.b", fire_writer.WireDType.FP32, (1, 2), 232, 8
                ),
            ),
        ]
        tensors = {
            shard_a: {"source.a": np.array([1.0, 2.0], dtype=np.float32)},
            shard_b: {"source.b": np.array([[3.0, 4.0]], dtype=np.float32)},
        }
        destination = BytesIO()
        writer = fire_writer.FireWriter()
        writer.write_header_and_directory(
            destination,
            [entry.tensor_info for entry in entries],
        )
        calls: list[Path] = []

        def fake_safe_open(
            path: Path,
            *,
            framework: str,
            device: str,
        ) -> _PayloadHandle:
            self.assertEqual((framework, device), ("pt", "cpu"))
            calls.append(path)
            return _PayloadHandle(tensors[path])

        with mock.patch.object(export_qwen3, "safe_open", side_effect=fake_safe_open):
            export_qwen3._write_payload(destination, entries, writer)

        self.assertEqual(calls, [shard_a, shard_b])
        self.assertEqual(destination.tell(), 240)
        self.assertEqual(
            np.frombuffer(destination.getvalue()[224:], dtype="<f4").tolist(),
            [1.0, 2.0, 3.0, 4.0],
        )

    def test_failure_after_output_creation_removes_partial_file(self) -> None:
        profile = export_qwen3.QWEN3_0_6B
        descriptor = export_qwen3._build_descriptor(profile)
        metadata = _metadata_for(descriptor)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_dir = root / "source"
            _write_config(source_dir, profile)
            (source_dir / "model.safetensors").touch()
            output_path = root / "qwen3.fire"

            with mock.patch.object(
                export_qwen3,
                "safe_open",
                return_value=_MetadataHandle(metadata),
            ), mock.patch.object(
                export_qwen3,
                "_write_payload",
                side_effect=OSError("injected payload failure"),
            ):
                with self.assertRaisesRegex(OSError, "injected payload failure"):
                    export_qwen3.export_qwen3(source_dir, output_path)

            self.assertFalse(output_path.exists())


class Qwen3V2ExportTest(unittest.TestCase):
    def test_int4_export_writes_quantized_linears_and_fp32_embedding(self) -> None:
        profile = export_qwen3._Qwen3Profile(
            model_id="test/Qwen3-tiny",
            expected_config={"model_type": "qwen3", "hidden_size": 128},
            vocab_size=8,
            hidden_size=128,
            intermediate_size=128,
            num_layers=1,
            num_attention_heads=1,
            num_kv_heads=1,
            head_dim=128,
        )
        descriptor = export_qwen3._build_descriptor(profile)
        metadata = _metadata_for(descriptor)
        tensors = {
            pattern.source_name: np.zeros(pattern.shape, dtype=np.float32)
            for pattern in descriptor
        }
        q_proj = tensors["model.layers.0.self_attn.q_proj.weight"]
        q_proj[0, 0] = -2.0
        q_proj[0, 1] = 1.0

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_dir = root / "source"
            _write_config(source_dir, profile)
            source_path = source_dir / "model.safetensors"
            source_path.touch()
            output_path = root / "tiny-int4.fire"

            def fake_safe_open(path: Path, *, framework: str, **kwargs: object):
                self.assertEqual(path, source_path)
                if framework == "numpy":
                    return _MetadataHandle(metadata)
                self.assertEqual((framework, kwargs), ("pt", {"device": "cpu"}))
                return _PayloadHandle(tensors)

            with mock.patch.object(export_qwen3, "SUPPORTED_PROFILES", (profile,)), mock.patch.object(
                export_qwen3, "safe_open", side_effect=fake_safe_open
            ):
                export_qwen3.export_qwen3(source_dir, output_path, quantization="int4")

            wire = output_path.read_bytes()

        magic, version, count, directory_offset, data_offset = struct.unpack_from(
            "<8sIIQQ", wire
        )
        self.assertEqual((magic, version, count, directory_offset, data_offset),
                         (b"FIRECKPT", 2, 30, 32, 2912))

        records = {}
        for index in range(count):
            offset = 32 + index * 96
            name = wire[offset:offset + 64].split(b"\0", 1)[0].decode("ascii")
            records[name] = struct.unpack_from("<QQIIBB6s", wire, offset + 64)

        self.assertEqual(records["tok_embeddings.weight"][4], 1)
        self.assertNotIn("layers.0.attention.wq.weight", records)
        qweight = records["layers.0.attention.wq.qweight"]
        scales = records["layers.0.attention.wq.scales"]
        zeros = records["layers.0.attention.wq.zero_points"]
        self.assertEqual(qweight[2:6], (128, 64, 2, 2))
        self.assertEqual(qweight[6], b"\x01\x80\x00\x00\x00\x00")
        self.assertEqual(scales[2:6], (128, 1, 1, 2))
        self.assertEqual(zeros[2:6], (128, 1, 2, 2))
        self.assertEqual(wire[qweight[0]], 0xF0)
        self.assertAlmostEqual(struct.unpack_from("<f", wire, scales[0])[0], 3 / 15)


class Qwen3AWQExportTest(unittest.TestCase):
    def test_8b_awq_descriptor_keeps_head_and_embedding_bf16(self) -> None:
        config = dict(export_qwen3.QWEN3_8B.expected_config)
        config["torch_dtype"] = "float16"
        config["quantization_config"] = {
            "quant_method": "awq", "version": "gemm", "bits": 4,
            "group_size": 128, "zero_point": True,
        }
        profile = export_qwen3._select_awq_profile(config)
        self.assertIs(profile, export_qwen3.QWEN3_8B)
        patterns = export_qwen3._build_awq_patterns(export_qwen3._build_descriptor(profile))
        by_name = {pattern.fire_name: pattern for pattern in patterns}
        self.assertEqual(len(patterns), 903)
        self.assertEqual(by_name["tok_embeddings.weight"].fire_dtype,
                         fire_writer.WireDType.BF16)
        self.assertEqual(by_name["output.weight"].fire_dtype,
                         fire_writer.WireDType.BF16)
        self.assertEqual(by_name["layers.0.attention.wq.qweight"].source_shape,
                         (4096, 512))
        self.assertEqual(by_name["layers.0.attention.wq.qweight"].fire_shape,
                         (4096, 2048))
        self.assertEqual(sum(name.endswith(".qweight") for name in by_name), 252)

    def test_awq_export_preserves_int4_values_and_bf16_bits(self) -> None:
        profile = export_qwen3._Qwen3Profile(
            model_id="test/Qwen3-AWQ-tiny",
            expected_config={"model_type": "qwen3", "hidden_size": 128},
            vocab_size=8, hidden_size=128, intermediate_size=128,
            num_layers=1, num_attention_heads=1, num_kv_heads=1, head_dim=128,
        )
        patterns = export_qwen3._build_awq_patterns(export_qwen3._build_descriptor(profile))
        tensors = {
            pattern.source_name: torch.zeros(
                pattern.source_shape,
                dtype=torch.int32 if pattern.source_dtype == "I32" else torch.bfloat16,
            )
            for pattern in patterns
        }
        awq_order = (0, 2, 4, 6, 1, 3, 5, 7)
        base = "model.layers.0.self_attn.q_proj"
        for k in range(128):
            values = [(k + out) % 16 for out in range(8)]
            word = sum(values[out] << (4 * index)
                       for index, out in enumerate(awq_order))
            tensors[base + ".qweight"][k, 0] = np.array(
                word, dtype=np.uint32
            ).view(np.int32).item()
        zero_word = sum((out + 8) << (4 * index)
                        for index, out in enumerate(awq_order))
        tensors[base + ".qzeros"][0, 0] = np.array(
            zero_word, dtype=np.uint32
        ).view(np.int32).item()
        tensors[base + ".scales"][0, :8] = 0.5
        tensors["model.embed_tokens.weight"].view(torch.uint16)[0, :2] = torch.tensor(
            [0x3F80, 0x4000], dtype=torch.uint16
        )
        tensors["lm_head.weight"].view(torch.uint16)[0, 0] = 0x7FC1
        tensors["model.norm.weight"][0] = 1.5

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_dir = root / "source"
            _write_config(source_dir, profile)
            config_path = source_dir / "config.json"
            config = json.loads(config_path.read_text())
            config["torch_dtype"] = "float16"
            config["quantization_config"] = {
                "quant_method": "awq", "version": "gemm", "bits": 4,
                "group_size": 128, "zero_point": True,
            }
            config_path.write_text(json.dumps(config))
            save_file(tensors, source_dir / "model.safetensors")
            source_bytes = sum(t.numel() * t.element_size() for t in tensors.values())
            (source_dir / "model.safetensors.index.json").write_text(json.dumps({
                "metadata": {"total_size": source_bytes + 2 * 1024 * 1024},
                "weight_map": {name: "model.safetensors" for name in tensors},
            }))
            output_path = root / "tiny-awq.fire"
            with mock.patch.object(export_qwen3, "SUPPORTED_PROFILES", (profile,)), \
                    mock.patch("sys.stderr", new_callable=StringIO) as stderr:
                export_qwen3.export_qwen3(source_dir, output_path, quantization="awq")
            self.assertIn("total_size differs", stderr.getvalue())
            wire = output_path.read_bytes()
            bad_tensors = dict(tensors)
            bad_tensors[base + ".scales"] = tensors[base + ".scales"].to(torch.float16)
            save_file(bad_tensors, source_dir / "model.safetensors")
            bad_output = root / "bad.fire"
            with mock.patch.object(export_qwen3, "SUPPORTED_PROFILES", (profile,)):
                with self.assertRaisesRegex(export_qwen3.ExportError, "dtype mismatch"):
                    export_qwen3.export_qwen3(source_dir, bad_output, quantization="awq")
            self.assertFalse(bad_output.exists())

        magic, version, count, _, _ = struct.unpack_from("<8sIIQQ", wire)
        self.assertEqual((magic, version, count), (b"FIRECKPT", 2, 28))
        records = {}
        for index in range(count):
            offset = 32 + index * 96
            name = wire[offset:offset + 64].split(b"\0", 1)[0].decode("ascii")
            records[name] = struct.unpack_from("<QQIIBB6s", wire, offset + 64)
        embedding = records["tok_embeddings.weight"]
        head = records["output.weight"]
        norm = records["norm.weight"]
        self.assertEqual((embedding[4], head[4], norm[4]), (3, 3, 1))
        self.assertEqual(struct.unpack_from("<2H", wire, embedding[0]), (0x3F80, 0x4000))
        self.assertEqual(struct.unpack_from("<H", wire, head[0])[0], 0x7FC1)
        self.assertEqual(struct.unpack_from("<f", wire, norm[0])[0], 1.5)

        qweight = records["layers.0.attention.wq.qweight"]
        scales = records["layers.0.attention.wq.scales"]
        zeros = records["layers.0.attention.wq.zero_points"]
        self.assertEqual(qweight[6], b"\x01\x80\x00\x00\x00\x00")
        self.assertEqual((wire[qweight[0]], wire[qweight[0] + 64]), (0x10, 0x21))
        self.assertEqual(wire[qweight[0] + 4], 0x98)
        self.assertEqual(wire[zeros[0]:zeros[0] + 8], bytes(range(8, 16)))
        self.assertEqual(struct.unpack_from("<f", wire, scales[0])[0], 0.5)

    def test_awq_rejects_unsupported_quantization_config(self) -> None:
        config = dict(export_qwen3.QWEN3_8B.expected_config)
        config["quantization_config"] = {
            "quant_method": "awq", "version": "gemv", "bits": 4,
            "group_size": 128, "zero_point": True,
        }
        with self.assertRaisesRegex(export_qwen3.ExportError, "quantization_config"):
            export_qwen3._select_awq_profile(config)


if __name__ == "__main__":
    unittest.main()
