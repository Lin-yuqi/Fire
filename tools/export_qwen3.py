"""Export supported dense Qwen3 checkpoints to a Fire v1 container.

The adapter recognizes an exact model profile from ``config.json`` and accepts
both a single ``model.safetensors`` file and Hugging Face sharded checkpoints.
Fire v1 currently stores FP32 payloads, so BF16 source tensors are converted
one at a time while each source shard is opened only once per export pass.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from io import BytesIO
import json
import math
from pathlib import Path
import sys
from typing import BinaryIO, Mapping, Sequence

from safetensors import safe_open

from fire_writer import (
    HEADER_SIZE,
    TENSOR_INFO_SIZE,
    FireWriter,
    QuantizationInfo,
    TensorInfo,
    WireDType,
)
from quant import pack_uint4, quantize_int4_groupwise


EXPECTED_SOURCE_DTYPE = "BF16"
INT4_GROUP_SIZE = 128
_LINEAR_SUFFIXES = (
    ".attention.wq.weight",
    ".attention.wk.weight",
    ".attention.wv.weight",
    ".attention.wo.weight",
    ".feed_forward.w1.weight",
    ".feed_forward.w2.weight",
    ".feed_forward.w3.weight",
)


class ExportError(RuntimeError):
    """A recoverable Qwen3 export failure suitable for CLI reporting."""


@dataclass(frozen=True)
class _Qwen3Profile:
    model_id: str
    expected_config: Mapping[str, object]
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_layers: int
    num_attention_heads: int
    num_kv_heads: int
    head_dim: int

    @property
    def q_dim(self) -> int:
        return self.num_attention_heads * self.head_dim

    @property
    def kv_dim(self) -> int:
        return self.num_kv_heads * self.head_dim

    @property
    def tensor_count(self) -> int:
        return self.num_layers * 11 + 3

    @property
    def element_count(self) -> int:
        per_layer = (
            self.hidden_size
            + self.q_dim * self.hidden_size
            + 2 * self.kv_dim * self.hidden_size
            + self.hidden_size * self.q_dim
            + 2 * self.head_dim
            + self.hidden_size
            + 3 * self.intermediate_size * self.hidden_size
        )
        return (
            2 * self.vocab_size * self.hidden_size
            + self.hidden_size
            + self.num_layers * per_layer
        )

    @property
    def source_payload_bytes(self) -> int:
        return self.element_count * 2

    @property
    def fp32_payload_bytes(self) -> int:
        return self.element_count * 4

    @property
    def data_offset(self) -> int:
        return HEADER_SIZE + self.tensor_count * TENSOR_INFO_SIZE

    @property
    def fire_file_bytes(self) -> int:
        return self.data_offset + self.fp32_payload_bytes


_COMMON_CONFIG: Mapping[str, object] = {
    "architectures": ["Qwen3ForCausalLM"],
    "attention_bias": False,
    "attention_dropout": 0.0,
    "head_dim": 128,
    "hidden_act": "silu",
    "max_position_embeddings": 40_960,
    "model_type": "qwen3",
    "num_key_value_heads": 8,
    "rms_norm_eps": 1e-6,
    "rope_scaling": None,
    "rope_theta": 1_000_000,
    "sliding_window": None,
    "torch_dtype": "bfloat16",
    "use_sliding_window": False,
    "vocab_size": 151_936,
}


def _profile_config(**overrides: object) -> Mapping[str, object]:
    config = dict(_COMMON_CONFIG)
    config.update(overrides)
    return config


QWEN3_0_6B = _Qwen3Profile(
    model_id="Qwen/Qwen3-0.6B",
    expected_config=_profile_config(
        hidden_size=1_024,
        intermediate_size=3_072,
        max_window_layers=28,
        num_attention_heads=16,
        num_hidden_layers=28,
        tie_word_embeddings=True,
    ),
    vocab_size=151_936,
    hidden_size=1_024,
    intermediate_size=3_072,
    num_layers=28,
    num_attention_heads=16,
    num_kv_heads=8,
    head_dim=128,
)

QWEN3_8B = _Qwen3Profile(
    model_id="Qwen/Qwen3-8B",
    expected_config=_profile_config(
        hidden_size=4_096,
        intermediate_size=12_288,
        max_window_layers=36,
        num_attention_heads=32,
        num_hidden_layers=36,
        tie_word_embeddings=False,
    ),
    vocab_size=151_936,
    hidden_size=4_096,
    intermediate_size=12_288,
    num_layers=36,
    num_attention_heads=32,
    num_kv_heads=8,
    head_dim=128,
)

SUPPORTED_PROFILES = (QWEN3_0_6B, QWEN3_8B)


@dataclass(frozen=True)
class _TensorPattern:
    source_name: str
    fire_name: str
    shape: tuple[int, ...]


@dataclass(frozen=True)
class _SourceIndex:
    weight_map: Mapping[str, Path]
    shard_paths: tuple[Path, ...]
    declared_source_bytes: int | None


@dataclass(frozen=True)
class _ExportEntry:
    source_name: str
    shard_path: Path
    tensor_info: TensorInfo


@dataclass(frozen=True)
class _V2ExportEntry:
    source_name: str
    shard_path: Path
    source_shape: tuple[int, ...]
    tensor_infos: tuple[TensorInfo, ...]


def _select_profile(config: Mapping[str, object]) -> _Qwen3Profile:
    for profile in SUPPORTED_PROFILES:
        if all(
            key in config and config[key] == value
            for key, value in profile.expected_config.items()
        ):
            return profile

    identity_keys = (
        "model_type",
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "vocab_size",
        "tie_word_embeddings",
    )
    identity = ", ".join(f"{key}={config.get(key)!r}" for key in identity_keys)
    raise ExportError(f"unsupported Qwen3 config: {identity}")


def _build_descriptor(profile: _Qwen3Profile) -> tuple[_TensorPattern, ...]:
    result = [
        _TensorPattern(
            "model.embed_tokens.weight",
            "tok_embeddings.weight",
            (profile.vocab_size, profile.hidden_size),
        )
    ]

    layer_patterns = (
        _TensorPattern(
            "model.layers.{layer}.input_layernorm.weight",
            "layers.{layer}.attention_norm.weight",
            (profile.hidden_size,),
        ),
        _TensorPattern(
            "model.layers.{layer}.self_attn.q_proj.weight",
            "layers.{layer}.attention.wq.weight",
            (profile.q_dim, profile.hidden_size),
        ),
        _TensorPattern(
            "model.layers.{layer}.self_attn.k_proj.weight",
            "layers.{layer}.attention.wk.weight",
            (profile.kv_dim, profile.hidden_size),
        ),
        _TensorPattern(
            "model.layers.{layer}.self_attn.v_proj.weight",
            "layers.{layer}.attention.wv.weight",
            (profile.kv_dim, profile.hidden_size),
        ),
        _TensorPattern(
            "model.layers.{layer}.self_attn.o_proj.weight",
            "layers.{layer}.attention.wo.weight",
            (profile.hidden_size, profile.q_dim),
        ),
        _TensorPattern(
            "model.layers.{layer}.self_attn.q_norm.weight",
            "layers.{layer}.attention.q_norm.weight",
            (profile.head_dim,),
        ),
        _TensorPattern(
            "model.layers.{layer}.self_attn.k_norm.weight",
            "layers.{layer}.attention.k_norm.weight",
            (profile.head_dim,),
        ),
        _TensorPattern(
            "model.layers.{layer}.post_attention_layernorm.weight",
            "layers.{layer}.ffn_norm.weight",
            (profile.hidden_size,),
        ),
        _TensorPattern(
            "model.layers.{layer}.mlp.gate_proj.weight",
            "layers.{layer}.feed_forward.w1.weight",
            (profile.intermediate_size, profile.hidden_size),
        ),
        _TensorPattern(
            "model.layers.{layer}.mlp.down_proj.weight",
            "layers.{layer}.feed_forward.w2.weight",
            (profile.hidden_size, profile.intermediate_size),
        ),
        _TensorPattern(
            "model.layers.{layer}.mlp.up_proj.weight",
            "layers.{layer}.feed_forward.w3.weight",
            (profile.intermediate_size, profile.hidden_size),
        ),
    )

    for layer in range(profile.num_layers):
        result.extend(
            _TensorPattern(
                pattern.source_name.format(layer=layer),
                pattern.fire_name.format(layer=layer),
                pattern.shape,
            )
            for pattern in layer_patterns
        )

    result.extend(
        (
            _TensorPattern(
                "model.norm.weight",
                "norm.weight",
                (profile.hidden_size,),
            ),
            _TensorPattern(
                "lm_head.weight",
                "output.weight",
                (profile.vocab_size, profile.hidden_size),
            ),
        )
    )

    if len(result) != profile.tensor_count:
        raise ExportError(
            "internal descriptor count mismatch for "
            f"{profile.model_id}: expected {profile.tensor_count}, got {len(result)}"
        )
    return tuple(result)


def _validate_shard_name(name: str) -> None:
    path = Path(name)
    if (
        not name
        or path.is_absolute()
        or path.name != name
        or "\\" in name
        or name in (".", "..")
    ):
        raise ExportError(f"unsafe shard path in model index: {name!r}")


def _build_source_index(
    source_dir: Path,
    descriptor: Sequence[_TensorPattern],
) -> _SourceIndex:
    expected_names = {pattern.source_name for pattern in descriptor}
    index_path = source_dir / "model.safetensors.index.json"

    if not index_path.is_file():
        source_path = source_dir / "model.safetensors"
        if not source_path.is_file():
            raise ExportError(
                "missing model.safetensors or model.safetensors.index.json in "
                f"{source_dir}"
            )
        return _SourceIndex(
            weight_map={name: source_path for name in expected_names},
            shard_paths=(source_path,),
            declared_source_bytes=None,
        )

    try:
        with index_path.open("r", encoding="utf-8") as file:
            index = json.load(file)
    except (OSError, json.JSONDecodeError) as error:
        raise ExportError(f"failed to read model index {index_path}: {error}") from error

    if not isinstance(index, dict) or not isinstance(index.get("weight_map"), dict):
        raise ExportError(f"invalid weight_map in model index: {index_path}")

    raw_weight_map = index["weight_map"]
    if not all(isinstance(name, str) and isinstance(shard, str)
               for name, shard in raw_weight_map.items()):
        raise ExportError(f"invalid weight_map entry in model index: {index_path}")

    actual_names = set(raw_weight_map)
    missing = expected_names - actual_names
    extra = actual_names - expected_names
    if missing:
        raise ExportError(f"missing source tensors in model index: {sorted(missing)}")
    if extra:
        raise ExportError(f"unexpected source tensors in model index: {sorted(extra)}")

    shard_paths: list[Path] = []
    resolved_weight_map: dict[str, Path] = {}
    seen_shards: set[Path] = set()
    for pattern in descriptor:
        shard_name = raw_weight_map[pattern.source_name]
        _validate_shard_name(shard_name)
        shard_path = source_dir / shard_name
        if not shard_path.is_file():
            raise ExportError(f"missing safetensors shard: {shard_path}")
        resolved_weight_map[pattern.source_name] = shard_path
        if shard_path not in seen_shards:
            seen_shards.add(shard_path)
            shard_paths.append(shard_path)

    declared_source_bytes: int | None = None
    metadata = index.get("metadata")
    if metadata is not None:
        if not isinstance(metadata, dict):
            raise ExportError(f"invalid metadata in model index: {index_path}")
        total_size = metadata.get("total_size")
        if total_size is not None:
            if not isinstance(total_size, int) or isinstance(total_size, bool) or total_size <= 0:
                raise ExportError(f"invalid metadata.total_size in model index: {index_path}")
            declared_source_bytes = total_size

    return _SourceIndex(
        weight_map=resolved_weight_map,
        shard_paths=tuple(shard_paths),
        declared_source_bytes=declared_source_bytes,
    )


def _build_export_entries(
    source_index: _SourceIndex,
    descriptor: Sequence[_TensorPattern],
    profile: _Qwen3Profile,
) -> list[_ExportEntry]:
    patterns_by_shard: dict[Path, list[_TensorPattern]] = defaultdict(list)
    for pattern in descriptor:
        patterns_by_shard[source_index.weight_map[pattern.source_name]].append(pattern)

    entries: list[_ExportEntry] = []
    next_offset = profile.data_offset
    source_bytes = 0

    for shard_path in source_index.shard_paths:
        patterns = patterns_by_shard[shard_path]
        expected_names = {pattern.source_name for pattern in patterns}
        with safe_open(shard_path, framework="numpy") as handle:
            actual_names = set(handle.keys())
            missing = expected_names - actual_names
            extra = actual_names - expected_names
            if missing:
                raise ExportError(
                    f"missing source tensors in {shard_path}: {sorted(missing)}"
                )
            if extra:
                raise ExportError(
                    f"unexpected source tensors in {shard_path}: {sorted(extra)}"
                )

            for pattern in patterns:
                tensor_slice = handle.get_slice(pattern.source_name)
                actual_dtype = tensor_slice.get_dtype()
                if actual_dtype != EXPECTED_SOURCE_DTYPE:
                    raise ExportError(
                        f"dtype mismatch for {pattern.source_name}: "
                        f"expected {EXPECTED_SOURCE_DTYPE}, got {actual_dtype}"
                    )

                actual_shape = tuple(tensor_slice.get_shape())
                if actual_shape != pattern.shape:
                    raise ExportError(
                        f"shape mismatch for {pattern.source_name}: "
                        f"expected {pattern.shape}, got {actual_shape}"
                    )

                elements = math.prod(pattern.shape)
                source_bytes += elements * 2
                byte_size = elements * 4
                entries.append(
                    _ExportEntry(
                        source_name=pattern.source_name,
                        shard_path=shard_path,
                        tensor_info=TensorInfo(
                            name=pattern.fire_name,
                            dtype=WireDType.FP32,
                            shape=pattern.shape,
                            byte_offset=next_offset,
                            byte_size=byte_size,
                        ),
                    )
                )
                next_offset += byte_size

    if len(entries) != profile.tensor_count:
        raise ExportError(
            f"tensor count mismatch for {profile.model_id}: "
            f"expected {profile.tensor_count}, got {len(entries)}"
        )
    if source_bytes != profile.source_payload_bytes:
        raise ExportError(
            f"BF16 source size mismatch for {profile.model_id}: "
            f"expected {profile.source_payload_bytes}, got {source_bytes}"
        )
    if (source_index.declared_source_bytes is not None
            and source_index.declared_source_bytes != source_bytes):
        raise ExportError(
            "model index total_size mismatch: "
            f"expected {source_bytes}, got {source_index.declared_source_bytes}"
        )
    if next_offset != profile.fire_file_bytes:
        raise ExportError(
            f"final .fire size mismatch for {profile.model_id}: "
            f"expected {profile.fire_file_bytes}, got {next_offset}"
        )
    return entries


def _load_config(source_dir: Path) -> Mapping[str, object]:
    config_path = source_dir / "config.json"
    if not config_path.is_file():
        raise ExportError(f"missing config.json: {config_path}")
    try:
        with config_path.open("r", encoding="utf-8") as file:
            config = json.load(file)
    except (OSError, json.JSONDecodeError) as error:
        raise ExportError(f"failed to read config {config_path}: {error}") from error
    if not isinstance(config, dict):
        raise ExportError(f"config root must be an object: {config_path}")
    return config


def _write_payload(
    destination: BinaryIO,
    entries: Sequence[_ExportEntry],
    writer: FireWriter,
) -> None:
    entries_by_shard: dict[Path, list[_ExportEntry]] = defaultdict(list)
    for entry in entries:
        entries_by_shard[entry.shard_path].append(entry)

    for shard_path, shard_entries in entries_by_shard.items():
        with safe_open(shard_path, framework="pt", device="cpu") as handle:
            for entry in shard_entries:
                tensor = handle.get_tensor(entry.source_name)
                actual_dtype = str(tensor.dtype)
                if actual_dtype not in ("torch.bfloat16", "bfloat16"):
                    raise ExportError(
                        f"payload dtype mismatch for {entry.source_name}: "
                        f"expected torch.bfloat16, got {actual_dtype}"
                    )

                actual_shape = tuple(tensor.shape)
                if actual_shape != entry.tensor_info.shape:
                    raise ExportError(
                        f"payload shape mismatch for {entry.source_name}: "
                        f"expected {entry.tensor_info.shape}, got {actual_shape}"
                    )

                expected_source_bytes = math.prod(actual_shape) * 2
                actual_source_bytes = tensor.numel() * tensor.element_size()
                if actual_source_bytes != expected_source_bytes:
                    raise ExportError(
                        f"payload byte size mismatch for {entry.source_name}: "
                        f"expected {expected_source_bytes}, got {actual_source_bytes}"
                    )

                fp32_tensor = tensor.float().contiguous()
                fp32 = fp32_tensor.numpy()
                writer.write_tensor(destination, entry.tensor_info, fp32)

                expected_position = (
                    entry.tensor_info.byte_offset + entry.tensor_info.byte_size
                )
                if destination.tell() != expected_position:
                    raise ExportError(
                        f"payload position mismatch for {entry.tensor_info.name}: "
                        f"expected {expected_position}, got {destination.tell()}"
                    )

                del fp32
                del fp32_tensor
                del tensor


def _build_int4_entries(
    source_index: _SourceIndex,
    descriptor: Sequence[_TensorPattern],
    profile: _Qwen3Profile,
) -> tuple[list[_V2ExportEntry], int]:
    # Reuse the source-checkpoint preflight, but compute v2 directory size and
    # payload offsets from the expanded set of physical tensors.
    source_entries = _build_export_entries(source_index, descriptor, profile)
    patterns = {pattern.source_name: pattern for pattern in descriptor}
    quantized_count = sum(
        pattern.fire_name == "output.weight"
        or pattern.fire_name.endswith(_LINEAR_SUFFIXES)
        for pattern in descriptor
    )
    output_count = len(descriptor) + 2 * quantized_count
    next_offset = HEADER_SIZE + output_count * TENSOR_INFO_SIZE
    result: list[_V2ExportEntry] = []

    for source in source_entries:
        pattern = patterns[source.source_name]
        shape = pattern.shape
        is_linear = pattern.fire_name == "output.weight" or pattern.fire_name.endswith(
            _LINEAR_SUFFIXES
        )
        definitions: tuple[tuple[str, WireDType, tuple[int, ...], QuantizationInfo | None], ...]
        if is_linear:
            out_features, in_features = shape
            if in_features % INT4_GROUP_SIZE != 0:
                raise ExportError(
                    f"INT4 group_size {INT4_GROUP_SIZE} does not divide K for "
                    f"{pattern.source_name}: {in_features}"
                )
            base = pattern.fire_name.removesuffix(".weight")
            group_shape = (out_features, in_features // INT4_GROUP_SIZE)
            definitions = (
                (base + ".qweight", WireDType.UINT8,
                 (out_features, in_features // 2), QuantizationInfo(INT4_GROUP_SIZE)),
                (base + ".scales", WireDType.FP32, group_shape, None),
                (base + ".zero_points", WireDType.UINT8, group_shape, None),
            )
        else:
            definitions = ((pattern.fire_name, WireDType.FP32, shape, None),)

        outputs = []
        for name, dtype, output_shape, quantization in definitions:
            next_offset = (next_offset + 3) & ~3
            byte_size = math.prod(output_shape) * (4 if dtype == WireDType.FP32 else 1)
            outputs.append(
                TensorInfo(name, dtype, output_shape, next_offset, byte_size, quantization)
            )
            next_offset += byte_size
        result.append(
            _V2ExportEntry(source.source_name, source.shard_path, shape, tuple(outputs))
        )

    return result, next_offset


def _write_int4_payload(
    destination: BinaryIO,
    entries: Sequence[_V2ExportEntry],
    writer: FireWriter,
) -> None:
    entries_by_shard: dict[Path, list[_V2ExportEntry]] = defaultdict(list)
    for entry in entries:
        entries_by_shard[entry.shard_path].append(entry)

    for shard_path, shard_entries in entries_by_shard.items():
        with safe_open(shard_path, framework="pt", device="cpu") as handle:
            for entry in shard_entries:
                tensor = handle.get_tensor(entry.source_name)
                if str(tensor.dtype) not in ("torch.bfloat16", "bfloat16"):
                    raise ExportError(f"payload dtype mismatch for {entry.source_name}")
                if tuple(tensor.shape) != entry.source_shape:
                    raise ExportError(f"payload shape mismatch for {entry.source_name}")
                expected_source_bytes = math.prod(entry.source_shape) * 2
                if tensor.numel() * tensor.element_size() != expected_source_bytes:
                    raise ExportError(f"payload byte size mismatch for {entry.source_name}")
                fp32_tensor = tensor.float().contiguous()
                fp32 = fp32_tensor.numpy()
                if len(entry.tensor_infos) == 3:
                    try:
                        unpacked, scales, zero_points = quantize_int4_groupwise(
                            fp32, group_size=INT4_GROUP_SIZE
                        )
                    except (TypeError, ValueError) as error:
                        raise ExportError(
                            f"cannot quantize {entry.source_name}: {error}"
                        ) from error
                    packed = pack_uint4(unpacked)
                    payloads = (packed, scales, zero_points)
                else:
                    payloads = (fp32,)
                for info, payload in zip(entry.tensor_infos, payloads):
                    writer.write_tensor(destination, info, payload)
                    if destination.tell() != info.byte_offset + info.byte_size:
                        raise ExportError(f"payload position mismatch for {info.name}")
                del payloads
                if len(entry.tensor_infos) == 3:
                    del unpacked, packed, scales, zero_points
                del fp32
                del fp32_tensor
                del tensor


def _export_qwen3_int4(source_dir: Path, output_path: Path) -> None:
    config = _load_config(source_dir)
    profile = _select_profile(config)
    descriptor = _build_descriptor(profile)
    source_index = _build_source_index(source_dir, descriptor)
    entries, expected_file_size = _build_int4_entries(source_index, descriptor, profile)
    tensor_infos = [info for entry in entries for info in entry.tensor_infos]
    writer = FireWriter(version=2)
    writer.write_header_and_directory(BytesIO(), tensor_infos)
    print(
        f"validated {len(entries)} source tensors and {len(tensor_infos)} Fire v2 "
        f"tensors for {profile.model_id} across {len(source_index.shard_paths)} shard(s)"
    )

    created = False
    completed = False
    try:
        with output_path.open("xb", buffering=0) as destination:
            created = True
            writer.write_header_and_directory(destination, tensor_infos)
            _write_int4_payload(destination, entries, writer)
            if destination.tell() != expected_file_size:
                raise ExportError(
                    f"final file size mismatch for {output_path}: "
                    f"expected {expected_file_size}, got {destination.tell()}"
                )
        completed = True
    finally:
        if created and not completed:
            try:
                output_path.unlink()
            except OSError as cleanup_error:
                print(
                    f"warning: failed to remove partial output {output_path}: "
                    f"{cleanup_error}", file=sys.stderr,
                )


def export_qwen3(
    source_dir: Path, output_path: Path, *, quantization: str = "none"
) -> None:
    """Validate and export one supported local Qwen3 checkpoint."""
    if sys.byteorder != "little":
        raise ExportError("Fire export requires a little-endian host")
    if quantization == "int4":
        _export_qwen3_int4(source_dir, output_path)
        return
    if quantization != "none":
        raise ExportError(f"unsupported quantization mode: {quantization}")

    config = _load_config(source_dir)
    profile = _select_profile(config)
    descriptor = _build_descriptor(profile)
    source_index = _build_source_index(source_dir, descriptor)
    entries = _build_export_entries(source_index, descriptor, profile)

    writer = FireWriter()
    tensor_infos = [entry.tensor_info for entry in entries]
    writer.write_header_and_directory(BytesIO(), tensor_infos)
    print(
        f"validated {len(entries)} tensors for {profile.model_id} "
        f"across {len(source_index.shard_paths)} shard(s)"
    )

    created = False
    completed = False
    try:
        with output_path.open("xb", buffering=0) as destination:
            created = True
            writer.write_header_and_directory(destination, tensor_infos)
            _write_payload(destination, entries, writer)
            if destination.tell() != profile.fire_file_bytes:
                raise ExportError(
                    f"final file size mismatch for {output_path}: "
                    f"expected {profile.fire_file_bytes}, got {destination.tell()}"
                )
        completed = True
    finally:
        if created and not completed:
            try:
                output_path.unlink()
            except OSError as cleanup_error:
                print(
                    f"warning: failed to remove partial output {output_path}: "
                    f"{cleanup_error}",
                    file=sys.stderr,
                )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export a supported Qwen3 checkpoint to Fire v1 or INT4 Fire v2",
    )
    parser.add_argument("output", type=Path, help="new .fire output path")
    parser.add_argument(
        "--hf",
        required=True,
        type=Path,
        help="local Qwen3 checkpoint directory",
    )
    parser.add_argument(
        "--quantization", choices=("none", "int4"), default="none",
        help="none writes FP32 Fire v1; int4 writes group-wise INT4 Fire v2",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        export_qwen3(args.hf, args.output, quantization=args.quantization)
    except ExportError as error:
        print(f"export failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
