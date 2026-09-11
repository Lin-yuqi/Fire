"""TinyLlama-1.1B-Chat-v1.0 to .fire exporter scaffold.

The adapter is intentionally model-specific. It will validate one local
``config.json`` plus one ``model.safetensors``, produce canonical TensorInfo
entries, and drive FireWriter's internal two-pass output sequence.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from io import BytesIO
import json
import math
from pathlib import Path
import sys
from typing import Mapping, Sequence

from safetensors import safe_open

from fire_writer import (
    HEADER_SIZE,
    TENSOR_INFO_SIZE,
    TensorInfo,
    WireDType,
    FireWriter
)


MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
EXPECTED_TENSOR_COUNT = 201
EXPECTED_FP32_PAYLOAD_BYTES = 4_400_193_536
EXPECTED_FIRE_FILE_BYTES = 4_400_212_864
EXPECTED_SOURCE_DTYPE = "BF16"
EXPECTED_HEAD_DIM = 64

# Only model-semantic fields are part of the v0.1 profile. Incidental fields
# such as transformers_version are deliberately excluded.
EXPECTED_CONFIG: Mapping[str, object] = {
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


class ExportError(RuntimeError):
    """A recoverable TinyLlama export failure suitable for CLI reporting."""


@dataclass(frozen=True)
class _TensorPattern:
    """One fixed HF-to-Fire name pattern and its expected source shape."""

    source_name: str
    fire_name: str
    shape: tuple[int, ...]


_EMBEDDING_TENSOR = _TensorPattern(
    "model.embed_tokens.weight",
    "tok_embeddings.weight",
    (32_000, 2_048),
)

_LAYER_TENSORS = (
    _TensorPattern(
        "model.layers.{layer}.input_layernorm.weight",
        "layers.{layer}.attention_norm.weight",
        (2_048,),
    ),
    _TensorPattern(
        "model.layers.{layer}.self_attn.q_proj.weight",
        "layers.{layer}.attention.wq.weight",
        (2_048, 2_048),
    ),
    _TensorPattern(
        "model.layers.{layer}.self_attn.k_proj.weight",
        "layers.{layer}.attention.wk.weight",
        (256, 2_048),
    ),
    _TensorPattern(
        "model.layers.{layer}.self_attn.v_proj.weight",
        "layers.{layer}.attention.wv.weight",
        (256, 2_048),
    ),
    _TensorPattern(
        "model.layers.{layer}.self_attn.o_proj.weight",
        "layers.{layer}.attention.wo.weight",
        (2_048, 2_048),
    ),
    _TensorPattern(
        "model.layers.{layer}.post_attention_layernorm.weight",
        "layers.{layer}.ffn_norm.weight",
        (2_048,),
    ),
    _TensorPattern(
        "model.layers.{layer}.mlp.gate_proj.weight",
        "layers.{layer}.feed_forward.w1.weight",
        (5_632, 2_048),
    ),
    _TensorPattern(
        "model.layers.{layer}.mlp.down_proj.weight",
        "layers.{layer}.feed_forward.w2.weight",
        (2_048, 5_632),
    ),
    _TensorPattern(
        "model.layers.{layer}.mlp.up_proj.weight",
        "layers.{layer}.feed_forward.w3.weight",
        (5_632, 2_048),
    ),
)

_FINAL_TENSORS = (
    _TensorPattern("model.norm.weight", "norm.weight", (2_048,)),
    _TensorPattern("lm_head.weight", "output.weight", (32_000, 2_048)),
)


@dataclass(frozen=True)
class _ExportEntry:
    """Private link between one source tensor and one Fire directory entry."""

    source_name: str
    tensor_info: TensorInfo


def _build_tinyllama_descriptor() -> tuple[_TensorPattern, ...]:
    """Expand the fixed patterns into 201 exact source/canonical entries."""
    result = []
    result.append(_EMBEDDING_TENSOR)

    for layer in range(22):
        for pattern in _LAYER_TENSORS :
            result.append(
                _TensorPattern(
                    source_name=pattern.source_name.format(layer=layer),
                    fire_name=pattern.fire_name.format(layer=layer),
                    shape=pattern.shape
                )
            )

    result.extend(_FINAL_TENSORS)
    return tuple(result)


def _validate_config(config: Mapping[str, object]) -> None:
    """Validate the model-semantic config fields before output creation."""
    for key,expected_val in EXPECTED_CONFIG.items():
        if key not in config :
            raise ExportError(f"missing config field: {key}")

        actual_value = config[key]

        if actual_value!=expected_val:
            raise ExportError(
                f"config mismatch for {key}: "
                f"expected {expected_val!r}, got {actual_value!r}"
            )   


def _build_export_entries(source_dir: Path) -> list[_ExportEntry]:
    """Read safetensors metadata and calculate every final byte offset."""
    source_path = source_dir / "model.safetensors"

    if not source_path.is_file():
        raise ExportError(f"missing model.safetensors: {source_path}")

    descriptor = _build_tinyllama_descriptor();

    if len(descriptor) != EXPECTED_TENSOR_COUNT:
        raise ExportError(
            f"internal descriptor count mismatch: "
            f"expected {EXPECTED_TENSOR_COUNT}, got {len(descriptor)}"
        )

    data_offset = HEADER_SIZE + len(descriptor) * TENSOR_INFO_SIZE
    next_offset = data_offset

    entries: list[_ExportEntry] = []

    with safe_open(source_path,framework = "numpy") as handle:
        expected_names = {
            pattern.source_name
            for pattern in descriptor
        }

        actual_names = set(handle.keys())

        missing = expected_names - actual_names
        extra = actual_names - expected_names

        if missing:
            raise ExportError(
                f"missing source tensors: {sorted(missing)}"
            )

        if extra:
            raise ExportError(
                f"unexpected source tensors: {sorted(extra)}"
            )

        for pattern in descriptor:
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

            byte_size = math.prod(pattern.shape) * 4

            info = TensorInfo(
                name=pattern.fire_name,
                dtype=WireDType.FP32,
                shape=pattern.shape,
                byte_offset=next_offset,
                byte_size=byte_size,
            )

            entries.append(_ExportEntry(source_name=pattern.source_name,tensor_info= info))

            next_offset+=byte_size

    payload_bytes = next_offset - data_offset
    if payload_bytes!=EXPECTED_FP32_PAYLOAD_BYTES:
            raise ExportError(
            f"FP32 payload size mismatch: "
            f"expected {EXPECTED_FP32_PAYLOAD_BYTES}, got {payload_bytes}"
        )

    if next_offset != EXPECTED_FIRE_FILE_BYTES:
        raise ExportError(
            f"final .fire size mismatch: "
            f"expected {EXPECTED_FIRE_FILE_BYTES}, got {next_offset}"
        )
    return entries
        




def export_tinyllama(source_dir: Path, output_path: Path) -> None:
    """Export the fixed TinyLlama v0.1 profile.

    Implementation sequence:
    1. Validate config and safetensors metadata without creating ``output``.
    2. Build the private ``_ExportEntry`` list and final byte offsets.
    3. Exclusive-create ``output`` and ask ``FireWriter`` for Header/Directory.
    4. Open one PyTorch ``safe_open`` context, convert/write one tensor at a
       time, and release each temporary tensor immediately.
    5. Remove the file best-effort after normal exceptions or ``Ctrl-C``.
    """
    if sys.byteorder != "little":
        raise ExportError("Fire v1 export requires a little-endian host")

    config_path = source_dir / "config.json"
    if not config_path.is_file():
        raise ExportError(f"missing config.json: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    _validate_config(config)

    entries = _build_export_entries(source_dir)

    print(f"validated {len(entries)} tensors")

    writer = FireWriter()
    tensor_infos = [entry.tensor_info for entry in entries]

    # FireWriter owns the canonical-name and wire-range checks. Exercise those
    # checks before exclusive-create so no metadata failure can leave output.
    writer.write_header_and_directory(BytesIO(), tensor_infos)

    created = False
    completed = False
    try:
        with output_path.open("xb", buffering=0) as dst:
            created = True
            writer.write_header_and_directory(dst, tensor_infos)

            with safe_open(
                source_dir / "model.safetensors",
                framework="pt",
                device="cpu",
            ) as handle:
                for entry in entries:
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
                    writer.write_tensor(dst, entry.tensor_info, fp32)

                    expected_position = (
                        entry.tensor_info.byte_offset + entry.tensor_info.byte_size
                    )
                    if dst.tell() != expected_position:
                        raise ExportError(
                            f"payload position mismatch for {entry.tensor_info.name}: "
                            f"expected {expected_position}, got {dst.tell()}"
                        )

                    del fp32
                    del fp32_tensor
                    del tensor

            expected_file_size = HEADER_SIZE + len(entries) * TENSOR_INFO_SIZE
            if entries:
                final_info = entries[-1].tensor_info
                expected_file_size = final_info.byte_offset + final_info.byte_size
            if dst.tell() != expected_file_size:
                raise ExportError(
                    f"final file size mismatch for {output_path}: "
                    f"expected {expected_file_size}, got {dst.tell()}"
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
        description=f"Export {MODEL_ID} to a Fire v1 tensor container",
    )
    parser.add_argument("output", type=Path, help="new .fire output path")
    parser.add_argument(
        "--hf",
        required=True,
        type=Path,
        help="local TinyLlama checkpoint directory",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        export_tinyllama(args.hf, args.output)
    except ExportError as error:
        print(f"export failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
