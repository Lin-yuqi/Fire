"""TinyLlama-1.1B-Chat-v1.0 to .fire exporter scaffold.

The adapter is intentionally model-specific. It will validate one local
``config.json`` plus one ``model.safetensors``, produce canonical TensorInfo
entries, and drive FireWriter's internal two-pass output sequence.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Mapping, Sequence

from fire_writer import TensorInfo


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
    raise ExportError("TinyLlama descriptor expansion is not implemented yet")


def _validate_config(config: Mapping[str, object]) -> None:
    """Validate the model-semantic config fields before output creation."""
    del config
    raise ExportError("TinyLlama config validation is not implemented yet")


def _build_export_entries(source_dir: Path) -> list[_ExportEntry]:
    """Read safetensors metadata and calculate every final byte offset."""
    del source_dir
    raise ExportError("TinyLlama metadata preflight is not implemented yet")


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
    del source_dir, output_path
    raise ExportError("TinyLlama export scaffold is present; export is not implemented yet")


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
