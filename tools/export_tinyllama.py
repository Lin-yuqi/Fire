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
from typing import Sequence

from fire_writer import TensorInfo


MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
EXPECTED_TENSOR_COUNT = 201
EXPECTED_FP32_PAYLOAD_BYTES = 4_400_193_536
EXPECTED_FIRE_FILE_BYTES = 4_400_212_864


class ExportError(RuntimeError):
    """A recoverable TinyLlama export failure suitable for CLI reporting."""


@dataclass(frozen=True)
class _ExportEntry:
    """Private link between one source tensor and one Fire directory entry."""

    source_name: str
    tensor_info: TensorInfo


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
