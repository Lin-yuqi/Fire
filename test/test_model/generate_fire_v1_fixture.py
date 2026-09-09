"""Generate the independent 260-byte Python/C++ .fire v1 wire fixture.

The implementation will use only Python's stdlib ``struct`` module and must
not import FireWriter. Keeping this oracle independent lets it catch matching
encoder/decoder mistakes later.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence


FIXTURE_SIZE = 260
DATA_OFFSET = 224
NORM_OFFSET = 224
ATTENTION_WQ_OFFSET = 236


def write_fixture(output_path: Path) -> None:
    """Write the two-tensor fixture described in model_export_v0_1.md."""
    del output_path
    raise NotImplementedError("the .fire v1 fixture encoder is not implemented yet")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    write_fixture(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
