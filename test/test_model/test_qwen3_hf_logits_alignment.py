"""Compare Fire Qwen3-0.6B logits with the Hugging Face reference model."""

from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path
import subprocess
import sys
import tempfile


_RUN_ENVIRONMENT_VARIABLE = "FIRE_RUN_QWEN3_HF_ALIGNMENT_TEST"
_TOKEN_IDS = (151_643, 9_707)
_VOCAB_SIZE = 151_936
_ABSOLUTE_TOLERANCE = 1e-4
_RELATIVE_TOLERANCE = 1e-4


def _parse_args() -> argparse.Namespace:
    repository = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runner",
        type=Path,
        default=repository / "build" / "test" / "qwen3_logits_runner",
    )
    parser.add_argument(
        "--hf-model",
        type=Path,
        default=repository / "models" / "Qwen3-0.6B",
    )
    parser.add_argument(
        "--fire-model",
        type=Path,
        default=repository / "tmp" / "qwen3-0.6b.fire",
    )
    return parser.parse_args()


def _hugging_face_logits(model_path: Path):
    try:
        import numpy as np
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM
    except ImportError as error:
        raise RuntimeError(
            "the alignment test requires numpy, torch, and transformers"
        ) from error

    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    config._attn_implementation = "eager"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        local_files_only=True,
        dtype=torch.float32,
    )
    model.eval()

    input_ids = torch.tensor([_TOKEN_IDS], dtype=torch.long)
    with torch.inference_mode():
        logits = model(input_ids=input_ids, use_cache=False).logits[0]
    reference = logits.detach().cpu().numpy().astype(np.float32, copy=True)

    del logits
    del input_ids
    del model
    gc.collect()
    return reference


def _fire_logits(runner: Path, model_path: Path, output_path: Path):
    import numpy as np

    subprocess.run(
        [
            str(runner),
            str(model_path),
            str(output_path),
            *(str(token_id) for token_id in _TOKEN_IDS),
        ],
        check=True,
    )

    expected_values = len(_TOKEN_IDS) * _VOCAB_SIZE
    logits = np.fromfile(output_path, dtype="<f4")
    if logits.size != expected_values:
        raise AssertionError(
            f"Fire runner wrote {logits.size} logits; expected {expected_values}"
        )
    return logits.reshape(len(_TOKEN_IDS), _VOCAB_SIZE)


def _assert_aligned(reference, actual) -> None:
    import numpy as np

    if reference.shape != actual.shape:
        raise AssertionError(
            f"logit shape mismatch: Hugging Face {reference.shape}, Fire {actual.shape}"
        )
    if not np.isfinite(reference).all() or not np.isfinite(actual).all():
        raise AssertionError("both implementations must produce only finite logits")

    for position, token_id in enumerate(_TOKEN_IDS):
        hf_row = reference[position]
        fire_row = actual[position]
        absolute_error = np.abs(hf_row - fire_row)
        hf_argmax = int(np.argmax(hf_row))
        fire_argmax = int(np.argmax(fire_row))
        print(
            f"position={position} token={token_id} "
            f"max_abs_error={float(absolute_error.max()):.8g} "
            f"mean_abs_error={float(absolute_error.mean()):.8g} "
            f"argmax={hf_argmax}/{fire_argmax}"
        )
        if hf_argmax != fire_argmax:
            raise AssertionError(
                f"argmax mismatch at position {position}: "
                f"Hugging Face {hf_argmax}, Fire {fire_argmax}"
            )

    np.testing.assert_allclose(
        actual,
        reference,
        rtol=_RELATIVE_TOLERANCE,
        atol=_ABSOLUTE_TOLERANCE,
        err_msg="Fire Qwen3 logits differ from Hugging Face reference logits",
    )


def main() -> int:
    args = _parse_args()
    if os.getenv(_RUN_ENVIRONMENT_VARIABLE) is None:
        print(
            f"skipped: set {_RUN_ENVIRONMENT_VARIABLE}=1 to run the Hugging Face alignment test"
        )
        return 77

    for description, path in (
        ("Qwen3 logits runner", args.runner),
        ("Hugging Face checkpoint", args.hf_model),
        ("Fire model", args.fire_model),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{description} is unavailable: {path}")

    reference = _hugging_face_logits(args.hf_model)
    with tempfile.TemporaryDirectory(prefix="fire-qwen3-hf-logits-") as temporary_directory:
        output_path = Path(temporary_directory) / "fire_logits.bin"
        actual = _fire_logits(args.runner, args.fire_model, output_path)
    _assert_aligned(reference, actual)
    return 0


if __name__ == "__main__":
    sys.exit(main())
