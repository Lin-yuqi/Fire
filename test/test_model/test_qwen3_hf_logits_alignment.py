"""Compare Fire Qwen3-0.6B logits with Hugging Face references."""

from __future__ import annotations

import argparse
import gc
import mmap
import os
from pathlib import Path
import struct
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
    parser.add_argument(
        "--quantized",
        action="store_true",
        help="compare GPU INT4 logits with the exported weights dequantized in Hugging Face",
    )
    return parser.parse_args()


def _load_fire_int4_weights(model, model_path: Path) -> None:
    """Replace HF linear weights with dequantized tensors from the Fire v2 file."""
    import numpy as np
    import torch

    repository = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repository / "tools"))
    from export_qwen3 import QWEN3_0_6B, _build_descriptor
    from quant import dequantize_int4_groupwise

    # Qwen3-0.6B ties these parameters in HF, while Fire keeps its embedding
    # FP32 and quantizes the output projection separately.
    if model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr():
        model.lm_head.weight = torch.nn.Parameter(
            model.lm_head.weight.detach().clone(), requires_grad=False
        )

    with model_path.open("rb") as file, mmap.mmap(
        file.fileno(), 0, access=mmap.ACCESS_READ
    ) as mapped:
        magic, version, tensor_count, directory_offset, _ = struct.unpack_from(
            "<8sIIQQ", mapped
        )
        if magic != b"FIRECKPT" or version != 2:
            raise AssertionError("quantized reference requires a Fire v2 file")

        entries = {}
        for index in range(tensor_count):
            name_bytes, offset, byte_size, dim0, dim1, wire_dtype, rank, metadata = (
                struct.unpack_from("<64sQQIIBB6s", mapped, directory_offset + 96 * index)
            )
            name = name_bytes.split(b"\0", 1)[0].decode("ascii")
            shape = (dim0,) if rank == 1 else (dim0, dim1)
            entries[name] = (offset, byte_size, shape, wire_dtype, metadata)

        expected_count = QWEN3_0_6B.tensor_count + 2 * (QWEN3_0_6B.num_layers * 7 + 1)
        if len(entries) != expected_count:
            raise AssertionError("unexpected number of Fire INT4 tensors")

        def fire_tensor(name: str):
            offset, byte_size, shape, wire_dtype, _ = entries[name]
            dtype = np.dtype("<f4") if wire_dtype == 1 else np.dtype("u1")
            array = np.frombuffer(mapped, dtype=dtype, count=int(np.prod(shape)), offset=offset)
            if array.nbytes != byte_size:
                raise AssertionError(f"Fire tensor byte size mismatch: {name}")
            return array.reshape(shape)

        quantized_count = 0
        for pattern in _build_descriptor(QWEN3_0_6B):
            parameter = model.get_parameter(pattern.source_name)
            base = pattern.fire_name.removesuffix(".weight")
            qweight_name = base + ".qweight"
            if qweight_name not in entries:
                expected = fire_tensor(pattern.fire_name)
                if not np.array_equal(parameter.detach().numpy(), expected):
                    raise AssertionError(f"FP32 Fire tensor differs: {pattern.fire_name}")
                del expected
                continue

            packed = fire_tensor(qweight_name)
            scales = fire_tensor(base + ".scales")
            zero_points = fire_tensor(base + ".zero_points")
            metadata = entries[qweight_name][4]
            if metadata[0] != 1:
                raise AssertionError(f"missing INT4 metadata: {qweight_name}")
            group_size = struct.unpack_from("<I", metadata, 1)[0]
            if tuple(parameter.shape) != (packed.shape[0], packed.shape[1] * 2):
                raise AssertionError(f"INT4 weight shape mismatch: {pattern.source_name}")

            with torch.no_grad():
                for start in range(0, packed.shape[0], 256):
                    end = min(start + 256, packed.shape[0])
                    weights = dequantize_int4_groupwise(
                        packed[start:end], scales[start:end], zero_points[start:end],
                        group_size=group_size,
                    )
                    parameter[start:end].copy_(torch.from_numpy(weights))
            quantized_count += 1
            del packed, scales, zero_points

        if quantized_count != QWEN3_0_6B.num_layers * 7 + 1:
            raise AssertionError("not all Qwen3 linear weights were quantized")


def _hugging_face_logits(model_path: Path, quantized_model_path: Path | None):
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

    quantized_reference = None
    if quantized_model_path is not None:
        _load_fire_int4_weights(model, quantized_model_path)
        with torch.inference_mode():
            logits = model(input_ids=input_ids, use_cache=False).logits[0]
        quantized_reference = logits.detach().cpu().numpy().astype(np.float32, copy=True)
        del logits

    del input_ids
    del model
    gc.collect()
    return reference, quantized_reference


def _fire_logits(runner: Path, model_path: Path, output_path: Path, *, quantized: bool):
    import numpy as np

    command = [str(runner), str(model_path), str(output_path)]
    if quantized:
        command.append("--gpu")
    command.extend(str(token_id) for token_id in _TOKEN_IDS)
    subprocess.run(command, check=True)

    expected_values = len(_TOKEN_IDS) * _VOCAB_SIZE
    logits = np.fromfile(output_path, dtype="<f4")
    if logits.size != expected_values:
        raise AssertionError(
            f"Fire runner wrote {logits.size} logits; expected {expected_values}"
        )
    return logits.reshape(len(_TOKEN_IDS), _VOCAB_SIZE)


def _report_quantization_effect(original, quantized) -> None:
    import numpy as np

    for position, token_id in enumerate(_TOKEN_IDS):
        baseline = original[position]
        quantized_row = quantized[position]
        absolute_error = np.abs(baseline - quantized_row)
        baseline64 = baseline.astype(np.float64)
        quantized64 = quantized_row.astype(np.float64)
        cosine = float(np.dot(baseline64, quantized64) /
                       (np.linalg.norm(baseline64) * np.linalg.norm(quantized64)))
        baseline_top10 = set(np.argpartition(baseline, -10)[-10:])
        quantized_top10 = set(np.argpartition(quantized_row, -10)[-10:])
        print(
            f"quantization position={position} token={token_id} "
            f"max_abs_error={float(absolute_error.max()):.8g} "
            f"mean_abs_error={float(absolute_error.mean()):.8g} "
            f"cosine={cosine:.8g} "
            f"argmax={int(np.argmax(baseline))}/{int(np.argmax(quantized_row))} "
            f"top10_overlap={len(baseline_top10 & quantized_top10)}/10"
        )


def _assert_aligned(reference, actual, *, reference_name: str) -> None:
    import numpy as np

    if reference.shape != actual.shape:
        raise AssertionError(
            f"logit shape mismatch: {reference_name} {reference.shape}, Fire {actual.shape}"
        )
    if not np.isfinite(reference).all() or not np.isfinite(actual).all():
        raise AssertionError("both implementations must produce only finite logits")

    for position, token_id in enumerate(_TOKEN_IDS):
        reference_row = reference[position]
        fire_row = actual[position]
        absolute_error = np.abs(reference_row - fire_row)
        expected_argmax = int(np.argmax(reference_row))
        fire_argmax = int(np.argmax(fire_row))
        print(
            f"alignment position={position} token={token_id} "
            f"max_abs_error={float(absolute_error.max()):.8g} "
            f"mean_abs_error={float(absolute_error.mean()):.8g} "
            f"argmax={expected_argmax}/{fire_argmax}"
        )
        if expected_argmax != fire_argmax:
            raise AssertionError(
                f"argmax mismatch at position {position}: "
                f"{reference_name} {expected_argmax}, Fire {fire_argmax}"
            )

    np.testing.assert_allclose(
        actual,
        reference,
        rtol=_RELATIVE_TOLERANCE,
        atol=_ABSOLUTE_TOLERANCE,
        err_msg=f"Fire Qwen3 logits differ from {reference_name} logits",
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

    original, quantized_reference = _hugging_face_logits(
        args.hf_model, args.fire_model if args.quantized else None
    )
    with tempfile.TemporaryDirectory(prefix="fire-qwen3-hf-logits-") as temporary_directory:
        output_path = Path(temporary_directory) / "fire_logits.bin"
        actual = _fire_logits(args.runner, args.fire_model, output_path,
                              quantized=args.quantized)
    if args.quantized:
        _report_quantization_effect(original, quantized_reference)
        _assert_aligned(quantized_reference, actual,
                        reference_name="quantized Hugging Face")
    else:
        _assert_aligned(original, actual, reference_name="Hugging Face")
    return 0


if __name__ == "__main__":
    sys.exit(main())
