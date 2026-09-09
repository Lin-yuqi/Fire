"""Acceptance-test placeholders for the TinyLlama exporter and FireWriter."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest


TOOLS_DIR = Path(__file__).resolve().parents[2] / "tools"
sys.path.insert(0, str(TOOLS_DIR))

import export_tinyllama  # noqa: E402,F401
import fire_writer  # noqa: E402,F401


@unittest.skip("export implementation intentionally deferred")
class TinyLlamaExportContractTest(unittest.TestCase):
    def test_descriptor_contains_the_exact_201_tensor_profile(self) -> None:
        self.fail(
            "TODO: enable _build_tinyllama_descriptor and verify names, shapes, and bytes"
        )

    def test_preflight_failure_does_not_create_output(self) -> None:
        self.fail("TODO: cover config, key, dtype, shape, and name failures")

    def test_existing_output_is_left_unchanged(self) -> None:
        self.fail("TODO: verify exclusive-create behavior")

    def test_payload_failure_removes_this_export_partial(self) -> None:
        self.fail("TODO: inject an ordinary write failure")

    def test_keyboard_interrupt_removes_this_export_partial(self) -> None:
        self.fail("TODO: inject KeyboardInterrupt during payload writing")

    def test_success_has_the_exact_expected_file_size(self) -> None:
        self.fail("TODO: verify the final file position and length")


if __name__ == "__main__":
    unittest.main()
