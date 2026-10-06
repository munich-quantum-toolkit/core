# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Execution capture consistency checks for the MQSF presentation."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import pytest


class ExecutionCaptureTest(unittest.TestCase):
    """Keep shot order, returned counts, and native evidence consistent."""

    def test_rejects_inconsistent_execution_evidence(self) -> None:
        """Reject independent evidence mismatches rather than accepting plausible counts."""
        script = Path(__file__).resolve().parents[3] / "presentations/mqsf2026/capture_execution.py"
        assert script.is_file(), "The execution capture is not implemented"
        spec = importlib.util.spec_from_file_location("capture_execution", script)
        assert spec is not None
        assert spec.loader is not None
        capture = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(capture)
        execution = {
            "shots": ["10", "00", "10"],
            "counts": {"10": 2, "00": 1},
            "num_shots": 3,
            "terminal_status": "DONE",
            "payload_identity_verified": True,
            "events": [
                {"time_ms": 0.0, "status": "QDMI_SUCCESS"},
                {"time_ms": 1.0, "status": "QDMI_SUCCESS"},
            ],
        }
        capture.validate_execution(execution)
        for field, invalid in (
            ("counts", {"10": 1, "00": 2}),
            ("terminal_status", "FAILED"),
            ("payload_identity_verified", False),
            ("payload_identity_verified", "false"),
            ("num_shots", 4),
        ):
            with (
                self.subTest(field=field),
                pytest.raises(ValueError, match=r"Execution must|Returned shot count|Ordered shots disagree"),
            ):
                capture.validate_execution(execution | {field: invalid})


if __name__ == "__main__":
    unittest.main()
