# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Check the boundaries between reported hardware and local compiler models."""

from __future__ import annotations

import json
import runpy
from pathlib import Path
from typing import Any

import pytest

SCRIPT = Path(__file__).resolve().parents[3] / "presentations/mqsf2026/capture_devices.py"


def test_emerald_disables_stale_calibration_edges(monkeypatch: pytest.MonkeyPatch) -> None:
    """A reported calibration must not add an edge absent from current topology."""
    monkeypatch.syspath_prepend(str(SCRIPT.parent))
    capture = runpy.run_path(str(SCRIPT))["braket_target"]
    capabilities = {
        "paradigm": {
            "qubitCount": 3,
            "nativeGateSet": ["prx", "cz"],
            "connectivity": {"connectivityGraph": {"1": ["2"], "2": ["1"], "3": []}},
        },
        "provider": {
            "properties": {
                "one_qubit": {"1": {"f1Q_simultaneous_RB": 0.99, "fRO": 0.98}},
                "two_qubit": {"1-2": {"fCZ": 0.97}, "2-3": {"fCZ": 0.96}},
            }
        },
        "service": {"updatedAt": "2026-10-09"},
    }
    raw = {
        "deviceCapabilities": json.dumps(capabilities),
        "deviceName": "Emerald",
        "deviceArn": "public-device-id",
        "deviceStatus": "ONLINE",
    }
    target = capture(raw)
    assert target["metadata"]["edges"] == [(0, 1)]
    assert target["metadata"]["sites"][0] == {"id": 0, "name": "QB1", "provider_id": 1}
    assert target["compiler_model"]["operations"][1]["siteOverrides"] == [{"sites": [0, 1], "fidelity": 0.97}]
    assert target["provenance"]["raw"] == raw


def test_metrics_follow_wire_dependencies_and_reject_control_flow(monkeypatch: pytest.MonkeyPatch) -> None:
    """Independent single-qubit gates occupy one layer; nested loops need a different metric."""
    monkeypatch.syspath_prepend(str(SCRIPT.parent))
    metrics = runpy.run_path(str(SCRIPT))["circuit_metrics"]
    circuit: dict[str, Any] = {
        "qubits": [{"id": 0}, {"id": 1}],
        "operations": [{"name": "h", "qubits": [0]}, {"name": "h", "qubits": [1]}, {"name": "cx", "qubits": [0, 1]}],
    }
    assert metrics(circuit) == {"operations": 3, "counts": {"h": 2, "cx": 1}, "depth": 2, "active_qubits": 2}
    with pytest.raises(ValueError, match="straight-line"):
        metrics({
            **circuit,
            "operations": [{"name": "for_loop", "qubits": [0], "blocks": [[{"name": "x", "qubits": [0]}]]}],
        })
