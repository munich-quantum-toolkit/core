#!/usr/bin/env python3
# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Run the same bounded circuits in separate native Core or pinned QCEC environments."""

from __future__ import annotations

import argparse
import json
import math
import platform
from importlib import import_module
from importlib.metadata import version
from pathlib import Path

from qiskit import QuantumCircuit, qasm2

parser = argparse.ArgumentParser()
parser.add_argument("backend", choices=("native", "qcec"))
parser.add_argument("output", type=Path)
arguments = parser.parse_args()


def circuit(phase: float = 0.0) -> QuantumCircuit:
    """Construct the one-qubit X reference with explicit metadata phase.

    Returns:
        The X circuit with the supplied global phase in radians.
    """
    value = QuantumCircuit(1)
    value.x(0)
    value.global_phase = phase
    return value


cases = []
a = circuit()
cases.append(("identical", a, a.copy(), "exact", 1e-12, "equal"))
b = a.copy()
b.ry(0.17, 0)
cases.append(("wrong_angle", a, b, "exact", 1e-12, "different"))
b = circuit(0.37)
cases.extend((
    ("metadata_phase_preserve", a, b, "exact", 1e-12, "native different; QCEC metadata-phase blind spot"),
    ("metadata_phase_ignore", a, b, "phase_ignore", 1e-12, "equal modulo phase"),
))
b = a.copy()
for gate in ("x", "z", "x", "z"):
    getattr(b, gate)(0)
cases.append(("gate_phase_ignore", a, b, "phase_ignore", 1e-12, "equal modulo phase"))
b = a.copy()
b.ry(math.pi / 3, 0)
cases.extend(
    (
        f"process_boundary_{threshold}",
        a,
        b,
        "process",
        threshold,
        "D=0.5; below rejects, near is native inconclusive, above accepts",
    )
    for threshold in (0.49, 0.5, 0.51)
)
a = QuantumCircuit(2, 4)
a.x(0)
a.measure([0, 1], [0, 3])
for physical_fault, label_fault in ((False, False), (True, False), (False, True)):
    b = QuantumCircuit(2, 4)
    b.x(1 if physical_fault else 0)
    b.swap(0, 1)
    b.measure([0, 1], [0, 3] if label_fault else [3, 0])
    cases.append((
        f"sparse_routing_physical_{physical_fault}_label_{label_fault}",
        a,
        b,
        "terminal",
        1e-12,
        "different" if physical_fault or label_fault else "equal observed probabilities for every input",
    ))

rows = []
for name, left, right, relation, tolerance, expected in cases:
    row = {
        "name": name,
        "relation": relation,
        "tolerance": tolerance,
        "expected": expected,
        "left_qasm2": qasm2.dumps(left),
        "right_qasm2": qasm2.dumps(right),
        "left_global_phase": float(left.global_phase),
        "right_global_phase": float(right.global_phase),
    }
    if arguments.backend == "native":
        from mqt.core.mlir import (
            QCProgram,
            VerificationMethod,
            VerificationOptions,
            VerificationPhase,
            VerificationRelation,
            verify,
        )

        options = VerificationOptions()
        options.method = VerificationMethod.CONSTRUCTION
        options.tolerance = tolerance
        options.numerical_tolerance = 1e-12
        if relation in {"phase_ignore", "process"}:
            options.phase = VerificationPhase.IGNORE
        if relation == "process":
            options.relation = VerificationRelation.PROCESS_DISTANCE
        elif relation == "terminal":
            options.relation = VerificationRelation.TERMINAL_PROBABILITIES
        lhs = QCProgram.from_qiskit(left).to_qco()
        rhs = QCProgram.from_qiskit(right).to_qco()
        row["left_qco"] = str(lhs)
        row["right_qco"] = str(rhs)
        result = verify(lhs, rhs, options=options)
        row.update(
            verdict=result.verdict.name,
            reason=result.reason,
            distance=result.distance,
            dd_tolerance=result.dd_tolerance,
            output_labels=result.output_labels,
            checkers=[
                {"verdict": value.verdict.name, "reason": value.reason, "distance": value.distance}
                for value in result.checkers
            ],
        )
    else:
        qcec_verify = import_module("mqt.qcec").verify

        options = {
            "method": "construction",
            "parallel": False,
            "seed": 42,
            "run_zx_checker": False,
            "trace_threshold": 1e-12,
            "numerical_tolerance": 1e-12,
        }
        if relation == "process":
            options.update(check_approximate_equivalence=True, approximate_checking_threshold=tolerance)
        elif relation == "terminal":
            options["check_partial_equivalence"] = True
        result = qcec_verify(left, right, **options)
        row.update(options=options, verdict=str(result.equivalence).split(".")[-1], details=json.loads(str(result)))
    rows.append(row)

packages = ["mqt-core", "qiskit"] + (["mqt-qcec"] if arguments.backend == "qcec" else [])
arguments.output.write_text(
    json.dumps(
        {
            "backend": arguments.backend,
            "python": platform.python_version(),
            "versions": {name: version(name) for name in packages},
            "cases": rows,
        },
        indent=2,
    )
    + "\n"
)
