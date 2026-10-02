# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Compare ancilla-free Core/Qiskit rotation synthesis in a common u,cx basis.

Run with the locally built package, for example::

    uv run --no-sync python test/bench/compare_controlled_rotations.py \
        --output build/bench/controlled-rotations.csv

Timing excludes input preparation, import/export, basis conversion, and routing.
Both outputs receive the same basis conversion and level-3 optimization, with
no assumption that input qubits start in zero. Small cases also check the
phase-sensitive operator, including symbolic binding at 2*pi.
"""

# Standalone benchmark executable; this directory is not a Python package.
# ruff: file-ignore[implicit-namespace-package]

from __future__ import annotations

import argparse
import csv
import hashlib
import logging
from importlib.metadata import version
from math import pi
from pathlib import Path
from statistics import median
from time import perf_counter_ns

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import AnnotatedOperation, ControlModifier, Parameter
from qiskit.circuit.library import RXGate, RYGate, RZGate
from qiskit.quantum_info import Operator

from mqt.core.mlir import QCOProgram, QCProgram

LOGGER = logging.getLogger(__name__)


def synthesize_core(source: str, samples: int) -> tuple[QuantumCircuit, float]:
    """Measure decomposition of fresh copies, excluding one warmup.

    Returns:
        The synthesized circuit and median synthesis time in milliseconds.
    """
    timings = []
    for _ in range(samples + 1):
        program = QCOProgram.from_mlir_str(source)
        start = perf_counter_ns()
        program.decompose_multi_controlled()
        timings.append((perf_counter_ns() - start) / 1e6)
    return program.to_qc().to_qiskit(), median(timings[1:])


def synthesize_qiskit(axis: str, controls: int, angle: float | Parameter, samples: int) -> tuple[QuantumCircuit, float]:
    """Measure the public no-ancilla synthesis methods, excluding one warmup.

    Returns:
        The synthesized circuit and median synthesis time in milliseconds.
    """
    timings = []
    for _ in range(samples + 1):
        circuit = QuantumCircuit(controls + 1)
        start = perf_counter_ns()
        if axis == "ry":
            circuit.mcry(angle, list(range(controls)), controls, mode="noancilla")
        else:
            getattr(circuit, f"mc{axis}")(angle, list(range(controls)), controls)
        timings.append((perf_counter_ns() - start) / 1e6)
    return circuit, median(timings[1:])


def metrics(circuit: QuantumCircuit, reference: QuantumCircuit) -> dict[str, int | float | str]:
    """Measure common-basis circuits and verify phase on small operators.

    Returns:
        Gate counts, depths, and the phase-sensitive error for small operators.
    """
    assert circuit.num_qubits == reference.num_qubits
    normalized = transpile(
        circuit, basis_gates=["u", "cx"], optimization_level=0, seed_transpiler=0, qubits_initially_zero=False
    )
    optimized = transpile(
        normalized, basis_gates=["u", "cx"], optimization_level=3, seed_transpiler=0, qubits_initially_zero=False
    )
    result: dict[str, int | float | str] = {}
    for prefix, output in (("raw", normalized), ("optimized", optimized)):
        assert set(output.count_ops()) <= {"u", "cx"}
        result[f"{prefix}_cx"] = output.count_ops().get("cx", 0)
        result[f"{prefix}_one_qubit"] = output.count_ops().get("u", 0)
        result[f"{prefix}_depth"] = output.depth()
        result[f"{prefix}_cx_depth"] = output.depth(lambda instruction: instruction.operation.num_qubits == 2)
    result["max_operator_error"] = ""
    if reference.num_qubits <= 6:
        error = 0.0
        for angle in (-0.61, 2 * pi) if reference.parameters else (0.73,):
            expected = reference.assign_parameters(dict.fromkeys(reference.parameters, angle))
            for output in (normalized, optimized):
                actual = output.assign_parameters(dict.fromkeys(output.parameters, angle))
                error = max(error, float(np.max(np.abs(Operator(actual).data - Operator(expected).data))))
        assert error <= 1e-10, f"phase-sensitive operator error: {error}"
        result["max_operator_error"] = error
    return result


def main() -> None:
    """Write quality and median synthesis times for numeric and symbolic gates."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controls", nargs="+", type=int, default=[2, 3, 4, 5, 6, 7, 8, 9, 10, 16, 32, 64])
    parser.add_argument("--samples", type=int, default=9)
    parser.add_argument("--output", type=Path, default=Path("build/bench/controlled-rotations.csv"))
    args = parser.parse_args()
    if args.samples < 1 or any(count < 2 for count in args.controls):
        parser.error("samples must be positive and control counts must be at least two")
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    LOGGER.setLevel(logging.INFO)
    source = Path(__file__).resolve().parents[2] / (
        "mlir/lib/Dialect/QCO/Transforms/Decomposition/DecomposeMultiControlled.cpp"
    )
    LOGGER.info("Core %s; Qiskit %s; samples=%s", version("mqt-core"), version("qiskit"), args.samples)
    LOGGER.info("DecomposeMultiControlled.cpp SHA256: %s", hashlib.sha256(source.read_bytes()).hexdigest())
    rows = []
    for axis, gate in (("rx", RXGate), ("ry", RYGate), ("rz", RZGate)):
        for controls in args.controls:
            for kind, angle in (("numeric", 0.73), ("symbolic", Parameter("theta"))):
                reference = QuantumCircuit(controls + 1)
                reference.append(AnnotatedOperation(gate(angle), ControlModifier(controls)), reference.qubits)
                source_ir = QCProgram.from_qiskit(reference).to_qco().ir
                for backend in ("core", "qiskit"):
                    circuit, milliseconds = (
                        synthesize_core(source_ir, args.samples)
                        if backend == "core"
                        else synthesize_qiskit(axis, controls, angle, args.samples)
                    )
                    rows.append({
                        "axis": axis,
                        "controls": controls,
                        "angle": kind,
                        "backend": backend,
                        "synthesis_ms": milliseconds,
                        **metrics(circuit, reference),
                    })
                LOGGER.info("%s, controls=%s, %s", axis, controls, kind)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
