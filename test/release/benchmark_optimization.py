#!/usr/bin/env python3
# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Deterministic, scalable workloads for local compiler optimization experiments."""

from __future__ import annotations

# Benchmark callbacks execute immediately within each loop iteration.
# ruff: file-ignore[missing-type-function-argument, missing-return-type-private-function, function-uses-loop-variable]
import argparse
import json
import math
import random
import statistics
import time
from pathlib import Path

import numpy as np

from mqt.core import dd, mlir
from mqt.core.qdmi import ProgramFormat
from mqt.core.qdmi.driver import open_device


def source(qubits: int, gates: int, *, training: bool) -> str:
    """Return different training/evaluation gate structures at a fixed size.

    Returns:
        OpenQASM circuit text.
    """
    parts = ['OPENQASM 3.0; include "stdgates.inc";', f"qubit[{qubits}] q;"]
    for i in range(gates):
        q = i % qubits
        other = (q + (1 if training else 3)) % qubits
        angle = (i % 31 + 1) * (0.11 if training else 0.17)
        if training:
            parts.append(f"h q[{q}]; rz({angle}) q[{q}]; cx q[{q}], q[{other}];")
        else:
            parts.append(f"ry({angle}) q[{q}]; cz q[{q}], q[{other}];")
    return " ".join(parts)


def main() -> None:
    """Time workloads and check results.

    Raises:
        RuntimeError: The expected wheel is not loaded.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", action="store_true")
    parser.add_argument("--repetitions", type=int, default=9)
    parser.add_argument("--sizes", default="32,128,512")
    parser.add_argument("--expected-root", type=Path)
    args = parser.parse_args()
    if args.expected_root and not Path(mlir.__file__).resolve().is_relative_to(args.expected_root.resolve()):
        msg = f"Loaded the wrong extension: {mlir.__file__}"
        raise RuntimeError(msg)
    results = {}

    def measure(name, operation, check, prepare=lambda: None) -> None:
        samples = []
        for iteration in range(args.repetitions + 1):
            prepare()
            started = time.perf_counter()
            result = operation()
            elapsed = time.perf_counter() - started
            check(result)
            if iteration:
                samples.append(elapsed)
            del result
        results[name] = {"seconds": statistics.median(samples), "samples": samples}

    def nonempty(program) -> None:
        assert program.ir

    if not args.training:
        rng = np.random.default_rng(8)
        vector = rng.normal(size=2**14) + 1j * rng.normal(size=2**14)
        vector /= np.linalg.norm(vector)
        package = dd.DDPackage(14)

        def prepare_vector() -> None:
            nonlocal package
            package = dd.DDPackage(14)

        measure(
            "vector_import_export",
            lambda: package.from_vector(vector).get_vector(),
            lambda x: np.testing.assert_allclose(x, vector),
            prepare_vector,
        )
        a = (rng.normal(size=(32, 32)) + 1j * rng.normal(size=(32, 32))) / 16
        b = (rng.normal(size=(32, 32)) + 1j * rng.normal(size=(32, 32))) / 16
        left, right = package.from_matrix(a), package.from_matrix(b)

        def prepare_matrix() -> None:
            nonlocal package, left, right
            package = dd.DDPackage(14)
            left, right = package.from_matrix(a), package.from_matrix(b)

        measure(
            "matrix_multiply",
            lambda: package.matrix_multiply(left, right),
            lambda x: np.testing.assert_allclose(x.get_matrix(5), a @ b, atol=1e-12),
            prepare_matrix,
        )
        continuity = 'OPENQASM 3.0; include "stdgates.inc"; qubit[12] q; ' + "".join(
            f"rx({0.13 * i}) q[{i % 12}]; cx q[{i % 12}],q[{(i + 1) % 12}];" for i in range(240)
        )
        measure("qasm_pipeline", lambda: mlir.compile_program(continuity, output=mlir.OutputFormat.QCO), nonempty)
        circuit = mlir.QCProgram.from_qasm_str(continuity).to_qiskit()

        def check_circuit(result) -> None:
            assert result == circuit

        measure("qiskit_roundtrip", lambda: mlir.QCProgram.from_qiskit(circuit).to_qiskit(), check_circuit)

    device = open_device("mqt.ddsim.default")
    device_target = mlir.CompilerTarget.from_device(device)
    for gates in map(int, args.sizes.split(",")):
        qubits = 6 if args.training else 8
        text = source(qubits, gates, training=args.training)
        qc = mlir.QCProgram.from_qasm_str(text)
        qco = qc.to_qco(copy=True)
        measure(f"parse/{gates}", lambda: mlir.QCProgram.from_qasm_str(text), nonempty)
        loop_body = "x q;" if args.training else "if (i % 2 == 0) { x q; } else { h q; h q; }"
        loop_source = (
            'OPENQASM 3.0; include "stdgates.inc"; qubit q; bit c; '
            f"for int i in [0:{gates - 1}] {{ {loop_body} }} c = measure q;"
        )

        def check_loop(program) -> None:
            expected_bit = (gates if args.training else (gates + 1) // 2) % 2
            assert program.to_qco(copy=True).sample(shots=1, seed=1) == {str(expected_bit): 1}

        measure(
            f"control_flow/{gates}", lambda: mlir.compile_program(loop_source, output=mlir.OutputFormat.QC), check_loop
        )
        measure(
            f"qco_optimize/{gates}",
            lambda: mlir.compile_program(text, output=mlir.OutputFormat.QCO_OPTIMIZED),
            nonempty,
        )
        target = mlir.CompilerTarget(
            qubits,
            connectivity=mlir.CompilerTarget.Connectivity([(i, i + 1) for i in range(qubits - 1)]),
            native_operations=mlir.CompilerTarget.NativeOperations.unrestricted(),
        )
        measure(
            f"mapping/{gates}",
            lambda: mlir.compile_program(qc, output=mlir.OutputFormat.QCO_OPTIMIZED, target=target),
            nonempty,
        )
        encoded = qco.copy().to_jeff().to_bytes()
        measure(
            f"jeff_roundtrip/{gates}",
            lambda: mlir.JeffProgram.from_bytes(encoded).to_qco().to_jeff().to_bytes(),
            lambda x: nonempty(mlir.JeffProgram.from_bytes(x).to_qco()),
        )
        measure(
            f"qir_lowering/{gates}",
            lambda: mlir.compile_program(text, output=mlir.OutputFormat.QIR_BASE, target=device_target),
            nonempty,
        )
        execution_source = f'OPENQASM 3.0; include "stdgates.inc"; qubit[{qubits}] q; bit[{qubits}] c; '
        bits = [0] * qubits
        rng = random.Random(17 if args.training else 31)  # ruff: ignore[suspicious-non-cryptographic-random-usage]
        for i in range(gates):
            q = rng.randrange(qubits)
            target_qubit = (q + (1 if args.training else 3)) % qubits
            control = (q + 2) % qubits
            other = (q + 5) % qubits
            angle = (i % 19 + 1) * (0.13 if args.training else 0.17)
            execution_source += (
                f"x q[{q}]; cx q[{q}],q[{target_qubit}]; cx q[{control}],q[{other}]; rz({angle}) q[{q}]; "
            )
            bits[q] ^= 1
            bits[target_qubit] ^= bits[q]
            bits[other] ^= bits[control]
        execution_source += "c = measure q;"
        expected_counts = {"".join(map(str, bits)): 256}
        qir = mlir.compile_program(execution_source, output=mlir.OutputFormat.QIR_BASE, target=device_target)

        def execute():
            job = device.submit_job(qir.llvm_ir, ProgramFormat.QIR_BASE_STRING, num_shots=256)
            assert job.wait()
            return job.get_counts()

        def check_state(counts) -> None:
            assert counts == expected_counts

        measure(f"qir_jit/{gates}", execute, check_state)
        expected = None
        if not args.training:
            expected = np.zeros(2**qubits, dtype=complex)
            expected[0] = 1
            indices = np.arange(2**qubits)
            for i in range(gates):
                bit = 1 << (i % qubits)
                lower = indices[(indices & bit) == 0]
                upper = lower | bit
                angle = (i % 31 + 1) * 0.17
                a, b = expected[lower].copy(), expected[upper].copy()
                expected[lower] = math.cos(angle / 2) * a - math.sin(angle / 2) * b
                expected[upper] = math.sin(angle / 2) * a + math.cos(angle / 2) * b
                other = 1 << ((i % qubits + 3) % qubits)
                expected[((indices & bit) != 0) & ((indices & other) != 0)] *= -1
            for program in [
                mlir.compile_program(text, output=mlir.OutputFormat.QCO_OPTIMIZED),
                mlir.JeffProgram.from_bytes(encoded).to_qco(),
            ]:
                validation_package = dd.DDPackage(qubits)
                state = program.simulate(validation_package.zero_state(0), validation_package, seed=1)
                np.testing.assert_allclose(state.get_vector(), expected, rtol=1e-10, atol=1e-10)
        dd_package = dd.DDPackage(qubits)

        def prepare_simulation() -> None:
            nonlocal dd_package
            dd_package = dd.DDPackage(qubits)

        def simulate():
            return qco.simulate(dd_package.zero_state(0), dd_package, seed=1)

        def check_norm(state) -> None:
            assert math.isclose(sum(abs(x) ** 2 for x in state.get_vector()), 1, abs_tol=1e-9)
            if expected is not None:
                np.testing.assert_allclose(state.get_vector(), expected, rtol=1e-10, atol=1e-10)

        measure(f"dd_simulation/{gates}", simulate, check_norm, prepare_simulation)

    print(json.dumps({"training": args.training, "results": results}), flush=True)  # ruff: ignore[print]


if __name__ == "__main__":
    main()
