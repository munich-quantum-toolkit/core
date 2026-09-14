#!/usr/bin/env python3
# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Train release PGO with the selected compiler, QIR, and DD workloads."""

from __future__ import annotations

# Callbacks execute immediately within each loop iteration.
# ruff: file-ignore[missing-type-function-argument, missing-return-type-private-function, function-uses-loop-variable, subprocess-without-shell-equals-true]
import argparse
import math
import random
import subprocess
import sys
from pathlib import Path

from mqt.core import dd, mlir
from mqt.core.qdmi import ProgramFormat
from mqt.core.qdmi.driver import open_device


def train(operation, check, prepare=lambda: None) -> None:
    """Keep the two executions per workload used to qualify the release recipe."""
    for _ in range(2):
        prepare()
        result = operation()
        check(result)
        del result


def nonempty(program) -> None:
    """Require successful compiler output."""
    assert program.ir


def main() -> None:
    """Run the fixed training corpus against the newly built wheel.

    Raises:
        RuntimeError: The expected wheel is not loaded.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-root", type=Path, required=True)
    args = parser.parse_args()
    if not Path(mlir.__file__).resolve().is_relative_to(args.expected_root.resolve()):
        msg = f"Loaded the wrong extension: {mlir.__file__}"
        raise RuntimeError(msg)
    subprocess.run([sys.executable, str(Path(__file__).with_name("train_bolt.py"))], check=True)
    device = open_device("mqt.ddsim.default")
    device_target = mlir.CompilerTarget.from_device(device)
    qubits = 6
    for gates in (32, 128, 512):
        parts = ['OPENQASM 3.0; include "stdgates.inc";', f"qubit[{qubits}] q;"]
        for i in range(gates):
            q = i % qubits
            angle = (i % 31 + 1) * 0.11
            parts.append(f"h q[{q}]; rz({angle}) q[{q}]; cx q[{q}], q[{(q + 1) % qubits}];")
        text = " ".join(parts)
        qc = mlir.QCProgram.from_openqasm_str(text)
        qco = qc.to_qco(copy=True)
        train(lambda: mlir.QCProgram.from_openqasm_str(text), nonempty)
        loop_source = (
            'OPENQASM 3.0; include "stdgates.inc"; qubit q; bit c; '
            f"for int i in [0:{gates - 1}] {{ x q; }} c = measure q;"
        )

        def check_loop(program) -> None:
            assert program.to_qco(copy=True).sample(shots=1, seed=1) == {str(gates % 2): 1}

        train(lambda: mlir.compile_program(loop_source, output=mlir.OutputFormat.QC), check_loop)
        train(lambda: mlir.compile_program(text, output=mlir.OutputFormat.QCO_OPTIMIZED), nonempty)
        target = mlir.CompilerTarget(
            qubits,
            connectivity=mlir.CompilerTarget.Connectivity([(i, i + 1) for i in range(qubits - 1)]),
            native_operations=mlir.CompilerTarget.NativeOperations.unrestricted(),
        )

        def map_program():
            mapped = mlir.compile_program(qc, output=mlir.OutputFormat.QCO_OPTIMIZED)
            mapped.compile_for_target(
                mlir.TargetEnvironment(target, mlir.PayloadSpecification(mlir.PayloadFormat("qir", "2.1.0", "base")))
            )
            return mapped

        train(map_program, nonempty)
        encoded = qco.copy().to_jeff().to_bytes()
        train(
            lambda: mlir.JeffProgram.from_bytes(encoded).to_qco().to_jeff().to_bytes(),
            lambda x: nonempty(mlir.JeffProgram.from_bytes(x).to_qco()),
        )
        train(lambda: mlir.compile_program(text, output=mlir.OutputFormat.QIR_BASE, target=device_target), nonempty)
        execution_source = f'OPENQASM 3.0; include "stdgates.inc"; qubit[{qubits}] q; bit[{qubits}] c; '
        bits = [0] * qubits
        rng = random.Random(17)  # ruff: ignore[suspicious-non-cryptographic-random-usage]
        for i in range(gates):
            q = rng.randrange(qubits)
            target_qubit = (q + 1) % qubits
            control = (q + 2) % qubits
            other = (q + 5) % qubits
            angle = (i % 19 + 1) * 0.13
            execution_source += (
                f"x q[{q}]; cx q[{q}],q[{target_qubit}]; cx q[{control}],q[{other}]; rz({angle}) q[{q}]; "
            )
            bits[q] ^= 1
            bits[target_qubit] ^= bits[q]
            bits[other] ^= bits[control]
        execution_source += "c = measure q;"
        expected_counts = {"".join(map(str, reversed(bits))): 256}
        qir = mlir.compile_program(execution_source, output=mlir.OutputFormat.QIR_BASE, target=device_target)

        def execute():
            job = device.submit_job(qir.llvm_ir, ProgramFormat.QIR_BASE_STRING, num_shots=256)
            assert job.wait()
            return job.get_counts()

        def check_state(counts) -> None:
            assert counts == expected_counts, (counts, expected_counts)

        train(execute, check_state)
        dd_package = dd.DDPackage(qubits)

        def prepare_simulation() -> None:
            nonlocal dd_package
            dd_package = dd.DDPackage(qubits)

        def simulate():
            return qco.simulate(dd_package.zero_state(0), dd_package, seed=1)

        def check_norm(state) -> None:
            assert math.isclose(dd_package.inner_product(state, state).real, 1, abs_tol=1e-9)

        train(simulate, check_norm, prepare_simulation)


if __name__ == "__main__":
    main()
