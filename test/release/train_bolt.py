#!/usr/bin/env python3
# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Bounded BOLT training with numerical, compiler, JIT, loading, and CLI checks."""

from __future__ import annotations

# Release checks execute trusted build artifacts and tools from the build environment.
# ruff: file-ignore[subprocess-without-shell-equals-true]
import json
import math
import os
import subprocess
import tempfile
from pathlib import Path

from mqt.core import dd, mlir
from mqt.core.qdmi import ProgramFormat
from mqt.core.qdmi.driver import open_device


def main() -> None:
    """Run the release optimization checks."""
    for qubits in (6, 8, 10):
        package = dd.DDPackage(qubits)
        ghz = package.ghz_state(qubits)
        w = package.w_state(qubits)
        assert abs(ghz.get_amplitude(qubits, "0" * qubits) - math.sqrt(0.5)) < 1e-10
        assert abs(ghz.get_amplitude(qubits, "1" * qubits) - math.sqrt(0.5)) < 1e-10
        assert abs(package.inner_product(ghz, w)) < 1e-10
        combined = package.vector_add(ghz, w)
        assert abs(package.inner_product(combined, combined) - 2) < 1e-10
        for _ in range(16):
            assert package.measure_all(ghz) in {"0" * qubits, "1" * qubits}
            assert package.measure_all(w).count("1") == 1
    for qubits in (4, 8, 12):
        program = 'OPENQASM 3.0; include "stdgates.inc"; ' + f"qubit[{qubits}] q; "
        program += "".join(
            f"h q[{i % qubits}]; t q[{i % qubits}]; cx q[{i % qubits}], q[{(i + 1) % qubits}];" for i in range(120)
        )
        for output in (mlir.OutputFormat.QCO, mlir.OutputFormat.JEFF):
            assert mlir.compile_program(program, output=output)
    device = open_device("mqt.ddsim.default")
    bell = 'OPENQASM 3.0; include "stdgates.inc"; qubit[2] q; bit[2] c; h q[0]; cx q[0],q[1]; c = measure q;'
    qir = mlir.compile_program(bell, output=mlir.OutputFormat.QIR_BASE, target=mlir.CompilerTarget.from_device(device))
    for program, format_ in ((bell, ProgramFormat.QASM3), (qir.llvm_ir, ProgramFormat.QIR_BASE_STRING)):
        job = device.submit_job(program, format_, num_shots=256)
        assert job.wait()
        counts = job.get_counts()
        assert set(counts) == {"00", "11"}
        assert sum(counts.values()) == 256
    executable = Path(mlir.__file__).parent / "bin" / ("mqt-core-bench.exe" if os.name == "nt" else "mqt-core-bench")
    with tempfile.TemporaryDirectory() as directory:
        work = Path(directory)
        specification = work / "qft.json"
        specification.write_text(
            json.dumps({
                "benchmark": "qft",
                "parameters": {"method": "standard", "period_exponent": 2, "qubits": 8},
                "schema_version": 1,
            })
        )
        for format_ in ("qc", "jeff"):
            subprocess.run(
                [
                    str(executable),
                    "generate",
                    f"--instance-specification={specification}",
                    f"--format={format_}",
                    f"--output={work / format_}",
                ],
                check=True,
            )


if __name__ == "__main__":
    main()
