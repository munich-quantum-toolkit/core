# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Record measured compiler examples for the MQSF machine-room slides."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
from datetime import UTC, datetime
from importlib.metadata import version
from math import pi
from pathlib import Path
from typing import Any

import numpy as np
from capture_programs import PARITY_SOURCE, ROOT, RUS_SOURCE, program_stage, stage
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from qiskit.synthesis import TwoQubitWeylDecomposition

OPTIMIZATION_SOURCE = """OPENQASM 3.1;
include "stdgates.inc";
qubit[2] q;
h q[0]; h q[0];
rx(pi/8) q[1]; rx(3*pi/8) q[1];
cx q[0], q[1]; cx q[0], q[1];
h q[0]; cx q[0], q[1];
"""
SOURCES = {
    "mapping": "https://arxiv.org/abs/2607.02616",
    "search_visual_inspiration": "https://arxiv.org/abs/2512.13790v1",
    "rus": "https://arxiv.org/abs/1311.1074",
    "rus_correction": "https://docs.quantinuum.com/guppy/guppylang/examples/repeat-until-success.html",
    "weyl": "https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.synthesis.TwoQubitWeylDecomposition",
}


def digest(source: str) -> str:
    """Return the SHA-256 digest of the exact recorded text."""
    return hashlib.sha256(source.encode()).hexdigest()


def grover_size(output: Path, qubits: tuple[int, ...] = tuple(range(2, 23, 2))) -> dict[str, Any]:
    """Measure emitted source size for preserved and fully unrolled Grover.

    Both paths start from the same MQT Core benchmark. Multi-controlled gates
    remain gates in both paths; this measures loop expansion, not synthesis.

    Returns:
        Byte measurements and representative sources with their hashes.
    """
    from mqt.core.bench import grover  # ruff: ignore[import-outside-top-level]

    points = []
    for width in qubits:
        benchmark = grover.Grover(grover.Options(marked_bitstring="0" * width))
        program = benchmark.generate().to_qco()
        preserved = program.to_qc(copy=True).to_openqasm3().source
        program.unroll_quantum_loops()
        program.cleanup()
        unrolled = program.to_qc(copy=True).to_openqasm3().source
        assert not re.search(r"\b(for|while)\b", unrolled), "Grover still contains a loop"
        point = {"qubits": width, "iterations": benchmark.options.iterations}
        for label, source in (("preserved", preserved), ("unrolled", unrolled)):
            name = f"grover-{width}-{label}.qasm"
            (output / name).write_text(source, encoding="utf-8")
            point.update({
                f"{label}_bytes": len(source.encode()),
                f"{label}_sha256": digest(source),
                f"{label}_file": name,
            })
        points.append(point)
    return {
        "metric": "UTF-8 bytes of emitted OpenQASM 3",
        "points": points,
        "method": "MQT Core Grover, automatic iteration count, marked all-zero state; full quantum-loop unrolling.",
        "scope": (
            "Measured source representation size; multi-controlled gates retained, "
            "no device synthesis or extrapolation."
        ),
        "representative": {
            label: stage(label, label.title(), "qasm", (output / f"grover-6-{label}.qasm").read_text())
            for label in ("preserved", "unrolled")
        }
        if 6 in qubits
        else {},
    }


def optimization() -> dict[str, Any]:
    """Capture canonicalization and rotation merging with unitary verification.

    Returns:
        Actual before/after programs and the semantic-check result.
    """
    from mqt.core.mlir import QCProgram  # ruff: ignore[import-outside-top-level]

    program = QCProgram.from_openqasm_str(OPTIMIZATION_SOURCE).to_qco()
    before = program_stage("before", "Before canonicalization", program)
    original = Operator(program.to_qiskit(copy=True))
    program.merge_single_qubit_rotation_gates()
    program.cleanup()
    after = program_stage("after", "After canonicalization and rotation merging", program)
    equivalent = original.equiv(Operator(program.to_qiskit()))
    assert equivalent, "Optimization changed the example's unitary"
    path = "mlir/include/mqt/Dialect/QCO/QCOUtils.h"
    implementation = (ROOT / path).read_text(encoding="utf-8")
    function = implementation.index("removeInversePairOneTargetZeroParameter(")
    start = implementation.rfind("template <", 0, function)
    end = implementation.index("\n}", function) + 2
    pattern = stage("pattern", "Actual MQT Core inverse-pair rewrite", "cpp", implementation[start:end])
    pattern["source_path"] = path
    return {
        "source": stage("source", "Optimization example", "qasm", OPTIMIZATION_SOURCE),
        "before": before,
        "after": after,
        "unitary_equivalent": equivalent,
        "passes": ["merge_single_qubit_rotation_gates", "cleanup"],
        "implementation": pattern,
    }


def syntax_checks() -> dict[str, Any]:
    """Record the supported compact syndrome syntax and its export boundary.

    Returns:
        Results of actual compiler import, QIR lowering, and circuit export.
    """
    from mqt.core.mlir import QCProgram, QIRProfile  # ruff: ignore[import-outside-top-level]

    program = QCProgram.from_openqasm_str(PARITY_SOURCE)
    qir = program.copy().to_qir(QIRProfile.ADAPTIVE).llvm_ir
    circuit = program.to_qiskit(copy=True)
    emitted = program.to_openqasm3().source
    QCProgram.from_openqasm_str(emitted).to_qiskit()
    assert circuit.count_ops()["h"] == 2
    return {
        "accepted": ["h data;", "cx data, flag;", "bit syndrome = measure flag;", "bit[2] result = measure data;"],
        "local_syndrome_declaration": {
            "import": True,
            "adaptive_qir": bool(qir),
            "qiskit": "supported",
            "openqasm3": True,
        },
        "all_register_h": "The two-qubit data register broadcasts H without changing the flag preparation.",
        "display_choice": "Local syndrome storage preserves the lexical scope in the compact source.",
    }


def rus_semantics() -> dict[str, Any]:
    """Verify the imported RUS attempt's Kraus operators on arbitrary data.

    Returns:
        The verified gate, success probability, and matrix residual.
    """
    from mqt.core.mlir import QCProgram  # ruff: ignore[import-outside-top-level]

    circuit = QCProgram.from_openqasm_str(RUS_SOURCE).to_qiskit()
    body = next(op.operation.blocks[0] for op in circuit.data if op.operation.name == "while_loop")
    first = QuantumCircuit(3)
    for instruction in body.data:
        if instruction.operation.name == "measure":
            break
        if instruction.operation.name not in {"reset", "store"}:
            first.append(instruction.operation, [body.find_bit(q).index for q in instruction.qubits])
    conditional = next(op for op in body.data if op.operation.name == "if_else")
    branch = conditional.operation.blocks[0]
    wires = [body.find_bit(q).index for q in conditional.qubits]
    attempt = first.copy()
    for instruction in branch.data:
        if instruction.operation.name == "measure":
            break
        if instruction.operation.name != "store":
            attempt.append(instruction.operation, [wires[branch.find_bit(q).index] for q in instruction.qubits])
    first_matrix = Operator(first).data
    attempt_matrix = Operator(attempt).data
    target = np.diag([1 + 2j, 1 - 2j]) / np.sqrt(5)
    identity = np.eye(2)
    retry = next(op for op in branch.data if op.operation.name == "if_else")
    correction = QuantumCircuit(1)
    retry_wires = [wires[branch.find_bit(q).index] for q in retry.qubits]
    for instruction in retry.operation.blocks[0].data:
        if instruction.qubits:
            assert [retry_wires[retry.operation.blocks[0].find_bit(q).index] for q in instruction.qubits] == [2]
            correction.append(instruction.operation, [0])

    def check(matrix: np.ndarray, reference: np.ndarray, probability: float) -> float:
        amplitude = np.trace(reference.conj().T @ matrix) / 2
        assert np.allclose(matrix, amplitude * reference, atol=1e-12)
        assert np.isclose(abs(amplitude) ** 2, probability, atol=1e-12)
        return float(np.max(abs(matrix - amplitude * reference)))

    def kraus(matrix: np.ndarray, ancillas: int) -> np.ndarray:
        return matrix[np.ix_([ancillas, ancillas + 4], [0, 4])]

    residuals = [check(kraus(attempt_matrix, 0), target, 5 / 8)]
    residuals.append(check(Operator(correction).data @ kraus(attempt_matrix, 2), identity, 1 / 8))
    residuals.extend(
        check(kraus(first_matrix, outcome), identity, probability) for outcome, probability in ((1, 0), (3, 1 / 4))
    )
    return {
        "gate": "V3 = (I + 2iZ)/sqrt(5)",
        "success_probability": 5 / 8,
        "mean_attempts": 8 / 5,
        "result_one_probability": 4 / 5,
        "check": (
            "Kraus matrices of imported compiler circuit: success sqrt(5/8) V3; corrected failures proportional to I."
        ),
        "max_residual": max(residuals),
        "sources": [SOURCES["rus"], SOURCES["rus_correction"]],
    }


def weyl(targets: list[dict[str, Any]]) -> dict[str, Any]:
    """Decompose actual fused compiler matrices into signed Weyl coordinates.

    Returns:
        Coordinates with the exact compiler matrices and their provenance.
    """
    target = next(item for item in targets if "iqm" in item["id"])
    trace = target["compilation"]["synthesis_trace"]
    assert trace, "The selected compilation did not record a fused matrix"
    points = []
    for index, block in enumerate(trace):
        values = np.asarray(block["matrix"])
        matrix = values[:, :, 0] + 1j * values[:, :, 1]
        assert matrix.shape == (4, 4)
        assert np.allclose(matrix.conj().T @ matrix, np.eye(4), atol=1e-12)
        decomposition = TwoQubitWeylDecomposition(matrix, fidelity=None)
        assert Operator(decomposition.circuit()).equiv(Operator(matrix))
        points.append({
            "fusion_index": index,
            "operations": block["operations"],
            "two_qubit_before": block["two_qubit_before"],
            "basis_uses_after": block["basis_uses_after"],
            "coordinates": [decomposition.a, decomposition.b, decomposition.c],
            "matrix": block["matrix"],
        })
    return {
        "coordinate_convention": "U = local · exp(i(a XX + b YY + c ZZ)) · local; pi/4 >= a >= b >= |c|.",
        "method": "Qiskit TwoQubitWeylDecomposition(fidelity=None) of exact native accepted-fusion matrix traces.",
        "scope": (
            "Every accepted two-qubit fusion in one H2 target compilation; successive passes may revisit a block. "
            "These are captured compiler events, not disjoint final-circuit blocks or a synthesis search trajectory."
        ),
        "target_id": target["id"],
        "input_sha256": digest(json.dumps(trace, sort_keys=True)),
        "program_sha256": target["compilation"]["source_sha256"],
        "vertices": [
            {"label": "I", "coordinates": [0, 0, 0]},
            {"label": "CZ / CX", "coordinates": [pi / 4, 0, 0]},
            {"label": "iSWAP", "coordinates": [pi / 4, pi / 4, 0]},
            {"label": "SWAP", "coordinates": [pi / 4, pi / 4, pi / 4]},
            {"label": "SWAP boundary", "coordinates": [pi / 4, pi / 4, -pi / 4]},
        ],
        "points": points,
        "source": SOURCES["weyl"],
    }


def validate_routing_trace(trace: list[dict[str, Any]], edges: list[list[int]]) -> None:
    """Check every recorded SWAP and A* parent edge against its actual layout."""
    coupling = {tuple(sorted(edge)) for edge in edges}
    for sweep in trace:
        if "steps" not in sweep:
            continue
        layout = list(sweep["before"])
        count = 0
        for step in sweep["steps"]:
            assert [layout[q] for q in step["logical"]] == step["sites"]
            if step["operation"] == "qco.swap":
                a, b = step["sites"]
                assert tuple(sorted((a, b))) in coupling
                layout = [b if q == a else a if q == b else q for q in layout]
                count += 1
        assert count == sweep["swaps"]
        assert layout == sweep["after"]
        search = sweep.get("search", {})
        nodes = {node["id"]: node for node in search.get("nodes", [])}
        for node in nodes.values():
            if node["parent"] is None:
                continue
            a, b = node["swap"]
            assert tuple(sorted((a, b))) in coupling
            assert node["layout"] == [b if q == a else a if q == b else q for q in nodes[node["parent"]]["layout"]]
        if search and not search["truncated"] and not search["fallback"]:
            goal = nodes[search["goal"]]
            assert tuple(sorted(goal["layout"][q] for q in search["front"])) in coupling
            assert search["visited"][-1] == search["goal"]


def main() -> None:
    """Save data and exact source files without making hardware requests."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "build/mqsf-click/compiler")
    parser.add_argument("--targets", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    git = shutil.which("git")
    assert git is not None, "Capturing provenance requires Git"
    revision = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true] - fixed Git arguments
        [git, "rev-parse", "HEAD"], cwd=ROOT, check=True, text=True, capture_output=True
    ).stdout.strip()
    result = {
        "provenance": {
            "core_revision": revision,
            "generated_at": datetime.now(UTC).isoformat(),
            "script_sha256": digest(Path(__file__).read_text(encoding="utf-8")),
            "qiskit_version": version("qiskit"),
            "numpy_version": version("numpy"),
        },
        "grover_size": grover_size(args.output),
        "optimization": optimization(),
        "syntax_checks": syntax_checks(),
        "rus": rus_semantics(),
        "sources": SOURCES,
    }
    if args.targets:
        target_source = args.targets.read_text()
        targets = json.loads(target_source)["targets"]
        result["provenance"]["targets_sha256"] = digest(target_source)
        result["weyl"] = weyl(targets)
        for target in targets:
            validate_routing_trace(target["compilation"]["routing_trace"], target["compiler_model"]["couplings"])
        result["routing_validation"] = {
            "passed": True,
            "targets": [target["id"] for target in targets],
            "checks": ["logical-to-physical roles", "SWAP adjacency", "sweep endpoints", "A* parent edges and goals"],
        }
    (args.output / "evidence.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
