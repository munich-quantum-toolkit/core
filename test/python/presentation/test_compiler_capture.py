# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Check that presentation extraction preserves compiler mapping evidence."""

from __future__ import annotations

import runpy
from pathlib import Path

import pytest
from qiskit import QuantumCircuit

SCRIPT = Path(__file__).resolve().parents[3] / "presentations/mqsf2026/capture_programs.py"


def test_extract_mapping_composes_permutations_and_keeps_branch_context() -> None:
    """Catch swapped mapping directions and invented executed SWAP ordering."""
    assert SCRIPT.is_file(), "The compiler capture entry point must exist"
    extract = runpy.run_path(str(SCRIPT))["extract_layout"]
    ir = """module attributes {mqt.layout = {
      initial = array<i64: 2, 0, 1>, input_count = 2 : i64,
      routing = array<i64: 1, 2, 0>, sites = array<i64: 7, 19, 42>}} {}
    """
    qasm = """OPENQASM 3.1;
if (result[0]) {
  swap $7, $42;
}
swap $19, $7;
"""
    layout = extract(ir, qasm)
    assert layout["initial"] == [42, 7]
    assert layout["final"] == [7, 19]
    assert layout["swaps"] == [[7, 42], [19, 7]]
    assert layout["swap_contexts"] == [
        {"line": 3, "regions": ["if (result[0]) {"]},
        {"line": 5, "regions": []},
    ]
    with pytest.raises(ValueError, match="permutation"):
        extract(ir.replace("routing = array<i64: 1, 2, 0>", "routing = array<i64: 1, 1, 0>"), qasm)


def test_pass_dumps_keep_exact_module_and_reject_missing_capture() -> None:
    """Catch accidental use of the final module as an intermediate stage."""
    assert SCRIPT.is_file(), "The compiler capture entry point must exist"
    extract = runpy.run_path(str(SCRIPT))["extract_pass_ir"]
    dumps = """// -----// IR Dump After MappingPass: place-and-route{ntrials=1} ('builtin.module' operation) //----- //
module {\n  %q = qco.static 7 : !qco.qubit\n}\n
// -----// IR Dump After TargetNativeSynthesis: target-native-synthesis ('builtin.module' operation) //----- //
module {\n  %q = qco.static 19 : !qco.qubit\n}\n
"""
    assert extract(dumps, "place-and-route") == "module {\n  %q = qco.static 7 : !qco.qubit\n}\n"
    with pytest.raises(ValueError, match="missing"):
        extract(dumps, "legalize-control-flow")


def test_qiskit_diagram_size_limit_preserves_small_circuits() -> None:
    """Prevent a successfully exported large circuit from bloating the offline deck."""
    helpers = runpy.run_path(str(SCRIPT))
    assert "render_qiskit" in helpers, "Capture must bound optional circuit drawings"
    render = helpers["render_qiskit"]
    circuit = QuantumCircuit(1)
    circuit.h(0)
    assert "H" in render(circuit)
    for _ in range(1000):
        circuit.x(0)
    with pytest.raises(ValueError, match=r"export succeeded.*text diagram omitted"):
        render(circuit)


@pytest.mark.parametrize(
    ("language", "code", "expected"),
    [
        ("cpp", "// copyright\n#include <array>\n\niterativeQPE(qc::QCProgramBuilder& builder) {\n}\n", 4),
        (
            "mlir",
            (
                "module {\n  func.func @helper() {\n    scf.for %i = %lb to %ub step %s {\n    }\n  }\n"
                "  func.func @main() attributes {mqt.entry_point} {\n    %c0 = arith.constant 0 : index\n"
                "    %q = qco.alloc : !qco.qubit\n    %result = qco.if %condition args(%arg0 = %q) {\n"
                "    }\n  }\n}\n"
            ),
            9,
        ),
    ],
)
def test_stage_focuses_algorithm_without_changing_captured_code(language: str, code: str, expected: int) -> None:
    """Skip copyrights, metadata, and helper functions when opening a capture."""
    artifact = runpy.run_path(str(SCRIPT))["stage"]("source", "Source", language, code)
    assert artifact.get("focus_line") == expected
    assert artifact["code"] == code
