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


def test_circuit_metadata_keeps_control_flow_and_physical_sites() -> None:
    """Draw nested gates on their actual parent wires and preserve site IDs."""
    capture = runpy.run_path(str(SCRIPT))["capture_circuit"]
    circuit = QuantumCircuit(3, 1)
    circuit.h(0)
    circuit.measure(2, 0)
    with circuit.if_test((circuit.clbits[0], 1)):
        circuit.cx(2, 1)
    diagram = capture(circuit, [7, 19, 42])
    assert [wire["site"] for wire in diagram["qubits"]] == [7, 19, 42]
    branch = diagram["operations"][-1]
    assert branch["name"] == "if_else"
    assert diagram["operations"][1]["clbits"] == [0]
    assert branch["condition_bits"] == [0]
    assert branch["blocks"][0][0]["qubits"] == [2, 1]
    loop = QuantumCircuit(54)
    body = QuantumCircuit(54)
    body.cx(2, 1)
    loop.for_loop(range(2), None, body, loop.qubits, [], label=None)
    diagram = capture(loop, list(range(54)))
    assert [wire["site"] for wire in diagram["qubits"]] == [1, 2]
    assert diagram["operations"][0]["blocks"][0][0]["qubits"] == [1, 0]


def test_parity_example_retains_loop_feedback_and_even_data_parity() -> None:
    """The small compiler story must be real and have a simple semantic check."""
    from mqt.core.mlir import QCProgram  # ruff: ignore[import-outside-top-level]

    helpers = runpy.run_path(str(SCRIPT))
    source = helpers["PARITY_SOURCE"]
    program = QCProgram.from_openqasm_str(source).to_qco()
    program.cleanup()
    assert "scf.for" in program.ir
    assert "qco.if" in program.ir
    circuit = helpers["capture_circuit"](program.to_qiskit())
    loop = next(op for op in circuit["operations"] if op["name"] == "for_loop")
    branch = next(op for op in loop["blocks"][0] if op["name"] == "if_else")
    assert branch["condition_bits"] == [0]
    counts = program.sample(128, 7)
    assert set(counts) == {"000", "110"}


def test_four_qubit_qpe_resolves_non_exact_phase() -> None:
    """Check the two dominant bins against the analytical QPE distribution."""
    import math  # ruff: ignore[import-outside-top-level]

    from mqt.core.mlir import QCProgram  # ruff: ignore[import-outside-top-level]

    source = runpy.run_path(str(SCRIPT))["qpe_source"]()
    program = QCProgram.from_openqasm_str(source).to_qco()
    counts = program.sample(2048, 7)
    for bin_value in (85, 86):
        difference = 1 / 3 - bin_value / 256
        expected = (math.sin(256 * math.pi * difference) / (256 * math.sin(math.pi * difference))) ** 2
        assert counts[f"{bin_value:08b}"] / 2048 == pytest.approx(expected, abs=0.05)


def test_repeat_until_success_retains_conditional_loop_and_heralded_state() -> None:
    """The retry loop must survive compilation and terminate in the heralded state."""
    from mqt.core.mlir import QCProgram  # ruff: ignore[import-outside-top-level]

    source = runpy.run_path(str(SCRIPT))["RUS_SOURCE"]
    program = QCProgram.from_openqasm_str(source).to_qco()
    program.cleanup()
    assert "scf.while" in program.ir
    assert program.sample(128, 7) == {"11": 128}


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
