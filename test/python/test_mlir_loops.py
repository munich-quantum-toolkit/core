# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Loop state, early exits, and observable behavior across translations."""

from __future__ import annotations

import math
import os

import pytest
import qiskit
from packaging.version import Version
from qiskit import QuantumCircuit, qasm3
from qiskit.circuit import Parameter
from qiskit.circuit.classical import expr, types

from mqt.core.dd import DDPackage
from mqt.core.mlir import JeffProgram, QCProgram

if not (
    Version("2.5.0") <= Version(qiskit.__version__) < Version("2.6.0")
    or qiskit.__version__ == os.environ.get("MQT_QISKIT_TEST_CANDIDATE_VERSION")
):
    pytest.skip(
        f"Loop interchange tests require Qiskit 2.5.x (installed: {qiskit.__version__})",
        allow_module_level=True,
    )


def observe(program: QCProgram) -> int:
    """Return the deterministic classical result using the existing interpreter."""
    counts = program.to_qco(copy=True).sample(shots=1, seed=1)
    assert len(counts) == 1
    return int(next(iter(counts)).replace(" ", ""), 2)


def amplitude(program: QCProgram) -> complex:
    """Return the single-qubit zero-state amplitude."""
    package = DDPackage(1)
    result = program.to_qco(copy=True).simulate(package.zero_state(1), package, seed=1)
    value = complex(result.get_vector()[0])
    package.dec_ref_vec(result)
    return value


def check_paths(program: QCProgram, expected: int) -> None:
    """Check observable behavior, serialized interchange, and source preservation."""
    original = program.ir
    assert observe(program) == expected
    source = program.to_openqasm3().source
    assert source == program.to_openqasm3().source
    assert observe(QCProgram.from_qasm_str(source)) == expected
    qiskit = program.to_qiskit()
    # Each export owns fresh native variables; their serialized structure is stable.
    assert qasm3.dumps(qiskit) == qasm3.dumps(program.to_qiskit())
    assert observe(QCProgram.from_qiskit(qiskit)) == expected
    jeff = program.to_qco(copy=True).to_jeff()
    restored = JeffProgram.from_bytes(jeff.to_bytes()).to_qco().to_qc()
    assert observe(restored) == expected
    assert observe(QCProgram.from_qiskit(restored.to_qiskit())) == expected
    assert observe(QCProgram.from_qasm_str(restored.to_openqasm3().source)) == expected
    assert program.ir == original


@pytest.mark.parametrize("iterations", [1, 3])
def test_do_while_round_trip(iterations: int) -> None:
    """A terminating post-test loop has one while and no exit-only if."""
    circuit = QuantumCircuit(1, 1)
    counter = circuit.add_var("counter", expr.lift(0, types.Uint(8)))
    with circuit.while_loop(expr.lift(True), None, None, None, label=None):  # ruff: ignore[boolean-positional-value-in-call] - Qiskit requires a positional value.
        circuit.x(0)
        circuit.store(counter, expr.add(counter, expr.lift(1, types.Uint(8))))
        with circuit.if_test(expr.greater_equal(counter, expr.lift(iterations, types.Uint(8)))):
            circuit.break_loop()
    circuit.measure(0, 0)
    program = QCProgram.from_qiskit(circuit)
    assert program.ir.count("scf.while") == 1
    assert "scf.if" not in program.ir
    restored = program.to_qiskit()
    assert restored.num_clbits == circuit.num_clbits
    assert restored.num_qubits == circuit.num_qubits
    assert restored.num_vars > 0
    loop = next(instruction.operation for instruction in restored.data if instruction.operation.name == "while_loop")
    assert loop.blocks[0].num_captured_vars > 0
    imported = QCProgram.from_qiskit(restored)
    assert imported.ir.count("scf.while") == 1
    assert "scf.if" not in imported.ir
    qasm = QCProgram.from_qasm_str(program.to_openqasm3().source)
    assert qasm.ir.count("scf.while") == 1
    assert "scf.if" not in qasm.ir
    check_paths(program, iterations % 2)


def test_general_while_preserves_after_region_global_phase() -> None:
    """A general while loop applies its after-region phase only while continuing."""
    program = QCProgram.from_qasm_str("""
OPENQASM 3.1;
include "stdgates.inc";
qubit q;
output bit result;
uint counter = 0;
while (counter < 2) {
  gphase(pi / 4);
  x q;
  counter += 1;
}
result = false;
""")
    restored = QCProgram.from_qiskit(program.to_qiskit())
    assert amplitude(program) == pytest.approx(1j)
    assert amplitude(restored) == pytest.approx(1j)


@pytest.mark.parametrize("stale", [False, True])
def test_wide_register_condition_in_do_while(*, stale: bool) -> None:
    """Preserve direct wide comparisons and reject snapshots read before a store."""
    read = "%bits = cbit.read %out : !cbit.reg<65> -> i65"
    program = QCProgram.from_mlir_str(f"""
module {{
  func.func @main() -> !cbit.reg<65> attributes {{mqt.entry_point}} {{
    %q = qc.alloc : !qc.qubit
    %out = cbit.alloc(#cbit.init<zero>) : !cbit.reg<65>
    %highest = arith.constant 64 : index
    %expected = arith.constant {1 << 64} : i65
    scf.while : () -> () {{
      qc.reset %q : !qc.qubit
      qc.x %q : !qc.qubit
      {read if stale else ""}
      %measured = qc.measure %q : !qc.qubit -> i1
      cbit.store %measured, %out[%highest] : !cbit.reg<65>
      {"" if stale else read}
      %continue = arith.cmpi ne, %bits, %expected : i65
      scf.condition(%continue)
    }} do {{
      scf.yield
    }}
    qc.dealloc %q : !qc.qubit
    return %out : !cbit.reg<65>
  }}
}}
""")
    if stale:
        with pytest.raises(RuntimeError, match="stale classical snapshot"):
            program.to_qiskit()
        with pytest.raises(RuntimeError, match="stale classical snapshot"):
            program.to_openqasm3()
    else:
        check_paths(program, 1 << 64)
        assert program.to_qiskit().num_clbits == 65


def test_unused_wide_qiskit_local_is_rejected() -> None:
    """Wide register comparison support does not permit wide scalar locals."""
    circuit = QuantumCircuit(1, 1)
    circuit.add_uninitialized_var(expr.Var.new("wide", types.Uint(65)))
    with pytest.raises(RuntimeError, match=r"local variables.*64"):
        QCProgram.from_qiskit(circuit)


def test_unequal_tuples_swaps_and_float_state() -> None:
    """Edge assignments use old values and preserve the exit tuple."""
    program = QCProgram.from_mlir_str("""
module {
  func.func @main() -> !cbit.reg<8> attributes {mqt.entry_point} {
    %q = qc.alloc : !qc.qubit
    %out = cbit.alloc(#cbit.init<zero>) : !cbit.reg<8>
    %zero = arith.constant 0 : i8
    %one = arith.constant 1 : i8
    %two = arith.constant 2 : i8
    %half = arith.constant 0.5 : f64
    %fstep = arith.constant 1.0 : f64
    %r:5 = scf.while (%a = %one, %b = %two, %i = %zero, %f = %half) : (i8, i8, i8, f64) -> (i8, i8, i8, f64, i1) {
      %next = arith.addi %i, %one : i8
      %nf = arith.addf %f, %fstep : f64
      %continue = arith.cmpi ult, %i, %two : i8
      scf.condition(%continue) %b, %a, %next, %nf, %continue : i8, i8, i8, f64, i1
    } do {
    ^bb0(%a: i8, %b: i8, %i: i8, %f: f64, %c: i1):
      scf.yield %a, %b, %i, %f : i8, i8, i8, f64
    }
    %expected = arith.constant 3.5 : f64
    %valid = arith.cmpf oeq, %r#3, %expected : f64
    %four = arith.constant 4 : i8
    %sixteen = arith.constant 16 : i8
    %b = arith.muli %r#1, %four : i8
    %i = arith.muli %r#2, %sixteen : i8
    %ab = arith.addi %r#0, %b : i8
    %result = arith.addi %ab, %i : i8
    %selected = arith.select %valid, %result, %zero : i8
    cbit.write %selected, %out : i8, !cbit.reg<8>
    qc.dealloc %q : !qc.qubit
    return %out : !cbit.reg<8>
  }
}
""")
    check_paths(program, 54)


@pytest.mark.parametrize(("condition", "expected"), [("false", 250), ("true", 4)])
def test_zero_iterations_and_narrow_overflow(condition: str, expected: int) -> None:
    """The initial value survives a zero-trip loop and i8 addition wraps."""
    program = QCProgram.from_qasm_str(f"""
OPENQASM 3.1;
output bit[8] result;
uint[8] value = 250;
while ({condition}) {{
  value += 10;
  break;
}}
result = bit[8](value);
""")
    check_paths(program, expected)


def test_for_continue_does_not_wrap_induction() -> None:
    """A singleton range stops before its positive step can overflow."""
    program = QCProgram.from_qasm_str("""
OPENQASM 3.1;
output bit[8] result;
uint[8] iterations = 0;
for int i in [1:9223372036854775807:1] {
  iterations += 1;
  continue;
}
result = bit[8](iterations);
""")
    assert observe(program) == 1


def test_nested_breaks_and_switches() -> None:
    """An inner break cannot escape an outer iteration or run its tail."""
    program = QCProgram.from_qasm_str("""
OPENQASM 3.1;
output bit[8] result;
uint[8] value = 0;
for int i in [0:4] {
  while (true) {
    value += 1;
    switch (value) {
      case 1 { break; }
      default { value += 2; break; }
    }
    value = 100;
  }
  if (value > 3) { break; }
  value += 1;
}
result = bit[8](value);
""")
    check_paths(program, 5)


@pytest.mark.parametrize("indexset", [range(3), [2, 4, 7, 9]])
@pytest.mark.parametrize("jump", ["break", "continue"])
def test_qiskit_for_jump(indexset: range | list[int], jump: str) -> None:
    """Both jumps skip the tail; continue advances the range or list iterator."""
    circuit = QuantumCircuit(1, 1)
    with circuit.for_loop(indexset, None, None, None, None, label=None):
        circuit.x(0)
        getattr(circuit, f"{jump}_loop")()
        circuit.x(0)
    circuit.measure(0, 0)
    check_paths(QCProgram.from_qiskit(circuit), 1 if jump == "break" else len(indexset) % 2)


def test_continue_in_nested_loop_and_switch() -> None:
    """Continue targets the inner loop and break still skips its remaining iterations."""
    program = QCProgram.from_qasm_str("""
OPENQASM 3.1;
output bit[8] result;
uint[8] value = 0;
uint[8] iteration = 0;
while (iteration < 4) {
  iteration += 1;
  if (iteration == 2) { continue; }
  for int inner in [0:2] {
    switch (inner) {
      case 0 { continue; }
      default { value += 1; }
    }
    if (inner == 1) { break; }
  }
}
result = bit[8](value);
""")
    check_paths(program, 3)


def test_qiskit_while_continue() -> None:
    """Continue carries updated scalar state back to the while condition."""
    circuit = QuantumCircuit(1, 1)
    counter = circuit.add_var("counter", expr.lift(0, types.Uint(8)))
    with circuit.while_loop(expr.less(counter, expr.lift(4, types.Uint(8))), None, None, None, label=None):
        circuit.store(counter, expr.add(counter, expr.lift(1, types.Uint(8))))
        with circuit.if_test(expr.equal(counter, expr.lift(2, types.Uint(8)))):
            circuit.continue_loop()
        circuit.x(0)
    circuit.measure(0, 0)
    check_paths(QCProgram.from_qiskit(circuit), 1)


@pytest.mark.parametrize("source", ["break;", "if (true) { break; }", "continue;", "if (true) { continue; }"])
def test_invalid_loop_jump_placement(source: str) -> None:
    """The source diagnostic distinguishes an invalid break from export limits."""
    with pytest.raises((ValueError, RuntimeError)):
        QCProgram.from_qasm_str("OPENQASM 3.1; " + source)


def test_qiskit_uninitialized_local() -> None:
    """Local variables need a definition on every reachable read path."""
    circuit = QuantumCircuit(1, 1)
    variable = expr.Var.new("value", types.Bool())
    circuit.add_uninitialized_var(variable)
    with circuit.while_loop((circuit.clbits[0], True), None, None, None, label=None):
        circuit.store(variable, expr.lift(True))  # ruff: ignore[boolean-positional-value-in-call] - Qiskit requires a positional value.
    with circuit.if_test(variable):
        circuit.x(0)
    with pytest.raises(RuntimeError, match="initialization"):
        QCProgram.from_qiskit(circuit)


@pytest.mark.parametrize("initialize_before", [False, True])
def test_qiskit_continue_initialization(*, initialize_before: bool) -> None:
    """Only assignments reached before continue initialize the eventual exit."""
    circuit = QuantumCircuit(1, 1)
    variable = expr.Var.new("value", types.Bool())
    circuit.add_uninitialized_var(variable)
    with circuit.for_loop(range(1), None, None, None, None, label=None):
        if initialize_before:
            circuit.store(variable, True)  # ruff: ignore[boolean-positional-value-in-call] - Qiskit requires a positional value.
        circuit.continue_loop()
        if not initialize_before:
            circuit.store(variable, True)  # ruff: ignore[boolean-positional-value-in-call] - Qiskit requires a positional value.
    with circuit.if_test(variable):
        circuit.x(0)
    circuit.measure(0, 0)
    if initialize_before:
        check_paths(QCProgram.from_qiskit(circuit), 1)
    else:
        with pytest.raises(RuntimeError, match="initialization"):
            QCProgram.from_qiskit(circuit)


@pytest.mark.parametrize("direct_measurement", [False, True])
def test_measurement_in_both_regions_and_snapshot(*, direct_measurement: bool) -> None:
    """Before runs on the final visit; after and its updates run only on continuation."""
    source = """
module {
  func.func @main() -> !cbit.reg<3> attributes {mqt.entry_point} {
    %q = qc.alloc : !qc.qubit
    %out = cbit.alloc(#cbit.init<zero>) : !cbit.reg<3>
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %two = arith.constant 2 : index
    scf.while : () -> () {
      qc.x %q : !qc.qubit
      %measured = qc.measure %q : !qc.qubit -> i1
      cbit.store %measured, %out[%zero] : !cbit.reg<3>
      %snapshot = cbit.load %out[%zero] : !cbit.reg<3>
      %false = arith.constant false
      cbit.store %false, %out[%zero] : !cbit.reg<3>
      scf.condition(%snapshot)
    } do {
      qc.z %q : !qc.qubit
      %body = qc.measure %q : !qc.qubit -> i1
      cbit.store %body, %out[%one] : !cbit.reg<3>
      scf.yield
    }
    %last = qc.measure %q : !qc.qubit -> i1
    cbit.store %last, %out[%two] : !cbit.reg<3>
    qc.dealloc %q : !qc.qubit
    return %out : !cbit.reg<3>
  }
}
"""
    if direct_measurement:
        source = source.replace("%snapshot = cbit.load %out[%zero] : !cbit.reg<3>", "")
        source = source.replace("scf.condition(%snapshot)", "scf.condition(%measured)")
    program = QCProgram.from_mlir_str(source)
    assert program.to_qiskit().num_clbits == 3
    check_paths(program, 2)


def test_initialization_on_first_body_and_multiple_exits() -> None:
    """The first body initializes every exit, while unreachable tails do not merge."""
    program = QCProgram.from_qasm_str("""
OPENQASM 3.1;
output bit[8] result;
uint[8] value;
uint[8] iteration = 0;
while (true) {
  iteration += 1;
  if (iteration == 2) { value = 7; break; }
  if (iteration > 3) { value = 9; break; }
  value = 3;
}
result = bit[8](value);
""")
    check_paths(program, 7)
    circuit = QuantumCircuit(1, 1)
    variable = expr.Var.new("initialized", types.Bool())
    circuit.add_uninitialized_var(variable)
    with circuit.while_loop(expr.lift(True), None, None, None, label=None):  # ruff: ignore[boolean-positional-value-in-call] - Qiskit requires a positional value.
        circuit.store(variable, expr.lift(True))  # ruff: ignore[boolean-positional-value-in-call] - Qiskit requires a positional value.
        circuit.break_loop()
    with circuit.if_test(variable):
        circuit.x(0)
    circuit.measure(0, 0)
    check_paths(QCProgram.from_qiskit(circuit), 1)
    with pytest.raises((ValueError, RuntimeError)):
        QCProgram.from_qasm_str("""
OPENQASM 3.1;
output bit result;
bool value;
while (false) { value = true; }
result = value;
""")


def test_loop_export_diagnostics_include_context() -> None:
    """Valid but unrepresentable loops report the target, feature and source location."""
    program = QCProgram.from_mlir_str("""
module {
  func.func @main() attributes {mqt.entry_point} {
    %q = qc.alloc : !qc.qubit
    %start = arith.constant 0 : i128
    %one = arith.constant 1 : i128
    %result = scf.while (%state = %start) : (i128) -> i128 {
      %next = arith.addi %state, %one : i128
      %condition = arith.cmpi ult, %state, %one : i128
      qc.x %q : !qc.qubit
      scf.condition(%condition) %next : i128
    } do {
    ^bb0(%state: i128):
      scf.yield %state : i128
    } loc("unsupported-loop.mlir":7:5)
    qc.dealloc %q : !qc.qubit
    return
  }
}
""")
    original = program.ir
    with pytest.raises(RuntimeError, match="OpenQASM") as qasm_error:
        program.to_openqasm3()
    assert "unsupported-loop.mlir" in str(qasm_error.value)
    assert "i128" in str(qasm_error.value)
    with pytest.raises(RuntimeError, match="Qiskit") as qiskit_error:
        program.to_qiskit()
    assert "scf.while" in str(qiskit_error.value)
    assert "unsupported-loop.mlir" in str(qiskit_error.value)
    assert program.ir == original


def test_runtime_gate_parameter_is_distinct_from_local_state() -> None:
    """Native classical storage does not make runtime gate angles available in Qiskit."""
    program = QCProgram.from_mlir_str("""
module {
  func.func @main() attributes {mqt.entry_point} {
    %q = qc.alloc : !qc.qubit
    %start = arith.constant 0.5 : f64
    %step = arith.constant 1.0 : f64
    %limit = arith.constant 2.0 : f64
    %result = scf.while (%angle = %start) : (f64) -> f64 {
      qc.rx(%angle) %q : !qc.qubit
      %next = arith.addf %angle, %step : f64
      %condition = arith.cmpf olt, %next, %limit : f64
      scf.condition(%condition) %next : f64
    } do {
    ^bb0(%angle: f64):
      scf.yield %angle : f64
    }
    qc.dealloc %q : !qc.qubit
    return
  }
}
""")
    with pytest.raises(RuntimeError, match="runtime classical gate parameters") as error:
        program.to_qiskit()
    assert "qc.rx" in str(error.value)
    assert "constant or symbolic" in str(error.value)


def test_loop_resource_allocation_is_actionable() -> None:
    """Constant-true loops are valid; resource allocation inside them is a target restriction."""
    program = QCProgram.from_mlir_str("""
module {
  func.func @main() attributes {mqt.entry_point} {
    %true = arith.constant true
    scf.while : () -> () {
      scf.condition(%true)
    } do {
      %q = qc.alloc : !qc.qubit
      qc.x %q : !qc.qubit
      qc.dealloc %q : !qc.qubit
      scf.yield
    }
    return
  }
}
""")
    with pytest.raises(RuntimeError, match="allocate") as qasm_error:
        program.to_openqasm3()
    assert "OpenQASM" in str(qasm_error.value)
    with pytest.raises(RuntimeError, match="allocate them before the loop") as qiskit_error:
        program.to_qiskit()
    assert "qc.alloc" in str(qiskit_error.value)


def test_first_measurement_initializes_do_while_output() -> None:
    """A measurement in the guaranteed first body defines both its test and output."""
    program = QCProgram.from_qasm_str("""
OPENQASM 3.1;
qubit q;
output bit result;
while (true) {
  x q;
  result = measure q;
  if (result) { break; }
}
""")
    check_paths(program, 1)


def test_qiskit_break_inside_switch() -> None:
    """Native switch exits carry scalar state and preserve nontrivial bit mappings."""
    circuit = QuantumCircuit(3, 2)
    counter = circuit.add_var("counter", expr.lift(0, types.Uint(8)))
    with circuit.while_loop(expr.lift(True), None, None, None, label=None):  # ruff: ignore[boolean-positional-value-in-call] - Qiskit requires a positional value.
        circuit.store(counter, expr.add(counter, expr.lift(1, types.Uint(8))))
        with circuit.switch(counter, None, None, None, label=None) as case:
            with case(1):
                circuit.store(counter, expr.lift(7, types.Uint(8)))
                circuit.x(2)
                circuit.break_loop()
            with case(case.DEFAULT):
                circuit.store(counter, expr.lift(9, types.Uint(8)))
                circuit.x(0)
                circuit.break_loop()
        circuit.x(1)
    with circuit.if_test(expr.equal(counter, expr.lift(7, types.Uint(8)))):
        circuit.x(1)
    circuit.measure(2, 0)
    circuit.measure(1, 1)
    check_paths(QCProgram.from_qiskit(circuit), 3)


def test_symbolic_gate_parameters_with_scalar_loops() -> None:
    """Introducing native loop variables preserves supported symbolic gate expressions."""
    circuit = QuantumCircuit(1, 1)
    angle = Parameter("angle")
    counter = circuit.add_var("counter", expr.lift(0, types.Uint(8)))
    circuit.x(0)
    with circuit.while_loop(expr.lift(True), None, None, None, label=None):  # ruff: ignore[boolean-positional-value-in-call] - Qiskit requires a positional value.
        circuit.store(counter, expr.add(counter, expr.lift(1, types.Uint(8))))
        with circuit.if_test(expr.greater_equal(counter, expr.lift(1, types.Uint(8)))):
            circuit.break_loop()
    circuit.rx(angle + 0.1, 0)
    circuit.measure(0, 0)
    program = QCProgram.from_qiskit(circuit)
    exported = program.to_qiskit()
    assert {parameter.name for parameter in exported.parameters} == {"angle"}
    bound = exported.assign_parameters({next(iter(exported.parameters)): math.pi - 0.1})
    check_paths(QCProgram.from_qiskit(bound), 0)
