# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Observable fixed-width integer behavior across compiler format boundaries."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from qiskit_support import supports_qiskit_translation

from mqt.core.mlir import JeffProgram, QCOProgram, QCProgram, sample

if TYPE_CHECKING:
    from qiskit import QuantumCircuit

    from mqt.core.mlir import OpenQASMProgram


def _program(width: int, body: str, result_width: int, initial: int) -> QCProgram:
    return QCProgram.from_mlir_str(f"""
module {{
  func.func @main() -> !cbit.reg<{result_width}> attributes {{mqt.entry_point}} {{
    %q = qc.alloc : !qc.qubit
    {"qc.x %q : !qc.qubit" if initial & 1 else ""}
    %source = cbit.alloc(#cbit.init<zero>) : !cbit.reg<{width}>
    %initial = arith.constant {initial} : i{width}
    cbit.write %initial, %source : i{width}, !cbit.reg<{width}>
    %index = arith.constant 0 : index
    %bit = qc.measure %q : !qc.qubit -> i1
    cbit.store %bit, %source[%index] : !cbit.reg<{width}>
    %value = cbit.read %source : !cbit.reg<{width}> -> i{width}
    %output = cbit.alloc(#cbit.init<zero>) : !cbit.reg<{result_width}>
    {body}
    cbit.write %result, %output : i{result_width}, !cbit.reg<{result_width}>
    qc.dealloc %q : !qc.qubit
    return %output : !cbit.reg<{result_width}>
  }}
}}
""")


def _observe(program: QCProgram | QuantumCircuit | OpenQASMProgram | JeffProgram, width: int) -> int:
    try:
        counts = sample(program, shots=1, seed=1)
    except ValueError as error:
        pytest.fail(f"{error}\n{program}")
    bits = next(iter(counts)).replace(" ", "")
    return int(bits[:width], 2)


def _check_paths(program: QCProgram, width: int, expected: int) -> None:
    assert _observe(program, width) == expected
    restored_qasm = program.to_openqasm3(copy=True)
    assert _observe(restored_qasm, width) == expected
    jeff = program.to_qco(copy=True).to_jeff()
    restored_jeff = JeffProgram.from_bytes(jeff.to_bytes())
    assert _observe(restored_jeff, width) == expected
    if supports_qiskit_translation():
        assert _observe(program.to_qiskit(copy=True), width) == expected


@pytest.mark.parametrize("width", [1, 3, 8, 9, 32, 64])
@pytest.mark.parametrize("predicate", ["eq", "ne", "ult", "ule", "ugt", "uge", "slt", "sle", "sgt", "sge"])
def test_integer_comparison_interchange(width: int, predicate: str) -> None:
    """Signedness belongs to the operation, including computed operands."""
    high = 1 << (width - 1)
    program = _program(
        width,
        f"""
      %zero = arith.constant 0 : i{width}
      %computed = arith.xori %value, %zero : i{width}
      %mask = arith.constant {(1 << width) - 1} : i{width}
      %other = arith.xori %value, %mask : i{width}
      %result = arith.cmpi {predicate}, %computed, %other : i{width}
    """,
        1,
        high,
    )
    expected = int(predicate in {"ne", "ugt", "uge", "slt", "sle"})
    _check_paths(program, 1, expected)
    program.cleanup()
    _check_paths(program, 1, expected)


@pytest.mark.parametrize("width", [1, 3, 8, 9, 32, 64])
def test_modular_arithmetic_and_selection(width: int) -> None:
    """Arithmetic and selection preserve the original width after promotion."""
    maximum = (1 << width) - 1
    program = _program(
        width,
        f"""
      %one = arith.constant 1 : i{width}
      %sum = arith.addi %value, %one : i{width}
      %condition = arith.cmpi ne, %sum, %one : i{width}
      %result = arith.select %condition, %sum, %value : i{width}
    """,
        width,
        maximum,
    )
    _check_paths(program, width, 0)


@pytest.mark.parametrize("width", [1, 3, 8, 9, 32, 64])
def test_logical_right_shift_high_bit(width: int) -> None:
    """The jeff adapter must not interpret a logical shift as signed."""
    high = 1 << (width - 1)
    program = _program(
        width,
        f"""
      %distance = arith.constant {width - 1} : i{width}
      %result = arith.shrui %value, %distance : i{width}
    """,
        width,
        high,
    )
    _check_paths(program, width, 1)


@pytest.mark.parametrize(("source_width", "target_width"), [(1, 3), (3, 8), (8, 9), (9, 32), (32, 64), (64, 3)])
@pytest.mark.parametrize("signed", [False, True])
def test_integer_width_casts(source_width: int, target_width: int, *, signed: bool) -> None:
    """Backend promotion must not change truncation or sign extension."""
    initial = (1 << source_width) - 1
    operation = "arith.trunci" if target_width < source_width else "arith.extsi" if signed else "arith.extui"
    program = _program(
        source_width,
        f"""
      %result = {operation} %value : i{source_width} to i{target_width}
    """,
        target_width,
        initial,
    )
    expected = (1 << target_width) - 1 if signed or target_width < source_width else initial
    _check_paths(program, target_width, expected)
    program.cleanup()
    _check_paths(program, target_width, expected)


@pytest.mark.parametrize("signed", [False, True])
def test_openqasm_wide_cast_narrowing_snapshot(*, signed: bool) -> None:
    """Truncate a wide register and retain its signed or unsigned snapshot."""
    cast_type = "int" if signed else "uint"
    initial = (1 << 149) | 129
    program = QCProgram.from_openqasm_str(f"""
OPENQASM 3.0;
qubit q;
bit[150] source = "{initial:0150b}";
{cast_type}[8] snapshot = {cast_type}[8]({cast_type}[150](source));
source = "{"0" * 150}";
output bit[2] result;
result[0] = snapshot == {(-127 if signed else 129)};
result[1] = snapshot < 0;
""")
    expected = 3 if signed else 1
    assert _observe(program, 2) == expected
    assert _observe(program.to_openqasm3(copy=True), 2) == expected
    program.cleanup()
    assert _observe(program, 2) == expected


@pytest.mark.parametrize("width", [1, 3, 8, 9, 32, 64])
@pytest.mark.parametrize("runtime", [False, True])
@pytest.mark.parametrize("direction", ["<<", ">>"])
def test_source_zero_filling_shifts(width: int, *, runtime: bool, direction: str) -> None:
    """Validate the original distance, including values lost when narrowing."""
    initial = 1 if direction == "<<" else 1 << (width - 1)
    distances = [0, width - 1, width, 256]
    for distance in distances:
        amount = "uint[16](distance)" if runtime else str(distance)
        program = QCProgram.from_openqasm_str(f"""
OPENQASM 3.0;
include "stdgates.inc";
qubit q;
{"x q;" if distance & 1 else ""}
bit[{width}] source = bit[{width}](uint[{width}]({initial}));
bit[16] distance = bit[16](uint[16]({distance}));
distance[0] = measure q;
bit[{width}] result;
result = source {direction} {amount};
""")
        expected = (initial << distance) & ((1 << width) - 1) if direction == "<<" else initial >> distance
        _check_paths(program, width, expected)
        program.cleanup()
        _check_paths(program, width, expected)


@pytest.mark.parametrize("width", [65, 150, 301])
@pytest.mark.parametrize("enabled", [False, True])
def test_wide_comparison_snapshot(width: int, *, enabled: bool) -> None:
    """Compare retained snapshots across storage mutation and format boundaries."""
    high = 1 << (width - 1)
    low_bits = high - 1
    value = low_bits | (high if enabled else 0)
    boundary = (1 << width) - 1
    signed_value = value - (1 << width) if enabled else value
    signed_boundary = boundary - (1 << width)
    predicates = ["eq", "ne", "ult", "ule", "ugt", "uge"]
    if width <= 256:
        # Local promotion also makes wide signed comparisons exportable to OpenQASM.
        predicates += ["slt", "sle", "sgt", "sge"]
    result_width = len(predicates)
    comparisons = []
    expected = 0
    for index, predicate in enumerate(predicates):
        left, right = (signed_value, signed_boundary) if predicate.startswith("s") else (value, boundary)
        lhs, rhs = ("%snapshot", "%boundary") if index % 2 == 0 else ("%boundary", "%snapshot")
        if index % 2:
            left, right = right, left
        result = {
            "eq": left == right,
            "ne": left != right,
            "lt": left < right,
            "le": left <= right,
            "gt": left > right,
            "ge": left >= right,
        }[predicate.removeprefix("u").removeprefix("s")]
        expected |= int(result) << index
        comparisons.append(f"""
    %result{index} = arith.cmpi {predicate}, {lhs}, {rhs} : i{width}
    %index{index} = arith.constant {index} : index
    cbit.store %result{index}, %output[%index{index}] : !cbit.reg<{result_width}>
""")
    initialize = "\n".join(
        f"%position{index} = arith.constant {index} : index\n"
        f"cbit.store %true, %source[%position{index}] : !cbit.reg<{width}>"
        for index in range(width - 1)
    )
    program = QCProgram.from_mlir_str(f"""
module {{
  func.func @main() -> !cbit.reg<{result_width}> attributes {{mqt.entry_point}} {{
    %q = qc.alloc : !qc.qubit
    {"qc.x %q : !qc.qubit" if enabled else ""}
    %bit = qc.measure %q : !qc.qubit -> i1
    %source = cbit.alloc(#cbit.init<zero>) : !cbit.reg<{width}>
    %true = arith.constant true
    %false = arith.constant false
    %low = arith.constant 0 : index
    {initialize}
    %position = arith.constant {width - 1} : index
    cbit.store %bit, %source[%position] : !cbit.reg<{width}>
    %snapshot = cbit.read %source : !cbit.reg<{width}> -> i{width}
    cbit.store %false, %source[%position] : !cbit.reg<{width}>
    cbit.store %false, %source[%low] : !cbit.reg<{width}>
    %boundary = arith.constant {boundary} : i{width}
    %output = cbit.alloc(#cbit.init<zero>) : !cbit.reg<{result_width}>
    {"".join(comparisons)}
    qc.dealloc %q : !qc.qubit
    return %output : !cbit.reg<{result_width}>
  }}
}}
""")
    _check_paths(program, result_width, expected)
    program.cleanup()
    if width <= 256:
        assert f"!cbit.reg<{width}>" not in str(program)
    _check_paths(program, result_width, expected)


@pytest.mark.parametrize("static_index", [False, True])
def test_wide_comparison_index_eligibility(*, static_index: bool) -> None:
    """Normalize static indices and retain packed reads for runtime indices."""
    program = QCOProgram.from_mlir_str(f"""module {{
      func.func @main() -> i1 attributes {{mqt.entry_point}} {{
        %q = qco.static 0 : !qco.qubit
        %flipped = qco.x %q : !qco.qubit
        %measured, %bit = qco.measure %flipped : !qco.qubit
        %zero = arith.constant 0 : index
        %one = arith.constant 1 : index
        %true = arith.constant true
        %index = scf.if {"%true" if static_index else "%bit"} -> index {{
          scf.yield %one : index
        }} else {{ scf.yield %zero : index }}
        %reg = cbit.alloc(#cbit.init<zero>) : !cbit.reg<150>
        cbit.store %bit, %reg[%index] : !cbit.reg<150>
        %snapshot = cbit.read %reg : !cbit.reg<150> -> i150
        %two = arith.constant 2 : i150
        %result = arith.cmpi eq, %snapshot, %two : i150
        qco.sink %measured : !qco.qubit
        return %result : i1
      }}
    }}""")
    program.run_pass_pipeline("promote-classical-memory")
    assert ("cbit.read" in str(program)) != static_index
    assert "cbit.load" not in str(program)
    assert sample(program, shots=1, seed=1) == {"1": 1}


@pytest.mark.parametrize("width", [1, 3, 8, 9, 32, 64])
def test_signed_right_shift_high_bit(width: int) -> None:
    """Arithmetic shifts retain the original sign bit in promoted integers."""
    program = _program(
        width,
        f"""
      %distance = arith.constant {width - 1} : i{width}
      %result = arith.shrsi %value, %distance : i{width}
    """,
        width,
        1 << (width - 1),
    )
    _check_paths(program, width, (1 << width) - 1)


@pytest.mark.parametrize("width", [1, 3, 8, 9, 32, 64])
@pytest.mark.parametrize("operation", ["popcount", "rotate_left", "rotate_right"])
def test_integer_intrinsics(width: int, operation: str) -> None:
    """Targets without integer intrinsics use the shared arithmetic lowering."""
    high = 1 << (width - 1)
    if operation == "popcount":
        body = f"%result = math.ctpop %value : i{width}"
        _check_paths(_program(width, body, width, (1 << width) - 1), width, width)
        return
    intrinsic = "fshl" if operation == "rotate_left" else "fshr"
    for distance in [0, 1, width]:
        body = f"""
          %amount = arith.constant {distance} : i{width}
          %result = llvm.intr.{intrinsic}(%value, %value, %amount) : (i{width}, i{width}, i{width}) -> i{width}
        """
        amount = distance % width
        expected = (
            ((high << amount) | (high >> (width - amount))) & ((1 << width) - 1)
            if operation == "rotate_left"
            else ((high >> amount) | (high << (width - amount))) & ((1 << width) - 1)
        )
        _check_paths(_program(width, body, width, high), width, expected)


@pytest.mark.parametrize("runtime", [False, True])
def test_narrow_unsigned_comparison_promotes(*, runtime: bool) -> None:
    """Constant analysis and runtime analysis use the same integer promotion."""
    operand = "uint[3](input_bits)" if runtime else "uint[3](7)"
    program = QCProgram.from_openqasm_str(f"""
OPENQASM 3.0;
include "stdgates.inc";
qubit q;
h q;
bit[3] input_bits = bit[3](uint[3](7));
bit[1] result;
result = bit[1](uint[1]({operand} > -1));
""")
    _check_paths(program, 1, 1)


@pytest.mark.parametrize("literal", ["255", "-1"])
def test_constant_integer_casts_truncate(literal: str) -> None:
    """Explicit narrowing also applies before any runtime IR is produced."""
    program = QCProgram.from_openqasm_str(f"""
OPENQASM 3.0;
include "stdgates.inc";
qubit q;
h q;
const uint[3] narrowed = uint[3]({literal});
bit[3] result = bit[3](narrowed);
""")
    _check_paths(program, 3, 7)


def test_simulator_aliases_survive_value_map_growth() -> None:
    """Constant-folded aliases must remain valid when the SSA map reallocates."""
    operations = ["%zero = arith.constant 0 : i8", "%alias0 = arith.ori %value, %zero : i8"]
    operations.extend(f"%alias{i} = arith.ori %alias{i - 1}, %zero : i8" for i in range(1, 1024))
    operations.append("%result = arith.ori %alias1023, %zero : i8")
    assert _observe(_program(8, "\n".join(operations), 8, 173), 8) == 173


def test_jeff_rejects_wider_general_integer_expressions() -> None:
    """The wide comparison fast path must not imply general multiword support."""
    program = _program(
        65,
        """
      %mask = arith.constant 3 : i65
      %result = arith.andi %value, %mask : i65
    """,
        65,
        1,
    )
    with pytest.raises(RuntimeError, match="Compiler action failed"):
        program.to_qco().to_jeff()


@pytest.mark.parametrize("initial", [0, 1 << 149])
def test_wide_qiskit_signed_values(initial: int) -> None:
    """Preserve sign bits through widening, arithmetic shifts, and comparison."""
    if not supports_qiskit_translation():
        pytest.skip("Qiskit translation is unavailable")
    program = _program(
        150,
        """
      %extended = arith.extsi %value : i150 to i151
      %one = arith.constant 1 : i151
      %half = arith.shrsi %extended, %one : i151
      %zero = arith.constant 0 : i151
      %result = arith.cmpi slt, %half, %zero : i151
    """,
        1,
        initial,
    )
    signed = initial - (1 << 150) if initial else 0
    expected = int((signed >> 1) < 0)
    assert _observe(program, 1) == expected
    assert _observe(program.to_qiskit(copy=True), 1) == expected
    program.cleanup()
    assert _observe(program.to_qiskit(), 1) == expected
