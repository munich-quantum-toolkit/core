# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Observable fixed-width integer behavior across compiler format boundaries."""

from __future__ import annotations

import os

import pytest
import qiskit
from packaging.version import Version

from mqt.core.mlir import JeffProgram, QCProgram

supports_qiskit_translation = Version("2.5.0") <= Version(qiskit.__version__) < Version(
    "2.6.0"
) or qiskit.__version__ == os.environ.get("MQT_QISKIT_TEST_CANDIDATE_VERSION")


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


def _observe(program: QCProgram, width: int) -> int:
    qco = program.to_qco(copy=True)
    try:
        counts = qco.sample(shots=1, seed=1)
    except ValueError as error:
        pytest.fail(f"{error}\n{qco}")
    bits = next(iter(counts)).replace(" ", "")
    return int(bits[:width], 2)


def _check_paths(program: QCProgram, width: int, expected: int) -> None:
    assert _observe(program, width) == expected
    restored_qasm = QCProgram.from_qasm_str(program.to_openqasm3().source)
    assert _observe(restored_qasm, width) == expected
    jeff = program.to_qco(copy=True).to_jeff()
    restored_jeff = JeffProgram.from_bytes(jeff.to_bytes()).to_qco().to_qc()
    assert _observe(restored_jeff, width) == expected
    assert _observe(QCProgram.from_qasm_str(restored_jeff.to_openqasm3().source), width) == expected
    if supports_qiskit_translation:
        for candidate in (program, restored_jeff):
            assert _observe(QCProgram.from_qiskit(candidate.to_qiskit()), width) == expected


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


@pytest.mark.parametrize("width", [1, 3, 8, 9, 32, 64])
@pytest.mark.parametrize("runtime", [False, True])
@pytest.mark.parametrize("direction", ["<<", ">>"])
def test_source_zero_filling_shifts(width: int, *, runtime: bool, direction: str) -> None:
    """Validate the original distance, including values lost when narrowing."""
    initial = 1 if direction == "<<" else 1 << (width - 1)
    distances = [0, width - 1, width, 256]
    for distance in distances:
        amount = "uint[16](distance)" if runtime else str(distance)
        program = QCProgram.from_qasm_str(f"""
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


@pytest.mark.parametrize("width", [65, 301])
def test_wide_comparison_snapshot_through_jeff(width: int) -> None:
    """Shared comparisons read the old register, even across a later store."""
    program = QCProgram.from_mlir_str(f"""
module {{
  func.func @main() -> !cbit.reg<1> attributes {{mqt.entry_point}} {{
    %q = qc.alloc : !qc.qubit
    qc.h %q : !qc.qubit
    %source = cbit.alloc(#cbit.init<zero>) : !cbit.reg<{width}>
    %position = arith.constant {width - 1} : index
    %true = arith.constant true
    %false = arith.constant false
    cbit.store %true, %source[%position] : !cbit.reg<{width}>
    %snapshot = cbit.read %source : !cbit.reg<{width}> -> i{width}
    cbit.store %false, %source[%position] : !cbit.reg<{width}>
    %high = arith.constant {1 << (width - 1)} : i{width}
    %zero = arith.constant 0 : i{width}
    %equal = arith.cmpi eq, %snapshot, %high : i{width}
    %positive = arith.cmpi ult, %zero, %snapshot : i{width}
    %result = arith.andi %equal, %positive : i1
    %output = cbit.alloc(#cbit.init<zero>) : !cbit.reg<1>
    cbit.write %result, %output : i1, !cbit.reg<1>
    qc.dealloc %q : !qc.qubit
    return %output : !cbit.reg<1>
  }}
}}
""")
    for cleanup in [False, True]:
        if cleanup:
            program.cleanup()
        assert _observe(program, 1) == 1
        jeff = program.to_qco(copy=True).to_jeff()
        restored = JeffProgram.from_bytes(jeff.to_bytes()).to_qco().to_qc()
        assert _observe(restored, 1) == 1


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
    program = QCProgram.from_qasm_str(f"""
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
    program = QCProgram.from_qasm_str(f"""
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
    with pytest.raises(RuntimeError, match="MLIR operation failed"):
        program.to_qco().to_jeff()
