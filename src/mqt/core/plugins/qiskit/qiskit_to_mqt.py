# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Functionality for translating from Qiskit to the MQT."""

from __future__ import annotations

import re
import warnings
from typing import TYPE_CHECKING, cast

from qiskit.circuit import AncillaRegister, Clbit, Qubit

from ...ir import QuantumComputation
from ...ir.operations import (
    CompoundOperation,
    Control,
    NonUnitaryOperation,
    OpType,
    StandardOperation,
    SymbolicOperation,
)
from ...ir.symbolic import Expression, Term, Variable

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from qiskit.circuit import Instruction, ParameterExpression, QuantumCircuit


__all__ = ["qiskit_to_mqt"]


def __dir__() -> list[str]:
    return __all__


def qiskit_to_mqt(circ: QuantumCircuit) -> QuantumComputation:
    """Convert a Qiskit :class:`~qiskit.circuit.QuantumCircuit` to a :class:`~mqt.core.ir.QuantumComputation` object.

    Args:
        circ: The Qiskit circuit to convert.

    Returns:
        The converted circuit.
    """
    qc = QuantumComputation()

    if circ.name is not None:
        qc.name = circ.name

    qubit_index = 0
    qubit_map: dict[Qubit, int] = {}
    for reg in circ.qregs:
        size = reg.size
        if size == 0:
            continue
        if isinstance(reg, AncillaRegister):
            qc.add_ancillary_register(size, reg.name)
        else:
            qc.add_qubit_register(size, reg.name)
        for qubit in reg:
            qubit_map[qubit] = qubit_index
            qubit_index += 1

    clbit_index = 0
    clbit_map: dict[Clbit, int] = {}
    for reg in circ.cregs:
        size = reg.size
        if size == 0:
            continue
        qc.add_classical_register(size, reg.name)
        for bit in reg:
            clbit_map[bit] = clbit_index
            clbit_index += 1

    try:
        qc.global_phase = circ.global_phase
    except TypeError:
        warnings.warn(
            "Symbolic global phase values are not supported yet. Setting global phase to 0.",
            RuntimeWarning,
            stacklevel=2,
        )
        qc.global_phase = 0

    for instruction in circ.data:
        symb_params = _emplace_operation(
            qc,
            instruction.operation,
            instruction.qubits,
            instruction.clbits,
            instruction.operation.params,
            qubit_map,
            clbit_map,
        )
        for symb_param in symb_params:
            qc.add_variable(symb_param)

    # import initial layout and output permutation if available
    if circ.layout is not None:
        _import_layouts(qc, circ)

    qc.initialize_io_mapping()
    return qc


_NATIVELY_SUPPORTED_GATES = frozenset({
    "i",
    "id",
    "iden",
    "x",
    "y",
    "z",
    "h",
    "s",
    "sdg",
    "t",
    "tdg",
    "p",
    "u1",
    "rx",
    "ry",
    "rz",
    "u2",
    "u",
    "u3",
    "cx",
    "cy",
    "cz",
    "cs",
    "csdg",
    "cp",
    "cu1",
    "ch",
    "crx",
    "cry",
    "crz",
    "cu3",
    "ccx",
    "swap",
    "cswap",
    "iswap",
    "sx",
    "sxdg",
    "csx",
    "mcx",
    "mcphase",
    "mcrx",
    "mcry",
    "mcrz",
    "dcx",
    "ecr",
    "rxx",
    "ryy",
    "rzx",
    "rzz",
    "xx_minus_yy",
    "xx_plus_yy",
    "reset",
    "barrier",
    "measure",
})


def _emplace_operation(
    qc: QuantumComputation | CompoundOperation,
    instr: Instruction,
    qargs: Sequence[Qubit],
    cargs: Sequence[Clbit],
    params: Sequence[float | ParameterExpression],
    qubit_map: Mapping[Qubit, int],
    clbit_map: Mapping[Clbit, int],
) -> list[float | ParameterExpression]:
    name = instr.name

    if name not in _NATIVELY_SUPPORTED_GATES:
        try:
            return _import_definition(qc, instr.definition, qargs, cargs, qubit_map, clbit_map)
        except Exception as ex:  # PRAGMA: NO COVER
            msg = f"Unsupported gate {name} with definition {instr.definition}"
            raise NotImplementedError(msg) from ex

    qubits = [qubit_map[qubit] for qubit in qargs]

    if name == "measure":
        clbits = [clbit_map[clbit] for clbit in cargs]
        qc.append(NonUnitaryOperation(qubits, clbits))
        return []

    if name == "reset":
        qc.append(NonUnitaryOperation(qubits))
        return []

    if name == "barrier":
        qc.append(StandardOperation(qubits, OpType.barrier))
        return []

    if name in {"i", "id", "iden"}:
        return _add_operation(qc, OpType.i, qargs, params, qubit_map)

    # we keep "mcx_gray" here for compatibility reasons with older Qiskit versions, e.g., 1.0.0 where "noancilla"
    # is the default for the argument "mode" which leads to the gate name "gray_code"
    if name in {"x", "cx", "ccx", "mcx", "mcx_gray"}:
        return _add_operation(qc, OpType.x, qargs, params, qubit_map)

    if name in {"y", "cy"}:
        return _add_operation(qc, OpType.y, qargs, params, qubit_map)

    if name in {"z", "cz"}:
        return _add_operation(qc, OpType.z, qargs, params, qubit_map)

    if name in {"h", "ch"}:
        return _add_operation(qc, OpType.h, qargs, params, qubit_map)

    if name in {"s", "cs"}:
        return _add_operation(qc, OpType.s, qargs, params, qubit_map)

    if name in {"sdg", "csdg"}:
        return _add_operation(qc, OpType.sdg, qargs, params, qubit_map)

    if name == "t":
        return _add_operation(qc, OpType.t, qargs, params, qubit_map)

    if name == "tdg":
        return _add_operation(qc, OpType.tdg, qargs, params, qubit_map)

    if name in {"sx", "csx"}:
        return _add_operation(qc, OpType.sx, qargs, params, qubit_map)

    if name == "sxdg":
        return _add_operation(qc, OpType.sxdg, qargs, params, qubit_map)

    if name in {"rx", "crx", "mcrx"}:
        return _add_operation(qc, OpType.rx, qargs, params, qubit_map)

    if name in {"ry", "cry", "mcry"}:
        return _add_operation(qc, OpType.ry, qargs, params, qubit_map)

    if name in {"rz", "crz", "mcrz"}:
        return _add_operation(qc, OpType.rz, qargs, params, qubit_map)

    if name in {"p", "u1", "cp", "cu1", "mcphase"}:
        return _add_operation(qc, OpType.p, qargs, params, qubit_map)

    if name == "u2":
        return _add_operation(qc, OpType.u2, qargs, params, qubit_map)

    if name in {"u", "u3", "cu3"}:
        return _add_operation(qc, OpType.u, qargs, params, qubit_map)

    if name in {"swap", "cswap"}:
        return _add_two_target_operation(qc, OpType.swap, qargs, params, qubit_map)

    if name == "iswap":
        return _add_two_target_operation(qc, OpType.iswap, qargs, params, qubit_map)

    if name == "dcx":
        return _add_two_target_operation(qc, OpType.dcx, qargs, params, qubit_map)

    if name == "ecr":
        return _add_two_target_operation(qc, OpType.ecr, qargs, params, qubit_map)

    if name == "rxx":
        return _add_two_target_operation(qc, OpType.rxx, qargs, params, qubit_map)

    if name == "ryy":
        return _add_two_target_operation(qc, OpType.ryy, qargs, params, qubit_map)

    if name == "rzz":
        return _add_two_target_operation(qc, OpType.rzz, qargs, params, qubit_map)

    if name == "rzx":
        return _add_two_target_operation(qc, OpType.rzx, qargs, params, qubit_map)

    if name == "xx_minus_yy":
        return _add_two_target_operation(qc, OpType.xx_minus_yy, qargs, params, qubit_map)

    if name == "xx_plus_yy":
        return _add_two_target_operation(qc, OpType.xx_plus_yy, qargs, params, qubit_map)

    msg = f"Unsupported gate {name}"  # pragma: no cover
    raise NotImplementedError(msg)


_SUM_REGEX = re.compile(r"[+|-]?[^+-]+")
_PROD_REGEX = re.compile(r"[*/]?[^*/]+")


def _parse_symbolic_expression(qiskit_expr: ParameterExpression | float) -> float | Expression:
    if isinstance(qiskit_expr, float):
        return qiskit_expr

    expr_str = str(qiskit_expr).strip()
    expr = Expression()
    is_const = True
    for summand in _SUM_REGEX.findall(expr_str):
        sign = 1
        summand_no_operator = summand
        if summand[0] == "+":
            summand_no_operator = summand[1:]
        elif summand[0] == "-":
            summand_no_operator = summand[1:]
            sign = -1

        coeff = 1.0
        var = ""
        for factor in _PROD_REGEX.findall(summand_no_operator):
            is_div = False
            factor_no_operator = factor
            if factor[0] == "*":
                factor_no_operator = factor[1:]
            elif factor[0] == "/":
                factor_no_operator = factor[1:]
                is_div = True

            factor_no_operator = factor_no_operator.strip()
            if factor_no_operator.replace(".", "").isnumeric():
                f = float(factor_no_operator)
                coeff *= 1.0 / f if is_div else f
            else:
                var = factor_no_operator

        if not var:
            expr += coeff
        else:
            is_const = False
            expr += Term(Variable(var), sign * coeff)

    if is_const:
        return expr.constant
    return expr


def _add_operation(
    qc: QuantumComputation | CompoundOperation,
    type_: OpType,
    qargs: Sequence[Qubit],
    params: Sequence[float | ParameterExpression],
    qubit_map: Mapping[Qubit, int],
) -> list[float | ParameterExpression]:
    qubits = [qubit_map[qubit] for qubit in qargs]
    target = qubits.pop()
    controls = {Control(qubit) for qubit in qubits}
    parameters = [_parse_symbolic_expression(param) for param in params]
    if any(isinstance(parameter, Expression) for parameter in parameters):
        qc.append(SymbolicOperation(controls, target, type_, parameters))
    else:
        qc.append(StandardOperation(controls, target, type_, cast("list[float]", parameters)))
    return parameters


def _add_two_target_operation(
    qc: QuantumComputation | CompoundOperation,
    type_: OpType,
    qargs: Sequence[Qubit],
    params: Sequence[float | ParameterExpression],
    qubit_map: Mapping[Qubit, int],
) -> list[float | ParameterExpression]:
    qubits = [qubit_map[qubit] for qubit in qargs]
    target2 = qubits.pop()
    target1 = qubits.pop()
    controls = {Control(qubit) for qubit in qubits}
    parameters = [_parse_symbolic_expression(param) for param in params]
    if any(isinstance(parameter, Expression) for parameter in parameters):
        qc.append(SymbolicOperation(controls, target1, target2, type_, parameters))
    else:
        qc.append(StandardOperation(controls, target1, target2, type_, cast("list[float]", parameters)))
    return parameters


def _import_layouts(qc: QuantumComputation, circ: QuantumCircuit) -> None:
    qc.initial_layout.clear()
    initial_index_layout = circ.layout.initial_index_layout()
    for virtual, physical in enumerate(initial_index_layout):
        qc.initial_layout[physical] = virtual

    # Properly mark ancillary qubits
    for register in circ.layout.initial_layout.get_registers():
        if register.name != "ancilla" and not isinstance(register, AncillaRegister):
            continue
        for qubit in reversed(register):
            physical_qubit = circ.layout.initial_layout[qubit]
            virtual_qubit = qc.initial_layout[physical_qubit]
            qc.set_circuit_qubit_ancillary(virtual_qubit)

    if circ.layout.final_layout is None:
        qc.output_permutation = qc.initial_layout
        return

    qc.output_permutation.clear()
    final_index_layout = circ.layout.final_index_layout()
    for virtual, physical in enumerate(final_index_layout):
        qc.output_permutation[physical] = virtual

    # Properly mark garbage qubits
    # Any qubit in the initial layout that is not in the final layout is garbage
    for virtual_qubit in range(len(final_index_layout), len(initial_index_layout)):
        qc.set_circuit_qubit_garbage(virtual_qubit)


def _import_definition(
    qc: QuantumComputation | CompoundOperation,
    circ: QuantumCircuit,
    qargs: Sequence[Qubit],
    cargs: Sequence[Clbit],
    qubit_map: Mapping[Qubit, int],
    clbit_map: Mapping[Clbit, int],
) -> list[float | ParameterExpression]:
    qarg_map = dict(zip(circ.qubits, qargs))
    carg_map = dict(zip(circ.clbits, cargs))

    qc.append(CompoundOperation())
    comp_op = cast("CompoundOperation", qc[-1])

    params = []
    for instruction in circ.data:
        mapped_qargs = [qarg_map[qarg] for qarg in instruction.qubits]
        mapped_cargs = [carg_map[carg] for carg in instruction.clbits]
        operation = instruction.operation
        new_params = _emplace_operation(
            comp_op,
            operation,
            mapped_qargs,
            mapped_cargs,
            operation.params,
            qubit_map,
            clbit_map,
        )
        params.extend(new_params)
    return params
