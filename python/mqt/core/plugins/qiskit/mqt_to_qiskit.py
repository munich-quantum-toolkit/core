# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Functionality for translating from the MQT to Qiskit."""

from __future__ import annotations

from typing import TYPE_CHECKING

from qiskit.circuit import (
    AncillaRegister,
    ClassicalRegister,
    Clbit,
    IfElseOp,
    QuantumCircuit,
    QuantumRegister,
    Qubit,
)
from qiskit.circuit.classical import expr
from qiskit.circuit.library import (
    DCXGate,
    ECRGate,
    HGate,
    IGate,
    PhaseGate,
    RXGate,
    RXXGate,
    RYGate,
    RYYGate,
    RZGate,
    RZXGate,
    RZZGate,
    SdgGate,
    SGate,
    SwapGate,
    SXdgGate,
    SXGate,
    TdgGate,
    TGate,
    U2Gate,
    U3Gate,
    XGate,
    XXMinusYYGate,
    XXPlusYYGate,
    YGate,
    ZGate,
    iSwapGate,
)
from qiskit.transpiler.layout import Layout, TranspileLayout

from ...ir import Permutation
from ...ir.operations import (
    ComparisonKind,
    CompoundOperation,
    Control,
    IfElseOperation,
    NonUnitaryOperation,
    Operation,
    OpType,
    StandardOperation,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from qiskit.circuit.singleton import SingletonGate

    from ...ir import QuantumComputation

__all__ = ["mqt_to_qiskit"]


def __dir__() -> list[str]:
    return __all__


def _translate_controls(controls: set[Control], qubit_map: Mapping[int, Qubit]) -> tuple[list[Qubit], str]:
    """Translate a set of :class:`~mqt.core.ir.operations.Control` to Qiskit.

    Args:
        controls: The controls to translate.
        qubit_map: A mapping from qubit indices to Qiskit :class:`~qiskit.circuit.Qubit`.

    Returns:
        A tuple containing the translated qubits and control states.
    """
    qubits: list[Qubit] = []
    ctrl_state: str = ""
    for control in controls:
        qubit = qubit_map[control.qubit]
        qubits.append(qubit)
        # MSB to the left
        ctrl_state = "1" + ctrl_state if control.type_ == Control.Type.Pos else "0" + ctrl_state
    return qubits, ctrl_state


def _translate_targets(targets: Sequence[int], qubit_map: Mapping[int, Qubit]) -> list[Qubit]:
    """Translate a sequence of target qubit indices to a list of Qiskit :class:`~qiskit.circuit.Qubit`.

    Args:
        targets: The target qubit indices to translate.
        qubit_map: A mapping from qubit indices to Qiskit :class:`~qiskit.circuit.Qubit`.

    Returns:
        The translated qubits.
    """
    return [qubit_map[target] for target in targets]


def _add_standard_operation(circ: QuantumCircuit, op: StandardOperation, qubit_map: Mapping[int, Qubit]) -> None:
    """Add a :class:`~mqt.core.ir.operations.StandardOperation`.

    Args:
        circ: The Qiskit circuit to add the operation to.
        op: The MQT operation to add.
        qubit_map: A mapping from qubit indices to Qiskit :class:`~qiskit.circuit.Qubit`.

    Raises:
        TypeError: If the operation type is not supported.
    """
    targets = _translate_targets(op.targets, qubit_map)

    if op.type_ == OpType.barrier:
        circ.barrier(targets)
        return

    controls, ctrl_state = _translate_controls(op.controls, qubit_map)

    gate_map_singleton: dict[OpType, SingletonGate] = {
        OpType.i: IGate(),
        OpType.x: XGate(),
        OpType.y: YGate(),
        OpType.z: ZGate(),
        OpType.h: HGate(),
        OpType.s: SGate(),
        OpType.sdg: SdgGate(),
        OpType.t: TGate(),
        OpType.tdg: TdgGate(),
        OpType.sx: SXGate(),
        OpType.sxdg: SXdgGate(),
        OpType.dcx: DCXGate(),
        OpType.ecr: ECRGate(),
        OpType.swap: SwapGate(),
        OpType.iswap: iSwapGate(),
    }

    if op.type_ in gate_map_singleton:
        gate = gate_map_singleton[op.type_]
        if len(controls) == 0:
            circ.append(gate, targets)
        else:
            circ.append(gate.control(len(controls), ctrl_state=ctrl_state), [*controls, *targets])
        return

    gate_map_single_param: dict[OpType, type] = {
        OpType.rx: RXGate,
        OpType.ry: RYGate,
        OpType.rz: RZGate,
        OpType.p: PhaseGate,
        OpType.rxx: RXXGate,
        OpType.ryy: RYYGate,
        OpType.rzz: RZZGate,
        OpType.rzx: RZXGate,
    }

    if op.type_ in gate_map_single_param:
        gate = gate_map_single_param[op.type_]
        parameter = op.parameter[0]
        if len(controls) == 0:
            circ.append(gate(parameter), targets)
        else:
            circ.append(gate(parameter).control(len(controls), ctrl_state=ctrl_state), [*controls, *targets])
        return

    gate_map_two_param: dict[OpType, type] = {
        OpType.u2: U2Gate,
        OpType.xx_plus_yy: XXPlusYYGate,
        OpType.xx_minus_yy: XXMinusYYGate,
    }

    if op.type_ in gate_map_two_param:
        gate = gate_map_two_param[op.type_]
        parameter1, parameter2 = op.parameter
        if len(controls) == 0:
            circ.append(gate(parameter1, parameter2), targets)
        else:
            circ.append(
                gate(parameter1, parameter2).control(len(controls), ctrl_state=ctrl_state), [*controls, *targets]
            )
        return

    gate_map_three_param: dict[OpType, type] = {
        OpType.u: U3Gate,
    }

    if op.type_ in gate_map_three_param:
        gate = gate_map_three_param[op.type_]
        parameter1, parameter2, parameter3 = op.parameter
        if len(controls) == 0:
            circ.append(gate(parameter1, parameter2, parameter3), targets)
        else:
            circ.append(
                gate(parameter1, parameter2, parameter3).control(len(controls), ctrl_state=ctrl_state),
                [*controls, *targets],
            )
        return

    msg = f"Unsupported operation type: {op.type_}"
    raise TypeError(msg)


def _add_non_unitary_operation(
    circ: QuantumCircuit,
    op: NonUnitaryOperation,
    qubit_map: Mapping[int, Qubit],
    clbit_map: Mapping[int, Clbit],
) -> None:
    """Add a :class:`~mqt.core.ir.operations.NonUnitaryOperation`.

    Args:
        circ: The Qiskit circuit to add the operation to.
        op: The MQT operation to add.
        qubit_map: A mapping from qubit indices to Qiskit :class:`~qiskit.circuit.Qubit`.
        clbit_map: A mapping from classical bit indices to Qiskit :class:`~qiskit.circuit.Clbit`.
    """
    if op.type_ == OpType.measure:
        for qubit, clbit in zip(op.targets, op.classics):
            circ.measure(qubit_map[qubit], clbit_map[clbit])
        return

    if op.type_ == OpType.reset:
        for qubit in op.targets:
            circ.reset(qubit_map[qubit])
        return


def _add_compound_operation(
    circ: QuantumCircuit,
    op: CompoundOperation,
    qubit_map: Mapping[int, Qubit],
    clbit_map: Mapping[int, Clbit],
) -> None:
    """Add a :class:`~mqt.core.ir.operations.CompoundOperation`.

    Args:
        circ: The Qiskit circuit to add the operation to.
        op: The MQT operation to add.
        qubit_map: A mapping from qubit indices to Qiskit :class:`~qiskit.circuit.Qubit`.
        clbit_map: A mapping from classical bit indices to Qiskit :class:`~qiskit.circuit.Clbit`.
    """
    inner_circ = QuantumCircuit(*circ.qregs, *circ.cregs)
    for inner_op in op:
        _add_operation(inner_circ, inner_op, qubit_map, clbit_map)
    circ.append(inner_circ.to_instruction(), circ.qubits, circ.clbits)


def _add_if_else_operation(
    circ: QuantumCircuit,
    op: IfElseOperation,
    qubit_map: Mapping[int, Qubit],
    clbit_map: Mapping[int, Clbit],
) -> None:
    """Add a :class:`~.IfElseOperation`.

    Args:
        circ: The Qiskit circuit to add the operation to.
        op: The :class:`~.IfElseOperation` operation to add.
        qubit_map: A mapping from qubit indices to Qiskit :class:`~qiskit.circuit.Qubit`.
        clbit_map: A mapping from classical bit indices to Qiskit :class:`~qiskit.circuit.Clbit`.
    """
    if op.control_register is not None:
        left_hand_side = next(reg for reg in circ.cregs if reg.name == op.control_register.name)
        right_hand_side = op.expected_value_register
    else:
        assert op.control_bit is not None
        left_hand_side = clbit_map[op.control_bit]
        right_hand_side = op.expected_value_bit

    if op.comparison_kind == ComparisonKind.eq:
        condition = expr.equal(left_hand_side, right_hand_side)
    elif op.comparison_kind == ComparisonKind.neq:
        condition = expr.not_equal(left_hand_side, right_hand_side)
    elif op.comparison_kind == ComparisonKind.lt:
        condition = expr.less(left_hand_side, right_hand_side)
    elif op.comparison_kind == ComparisonKind.leq:
        condition = expr.less_equal(left_hand_side, right_hand_side)
    elif op.comparison_kind == ComparisonKind.gt:
        condition = expr.greater(left_hand_side, right_hand_side)
    elif op.comparison_kind == ComparisonKind.geq:
        condition = expr.greater_equal(left_hand_side, right_hand_side)

    then_circ = QuantumCircuit(*circ.qregs, *circ.cregs)
    _add_operation(then_circ, op.then_operation, qubit_map, clbit_map)

    else_circ: QuantumCircuit | None = None
    if op.else_operation is not None:
        else_circ = QuantumCircuit(*circ.qregs, *circ.cregs)
        _add_operation(else_circ, op.else_operation, qubit_map, clbit_map)

    circ.append(IfElseOp(condition, then_circ, else_circ), circ.qubits, circ.clbits)


def _add_operation(
    circ: QuantumCircuit,
    op: Operation,
    qubit_map: Mapping[int, Qubit],
    clbit_map: Mapping[int, Clbit],
) -> None:
    """Add an operation to a Qiskit circuit.

    Args:
        circ: The Qiskit circuit to add the operation to.
        op: The MQT operation to add.
        qubit_map: A mapping from qubit indices to Qiskit :class:`~qiskit.circuit.Qubit`.
        clbit_map: A mapping from classical bit indices to Qiskit :class:`~qiskit.circuit.Clbit`.

    Raises:
        TypeError: If the operation type is not supported.
    """
    if isinstance(op, StandardOperation):
        _add_standard_operation(circ, op, qubit_map)
    elif isinstance(op, NonUnitaryOperation):
        _add_non_unitary_operation(circ, op, qubit_map, clbit_map)
    elif isinstance(op, CompoundOperation):
        _add_compound_operation(circ, op, qubit_map, clbit_map)
    elif isinstance(op, IfElseOperation):
        _add_if_else_operation(circ, op, qubit_map, clbit_map)
    else:
        msg = f"Unsupported operation type: {type(op)}"
        raise TypeError(msg)


def mqt_to_qiskit(qc: QuantumComputation, *, set_layout: bool = False) -> QuantumCircuit:
    """Convert a :class:`~mqt.core.ir.QuantumComputation` to a Qiskit :class:`~qiskit.circuit.QuantumCircuit`.

    Args:
        qc: The MQT circuit to convert.
        set_layout: If true, the :attr:`~qiskit.circuit.QuantumCircuit.layout` property is populated with the
                    initial layout and output permutation of the MQT circuit.

    Returns:
        The converted circuit.

    Raises:
        NotImplementedError: If the MQT circuit contains variables.
    """
    if not qc.is_variable_free():
        msg = "Converting symbolic circuits with variables to Qiskit is not yet supported."
        raise NotImplementedError(msg)

    circ = QuantumCircuit()

    if qc.name is not None:
        circ.name = qc.name

    qregs = sorted((qc.qregs | qc.ancregs).values(), key=lambda reg: reg.start)
    qubit_map: dict[int, Qubit] = {}
    for qreg in qregs:
        qiskit_reg = (
            QuantumRegister(size=qreg.size, name=qreg.name)
            if qreg.name in qc.qregs
            else AncillaRegister(size=qreg.size, name=qreg.name)
        )
        circ.add_register(qiskit_reg)
        for i, qubit in enumerate(qiskit_reg):
            qubit_map[qreg.start + i] = qubit

    cregs = sorted(qc.cregs.values(), key=lambda reg: reg.start)
    clbit_map: dict[int, Clbit] = {}
    for creg in cregs:
        qiskit_creg = ClassicalRegister(size=creg.size, name=creg.name)
        circ.add_register(qiskit_creg)
        for i, clbit in enumerate(qiskit_creg):
            clbit_map[creg.start + i] = clbit

    for op in qc:
        _add_operation(circ, op, qubit_map, clbit_map)

    if not set_layout:
        return circ

    # create a list of physical qubits initialized to none, but with the correct length
    p2v: list[Qubit | None] = [None] * len(circ.qubits)
    # fill the list with the correct virtual qubits
    for virtual, physical in qc.initial_layout.items():
        p2v[virtual] = qubit_map[physical]
    initial_layout = Layout().from_qubit_list(p2v, *circ.qregs)

    # reconstruct the final layout, which is the permutation between the initial layout and the output permutation
    permutation = Permutation()
    for physical, virtual in qc.output_permutation.items():
        # find the virtual qubit in the initial layout and store the corresponding physical qubit
        for p, v in qc.initial_layout.items():
            if v == virtual:
                permutation[p] = physical
                continue

    p2v = [None] * len(circ.qubits)
    # fill the list with the correct virtual qubits
    for physical, virtual in permutation.items():
        p2v[virtual] = qubit_map[physical]
    final_layout = Layout().from_qubit_list(p2v, *circ.qregs)

    circ._layout = TranspileLayout(  # noqa: SLF001
        initial_layout=initial_layout,
        input_qubit_mapping={qubit: idx for idx, qubit in qubit_map.items()},
        final_layout=final_layout,
        _input_qubit_count=qc.num_qubits,
        _output_qubit_list=list(final_layout.get_virtual_bits()),
    )

    return circ
