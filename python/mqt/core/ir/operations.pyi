# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

import enum
from collections.abc import Iterable, Mapping, MutableSequence, Sequence
from collections.abc import Set as AbstractSet
from typing import overload

import mqt.core.ir.registers
import mqt.core.ir.symbolic

class OpType(enum.Enum):
    """Enumeration of operation types."""

    none = 0
    """
    A placeholder operation.

    It is used to represent an operation that is not yet defined.
    """

    gphase = 4
    """
    A global phase operation.

    See Also:
        :meth:`mqt.core.ir.QuantumComputation.gphase`
    """

    i = 10
    """
    An identity operation.

    See Also:
        :meth:`mqt.core.ir.QuantumComputation.i`
    """

    h = 16
    """
    A Hadamard gate.

    See Also:
        :meth:`mqt.core.ir.QuantumComputation.h`
    """

    x = 20
    """
    An X gate.

    See Also:
        :meth:`mqt.core.ir.QuantumComputation.x`
    """

    y = 24
    """
    A Y gate.

    See Also:
        :meth:`mqt.core.ir.QuantumComputation.y`
    """

    z = 30
    """
    A Z gate.

    See Also:
        :meth:`mqt.core.ir.QuantumComputation.z`
    """

    s = 34
    """
    An S gate.

    See Also:
        :meth:`mqt.core.ir.QuantumComputation.s`
    """

    sdg = 35
    r"""
    An :math:`S^\dagger` gate.

    See Also:
        :meth:`mqt.core.ir.QuantumComputation.sdg`
    """

    t = 38
    """
    A T gate.

    See Also:
        :meth:`mqt.core.ir.QuantumComputation.t`
    """

    tdg = 39
    r"""
    A :math:`T^\dagger` gate.

    See Also:
        :meth:`mqt.core.ir.QuantumComputation.tdg`
    """

    v = 40
    """
    A V gate.

    See Also:
        :meth:`mqt.core.ir.QuantumComputation.v`
    """

    vdg = 41
    r"""
    A :math:`V^\dagger` gate.

    See Also:
        :meth:`mqt.core.ir.QuantumComputation.vdg`
    """

    u = 44
    """
    A U gate.

    See Also:
        :meth:`mqt.core.ir.QuantumComputation.u`
    """

    u2 = 48
    """
    A U2 gate.

    See Also:
        :meth:`mqt.core.ir.QuantumComputation.u2`
    """

    p = 54
    """
    A phase gate.

    See Also:
        :meth:`mqt.core.ir.QuantumComputation.p`
    """

    sx = 56
    r"""
    A :math:`\sqrt{X}` gate.

    See Also:
        :meth:`mqt.core.ir.QuantumComputation.sx`
    """

    sxdg = 57
    r"""
    A :math:`\sqrt{X}^\dagger` gate.

    See Also:
        :meth:`mqt.core.ir.QuantumComputation.sxdg`
    """

    rx = 60
    """
    A :math:`R_x` gate.

    See Also:
        :meth:`mqt.core.ir.QuantumComputation.rx`
    """

    ry = 64
    """
    A :math:`R_y` gate.

    See Also:
        :meth:`mqt.core.ir.QuantumComputation.ry`
    """

    rz = 70
    """
    A :math:`R_z` gate.

    See Also:
        :meth:`mqt.core.ir.QuantumComputation.rz`
    """

    r = 164
    """
    An :math:`R` gate.

    See Also:
        :meth:`mqt.core.ir.QuantumComputation.r`
    """

    swap = 72
    """
    A SWAP gate.

    See Also:
        :meth:`mqt.core.ir.QuantumComputation.swap`
    """

    iswap = 76
    """
    A iSWAP gate.

    See Also:
        :meth:`mqt.core.ir.QuantumComputation.iswap`
    """

    iswapdg = 77
    r"""
    A :math:`i\text{SWAP}^\dagger` gate.

    See Also:
        :meth:`mqt.core.ir.QuantumComputation.iswapdg`
    """

    peres = 80
    """
    A Peres gate.

    See Also:
        :meth:`mqt.core.ir.QuantumComputation.peres`
    """

    peresdg = 81
    r"""
    A :math:`\text{Peres}^\dagger` gate.

    See Also:
        :meth:`mqt.core.ir.QuantumComputation.peresdg`
    """

    dcx = 84
    """
    A DCX gate.

    See Also:
        :meth:`mqt.core.ir.QuantumComputation.dcx`
    """

    ecr = 88
    """
    An ECR gate.

    See Also:
        :meth:`mqt.core.ir.QuantumComputation.ecr`
    """

    rxx = 92
    """
    A :math:`R_{xx}` gate.

    See Also:
        :meth:`mqt.core.ir.QuantumComputation.rxx`
    """

    ryy = 96
    """
    A :math:`R_{yy}` gate.

    See Also:
        :meth:`mqt.core.ir.QuantumComputation.ryy`
    """

    rzz = 102
    """
    A :math:`R_{zz}` gate.

    See Also:
        :meth:`mqt.core.ir.QuantumComputation.rzz`
    """

    rzx = 104
    """
    A :math:`R_{zx}` gate.

    See Also:
        :meth:`mqt.core.ir.QuantumComputation.rzx`
    """

    xx_minus_yy = 108
    """
    A :math:`R_{XX - YY}` gate.

    See Also:
        :meth:`mqt.core.ir.QuantumComputation.xx_minus_yy`
    """

    xx_plus_yy = 112
    """
    A :math:`R_{XX + YY}` gate.

    See Also:
        :meth:`mqt.core.ir.QuantumComputation.xx_plus_yy`
    """

    compound = 116
    """
    A compound operation.

    It is used to group multiple operations into a single operation.

    See also :class:`.CompoundOperation`
    """

    measure = 120
    """
    A measurement operation.

    See Also:
        :meth:`mqt.core.ir.QuantumComputation.measure`
    """

    reset = 124
    """
    A reset operation.

    See Also:
        :meth:`mqt.core.ir.QuantumComputation.reset`
    """

    barrier = 14
    """
    A barrier operation.

    It is used to separate operations in the circuit.

    See Also:
        :meth:`mqt.core.ir.QuantumComputation.barrier`
    """

    if_else = 128
    """
    An if-else operation.

    It is used to control the execution of an operation based on the value of a classical register.

    See Also:
        :meth:`mqt.core.ir.QuantumComputation.if_else`
    """

class Control:
    """A control is a pair of a qubit and a type. The type can be either positive or negative.

    Args:
        qubit: The qubit that is the control.
        type_: The type of the control.
    """

    def __init__(self, qubit: int, type_: Control.Type = ...) -> None: ...

    class Type(enum.Enum):
        """Enumeration of control types."""

        Pos = 1

        Neg = 0

    @property
    def qubit(self) -> int:
        """The qubit that is the control."""

    @property
    def type_(self) -> Control.Type:
        """The type of the control."""

    def __eq__(self, arg: object, /) -> bool: ...
    def __ne__(self, arg: object, /) -> bool: ...
    def __hash__(self) -> int: ...

class Operation:
    @property
    def name(self) -> str:
        """The name of the operation."""

    @property
    def type_(self) -> OpType:
        """The type of the operation."""

    @type_.setter
    def type_(self, arg: OpType, /) -> None: ...
    @property
    def targets(self) -> list[int]:
        """The targets of the operation.

        Note:
            The notion of a target might not make sense for all types of operations.
        """

    @targets.setter
    def targets(self, arg: Sequence[int], /) -> None: ...
    @property
    def num_targets(self) -> int:
        """The number of targets of the operation."""

    @property
    def controls(self) -> set[Control]:
        """The controls of the operation.

        Note:
            The notion of a control might not make sense for all types of operations.
        """

    @controls.setter
    def controls(self, arg: AbstractSet[Control], /) -> None: ...
    @property
    def num_controls(self) -> int:
        """The number of controls of the operation."""

    def add_control(self, control: Control) -> None:
        """Add a control to the operation.

        Args:
            control: The control to add.
        """

    def add_controls(self, controls: AbstractSet[Control]) -> None:
        """Add multiple controls to the operation.

        Args:
            controls: The controls to add.
        """

    def clear_controls(self) -> None:
        """Clear all controls of the operation."""

    def remove_control(self, control: Control) -> None:
        """Remove a control from the operation.

        Args:
            control: The control to remove.
        """

    def remove_controls(self, controls: AbstractSet[Control]) -> None:
        """Remove multiple controls from the operation.

        Args:
            controls: The controls to remove.
        """

    def get_used_qubits(self) -> set[int]:
        """Get the qubits that are used by the operation.

        Returns:
            The set of qubits that are used by the operation.
        """

    def acts_on(self, qubit: int) -> bool:
        """Check if the operation acts on a specific qubit.

        Args:
            qubit: The qubit to check.

        Returns:
            True if the operation acts on the qubit, False otherwise.
        """

    @property
    def parameter(self) -> list[float]:
        """The parameters of the operation.

        Note:
            The notion of a parameter might not make sense for all types of operations.
        """

    @parameter.setter
    def parameter(self, arg: Sequence[float], /) -> None: ...
    def is_unitary(self) -> bool:
        """Check if the operation is unitary.

        Returns:
            True if the operation is unitary, False otherwise.
        """

    def is_standard_operation(self) -> bool:
        """Check if the operation is a :class:`StandardOperation`.

        Returns:
            True if the operation is a :class:`StandardOperation`, False otherwise.
        """

    def is_compound_operation(self) -> bool:
        """Check if the operation is a :class:`CompoundOperation`.

        Returns:
            True if the operation is a :class:`CompoundOperation`, False otherwise.
        """

    def is_non_unitary_operation(self) -> bool:
        """Check if the operation is a :class:`NonUnitaryOperation`.

        Returns:
            True if the operation is a :class:`NonUnitaryOperation`, False otherwise.
        """

    def is_if_else_operation(self) -> bool:
        """Check if the operation is a :class:`IfElseOperation`.

        Returns:
            True if the operation is a :class:`IfElseOperation`, False otherwise.
        """

    def is_symbolic_operation(self) -> bool:
        """Check if the operation is a :class:`SymbolicOperation`.

        Returns:
            True if the operation is a :class:`SymbolicOperation`, False otherwise.
        """

    def is_controlled(self) -> bool:
        """Check if the operation is controlled.

        Returns:
            True if the operation is controlled, False otherwise.
        """

    def get_inverted(self) -> Operation:
        """Get the inverse of the operation.

        Returns:
            The inverse of the operation.
        """

    def invert(self) -> None:
        """Invert the operation (in-place)."""

    def __eq__(self, arg: object, /) -> bool: ...
    def __ne__(self, arg: object, /) -> bool: ...
    def __hash__(self) -> int: ...

class StandardOperation(Operation):
    """Standard quantum operation.

    This class is used to represent all standard quantum operations, i.e., operations that are unitary.
    This includes all possible quantum gates.
    Such Operations are defined by their :class:`OpType`, the qubits (controls and targets) they act on, and their parameters.

    Args:
        control: The control qubit(s) of the operation (if any).
        target: The target qubit(s) of the operation.
        op_type: The type of the operation.
        params: The parameters of the operation (if any).
    """

    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(self, target: int, op_type: OpType, params: Sequence[float] = ...) -> None: ...
    @overload
    def __init__(self, targets: Sequence[int], op_type: OpType, params: Sequence[float] = ...) -> None: ...
    @overload
    def __init__(self, control: Control, target: int, op_type: OpType, params: Sequence[float] = ...) -> None: ...
    @overload
    def __init__(
        self, control: Control, targets: Sequence[int], op_type: OpType, params: Sequence[float] = ...
    ) -> None: ...
    @overload
    def __init__(
        self, controls: AbstractSet[Control], target: int, op_type: OpType, params: Sequence[float] = ...
    ) -> None: ...
    @overload
    def __init__(
        self, controls: AbstractSet[Control], targets: Sequence[int], op_type: OpType, params: Sequence[float] = ...
    ) -> None: ...
    @overload
    def __init__(
        self, controls: AbstractSet[Control], target0: int, target1: int, op_type: OpType, params: Sequence[float] = ...
    ) -> None: ...

class CompoundOperation(Operation, MutableSequence[Operation]):
    """Compound quantum operation.

    This class is used to aggregate and group multiple operations into a single object.
    This is useful for optimizations and for representing complex quantum functionality.
    A :class:`CompoundOperation` can contain any number of operations, including other :class:`CompoundOperation`'s.

    Args:
        ops: The operations that are part of the compound operation.
    """

    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(self, ops: Sequence[Operation]) -> None: ...
    def __len__(self) -> int:
        """The number of operations in the compound operation."""

    @overload
    def __getitem__(self, index: int) -> Operation:
        """Get the operation at the given index.

        Note:
            This gives direct access to the operations in the compound operation

        Args:
            index: The index of the operation to get.

        Returns:
            The operation at the given index.
        """

    @overload
    def __getitem__(self, index: slice) -> list[Operation]:
        """Get the operations in the given slice.

        Note:
            This gives direct access to the operations in the compound operation.

        Args:
            index: The slice of the operations to get.

        Returns:
            The operations in the given slice.
        """

    @overload
    def __setitem__(self, index: int, value: Operation) -> None:
        """Set the operation at the given index.

        Args:
            index: The index of the operation to set.
            value: The operation to set at the given index.
        """

    @overload
    def __setitem__(self, index: slice, value: Iterable[Operation]) -> None:
        """Set the operations in the given slice.

        Args:
            index: The slice of operations to set.
            value: The operations to set in the given slice.
        """

    @overload
    def __delitem__(self, index: int) -> None:
        """Delete the operation at the given index.

        Args:
            index: The index of the operation to delete.
        """

    @overload
    def __delitem__(self, index: slice) -> None:
        """Delete the operations in the given slice.

        Args:
            index: The slice of operations to delete.
        """

    def append(self, value: Operation) -> None:
        """Append an operation to the compound operation."""

    def insert(self, index: int, value: Operation) -> None:
        """Insert an operation at the given index.

        Args:
            index: The index to insert the operation at.
            value: The operation to insert.
        """

    def empty(self) -> bool:
        """Check if the compound operation is empty."""

    def clear(self) -> None:
        """Clear all operations in the compound operation."""

class NonUnitaryOperation(Operation):
    """Non-unitary operation.

    This class is used to represent all non-unitary operations, i.e., operations that are not reversible.
    This includes measurements and resets.

    Args:
        targets: The target qubit(s) of the operation.
        classics: The classical bit(s) that are associated with the operation (only relevant for measurements).
        op_type: The type of the operation.
    """

    @overload
    def __init__(self, targets: Sequence[int], classics: Sequence[int]) -> None: ...
    @overload
    def __init__(self, target: int, classic: int) -> None: ...
    @overload
    def __init__(self, targets: Sequence[int], op_type: OpType = ...) -> None: ...
    @property
    def classics(self) -> list[int]:
        """The classical bits that are associated with the operation."""

class SymbolicOperation(StandardOperation):
    """Symbolic quantum operation.

    This class is used to represent quantum operations that are not yet fully defined.
    This can be useful for representing operations that depend on parameters that are not yet known.
    A :class:`SymbolicOperation` is defined by its :class:`OpType`, the qubits (controls and targets) it acts on, and its parameters.
    The parameters can be either fixed values or symbolic expressions.

    Args:
         controls: The control qubit(s) of the operation (if any).
         targets: The target qubit(s) of the operation.
         op_type: The type of the operation.
         params: The parameters of the operation (if any).
    """

    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(
        self, target: int, op_type: OpType, params: Sequence[mqt.core.ir.symbolic.Expression | float] = ...
    ) -> None: ...
    @overload
    def __init__(
        self, targets: Sequence[int], op_type: OpType, params: Sequence[mqt.core.ir.symbolic.Expression | float] = ...
    ) -> None: ...
    @overload
    def __init__(
        self,
        control: Control,
        target: int,
        op_type: OpType,
        params: Sequence[mqt.core.ir.symbolic.Expression | float] = ...,
    ) -> None: ...
    @overload
    def __init__(
        self,
        control: Control,
        targets: Sequence[int],
        op_type: OpType,
        params: Sequence[mqt.core.ir.symbolic.Expression | float] = ...,
    ) -> None: ...
    @overload
    def __init__(
        self,
        controls: AbstractSet[Control],
        target: int,
        op_type: OpType,
        params: Sequence[mqt.core.ir.symbolic.Expression | float] = ...,
    ) -> None: ...
    @overload
    def __init__(
        self,
        controls: AbstractSet[Control],
        targets: Sequence[int],
        op_type: OpType,
        params: Sequence[mqt.core.ir.symbolic.Expression | float] = ...,
    ) -> None: ...
    @overload
    def __init__(
        self,
        controls: AbstractSet[Control],
        target0: int,
        target1: int,
        op_type: OpType,
        params: Sequence[mqt.core.ir.symbolic.Expression | float] = ...,
    ) -> None: ...
    def get_parameter(self, index: int) -> mqt.core.ir.symbolic.Expression | float:
        """Get the parameter at the given index.

        Args:
             index: The index of the parameter to get.

        Returns:
             The parameter at the given index.
        """

    def get_parameters(self) -> list[mqt.core.ir.symbolic.Expression | float]:
        """Get all parameters of the operation.

        Returns:
             The parameters of the operation.
        """

    def get_instantiated_operation(
        self, assignment: Mapping[mqt.core.ir.symbolic.Variable, float]
    ) -> StandardOperation:
        """Get the instantiated operation.

        Args:
             assignment: The assignment of the symbolic parameters.

        Returns:
             The instantiated operation.
        """

    def instantiate(self, assignment: Mapping[mqt.core.ir.symbolic.Variable, float]) -> None:
        """Instantiate the operation (in-place).

        Args:
             assignment: The assignment of the symbolic parameters.
        """

class ComparisonKind(enum.Enum):
    """Enumeration of comparison types for classic-controlled operations."""

    eq = 0
    """Equality comparison."""

    neq = 1
    """Inequality comparison."""

    lt = 2
    """Less-than comparison."""

    leq = 3
    """Less-than-or-equal comparison."""

    gt = 4
    """Greater-than comparison."""

    geq = 5
    """Greater-than-or-equal comparison."""

class IfElseOperation(Operation):
    """If-else quantum operation.

    This class is used to represent an if-else operation.
    The then operation is executed if the value of the classical register matches the expected value.
    Otherwise, the else operation is executed.

    Args:
        then_operation: The operation that is executed if the condition is met.
        else_operation: The operation that is executed if the condition is not met.
        control_register: The classical register that controls the operation.
        expected_value: The expected value of the classical register.
        comparison_kind: The kind of comparison (default is equality).
    """

    @overload
    def __init__(
        self,
        then_operation: Operation,
        else_operation: Operation | None,
        control_register: mqt.core.ir.registers.ClassicalRegister,
        expected_value: int = 1,
        comparison_kind: ComparisonKind = ...,
    ) -> None: ...
    @overload
    def __init__(
        self,
        then_operation: Operation,
        else_operation: Operation | None,
        control_bit: int,
        expected_value: bool = True,
        comparison_kind: ComparisonKind = ...,
    ) -> None: ...
    @property
    def then_operation(self) -> Operation:
        """The operation that is executed if the condition is met."""

    @property
    def else_operation(self) -> Operation | None:
        """The operation that is executed if the condition is not met."""

    @property
    def control_register(self) -> mqt.core.ir.registers.ClassicalRegister | None:
        """The classical register that controls the operation."""

    @property
    def control_bit(self) -> int | None:
        """The classical bit that controls the operation."""

    @property
    def expected_value_register(self) -> int:
        """The expected value of the classical register.

        The then-operation is executed if the value of the classical register matches the expected value based on the kind of comparison.
        The expected value is an integer that is interpreted as a binary number, where the least significant bit is at the start index of the classical register.
        """

    @property
    def expected_value_bit(self) -> bool:
        """The expected value of the classical bit.

        The then-operation is executed if the value of the classical bit matches the expected value based on the kind of comparison.
        """

    @property
    def comparison_kind(self) -> ComparisonKind:
        """The kind of comparison.

        The then-operation is executed if the value of the control matches the expected value based on the kind of comparison.
        """
