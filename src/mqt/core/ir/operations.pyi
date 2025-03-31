# Copyright (c) 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, MutableSequence, Sequence
from typing import ClassVar, overload

from .registers import ClassicalRegister
from .symbolic import Expression, Variable

__all__ = [
    "ClassicControlledOperation",
    "ComparisonKind",
    "CompoundOperation",
    "Control",
    "NonUnitaryOperation",
    "OpType",
    "Operation",
    "StandardOperation",
    "SymbolicOperation",
]

class Control:
    """A control is a pair of a qubit and a type. The type can be either positive or negative.

    Args:
        qubit: The qubit that is the control.
        type_: The type of the control (default is positive).
    """

    class Type:
        """The type of a control. It can be either positive or negative."""

        __members__: ClassVar[dict[Control.Type, str]]
        Neg: ClassVar[Control.Type]
        r"""A negative control.

        The operation that is controlled on this qubit is only executed if the qubit is in the :math:`|0\rangle` state.
        """
        Pos: ClassVar[Control.Type]
        r"""A positive control.

        The operation that is controlled on this qubit is only executed if the qubit is in the :math:`|1\rangle` state.
        """

        def __eq__(self, other: object) -> bool: ...
        def __getstate__(self) -> int: ...
        def __hash__(self) -> int: ...
        def __index__(self) -> int: ...
        def __init__(self, value: int) -> None: ...
        def __int__(self) -> int: ...
        def __ne__(self, other: object) -> bool: ...
        def __setstate__(self, state: int) -> None: ...
        @property
        def name(self) -> str: ...
        @property
        def value(self) -> int: ...

    qubit: int
    type_: Type

    def __init__(self, qubit: int, type_: Type = ...) -> None:
        """Initialize a control.

        Args:
            qubit: The qubit that is the control.
            type_: The type of the control (default is positive).
        """

    def __eq__(self, other: object) -> bool:
        """Check if two controls are equal."""

    def __ne__(self, other: object) -> bool:
        """Check if two controls are not equal."""

    def __hash__(self) -> int:
        """Get the hash of the control."""

class OpType:
    """An Enum-like class that represents the type of an operation."""

    __members__: ClassVar[dict[OpType, str]]  # readonly
    barrier: ClassVar[OpType]
    """
    A barrier operation. It is used to separate operations in the circuit.

    See Also:
        :meth:`mqt.core.ir.QuantumComputation.barrier`
    """
    classic_controlled: ClassVar[OpType]
    """
    A classic controlled operation.

    It is used to control the execution of an operation based on the value of a classical register.

    See Also:
        :meth:`mqt.core.ir.QuantumComputation.classic_controlled`
    """
    compound: ClassVar[OpType]
    """
    A compound operation. It is used to group multiple operations into a single operation.

    See Also:
        :class:`.CompoundOperation`
    """
    dcx: ClassVar[OpType]
    """
    A DCX gate.

    See Also:
        :meth:`mqt.core.ir.QuantumComputation.dcx`
    """
    ecr: ClassVar[OpType]
    """
    An ECR gate.

    See Also:
        :meth:`mqt.core.ir.QuantumComputation.ecr`
    """
    gphase: ClassVar[OpType]
    """
    A global phase operation.

    See Also:
        :meth:`mqt.core.ir.QuantumComputation.gphase`
    """
    h: ClassVar[OpType]
    """
    A Hadamard gate.

    See Also:
        :meth:`mqt.core.ir.QuantumComputation.h`
    """
    i: ClassVar[OpType]
    """
    An identity operation.

    See Also:
        :meth:`mqt.core.ir.QuantumComputation.i`
    """
    iswap: ClassVar[OpType]
    """
    An iSWAP gate.

    See Also:
        :meth:`mqt.core.ir.QuantumComputation.iswap`
    """
    iswapdg: ClassVar[OpType]
    r"""
    An :math:`i\text{SWAP}^\dagger` gate.

    See Also:
        :meth:`mqt.core.ir.QuantumComputation.iswapdg`
    """
    measure: ClassVar[OpType]
    """
    A measurement operation.

    See Also:
        :meth:`mqt.core.ir.QuantumComputation.measure`
    """
    none: ClassVar[OpType]
    """
    A placeholder operation. It is used to represent an operation that is not yet defined.
    """
    peres: ClassVar[OpType]
    """
    A Peres gate.

    See Also:
        :meth:`mqt.core.ir.QuantumComputation.peres`
    """
    peresdg: ClassVar[OpType]
    r"""
    A :math:`\text{Peres}^\dagger` gate.

    See Also:
        :meth:`mqt.core.ir.QuantumComputation.peresdg`
    """
    p: ClassVar[OpType]
    """
    A phase gate.

    See Also:
        :meth:`mqt.core.ir.QuantumComputation.p`
    """
    reset: ClassVar[OpType]
    """
    A reset operation.

    See Also:
        :meth:`mqt.core.ir.QuantumComputation.reset`
    """
    rx: ClassVar[OpType]
    r"""
    An :math:`R_x` gate.

    See Also:
        :meth:`mqt.core.ir.QuantumComputation.rx`
    """
    rxx: ClassVar[OpType]
    r"""
    An :math:`R_{xx}` gate.

    See Also:
        :meth:`mqt.core.ir.QuantumComputation.rxx`
    """
    ry: ClassVar[OpType]
    r"""
    An :math:`R_y` gate.

    See Also:
        :meth:`mqt.core.ir.QuantumComputation.ry`
    """
    ryy: ClassVar[OpType]
    r"""
    An :math:`R_{yy}` gate.

    See Also:
        :meth:`mqt.core.ir.QuantumComputation.ryy`
    """
    rz: ClassVar[OpType]
    r"""
    An :math:`R_z` gate.

    See Also:
        :meth:`mqt.core.ir.QuantumComputation.rz`
    """
    rzx: ClassVar[OpType]
    r"""
    An :math:`R_{zx}` gate.

    See Also:
        :meth:`mqt.core.ir.QuantumComputation.rzx`
    """
    rzz: ClassVar[OpType]
    r"""
    An :math:`R_{zz}` gate.

    See Also:
        :meth:`mqt.core.ir.QuantumComputation.rzz`
    """
    s: ClassVar[OpType]
    """
    An S gate.

    See Also:
        :meth:`mqt.core.ir.QuantumComputation.s`
    """
    sdg: ClassVar[OpType]
    r"""
    An :math:`S^\dagger` gate.

    See Also:
        :meth:`mqt.core.ir.QuantumComputation.sdg`
    """
    swap: ClassVar[OpType]
    """
    A SWAP gate.

    See Also:
        :meth:`mqt.core.ir.QuantumComputation.swap`
    """
    sx: ClassVar[OpType]
    r"""
    A :math:`\sqrt{X}` gate.

    See Also:
        :meth:`mqt.core.ir.QuantumComputation.sx`
    """
    sxdg: ClassVar[OpType]
    r"""
    A :math:`\sqrt{X}^\dagger` gate.

    See Also:
        :meth:`mqt.core.ir.QuantumComputation.sxdg`
    """
    t: ClassVar[OpType]
    """
    A T gate.

    See Also:
        :meth:`mqt.core.ir.QuantumComputation.t`
    """
    tdg: ClassVar[OpType]
    r"""
    A :math:`T^\dagger` gate.

    See Also:
        :meth:`mqt.core.ir.QuantumComputation.tdg`
    """
    u2: ClassVar[OpType]
    """
    A U2 gate.

    See Also:
        :meth:`mqt.core.ir.QuantumComputation.u2`
    """
    u: ClassVar[OpType]
    """
    A U gate.

    See Also:
        :meth:`mqt.core.ir.QuantumComputation.u`
    """
    v: ClassVar[OpType]
    """
    A V gate.

    See Also:
        :meth:`mqt.core.ir.QuantumComputation.v`
    """
    vdg: ClassVar[OpType]
    r"""
    A :math:`V^\dagger` gate.

    See Also:
        :meth:`mqt.core.ir.QuantumComputation.vdg`
    """
    x: ClassVar[OpType]
    """
    An X gate.

    See Also:
        :meth:`mqt.core.ir.QuantumComputation.x`
    """
    xx_minus_yy: ClassVar[OpType]
    r"""
    An :math:`R_{XX - YY}` gate.

    See Also:
        :meth:`mqt.core.ir.QuantumComputation.xx_minus_yy`
    """
    xx_plus_yy: ClassVar[OpType]
    r"""
    An :math:`R_{XX + YY}` gate.

    See Also:
        :meth:`mqt.core.ir.QuantumComputation.xx_plus_yy`
    """
    y: ClassVar[OpType]
    """
    A Y gate.

    See Also:
        :meth:`mqt.core.ir.QuantumComputation.y`
    """
    z: ClassVar[OpType]
    """
    A Z gate.

    See Also:
        :meth:`mqt.core.ir.QuantumComputation.z`
    """

    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...
    def __eq__(self, other: object) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    @overload
    def __init__(self, value: int) -> None: ...
    @overload
    def __init__(self, arg0: str) -> None: ...
    def __int__(self) -> int: ...
    def __ne__(self, other: object) -> bool: ...
    def __setstate__(self, state: int) -> None: ...

class Operation(ABC):
    """An abstract base class for operations that can be added to a :class:`~mqt.core.ir.QuantumComputation`."""

    type_: OpType
    """
    The type of the operation.
    """
    controls: set[Control]
    """
    The controls of the operation.

    Note:
        The notion of a control might not make sense for all types of operations.
    """
    targets: list[int]
    """
    The targets of the operation.

    Note:
        The notion of a target might not make sense for all types of operations.
    """
    parameter: list[float]
    """
    The parameters of the operation.

    Note:
        The notion of a parameter might not make sense for all types of operations.
    """

    @property
    def name(self) -> str:
        """The name of the operation."""

    @property
    def num_targets(self) -> int:
        """The number of targets of the operation."""

    @property
    def num_controls(self) -> int:
        """The number of controls of the operation."""

    @abstractmethod
    def add_control(self, control: Control) -> None:
        """Add a control to the operation.

        Args:
            control: The control to add.
        """

    def add_controls(self, controls: set[Control]) -> None:
        """Add multiple controls to the operation.

        Args:
            controls: The controls to add.
        """

    @abstractmethod
    def clear_controls(self) -> None:
        """Clear all controls of the operation."""

    @abstractmethod
    def remove_control(self, control: Control) -> None:
        """Remove a control from the operation.

        Args:
            control: The control to remove.
        """

    def remove_controls(self, controls: set[Control]) -> None:
        """Remove multiple controls from the operation.

        Args:
            controls: The controls to remove.
        """

    def acts_on(self, qubit: int) -> bool:
        """Check if the operation acts on a specific qubit.

        Args:
            qubit: The qubit to check.

        Returns:
            True if the operation acts on the qubit, False otherwise.
        """

    def get_used_qubits(self) -> set[int]:
        """Get the qubits that are used by the operation.

        Returns:
            The set of qubits that are used by the operation.
        """

    def is_classic_controlled_operation(self) -> bool:
        """Check if the operation is a :class:`ClassicControlledOperation`.

        Returns:
            True if the operation is a :class:`ClassicControlledOperation`, False otherwise.
        """

    def is_compound_operation(self) -> bool:
        """Check if the operation is a :class:`CompoundOperation`.

        Returns:
            True if the operation is a :class:`CompoundOperation`, False otherwise.
        """

    def is_controlled(self) -> bool:
        """Check if the operation is controlled.

        Returns:
            True if the operation is controlled, False otherwise.
        """

    def is_non_unitary_operation(self) -> bool:
        """Check if the operation is a :class:`NonUnitaryOperation`.

        Returns:
            True if the operation is a :class:`NonUnitaryOperation`, False otherwise.
        """

    def is_standard_operation(self) -> bool:
        """Check if the operation is a :class:`StandardOperation`.

        Returns:
            True if the operation is a :class:`StandardOperation`, False otherwise.
        """

    def is_symbolic_operation(self) -> bool:
        """Check if the operation is a :class:`SymbolicOperation`.

        Returns:
            True if the operation is a :class:`SymbolicOperation`, False otherwise.
        """

    def is_unitary(self) -> bool:
        """Check if the operation is unitary.

        Returns:
            True if the operation is unitary, False otherwise.
        """

    def get_inverted(self) -> Operation:
        """Get the inverse of the operation.

        Returns:
            The inverse of the operation.
        """

    @abstractmethod
    def invert(self) -> None:
        """Invert the operation (in-place)."""

    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...
    def __ne__(self, other: object) -> bool: ...

class StandardOperation(Operation):
    """Standard quantum operation.

    This class is used to represent all standard quantum operations, i.e.,
    operations that are unitary. This includes all possible quantum gates.
    Such Operations are defined by their :class:`OpType`, the qubits (controls
    and targets) they act on, and their parameters.

    Args:
        control: The control qubit(s) of the operation (if any).
        target: The target qubit(s) of the operation.
        op_type: The type of the operation.
        params: The parameters of the operation (if any).
    """

    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(
        self,
        target: int,
        op_type: OpType,
        params: Sequence[float] | None = None,
    ) -> None: ...
    @overload
    def __init__(
        self,
        targets: Sequence[int],
        op_type: OpType,
        params: Sequence[float] | None = None,
    ) -> None: ...
    @overload
    def __init__(
        self,
        control: Control,
        target: int,
        op_type: OpType,
        params: Sequence[float] | None = None,
    ) -> None: ...
    @overload
    def __init__(
        self,
        control: Control,
        targets: Sequence[int],
        op_type: OpType,
        params: Sequence[float] | None = None,
    ) -> None: ...
    @overload
    def __init__(
        self,
        controls: set[Control],
        target: int,
        op_type: OpType,
        params: Sequence[float] | None = None,
    ) -> None: ...
    @overload
    def __init__(
        self,
        controls: set[Control],
        targets: Sequence[int],
        op_type: OpType,
        params: Sequence[float] | None = None,
    ) -> None: ...
    @overload
    def __init__(
        self,
        controls: set[Control],
        target0: int,
        target1: int,
        op_type: OpType,
        params: Sequence[float] | None = None,
    ) -> None: ...
    def add_control(self, control: Control) -> None:
        """Add a control to the operation.

        :class:`StandardOperation` supports arbitrarily many controls per operation.

        Args:
            control: The control to add.
        """

    def clear_controls(self) -> None:
        """Clear all controls of the operation."""

    def remove_control(self, control: Control) -> None:
        """Remove a control from the operation.

        Args:
            control: The control to remove.
        """

    def invert(self) -> None:
        """Invert the operation (in-place).

        Since any :class:`StandardOperation` is unitary, the inverse is simply the
        conjugate transpose of the operation's matrix representation.
        """

class NonUnitaryOperation(Operation):
    """Non-unitary operation.

    This class is used to represent all non-unitary operations, i.e., operations
    that are not reversible. This includes measurements and resets.

    Args:
        targets: The target qubit(s) of the operation.
        classics: The classical bit(s) that are associated with the operation (only relevant for measurements).
        op_type: The type of the operation.
    """

    @property
    def classics(self) -> list[int]:
        """The classical registers that are associated with the operation."""

    @overload
    def __init__(self, targets: Sequence[int], classics: Sequence[int]) -> None: ...
    @overload
    def __init__(self, target: int, classic: int) -> None: ...
    @overload
    def __init__(self, targets: Sequence[int], op_type: OpType = ...) -> None: ...
    def add_control(self, control: Control) -> None:
        """Adding controls to a non-unitary operation is not supported."""

    def clear_controls(self) -> None:
        """Cannot clear controls of a non-unitary operation."""

    def remove_control(self, control: Control) -> None:
        """Removing controls from a non-unitary operation is not supported."""

    def invert(self) -> None:
        """Non-unitary operations are, per definition, not invertible."""

class CompoundOperation(Operation, MutableSequence[Operation]):
    """Compound quantum operation.

    This class is used to aggregate and group multiple operations into a single
    object. This is useful for optimizations and for representing complex
    quantum functionality. A :class:`CompoundOperation` can contain any number
    of operations, including other :class:`CompoundOperation`'s.

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
    def __getitem__(self, idx: int) -> Operation:
        """Get the operation at the given index.

        Args:
            idx: The index of the operation to get.

        Returns:
            The operation at the given index.

        Notes:
            This gives direct access to the operations in the compound operation.
        """

    @overload
    def __getitem__(self, idx: slice) -> list[Operation]:
        """Get the operations in the given slice.

        Args:
            idx: The slice of the operations to get.

        Returns:
            The operations in the given slice.

        Notes:
            This gives direct access to the operations in the compound operation.
        """

    @overload
    def __setitem__(self, idx: int, op: Operation) -> None:
        """Set the operation at the given index.

        Args:
            idx: The index of the operation to set.
            op: The operation to set at the given index.
        """

    @overload
    def __setitem__(self, idx: slice, ops: Iterable[Operation]) -> None:
        """Set the operations in the given slice.

        Args:
            idx: The slice of operations to set.
            ops: The operations to set in the given slice.
        """

    @overload
    def __delitem__(self, idx: int) -> None:
        """Delete the operation at the given index.

        Args:
            idx: The index of the operation to delete.
        """

    @overload
    def __delitem__(self, idx: slice) -> None:
        """Delete the operations in the given slice.

        Args:
            idx: The slice of operations to delete.
        """

    def insert(self, idx: int, op: Operation) -> None:
        """Insert an operation at the given index.

        Args:
            idx: The index to insert the operation at.
            op: The operation to insert.
        """

    def append(self, op: Operation) -> None:
        """Append an operation to the compound operation."""

    def empty(self) -> bool:
        """Check if the compound operation is empty."""

    def clear(self) -> None:
        """Clear all operations in the compound operation."""

    def add_control(self, control: Control) -> None:
        """Add a control to the operation.

        This will add the control to all operations in the compound operation.
        Additionally, the control is added to the compound operation itself to
        keep track of all controls that are applied to the compound operation.

        Args:
            control: The control to add.
        """

    def clear_controls(self) -> None:
        """Clear all controls of the operation.

        This will clear all controls that have been tracked in the compound
        operation itself and will clear these controls of all operations that are
        part of the compound operation.
        """

    def remove_control(self, control: Control) -> None:
        """Remove a control from the operation.

        This will remove the control from all operations in the compound operation.
        Additionally, the control is removed from the compound operation itself to
        keep track of all controls that are applied to the compound operation.

        Args:
            control: The control to remove.
        """

    def invert(self) -> None:
        """Invert the operation (in-place).

        This will invert all operations in the compound operation and reverse
        the order of the operations. This only works if all operations in the
        compound operation are invertible and will throw an error otherwise.
        """

class SymbolicOperation(StandardOperation):
    """Symbolic quantum operation.

    This class is used to represent quantum operations that are not yet fully
    defined. This can be useful for representing operations that depend on
    parameters that are not yet known. A :class:`SymbolicOperation` is defined
    by its :class:`OpType`, the qubits (controls and targets) it acts on, and
    its parameters. The parameters can be either fixed values or symbolic
    expressions.

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
        self,
        target: int,
        op_type: OpType,
        params: Sequence[Expression | float] | None = None,
    ) -> None: ...
    @overload
    def __init__(
        self,
        targets: Sequence[int],
        op_type: OpType,
        params: Sequence[Expression | float] | None = None,
    ) -> None: ...
    @overload
    def __init__(
        self,
        control: Control,
        target: int,
        op_type: OpType,
        params: Sequence[Expression | float] | None = None,
    ) -> None: ...
    @overload
    def __init__(
        self,
        control: Control,
        targets: Sequence[int],
        op_type: OpType,
        params: Sequence[Expression | float] | None = None,
    ) -> None: ...
    @overload
    def __init__(
        self,
        controls: set[Control],
        target: int,
        op_type: OpType,
        params: Sequence[Expression | float] | None = None,
    ) -> None: ...
    @overload
    def __init__(
        self,
        controls: set[Control],
        targets: Sequence[int],
        op_type: OpType,
        params: Sequence[Expression | float] | None = None,
    ) -> None: ...
    @overload
    def __init__(
        self,
        controls: set[Control],
        target0: int,
        target1: int,
        op_type: OpType,
        params: Sequence[Expression | float] | None = None,
    ) -> None: ...
    def get_parameter(self, idx: int) -> Expression | float:
        """Get the parameter at the given index.

        Args:
            idx: The index of the parameter to get.

        Returns:
            The parameter at the given index.
        """

    def get_parameters(self) -> list[Expression | float]:
        """Get all parameters of the operation.

        Returns:
            The parameters of the operation.
        """

    def get_instantiated_operation(self, assignment: Mapping[Variable, float]) -> StandardOperation:
        """Get the instantiated operation.

        Args:
            assignment: The assignment of the symbolic parameters.

        Returns:
            The instantiated operation.
        """

    def instantiate(self, assignment: Mapping[Variable, float]) -> None:
        """Instantiate the operation (in-place).

        Args:
            assignment: The assignment of the symbolic parameters.
        """

class ComparisonKind:
    """An Enum-like class that represents the kind of comparison for classic controlled operations."""

    __members__: ClassVar[dict[ComparisonKind, str]]  # readonly
    eq: ClassVar[ComparisonKind]
    """Equality comparison."""
    neq: ClassVar[ComparisonKind]
    """Inequality comparison."""
    lt: ClassVar[ComparisonKind]
    """Less than comparison."""
    leq: ClassVar[ComparisonKind]
    """Less than or equal comparison."""
    gt: ClassVar[ComparisonKind]
    """Greater than comparison."""
    geq: ClassVar[ComparisonKind]
    """Greater than or equal comparison."""

class ClassicControlledOperation(Operation):
    """Classic controlled quantum operation.

    This class is used to represent quantum operations that are controlled by
    the value of a classical register. The operation is only executed if the
    value of the classical register matches the expected value.

    Args:
        operation: The operation that is controlled.
        control_register: The classical register that controls the operation.
        expected_value: The expected value of the classical register.
        comparison_kind: The kind of comparison (default is equality).
    """

    @overload
    def __init__(
        self,
        operation: Operation,
        control_register: ClassicalRegister,
        expected_value: int = 1,
        comparison_kind: ComparisonKind = ...,
    ) -> None: ...
    @overload
    def __init__(
        self,
        operation: Operation,
        control_bit: int,
        expected_value: int = 1,
        comparison_kind: ComparisonKind = ...,
    ) -> None: ...
    @property
    def operation(self) -> Operation:
        """The operation that is classically controlled."""

    @property
    def control_register(self) -> ClassicalRegister | None:
        """The classical register that controls the operation."""

    @property
    def control_bit(self) -> int | None:
        """The classical bit that controls the operation."""

    @property
    def expected_value(self) -> int:
        """The expected value of the classical register.

        The operation is only executed if the value of the classical register matches the expected value
        based on the kind of comparison.
        If the classical register is a single bit, the expected value is either 0 or 1.
        Otherwise, the expected value is an integer that is interpreted as a binary number, where
        the least significant bit is at the start index of the classical register.
        """

    @property
    def comparison_kind(self) -> ComparisonKind:
        """The kind of comparison.

        The operation is only executed if the value of the classical register matches the expected value
        based on the kind of comparison.
        """

    def add_control(self, control: Control) -> None:
        """Adds a control to the underlying operation.

        Args:
            control: The control to add.

        See Also:
            :meth:`Operation.add_control`
        """

    def clear_controls(self) -> None:
        """Clears the controls of the underlying operation.

        See Also:
            :meth:`Operation.clear_controls`
        """

    def remove_control(self, control: Control) -> None:
        """Removes a control from the underlying operation.

        Args:
            control: The control to remove.

        See Also:
            :meth:`Operation.remove_control`
        """

    def invert(self) -> None:
        """Inverts the underlying operation.

        See Also:
            :meth:`Operation.invert`
        """
