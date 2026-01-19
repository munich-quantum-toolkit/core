# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

import os
from collections.abc import ItemsView, Iterable, Iterator, Mapping, MutableMapping, MutableSequence, Sequence
from collections.abc import Set as AbstractSet
from typing import overload

from . import operations as operations
from . import registers as registers
from . import symbolic as symbolic

class Permutation(MutableMapping[int, int]):
    """A class to represent a permutation of the qubits in a quantum circuit.

    Args:
        permutation: The permutation to initialize the object with.
    """

    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(self, permutation: dict[int, int]) -> None:
        """Create a permutation from a dictionary."""

    @overload
    def apply(self, controls: AbstractSet[operations.Control]) -> set[operations.Control]:
        """Apply the permutation to a set of controls.

        Args:
            controls: The set of controls to apply the permutation to.

        Returns:
            The set of controls with the permutation applied.
        """

    @overload
    def apply(self, targets: Sequence[int]) -> list[int]:
        """Apply the permutation to a list of targets.

        Args:
            targets: The list of targets to apply the permutation to.

        Returns:
            The list of targets with the permutation applied.
        """

    def clear(self) -> None:
        """Clear the permutation of all indices and values."""

    def __getitem__(self, index: int) -> int:
        """Get the value of the permutation at the given index.

        Args:
            index: The index to get the value of the permutation at.

        Returns:
            The value of the permutation at the given index.
        """

    def __setitem__(self, index: int, value: int) -> None:
        """Set the value of the permutation at the given index.

        Args:
            index: The index to set the value of the permutation at.
            value: The value to set the permutation at the given index to.
        """

    def __delitem__(self, index: int) -> None:
        """Delete the value of the permutation at the given index.

        Args:
            index: The index to delete the value of the permutation at.
        """

    def __len__(self) -> int:
        """Return the number of indices in the permutation."""

    def __iter__(self) -> Iterator[int]: ...
    def items(self) -> ItemsView[int, int]: ...
    def __eq__(self, arg: object, /) -> bool: ...
    def __ne__(self, arg: object, /) -> bool: ...

class QuantumComputation(MutableSequence[operations.Operation]):
    """The main class for representing quantum computations within the MQT.

    Acts as mutable sequence of :class:`~mqt.core.ir.operations.Operation` objects, which represent the individual operations in the quantum computation.

    Args:
        nq: The number of qubits in the quantum computation.
        nc: The number of classical bits in the quantum computation.
        seed: The seed to use for the internal random number generator.
    """

    def __init__(self, nq: int = 0, nc: int = 0, seed: int = 0) -> None: ...
    @staticmethod
    def from_qasm_str(qasm: str) -> QuantumComputation:
        """Create a QuantumComputation object from an OpenQASM string.

        Args:
            qasm: The OpenQASM string to create the QuantumComputation object from.

        Returns:
            The QuantumComputation object created from the OpenQASM string.
        """

    @staticmethod
    def from_qasm(filename: str) -> QuantumComputation:
        """Create a QuantumComputation object from an OpenQASM file.

        Args:
            filename: The filename of the OpenQASM file to create the QuantumComputation object from.

        Returns:
            The QuantumComputation object created from the OpenQASM file.
        """

    @property
    def name(self) -> str:
        """The name of the quantum computation."""

    @name.setter
    def name(self, arg: str, /) -> None: ...
    @property
    def num_qubits(self) -> int:
        """The total number of qubits in the quantum computation."""

    @property
    def num_ancilla_qubits(self) -> int:
        r"""The number of ancilla qubits in the quantum computation.

        Note:
            Ancilla qubits are qubits that always start in a fixed state (usually :math:`|0\\rangle`).
        """

    @property
    def num_garbage_qubits(self) -> int:
        """The number of garbage qubits in the quantum computation.

        Note:
            Garbage qubits are qubits whose final state is not relevant for the computation.
        """

    @property
    def num_measured_qubits(self) -> int:
        r"""The number of qubits that are measured in the quantum computation.

        Computed as :math:`| \text{qubits} | - | \text{garbage} |`.
        """

    @property
    def num_data_qubits(self) -> int:
        r"""The number of data qubits in the quantum computation.

        Computed as :math:`| \text{qubits} | - | \text{ancilla} |`.
        """

    @property
    def num_classical_bits(self) -> int:
        """The number of classical bits in the quantum computation."""

    @property
    def num_ops(self) -> int:
        """The number of operations in the quantum computation."""

    def num_single_qubit_ops(self) -> int:
        """Return the number of single-qubit operations in the quantum computation."""

    def num_total_ops(self) -> int:
        """Return the total number of operations in the quantum computation.

        Recursively counts sub-operations (e.g., from :class:`~mqt.core.ir.operations.CompoundOperation` objects).
        """

    def depth(self) -> int:
        """Return the depth of the quantum computation."""

    @property
    def global_phase(self) -> float:
        """The global phase of the quantum computation."""

    @global_phase.setter
    def global_phase(self, arg: float, /) -> None: ...
    def invert(self) -> None:
        """Invert the quantum computation in-place by inverting each operation and reversing the order of operations."""

    def to_operation(self) -> operations.Operation:
        """Convert the quantum computation to a single operation.

        This gives ownership of the operations to the resulting operation, so the quantum computation will be empty after this operation.

        When the quantum computation contains more than one operation, the resulting operation is a :class:`~mqt.core.ir.operations.CompoundOperation`.

        Returns:
            The operation representing the quantum computation.
        """

    @overload
    def __getitem__(self, index: int) -> operations.Operation:
        """Get the operation at the given index.

        Note:
            This gives write access to the operation at the given index.

        Args:
            index: The index of the operation to get.

        Returns:
            The operation at the given index.
        """

    @overload
    def __getitem__(self, index: slice) -> list[operations.Operation]:
        """Get a slice of operations from the quantum computation.

        Note:
            This gives write access to the operations in the given slice.

        Args:
            index: The slice of operations to get.

        Returns:
            The operations in the given slice.
        """

    @overload
    def __setitem__(self, index: int, value: operations.Operation) -> None:
        """Set the operation at the given index.

        Args:
            index: The index of the operation to set.
            value: The operation to set at the given index.
        """

    @overload
    def __setitem__(self, index: slice, value: Iterable[operations.Operation]) -> None:
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

    def __len__(self) -> int:
        """Return the number of operations in the quantum computation."""

    def insert(self, index: int, value: operations.Operation) -> None:
        """Insert an operation at the given index.

        Args:
            index: The index to insert the operation at.
            value: The operation to insert.
        """

    def append(self, value: operations.Operation) -> None:
        """Append an operation to the end of the quantum computation.

        Args:
            value: The operation to append.
        """

    def reverse(self) -> None:
        """Reverse the order of the operations in the quantum computation (in-place)."""

    def clear(self) -> None:
        """Clear the quantum computation of all operations."""

    def add_qubit_register(self, n: int, name: str = "q") -> registers.QuantumRegister:
        """Add a qubit register to the quantum computation.

        Args:
            n: The number of qubits in the qubit register.
            name: The name of the qubit register.

        Returns:
            The qubit register added to the quantum computation.
        """

    def add_classical_register(self, n: int, name: str = "c") -> registers.ClassicalRegister:
        """Add a classical register to the quantum computation.

        Args:
            n: The number of bits in the classical register.
            name: The name of the classical register.

        Returns:
            The classical register added to the quantum computation.
        """

    def add_ancillary_register(self, n: int, name: str = "anc") -> registers.QuantumRegister:
        """Add an ancillary register to the quantum computation.

        Args:
            n: The number of qubits in the ancillary register.
            name: The name of the ancillary register.

        Returns:
            The ancillary register added to the quantum computation.
        """

    def unify_quantum_registers(self, name: str = "q") -> registers.QuantumRegister:
        """Unify all quantum registers in the quantum computation.

        Args:
            name: The name of the unified quantum register.

        Returns:
            The unified quantum register.
        """

    @property
    def qregs(self) -> dict[str, registers.QuantumRegister]:
        """The quantum registers in the quantum computation."""

    @property
    def cregs(self) -> dict[str, registers.ClassicalRegister]:
        """The classical registers in the quantum computation."""

    @property
    def ancregs(self) -> dict[str, registers.QuantumRegister]:
        """The ancillary registers in the quantum computation."""

    @property
    def initial_layout(self) -> Permutation:
        """The initial layout of the qubits in the quantum computation.

        This is a permutation of the qubits in the quantum computation.
        It is mainly used to track the mapping of circuit qubits to device qubits during quantum circuit compilation.
        The keys are the device qubits (in which a compiled circuit is expressed in), and the values are the circuit qubits (in which the original quantum circuit is expressed in).

        Any operations in the quantum circuit are expected to be expressed in terms of the keys of the initial layout.

        Examples:
            - If no initial layout is explicitly specified (which is the default), the initial layout is assumed to be the identity permutation.
            - Assume a three-qubit circuit has been compiled to a four qubit device and circuit qubit 0 is mapped to device qubit 1, circuit qubit 1 is mapped to device qubit 2, and circuit qubit 2 is mapped to device qubit 3.
              Then the initial layout is {1: 0, 2: 1, 3: 2}.
        """

    @initial_layout.setter
    def initial_layout(self, arg: Permutation, /) -> None: ...
    @property
    def output_permutation(self) -> Permutation:
        """The output permutation of the qubits in the quantum computation.

        This is a permutation of the qubits in the quantum computation.
        It is mainly used to track where individual qubits end up at the end of the quantum computation, for example after a circuit has been compiled to a specific device and SWAP gates have been inserted, which permute the qubits.
        Similar to the initial layout, the keys are the qubits in the circuit and the values are the qubits in the "original" circuit.

        Examples:
            - If no output permutation is explicitly specified and the circuit does not contain measurements at the end, the output permutation is assumed to be the identity permutation.
            - If the circuit contains measurements at the end, these measurements are used to infer the output permutation.
              Assume a three-qubit circuit has been compiled to a four qubit device and, at the end of the circuit, circuit qubit 0 is measured into classical bit 2, circuit qubit 1 is measured into classical bit 1, and circuit qubit 3 is measured into classical bit 0.
              Then the output permutation is {0: 2, 1: 1, 3: 0}.
        """

    @output_permutation.setter
    def output_permutation(self, arg: Permutation, /) -> None: ...
    def initialize_io_mapping(self) -> None:
        """Initialize the I/O mapping of the quantum computation.

        If no initial layout is explicitly specified, the initial layout is assumed to be the identity permutation.
        If the circuit contains measurements at the end, these measurements are used to infer the output permutation.
        """

    @property
    def ancillary(self) -> list[bool]:
        """A list of booleans indicating whether each qubit is ancillary."""

    def set_circuit_qubit_ancillary(self, q: int) -> None:
        """Set a circuit (i.e., logical) qubit to be ancillary.

        Args:
            q: The index of the circuit qubit to set as ancillary.
        """

    def set_circuit_qubits_ancillary(self, q_min: int, q_max: int) -> None:
        """Set a range of circuit (i.e., logical) qubits to be ancillary.

        Args:
            q_min: The minimum index of the circuit qubits to set as ancillary.
            q_max: The maximum index of the circuit qubits to set as ancillary.
        """

    def is_circuit_qubit_ancillary(self, q: int) -> bool:
        """Check if a circuit (i.e., logical) qubit is ancillary.

        Args:
            q: The index of the circuit qubit to check.

        Returns:
            True if the circuit qubit is ancillary, False otherwise.
        """

    @property
    def garbage(self) -> list[bool]:
        """A list of booleans indicating whether each qubit is garbage."""

    def set_circuit_qubit_garbage(self, q: int) -> None:
        """Set a circuit (i.e., logical) qubit to be garbage.

        Args:
            q: The index of the circuit qubit to set as garbage.
        """

    def set_circuit_qubits_garbage(self, q_min: int, q_max: int) -> None:
        """Set a range of circuit (i.e., logical) qubits to be garbage.

        Args:
            q_min: The minimum index of the circuit qubits to set as garbage.
            q_max: The maximum index of the circuit qubits to set as garbage.
        """

    def is_circuit_qubit_garbage(self, q: int) -> bool:
        """Check if a circuit (i.e., logical) qubit is garbage.

        Args:
            q: The index of the circuit qubit to check.

        Returns:
            True if the circuit qubit is garbage, False otherwise.
        """

    @property
    def variables(self) -> set[symbolic.Variable]:
        """The set of variables in the quantum computation."""

    def add_variable(self, var: symbolic.Expression | float) -> None:
        """Add a variable to the quantum computation.

        Args:
            var: The variable to add.
        """

    def add_variables(self, vars_: Sequence[symbolic.Expression | float]) -> None:
        """Add multiple variables to the quantum computation.

        Args:
            vars_: The variables to add.
        """

    def is_variable_free(self) -> bool:
        """Check if the quantum computation is free of variables.

        Returns:
            True if the quantum computation is free of variables, False otherwise.
        """

    def instantiate(self, assignment: Mapping[symbolic.Variable, float]) -> QuantumComputation:
        """Instantiate the quantum computation with the given variable assignment.

        Args:
            assignment: The variable assignment to instantiate the quantum computation with.

        Returns:
            The instantiated quantum computation.
        """

    def instantiate_inplace(self, assignment: Mapping[symbolic.Variable, float]) -> None:
        """Instantiate the quantum computation with the given variable assignment in-place.

        Args:
            assignment: The variable assignment to instantiate the quantum computation with.
        """

    def qasm2_str(self) -> str:
        """Return the OpenQASM2 representation of the quantum computation as a string.

        Note:
            This uses some custom extensions to OpenQASM 2.0 that allow for easier definition of multi-controlled gates.
            These extensions might not be supported by all OpenQASM 2.0 parsers.
            Consider using the :meth:`qasm3_str` method instead, which uses OpenQASM 3.0 that natively supports multi-controlled gates.
            The export also assumes the bigger, non-standard `qelib1.inc` from Qiskit is available.

        Returns:
            The OpenQASM2 representation of the quantum computation as a string.
        """

    def qasm2(self, filename: os.PathLike[str] | str) -> None:
        """Write the OpenQASM2 representation of the quantum computation to a file.

        See Also:
            :meth:`qasm2_str`

        Args:
            filename: The filename of the file to write the OpenQASM2 representation to.
        """

    def qasm3_str(self) -> str:
        """Return the OpenQASM3 representation of the quantum computation as a string.

        Returns:
            The OpenQASM3 representation of the quantum computation as a string.
        """

    def qasm3(self, filename: os.PathLike[str] | str) -> None:
        """Write the OpenQASM3 representation of the quantum computation to a file.

        See Also:
            :meth:`qasm3_str`

        Args:
            filename: The filename of the file to write the OpenQASM3 representation to.
        """

    def i(self, q: int) -> None:
        r"""Apply an identity operation.

        .. math::
            I = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}

        Args:
            q: The target qubit
        """

    def ci(self, control: operations.Control | int, target: int) -> None:
        """Apply a controlled identity operation.

        See Also:
            :meth:`i`

        Args:
            control: The control qubit
            target: The target qubit
        """

    def mci(self, controls: AbstractSet[operations.Control | int], target: int) -> None:
        """Apply a multi-controlled identity operation.

        See Also:
            :meth:`i`

        Args:
            controls: The control qubits
            target: The target qubit
        """

    def x(self, q: int) -> None:
        r"""Apply a Pauli-X gate.

        .. math::
            X = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}

        Args:
            q: The target qubit
        """

    def cx(self, control: operations.Control | int, target: int) -> None:
        """Apply a controlled Pauli-X (i.e., CNOT or CX) gate.

        See Also:
            :meth:`x`

        Args:
            control: The control qubit
            target: The target qubit
        """

    def mcx(self, controls: AbstractSet[operations.Control | int], target: int) -> None:
        """Apply a multi-controlled Pauli-X (i.e., Toffoli or MCX) gate.

        See Also:
            :meth:`x`

        Args:
            controls: The control qubits
            target: The target qubit
        """

    def y(self, q: int) -> None:
        r"""Apply a Pauli-Y gate.

        .. math::
            Y = \begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix}

        Args:
            q: The target qubit
        """

    def cy(self, control: operations.Control | int, target: int) -> None:
        """Apply a controlled Pauli-Y gate.

        See Also:
            :meth:`y`

        Args:
            control: The control qubit
            target: The target qubit
        """

    def mcy(self, controls: AbstractSet[operations.Control | int], target: int) -> None:
        """Apply a multi-controlled Pauli-Y gate.

        See Also:
            :meth:`y`

        Args:
            controls: The control qubits
            target: The target qubit
        """

    def z(self, q: int) -> None:
        r"""Apply a Pauli-Z gate.

        .. math::
            Z = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}

        Args:
            q: The target qubit
        """

    def cz(self, control: operations.Control | int, target: int) -> None:
        """Apply a controlled Pauli-Z gate.

        See Also:
            :meth:`z`

        Args:
            control: The control qubit
            target: The target qubit
        """

    def mcz(self, controls: AbstractSet[operations.Control | int], target: int) -> None:
        """Apply a multi-controlled Pauli-Z gate.

        See Also:
            :meth:`z`

        Args:
            controls: The control qubits
            target: The target qubit
        """

    def h(self, q: int) -> None:
        r"""Apply a Hadamard gate.

        .. math::
            H = \frac{1}{\sqrt{2}} \begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}

        Args:
            q: The target qubit
        """

    def ch(self, control: operations.Control | int, target: int) -> None:
        """Apply a controlled Hadamard gate.

        See Also:
            :meth:`h`

        Args:
            control: The control qubit
            target: The target qubit
        """

    def mch(self, controls: AbstractSet[operations.Control | int], target: int) -> None:
        """Apply a multi-controlled Hadamard gate.

        See Also:
            :meth:`h`

        Args:
            controls: The control qubits
            target: The target qubit
        """

    def s(self, q: int) -> None:
        r"""Apply an S (i.e., phase) gate.

        .. math::
            S = \begin{pmatrix} 1 & 0 \\ 0 & i \end{pmatrix}

        Args:
            q: The target qubit
        """

    def cs(self, control: operations.Control | int, target: int) -> None:
        """Apply a controlled S gate.

        See Also:
            :meth:`s`

        Args:
            control: The control qubit
            target: The target qubit
        """

    def mcs(self, controls: AbstractSet[operations.Control | int], target: int) -> None:
        """Apply a multi-controlled S gate.

        See Also:
            :meth:`s`

        Args:
            controls: The control qubits
            target: The target qubit
        """

    def sdg(self, q: int) -> None:
        r"""Apply an :math:`S^\dagger` gate.

        .. math::
            S^\dagger = \begin{pmatrix} 1 & 0 \\ 0 & -i \end{pmatrix}

        Args:
            q: The target qubit
        """

    def csdg(self, control: operations.Control | int, target: int) -> None:
        r"""Apply a controlled :math:`S^\dagger` gate.

        See Also:
            :meth:`sdg`

        Args:
            control: The control qubit
            target: The target qubit
        """

    def mcsdg(self, controls: AbstractSet[operations.Control | int], target: int) -> None:
        r"""Apply a multi-controlled :math:`S^\dagger` gate.

        See Also:
            :meth:`sdg`

        Args:
            controls: The control qubits
            target: The target qubit
        """

    def t(self, q: int) -> None:
        r"""Apply a T gate.

        .. math::
            T = \begin{pmatrix} 1 & 0 \\ 0 & e^{i \pi / 4} \end{pmatrix}

        Args:
            q: The target qubit
        """

    def ct(self, control: operations.Control | int, target: int) -> None:
        """Apply a controlled T gate.

        See Also:
            :meth:`t`

        Args:
            control: The control qubit
            target: The target qubit
        """

    def mct(self, controls: AbstractSet[operations.Control | int], target: int) -> None:
        """Apply a multi-controlled T gate.

        See Also:
            :meth:`t`

        Args:
            controls: The control qubits
            target: The target qubit
        """

    def tdg(self, q: int) -> None:
        r"""Apply a :math:`T^\dagger` gate.

        .. math::
            T^\dagger = \begin{pmatrix} 1 & 0 \\ 0 & e^{-i \pi / 4} \end{pmatrix}

        Args:
            q: The target qubit
        """

    def ctdg(self, control: operations.Control | int, target: int) -> None:
        r"""Apply a controlled :math:`T^\dagger` gate.

        See Also:
            :meth:`tdg`

        Args:
            control: The control qubit
            target: The target qubit
        """

    def mctdg(self, controls: AbstractSet[operations.Control | int], target: int) -> None:
        r"""Apply a multi-controlled :math:`T^\dagger` gate.

        See Also:
            :meth:`tdg`

        Args:
            controls: The control qubits
            target: The target qubit
        """

    def v(self, q: int) -> None:
        r"""Apply a V gate.

        .. math::
            V = \frac{1}{\sqrt{2}} \begin{pmatrix} 1 & -i \\ -i & 1 \end{pmatrix}

        Args:
            q: The target qubit
        """

    def cv(self, control: operations.Control | int, target: int) -> None:
        """Apply a controlled V gate.

        See Also:
            :meth:`v`

        Args:
            control: The control qubit
            target: The target qubit
        """

    def mcv(self, controls: AbstractSet[operations.Control | int], target: int) -> None:
        """Apply a multi-controlled V gate.

        See Also:
            :meth:`v`

        Args:
            controls: The control qubits
            target: The target qubit
        """

    def vdg(self, q: int) -> None:
        r"""Apply a :math:`V^\dagger` gate.

        .. math::
            V^\dagger = \frac{1}{\sqrt{2}} \begin{pmatrix} 1 & i \\ i & 1 \end{pmatrix}

        Args:
            q: The target qubit
        """

    def cvdg(self, control: operations.Control | int, target: int) -> None:
        r"""Apply a controlled :math:`V^\dagger` gate.

        See Also:
            :meth:`vdg`

        Args:
            control: The control qubit
            target: The target qubit
        """

    def mcvdg(self, controls: AbstractSet[operations.Control | int], target: int) -> None:
        r"""Apply a multi-controlled :math:`V^\dagger` gate.

        See Also:
            :meth:`vdg`

        Args:
            controls: The control qubits
            target: The target qubit
        """

    def sx(self, q: int) -> None:
        r"""Apply a :math:`\sqrt{X}` gate.

        .. math::
            \sqrt{X} = \frac{1}{2} \begin{pmatrix} 1 + i & 1 - i \\ 1 - i & 1 + i \end{pmatrix}

        Args:
            q: The target qubit
        """

    def csx(self, control: operations.Control | int, target: int) -> None:
        r"""Apply a controlled :math:`\sqrt{X}` gate.

        See Also:
            :meth:`sx`

        Args:
            control: The control qubit
            target: The target qubit
        """

    def mcsx(self, controls: AbstractSet[operations.Control | int], target: int) -> None:
        r"""Apply a multi-controlled :math:`\sqrt{X}` gate.

        See Also:
            :meth:`sx`

        Args:
            controls: The control qubits
            target: The target qubit
        """

    def sxdg(self, q: int) -> None:
        r"""Apply a :math:`\sqrt{X}^\dagger` gate.

        .. math::
            \sqrt{X}^{\dagger} = \frac{1}{2} \begin{pmatrix} 1 - i & 1 + i \\ 1 + i & 1 - i \end{pmatrix}

        Args:
            q: The target qubit
        """

    def csxdg(self, control: operations.Control | int, target: int) -> None:
        r"""Apply a controlled :math:`\sqrt{X}^\dagger` gate.

        See Also:
            :meth:`sxdg`

        Args:
            control: The control qubit
            target: The target qubit
        """

    def mcsxdg(self, controls: AbstractSet[operations.Control | int], target: int) -> None:
        r"""Apply a multi-controlled :math:`\sqrt{X}^\dagger` gate.

        See Also:
            :meth:`sxdg`

        Args:
            controls: The control qubits
            target: The target qubit
        """

    def rx(self, theta: symbolic.Expression | float, q: int) -> None:
        r"""Apply an :math:`R_x(\theta)` gate.

        .. math::
            R_x(\theta) = e^{-i \theta X / 2} = \cos(\theta / 2) I - i \sin(\theta / 2) X
                        = \begin{pmatrix} \cos(\theta / 2) & -i \sin(\theta / 2) \\ -i \sin(\theta / 2) & \cos(\theta / 2) \end{pmatrix}

        Args:
            theta: The rotation angle
            q: The target qubit
        """

    def crx(self, theta: symbolic.Expression | float, control: operations.Control | int, target: int) -> None:
        r"""Apply a controlled :math:`R_x(\theta)` gate.

        See Also:
            :meth:`rx`

        Args:
            theta: The rotation angle
            control: The control qubit
            target: The target qubit
        """

    def mcrx(
        self, theta: symbolic.Expression | float, controls: AbstractSet[operations.Control | int], target: int
    ) -> None:
        r"""Apply a multi-controlled :math:`R_x(\theta)` gate.

        See Also:
            :meth:`rx`

        Args:
            theta: The rotation angle
            controls: The control qubits
            target: The target qubit
        """

    def ry(self, theta: symbolic.Expression | float, q: int) -> None:
        r"""Apply an :math:`R_y(\theta)` gate.

        .. math::
            R_y(\theta) = e^{-i \theta Y / 2} = \cos(\theta / 2) I - i \sin(\theta / 2) Y
                        = \begin{pmatrix} \cos(\theta / 2) & -\sin(\theta / 2) \\ \sin(\theta / 2) & \cos(\theta / 2) \end{pmatrix}

        Args:
            theta: The rotation angle
            q: The target qubit
        """

    def cry(self, theta: symbolic.Expression | float, control: operations.Control | int, target: int) -> None:
        r"""Apply a controlled :math:`R_y(\theta)` gate.

        See Also:
            :meth:`ry`

        Args:
            theta: The rotation angle
            control: The control qubit
            target: The target qubit
        """

    def mcry(
        self, theta: symbolic.Expression | float, controls: AbstractSet[operations.Control | int], target: int
    ) -> None:
        r"""Apply a multi-controlled :math:`R_y(\theta)` gate.

        See Also:
            :meth:`ry`

        Args:
            theta: The rotation angle
            controls: The control qubits
            target: The target qubit
        """

    def rz(self, theta: symbolic.Expression | float, q: int) -> None:
        r"""Apply an :math:`R_z(\theta)` gate.

        .. math::
            R_z(\theta) = e^{-i \theta Z / 2} = \begin{pmatrix} e^{-i \theta / 2} & 0 \\ 0 & e^{i \theta / 2} \end{pmatrix}

        Args:
            theta: The rotation angle
            q: The target qubit
        """

    def crz(self, theta: symbolic.Expression | float, control: operations.Control | int, target: int) -> None:
        r"""Apply a controlled :math:`R_z(\theta)` gate.

        See Also:
            :meth:`rz`

        Args:
            theta: The rotation angle
            control: The control qubit
            target: The target qubit
        """

    def mcrz(
        self, theta: symbolic.Expression | float, controls: AbstractSet[operations.Control | int], target: int
    ) -> None:
        r"""Apply a multi-controlled :math:`R_z(\theta)` gate.

        See Also:
            :meth:`rz`

        Args:
            theta: The rotation angle
            controls: The control qubits
            target: The target qubit
        """

    def p(self, theta: symbolic.Expression | float, q: int) -> None:
        r"""Apply a phase gate.

        .. math::
            P(\theta) = \begin{pmatrix} 1 & 0 \\ 0 & e^{i \theta} \end{pmatrix}

        Args:
            theta: The rotation angle
            q: The target qubit
        """

    def cp(self, theta: symbolic.Expression | float, control: operations.Control | int, target: int) -> None:
        """Apply a controlled phase gate.

        See Also:
            :meth:`p`

        Args:
            theta: The rotation angle
            control: The control qubit
            target: The target qubit
        """

    def mcp(
        self, theta: symbolic.Expression | float, controls: AbstractSet[operations.Control | int], target: int
    ) -> None:
        """Apply a multi-controlled phase gate.

        See Also:
            :meth:`p`

        Args:
            theta: The rotation angle
            controls: The control qubits
            target: The target qubit
        """

    def u2(self, phi: symbolic.Expression | float, lambda_: symbolic.Expression | float, q: int) -> None:
        r"""Apply a :math:`U_2(\phi, \lambda)` gate.

        .. math::
            U_2(\phi, \lambda) = \frac{1}{\sqrt{2}} \begin{pmatrix} 1 & -e^{i \lambda} \\ e^{i \phi} & e^{i (\phi + \lambda)} \end{pmatrix}

        Args:
            phi: The rotation angle
            lambda_: The rotation angle
            q: The target qubit
        """

    def cu2(
        self,
        phi: symbolic.Expression | float,
        lambda_: symbolic.Expression | float,
        control: operations.Control | int,
        target: int,
    ) -> None:
        r"""Apply a controlled :math:`U_2(\phi, \lambda)` gate.

        See Also:
            :meth:`u2`

        Args:
            phi: The rotation angle
            lambda_: The rotation angle
            control: The control qubit
            target: The target qubit
        """

    def mcu2(
        self,
        phi: symbolic.Expression | float,
        lambda_: symbolic.Expression | float,
        controls: AbstractSet[operations.Control | int],
        target: int,
    ) -> None:
        r"""Apply a multi-controlled :math:`U_2(\phi, \lambda)` gate.

        See Also:
            :meth:`u2`

        Args:
            phi: The rotation angle
            lambda_: The rotation angle
            controls: The control qubits
            target: The target qubit
        """

    def r(self, theta: symbolic.Expression | float, phi: symbolic.Expression | float, q: int) -> None:
        r"""Apply an :math:`R(\theta, \phi)` gate.

        .. math::
            R(\theta, \phi) = e^{-i \frac{\theta}{2} (\cos(\phi) X + \sin(\phi) Y)}
                            = \begin{pmatrix} \cos(\theta / 2) & -i e^{-i \phi} \sin(\theta / 2) \\ -i e^{i \phi} \sin(\theta / 2) & \cos(\theta / 2) \end{pmatrix}

        Args:
            theta: The rotation angle
            phi: The rotation angle
            q: The target qubit
        """

    def cr(
        self,
        theta: symbolic.Expression | float,
        phi: symbolic.Expression | float,
        control: operations.Control | int,
        target: int,
    ) -> None:
        r"""Apply a controlled :math:`R(\theta, \phi)` gate.

        See Also:
            :meth:`r`

        Args:
            theta: The rotation angle
            phi: The rotation angle
            control: The control qubit
            target: The target qubit
        """

    def mcr(
        self,
        theta: symbolic.Expression | float,
        phi: symbolic.Expression | float,
        controls: AbstractSet[operations.Control | int],
        target: int,
    ) -> None:
        r"""Apply a multi-controlled :math:`R(\theta, \phi)` gate.

        See Also:
            :meth:`r`

        Args:
            theta: The rotation angle
            phi: The rotation angle
            controls: The control qubits
            target: The target qubit
        """

    def u(
        self,
        theta: symbolic.Expression | float,
        phi: symbolic.Expression | float,
        lambda_: symbolic.Expression | float,
        q: int,
    ) -> None:
        r"""Apply a :math:`U(\theta, \phi, \lambda)` gate.

        .. math::
            U(\theta, \phi, \lambda) = \begin{pmatrix} \cos(\theta / 2) & -e^{i \lambda} \sin(\theta / 2) \\ e^{i \phi} \sin(\theta / 2) & e^{i (\phi + \lambda)}\cos(\theta / 2) \end{pmatrix}

        Args:
            theta: The rotation angle
            phi: The rotation angle
            lambda_: The rotation angle
            q: The target qubit
        """

    def cu(
        self,
        theta: symbolic.Expression | float,
        phi: symbolic.Expression | float,
        lambda_: symbolic.Expression | float,
        control: operations.Control | int,
        target: int,
    ) -> None:
        r"""Apply a controlled :math:`U(\theta, \phi, \lambda)` gate.

        See Also:
            :meth:`u`

        Args:
            theta: The rotation angle
            phi: The rotation angle
            lambda_: The rotation angle
            control: The control qubit
            target: The target qubit
        """

    def mcu(
        self,
        theta: symbolic.Expression | float,
        phi: symbolic.Expression | float,
        lambda_: symbolic.Expression | float,
        controls: AbstractSet[operations.Control | int],
        target: int,
    ) -> None:
        r"""Apply a multi-controlled :math:`U(\theta, \phi, \lambda)` gate.

        See Also:
            :meth:`u`

        Args:
            theta: The rotation angle
            phi: The rotation angle
            lambda_: The rotation angle
            controls: The control qubits
            target: The target qubit
        """

    def swap(self, target1: int, target2: int) -> None:
        r"""Apply a SWAP gate.

        .. math::
            \text{SWAP} = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 \end{pmatrix}

        Args:
            target1: The first target qubit
            target2: The second target qubit
        """

    def cswap(self, control: operations.Control | int, target1: int, target2: int) -> None:
        """Apply a controlled SWAP gate.

        See Also:
            :meth:`swap`

        Args:
            control: The control qubit
            target1: The first target qubit
            target2: The second target qubit
        """

    def mcswap(self, controls: AbstractSet[operations.Control | int], target1: int, target2: int) -> None:
        """Apply a multi-controlled SWAP gate.

        See Also:
            :meth:`swap`

        Args:
            controls: The control qubits
            target1: The first target qubit
            target2: The second target qubit
        """

    def dcx(self, target1: int, target2: int) -> None:
        r"""Apply a DCX (i.e., double CNOT) gate.

        .. math::
            DCX = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 0 & 0 & 1 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 1 & 0 \end{pmatrix}

        Args:
            target1: The first target qubit
            target2: The second target qubit
        """

    def cdcx(self, control: operations.Control | int, target1: int, target2: int) -> None:
        """Apply a controlled DCX gate.

        See Also:
            :meth:`dcx`

        Args:
            control: The control qubit
            target1: The first target qubit
            target2: The second target qubit
        """

    def mcdcx(self, controls: AbstractSet[operations.Control | int], target1: int, target2: int) -> None:
        """Apply a multi-controlled DCX gate.

        See Also:
            :meth:`dcx`

        Args:
            controls: The control qubits
            target1: The first target qubit
            target2: The second target qubit
        """

    def ecr(self, target1: int, target2: int) -> None:
        r"""Apply a ECR (echoed cross-resonance) gate.

        .. math::
            ECR = \frac{1}{\sqrt{2}} \begin{pmatrix} 0 & 0 & 1 & i \\ 0 & 0 & i & 1 \\ 1 & -i & 0 & 0 \\ -i & 1 & 0 & 0 \end{pmatrix}

        Args:
            target1: The first target qubit
            target2: The second target qubit
        """

    def cecr(self, control: operations.Control | int, target1: int, target2: int) -> None:
        """Apply a controlled ECR gate.

        See Also:
            :meth:`ecr`

        Args:
            control: The control qubit
            target1: The first target qubit
            target2: The second target qubit
        """

    def mcecr(self, controls: AbstractSet[operations.Control | int], target1: int, target2: int) -> None:
        """Apply a multi-controlled ECR gate.

        See Also:
            :meth:`ecr`

        Args:
            controls: The control qubits
            target1: The first target qubit
            target2: The second target qubit
        """

    def iswap(self, target1: int, target2: int) -> None:
        r"""Apply a :math:`i\text{SWAP}` gate.

        .. math::
            i\text{SWAP} = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 0 & i & 0 \\ 0 & i & 0 & 0 \\ 0 & 0 & 0 & 1 \end{pmatrix}

        Args:
            target1: The first target qubit
            target2: The second target qubit
        """

    def ciswap(self, control: operations.Control | int, target1: int, target2: int) -> None:
        r"""Apply a controlled :math:`i\text{SWAP}` gate.

        See Also:
            :meth:`iswap`

        Args:
            control: The control qubit
            target1: The first target qubit
            target2: The second target qubit
        """

    def mciswap(self, controls: AbstractSet[operations.Control | int], target1: int, target2: int) -> None:
        r"""Apply a multi-controlled :math:`i\text{SWAP}` gate.

        See Also:
            :meth:`iswap`

        Args:
            controls: The control qubits
            target1: The first target qubit
            target2: The second target qubit
        """

    def iswapdg(self, target1: int, target2: int) -> None:
        r"""Apply a :math:`i\text{SWAP}^\dagger` gate.

        .. math::
            i\text{SWAP}^\dagger = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 0 & -i & 0 \\ 0 & -i & 0 & 0 \\ 0 & 0 & 0 & 1 \end{pmatrix}

        Args:
            target1: The first target qubit
            target2: The second target qubit
        """

    def ciswapdg(self, control: operations.Control | int, target1: int, target2: int) -> None:
        r"""Apply a controlled :math:`i\text{SWAP}^\dagger` gate.

        See Also:
            :meth:`iswapdg`

        Args:
            control: The control qubit
            target1: The first target qubit
            target2: The second target qubit
        """

    def mciswapdg(self, controls: AbstractSet[operations.Control | int], target1: int, target2: int) -> None:
        r"""Apply a multi-controlled :math:`i\text{SWAP}^\dagger` gate.

        See Also:
            :meth:`iswapdg`

        Args:
            controls: The control qubits
            target1: The first target qubit
            target2: The second target qubit
        """

    def peres(self, target1: int, target2: int) -> None:
        r"""Apply a Peres gate.

        .. math::
            \text{Peres} = \begin{pmatrix} 0 & 0 & 0 & 1 \\ 0 & 0 & 1 & 0 \\ 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \end{pmatrix}

        Args:
            target1: The first target qubit
            target2: The second target qubit
        """

    def cperes(self, control: operations.Control | int, target1: int, target2: int) -> None:
        """Apply a controlled Peres gate.

        See Also:
            :meth:`peres`

        Args:
            control: The control qubit
            target1: The first target qubit
            target2: The second target qubit
        """

    def mcperes(self, controls: AbstractSet[operations.Control | int], target1: int, target2: int) -> None:
        """Apply a multi-controlled Peres gate.

        See Also:
            :meth:`peres`

        Args:
            controls: The control qubits
            target1: The first target qubit
            target2: The second target qubit
        """

    def peresdg(self, target1: int, target2: int) -> None:
        r"""Apply a :math:`\text{Peres}^\dagger` gate.

        .. math::
            \text{Peres}^\dagger = \begin{pmatrix} 0 & 0 & 0 & 1 \\ 0 & 0 & 1 & 0 \\ 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \end{pmatrix}

        Args:
            target1: The first target qubit
            target2: The second target qubit
        """

    def cperesdg(self, control: operations.Control | int, target1: int, target2: int) -> None:
        r"""Apply a controlled :math:`\text{Peres}^\dagger` gate.

        See Also:
            :meth:`peresdg`

        Args:
            control: The control qubit
            target1: The first target qubit
            target2: The second target qubit
        """

    def mcperesdg(self, controls: AbstractSet[operations.Control | int], target1: int, target2: int) -> None:
        r"""Apply a multi-controlled :math:`\text{Peres}^\dagger` gate.

        See Also:
            :meth:`peresdg`

        Args:
            controls: The control qubits
            target1: The first target qubit
            target2: The second target qubit
        """

    def rxx(self, theta: symbolic.Expression | float, target1: int, target2: int) -> None:
        r"""Apply an :math:`R_{xx}(\theta)` gate.

        .. math::
            R_{xx}(\theta) = e^{-i \theta XX / 2} = \cos(\theta / 2) I \otimes I - i \sin(\theta / 2) X \otimes X
                           = \begin{pmatrix} \cos(\theta / 2) & 0 & 0 & -i \sin(\theta / 2) \\
                             0 & \cos(\theta / 2) & -i \sin(\theta / 2) & 0 \\
                             0 & -i \sin(\theta / 2) & \cos(\theta / 2) & 0 \\
                             -i \sin(\theta / 2) & 0 & 0 & \cos(\theta / 2) \end{pmatrix}

        Args:
            theta: The rotation angle
            target1: The first target qubit
            target2: The second target qubit
        """

    def crxx(
        self, theta: symbolic.Expression | float, control: operations.Control | int, target1: int, target2: int
    ) -> None:
        r"""Apply a controlled :math:`R_{xx}(\theta)` gate.

        See Also:
            :meth:`rxx`

        Args:
            theta: The rotation angle
            control: The control qubit
            target1: The first target qubit
            target2: The second target qubit
        """

    def mcrxx(
        self,
        theta: symbolic.Expression | float,
        controls: AbstractSet[operations.Control | int],
        target1: int,
        target2: int,
    ) -> None:
        r"""Apply a multi-controlled :math:`R_{xx}(\theta)` gate.

        See Also:
            :meth:`rxx`

        Args:
            theta: The rotation angle
            controls: The control qubits
            target1: The first target qubit
            target2: The second target qubit
        """

    def ryy(self, theta: symbolic.Expression | float, target1: int, target2: int) -> None:
        r"""Apply an :math:`R_{yy}(\theta)` gate.

        .. math::
            R_{yy}(\theta) = e^{-i \theta YY / 2} = \cos(\theta / 2) I \otimes I - i \sin(\theta / 2) Y \otimes Y
                           = \begin{pmatrix} \cos(\theta / 2) & 0 & 0 & i \sin(\theta / 2) \\
                             0 & \cos(\theta / 2) & -i \sin(\theta / 2) & 0 \\
                             0 & -i \sin(\theta / 2) & \cos(\theta / 2) & 0 \\
                             i \sin(\theta / 2) & 0 & 0 & \cos(\theta / 2) \end{pmatrix}

        Args:
            theta: The rotation angle
            target1: The first target qubit
            target2: The second target qubit
        """

    def cryy(
        self, theta: symbolic.Expression | float, control: operations.Control | int, target1: int, target2: int
    ) -> None:
        r"""Apply a controlled :math:`R_{yy}(\theta)` gate.

        See Also:
            :meth:`ryy`

        Args:
            theta: The rotation angle
            control: The control qubit
            target1: The first target qubit
            target2: The second target qubit
        """

    def mcryy(
        self,
        theta: symbolic.Expression | float,
        controls: AbstractSet[operations.Control | int],
        target1: int,
        target2: int,
    ) -> None:
        r"""Apply a multi-controlled :math:`R_{yy}(\theta)` gate.

        See Also:
            :meth:`ryy`

        Args:
            theta: The rotation angle
            controls: The control qubits
            target1: The first target qubit
            target2: The second target qubit
        """

    def rzx(self, theta: symbolic.Expression | float, target1: int, target2: int) -> None:
        r"""Apply an :math:`R_{zx}(\theta)` gate.

        .. math::
            R_{zx}(\theta) = e^{-i \theta ZX / 2} = \cos(\theta / 2) I \otimes I - i \sin(\theta / 2) Z \otimes X
                           = \begin{pmatrix} \cos(\theta/2) & -i \sin(\theta/2) & 0 & 0 \\
                             -i \sin(\theta/2) & \cos(\theta/2) & 0 & 0 \\
                             0 & 0 & \cos(\theta/2) & i \sin(\theta/2) \\
                             0 & 0 & i \sin(\theta/2) & \cos(\theta/2) \end{pmatrix}

        Args:
            theta: The rotation angle
            target1: The first target qubit
            target2: The second target qubit
        """

    def crzx(
        self, theta: symbolic.Expression | float, control: operations.Control | int, target1: int, target2: int
    ) -> None:
        r"""Apply a controlled :math:`R_{zx}(\theta)` gate.

        See Also:
            :meth:`rzx`

        Args:
            theta: The rotation angle
            control: The control qubit
            target1: The first target qubit
            target2: The second target qubit
        """

    def mcrzx(
        self,
        theta: symbolic.Expression | float,
        controls: AbstractSet[operations.Control | int],
        target1: int,
        target2: int,
    ) -> None:
        r"""Apply a multi-controlled :math:`R_{zx}(\theta)` gate.

        See Also:
            :meth:`rzx`

        Args:
            theta: The rotation angle
            controls: The control qubits
            target1: The first target qubit
            target2: The second target qubit
        """

    def rzz(self, theta: symbolic.Expression | float, target1: int, target2: int) -> None:
        r"""Apply an :math:`R_{zz}(\theta)` gate.

        .. math::
            R_{zz}(\theta) = e^{-i \theta ZZ / 2}
                           = \begin{pmatrix} e^{-i \theta / 2} & 0 & 0 & 0 \\
                             0 & e^{i \theta / 2} & 0 & 0 \\
                             0 & 0 & e^{i \theta / 2} & 0 \\
                             0 & 0 & 0 & e^{-i \theta / 2} \end{pmatrix}

        Args:
            theta: The rotation angle
            target1: The first target qubit
            target2: The second target qubit
        """

    def crzz(
        self, theta: symbolic.Expression | float, control: operations.Control | int, target1: int, target2: int
    ) -> None:
        r"""Apply a controlled :math:`R_{zz}(\theta)` gate.

        See Also:
            :meth:`rzz`

        Args:
            theta: The rotation angle
            control: The control qubit
            target1: The first target qubit
            target2: The second target qubit
        """

    def mcrzz(
        self,
        theta: symbolic.Expression | float,
        controls: AbstractSet[operations.Control | int],
        target1: int,
        target2: int,
    ) -> None:
        r"""Apply a multi-controlled :math:`R_{zz}(\theta)` gate.

        See Also:
            :meth:`rzz`

        Args:
            theta: The rotation angle
            controls: The control qubits
            target1: The first target qubit
            target2: The second target qubit
        """

    def xx_minus_yy(
        self, theta: symbolic.Expression | float, beta: symbolic.Expression | float, target1: int, target2: int
    ) -> None:
        r"""Apply an :math:`R_{XX - YY}(\theta, \beta)` gate.

        .. math::
            R_{XX - YY}(\theta, \beta) = R_{z_2}(\beta) \cdot e^{-i \frac{\theta}{2} \frac{XX - YY}{2}} \cdot R_{z_2}(-\beta)
                                       = \begin{pmatrix} \cos(\theta / 2) & 0 & 0 & -i \sin(\theta / 2) e^{-i \beta} \\
                                         0 & 1 & 0 & 0 \\
                                         0 & 0 & 1 & 0 \\
                                         -i \sin(\theta / 2) e^{i \beta} & 0 & 0 & \cos(\theta / 2) \end{pmatrix}

        Args:
            theta: The rotation angle
            beta: The rotation angle
            target1: The first target qubit
            target2: The second target qubit
        """

    def cxx_minus_yy(
        self,
        theta: symbolic.Expression | float,
        beta: symbolic.Expression | float,
        control: operations.Control | int,
        target1: int,
        target2: int,
    ) -> None:
        r"""Apply a controlled :math:`R_{XX - YY}(\theta, \beta)` gate.

        See Also:
            :meth:`xx_minus_yy`

        Args:
            theta: The rotation angle
            beta: The rotation angle
            control: The control qubit
            target1: The first target qubit
            target2: The second target qubit
        """

    def mcxx_minus_yy(
        self,
        theta: symbolic.Expression | float,
        beta: symbolic.Expression | float,
        controls: AbstractSet[operations.Control | int],
        target1: int,
        target2: int,
    ) -> None:
        r"""Apply a multi-controlled :math:`R_{XX - YY}(\theta, \beta)` gate.

        See Also:
            :meth:`xx_minus_yy`

        Args:
            theta: The rotation angle
            beta: The rotation angle
            controls: The control qubits
            target1: The first target qubit
            target2: The second target qubit
        """

    def xx_plus_yy(
        self, theta: symbolic.Expression | float, beta: symbolic.Expression | float, target1: int, target2: int
    ) -> None:
        r"""Apply an :math:`R_{XX + YY}(\theta, \beta)` gate.

        .. math::
            R_{XX + YY}(\theta, \beta) = R_{z_1}(\beta) \cdot e^{-i \frac{\theta}{2} \frac{XX + YY}{2}} \cdot R_{z_1}(-\beta)
                                       = \begin{pmatrix} 1 & 0 & 0 & 0 \\
                                         0 & \cos(\theta / 2) & -i \sin(\theta / 2) e^{-i \beta} & 0 \\
                                         0 & -i \sin(\theta / 2) e^{i \beta} & \cos(\theta / 2) & 0 \\
                                         0 & 0 & 0 & 1 \end{pmatrix}

        Args:
            theta: The rotation angle
            beta: The rotation angle
            target1: The first target qubit
            target2: The second target qubit
        """

    def cxx_plus_yy(
        self,
        theta: symbolic.Expression | float,
        beta: symbolic.Expression | float,
        control: operations.Control | int,
        target1: int,
        target2: int,
    ) -> None:
        r"""Apply a controlled :math:`R_{XX + YY}(\theta, \beta)` gate.

        See Also:
            :meth:`xx_plus_yy`

        Args:
            theta: The rotation angle
            beta: The rotation angle
            control: The control qubit
            target1: The first target qubit
            target2: The second target qubit
        """

    def mcxx_plus_yy(
        self,
        theta: symbolic.Expression | float,
        beta: symbolic.Expression | float,
        controls: AbstractSet[operations.Control | int],
        target1: int,
        target2: int,
    ) -> None:
        r"""Apply a multi-controlled :math:`R_{XX + YY}(\theta, \beta)` gate.

        See Also:
            :meth:`xx_plus_yy`

        Args:
            theta: The rotation angle
            beta: The rotation angle
            controls: The control qubits
            target1: The first target qubit
            target2: The second target qubit
        """

    def gphase(self, phase: float) -> None:
        r"""Apply a global phase gate.

        .. math::
            GPhase(\theta) = (e^{i \theta})

        Args:
            phase: The rotation angle
        """

    @overload
    def measure(self, qubit: int, cbit: int) -> None:
        """Measure a qubit and store the result in a classical bit.

        Args:
            qubit: The qubit to measure
            cbit: The classical bit to store the result
        """

    @overload
    def measure(self, qubits: Sequence[int], cbits: Sequence[int]) -> None:
        """Measure multiple qubits and store the results in classical bits.

        This method is equivalent to calling :meth:`measure` multiple times.

        Args:
            qubits: The qubits to measure
            cbits: The classical bits to store the results
        """

    def measure_all(self, *, add_bits: bool = True) -> None:
        """Measure all qubits and store the results in classical bits.

        Details:
            If `add_bits` is `True`, a new classical register (named "`meas`") with the same size as the number of qubits will be added to the circuit and the results will be stored in it.
            If `add_bits` is `False`, the classical register must already exist and have a sufficient number of bits to store the results.

        Args:
            add_bits: Whether to explicitly add a classical register
        """

    @overload
    def reset(self, q: int) -> None:
        """Add a reset operation to the circuit.

        Args:
            q: The qubit to reset
        """

    @overload
    def reset(self, qubits: Sequence[int]) -> None:
        """Add a reset operation to the circuit.

        Args:
            qubits: The qubits to reset
        """

    @overload
    def barrier(self) -> None:
        """Add a barrier to the circuit."""

    @overload
    def barrier(self, q: int) -> None:
        """Add a barrier to the circuit.

        Args:
            q: The qubit to add the barrier to
        """

    @overload
    def barrier(self, qubits: Sequence[int]) -> None:
        """Add a barrier to the circuit.

        Args:
            qubits: The qubits to add the barrier to
        """

    @overload
    def if_else(
        self,
        then_operation: operations.Operation,
        else_operation: operations.Operation,
        control_register: registers.ClassicalRegister,
        expected_value: int = 1,
        comparison_kind: operations.ComparisonKind = ...,
    ) -> None:
        """Add an if-else operation to the circuit.

        Args:
            then_operation: The operation to apply if the condition is met
            else_operation: The operation to apply if the condition is not met
            control_register: The classical register to check against
            expected_value: The expected value of the control register
            comparison_kind: The kind of comparison to perform
        """

    @overload
    def if_else(
        self,
        then_operation: operations.Operation,
        else_operation: operations.Operation,
        control_bit: int,
        expected_value: int = 1,
        comparison_kind: operations.ComparisonKind = ...,
    ) -> None:
        """Add an if-else operation to the circuit.

        Args:
            then_operation: The operation to apply if the condition is met
            else_operation: The operation to apply if the condition is not met
            control_bit: The index of the classical bit to check against
            expected_value: The expected value of the control bit
            comparison_kind: The kind of comparison to perform
        """

    @overload
    def if_(
        self,
        op_type: operations.OpType,
        target: int,
        control_register: registers.ClassicalRegister,
        expected_value: int = 1,
        comparison_kind: operations.ComparisonKind = ...,
        params: Sequence[float] = ...,
    ) -> None:
        """Add an if operation to the circuit.

        Args:
            op_type: The operation to apply
            target: The target qubit
            control_register: The classical register to check against
            expected_value: The expected value of the control register
            comparison_kind: The kind of comparison to perform
            params: The parameters of the operation
        """

    @overload
    def if_(
        self,
        op_type: operations.OpType,
        target: int,
        control: operations.Control | int,
        control_register: registers.ClassicalRegister,
        expected_value: int = 1,
        comparison_kind: operations.ComparisonKind = ...,
        params: Sequence[symbolic.Expression | float] = ...,
    ) -> None:
        """Add an if operation to the circuit.

        Args:
            op_type: The operation to apply
            target: The target qubit
            control: The control qubit
            control_register: The classical register to check against
            expected_value: The expected value of the control register
            comparison_kind: The kind of comparison to perform
            params: The parameters of the operation.
        """

    @overload
    def if_(
        self,
        op_type: operations.OpType,
        target: int,
        controls: AbstractSet[operations.Control | int],
        control_register: registers.ClassicalRegister,
        expected_value: int = 1,
        comparison_kind: operations.ComparisonKind = ...,
        params: Sequence[symbolic.Expression | float] = ...,
    ) -> None:
        """Add an if operation to the circuit.

        Args:
            op_type: The operation to apply
            target: The target qubit
            controls: The control qubits
            control_register: The classical register to check against
            expected_value: The expected value of the control register
            comparison_kind: The kind of comparison to perform
            params: The parameters of the operation.
        """

    @overload
    def if_(
        self,
        op_type: operations.OpType,
        target: int,
        control_bit: int,
        expected_value: bool = True,
        comparison_kind: operations.ComparisonKind = ...,
        params: Sequence[float] = ...,
    ) -> None:
        """Add an if operation to the circuit.

        Args:
            op_type: The operation to apply
            target: The target qubit
            control_bit: The index of the classical bit to check against
            expected_value: The expected value of the control bit
            comparison_kind: The kind of comparison to perform
            params: The parameters of the operation.
        """

    @overload
    def if_(
        self,
        op_type: operations.OpType,
        target: int,
        control: operations.Control | int,
        control_bit: int,
        expected_value: bool = True,
        comparison_kind: operations.ComparisonKind = ...,
        params: Sequence[symbolic.Expression | float] = ...,
    ) -> None:
        """Add an if operation to the circuit.

        Args:
            op_type: The operation to apply
            target: The target qubit
            control: The control qubit
            control_bit: The index of the classical bit to check against
            expected_value: The expected value of the control bit
            comparison_kind: The kind of comparison to perform
            params: The parameters of the operation.
        """

    @overload
    def if_(
        self,
        op_type: operations.OpType,
        target: int,
        controls: AbstractSet[operations.Control | int],
        control_bit: int,
        expected_value: bool = True,
        comparison_kind: operations.ComparisonKind = ...,
        params: Sequence[symbolic.Expression | float] = ...,
    ) -> None:
        """Add an if operation to the circuit.

        Args:
            op_type: The operation to apply
            target: The target qubit
            controls: The control qubits
            control_bit: The index of the classical bit to check against
            expected_value: The expected value of the control bit
            comparison_kind: The kind of comparison to perform
            params: The parameters of the operation.
        """
