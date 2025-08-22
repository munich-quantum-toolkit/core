# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""MQT Core DD - The MQT Decision Diagram Package."""

from collections.abc import Iterable
from enum import Enum
from typing import Any

import numpy as np
import numpy.typing as npt

from .ir import Permutation, QuantumComputation
from .ir.operations import Control, IfElseOperation, NonUnitaryOperation, Operation

__all__ = [
    "BasisStates",
    "DDPackage",
    "Matrix",
    "MatrixDD",
    "Vector",
    "VectorDD",
    "build_functionality",
    "build_unitary",
    "sample",
    "simulate",
    "simulate_statevector",
]

def sample(qc: QuantumComputation, shots: int = 1024, seed: int = 0) -> dict[str, int]:
    """Sample from the output distribution of a quantum computation.

    This function classically simulates the quantum computation and repeatedly
    samples from the output distribution. It supports mid-circuit measurements,
    resets, and classical control.

    Args:
        qc: The quantum computation.
        shots: The number of samples to take. If the quantum computation contains
               no mid-circuit measurements or resets, the circuit is simulated
               once and the samples are drawn from the final state. Otherwise,
               the circuit is simulated once for each sample. Defaults to 1024.
        seed: The seed for the random number generator. If set to a specific
              non-zero value, the simulation is deterministic. If set to 0, the
              RNG is randomly seeded. Defaults to 0.

    Returns:
        A histogram of the samples. Each sample is a bitstring representing the
        measurement outcomes of the qubits in the quantum computation. The
        leftmost bit corresponds to the most significant qubit, that is, the
        qubit with the highest index (big-endian). If the circuit contains
        measurements, only the qubits that are actively measured are included in
        the output distribution. Otherwise, all qubits in the circuit are measured.
    """

def simulate_statevector(qc: QuantumComputation) -> Vector:
    """Simulate the quantum computation and return the final state vector.

    This function classically simulates the quantum computation and returns the
    state vector of the final state.
    It does not support measurements, resets, or classical control.

    Since the state vector is guaranteed to be exponentially large in the number
    of qubits, this function is only suitable for small quantum computations.
    Consider using the :func:`~mqt.core.dd.simulate` or the
    :func:`~mqt.core.dd.sample` functions, which never explicitly construct
    the state vector, for larger quantum computations.

    Args:
        qc: The quantum computation.
            Must only contain unitary operations.

    Returns:
        The state vector of the final state.

    Notes:
        This function internally constructs a :class:`~mqt.core.dd.DDPackage`, creates the
        zero state, and simulates the quantum computation via the :func:`simulate`
        function.
        The state vector is then extracted from the resulting DD via the :meth:`~mqt.core.dd.VectorDD.get_vector`
        method.
        The resulting :class:`~mqt.core.dd.Vector` can be converted to a NumPy array without copying
        the data by calling :func:`numpy.array` with the `copy=False` argument.
    """

def build_unitary(qc: QuantumComputation, recursive: bool = False) -> Matrix:
    """Build a unitary matrix representation of a quantum computation.

    This function builds a matrix representation of the unitary representing the
    functionality of a quantum computation.
    This function does not support measurements, resets, or classical control,
    as the corresponding operations are non-unitary.

    Since the unitary matrix is guaranteed to be exponentially large in the number
    of qubits, this function is only suitable for small quantum computations.
    Consider using the :func:`~mqt.core.dd.build_functionality` function, which
    never explicitly constructs the unitary matrix, for larger quantum computations.

    Args:
        qc: The quantum computation.
            Must only contain unitary operations.
        recursive: Whether to build the unitary matrix recursively.
                   If set to True, the unitary matrix is built recursively by
                   pairwise grouping the operations of the quantum computation.
                   If set to False, the unitary matrix is built by sequentially
                   applying the operations of the quantum computation to the
                   identity matrix.
                   Defaults to False.

    Returns:
        The unitary matrix representing the functionality of the quantum computation.

    Notes:
        This function internally constructs a :class:`~mqt.core.dd.DDPackage`, creates the
        identity matrix, and builds the unitary matrix via the :func:`~mqt.core.dd.build_functionality`
        function.
        The unitary matrix is then extracted from the resulting DD via the :meth:`~mqt.core.dd.MatrixDD.get_matrix`
        method.
        The resulting :class:`~mqt.core.dd.Matrix` can be converted to a NumPy array without copying
        the data by calling :func:`numpy.array` with the `copy=False` argument.
    """

def simulate(qc: QuantumComputation, initial_state: VectorDD, dd_package: DDPackage) -> VectorDD:
    """Simulate a quantum computation.

    This function classically simulates a quantum computation for a given initial
    state and returns the final state (represented as a DD). Compared to the
    `sample` function, this function does not support measurements, resets, or
    classical control. It only supports unitary operations.

    The simulation is effectively computed by sequentially applying the operations
    of the quantum computation to the initial state.

    Args:
        qc: The quantum computation.
            Must only contain unitary operations.
        initial_state: The initial state as a DD. Must have the same number of qubits
                       as the quantum computation. The reference count of the initial
                       state is decremented during the simulation, so the caller must
                       ensure that the initial state has a non-zero reference count.
        dd_package: The DD package. Must be configured with a sufficient number of
                    qubits to accommodate the quantum computation.

    Returns:
        The final state as a DD. The reference count of the final state is non-zero
        and must be manually decremented by the caller if it is no longer needed.
    """

def build_functionality(qc: QuantumComputation, dd_package: DDPackage, recursive: bool = False) -> MatrixDD:
    """Build a functional representation of a quantum computation.

    This function builds a matrix DD representation of the unitary representing
    the functionality of a quantum computation. This function does not support
    measurements, resets, or classical control, as the corresponding operations
    are non-unitary.

    Args:
        qc: The quantum computation.
            Must only contain unitary operations.
        dd_package: The DD package.
                    Must be configured with a sufficient number of qubits to
                    accommodate the quantum computation.
        recursive: Whether to build the functionality matrix recursively. If set
                   to True, the functionality matrix is built recursively by
                   pairwise grouping the operations of the quantum computation.
                   If set to False, the functionality matrix is built by
                   sequentially applying the operations of the quantum
                   computation to the identity matrix. Defaults to False.

    Returns:
        The functionality as a DD. The reference count of the result is non-zero
        and must be manually decremented by the caller if it is no longer needed.
    """

class DDPackage:
    """The central manager for performing computations on decision diagrams.

    It drives all computation on decision diagrams and maintains the necessary
    data structures for this purpose.
    Specifically, it

    - manages the memory for the decision diagram nodes (Memory Manager),
    - ensures the canonical representation of decision diagrams (Unique Table),
    - ensures the efficiency of decision diagram operations (Compute Table),
    - provides methods for creating quantum states and operations from various sources,
    - provides methods for various operations on quantum states and operations, and
    - provides means for reference counting and garbage collection.

    Args:
        num_qubits: The maximum number of qubits that the DDPackage can handle.
                    Mainly influences the size of the unique tables.
                    Can be adjusted dynamically using the `resize` method.
                    Since resizing the DDPackage can be expensive, it is recommended
                    to choose a value that is large enough for the quantum computations
                    that are to be performed, but not unnecessarily large.
                    Default is 32.

    Notes:
        It is undefined behavior to pass VectorDD or MatrixDD objects that were
        created with a different DDPackage to the methods of the DDPackage.

    """

    def __init__(self, num_qubits: int = 32) -> None: ...
    def resize(self, num_qubits: int) -> None:
        """Resize the DDPackage to accommodate a different number of qubits.

        Args:
            num_qubits: The new number of qubits.
                        Must be greater than zero.
                        It is undefined behavior to resize the DDPackage to a
                        smaller number of qubits and then perform operations
                        on decision diagrams that are associated with qubits
                        that are no longer present.
        """

    @property
    def max_qubits(self) -> int:
        """The maximum number of qubits that the DDPackage can handle."""

    def zero_state(self, num_qubits: int) -> VectorDD:
        r"""Create the DD for the zero state :math:`|0\ldots 0\rangle`.

        Args:
            num_qubits: The number of qubits.
                        Must not be greater than the number of qubits the DDPackage is configured with.

        Returns:
            The DD for the zero state.
            The resulting state is guaranteed to have its reference count increased.
        """

    def computational_basis_state(self, num_qubits: int, state: list[bool]) -> VectorDD:
        r"""Create the DD for the computational basis state :math:`|b_{n-1} \ldots b_0\rangle`.

        Args:
            num_qubits: The number of qubits.
                        Must not be greater than the number of qubits the DDPackage is configured with.
            state: The state as a list of booleans.
                   Must be at least `num_qubits` long.

        Returns:
            The DD for the computational basis state.
            The resulting state is guaranteed to have its reference count increased.
        """

    def basis_state(self, num_qubits: int, state: Iterable[BasisStates]) -> VectorDD:
        r"""Create the DD for the basis state :math:`|B_{n-1} \ldots B_0\rangle`, where :math:`B_i\in\{0,1,+\,-,L,R\}`.

        Args:
            num_qubits: The number of qubits.
                        Must not be greater than the number of qubits the DDPackage is configured with.
            state: The state as an iterable of :class:`BasisStates`.
                   Must be at least `num_qubits` long.

        Returns:
            The DD for the basis state.
            The resulting state is guaranteed to have its reference count increased.
        """

    def ghz_state(self, num_qubits: int) -> VectorDD:
        r"""Create the DD for the GHZ state :math:`\frac{1}{\sqrt{2}}(|0\ldots 0\rangle + |1\ldots 1\rangle)`.

        Args:
            num_qubits: The number of qubits.
                        Must not be greater than the number of qubits the DDPackage is configured with.

        Returns:
            The DD for the GHZ state.
            The resulting state is guaranteed to have its reference count increased.
        """

    def w_state(self, num_qubits: int) -> VectorDD:
        r"""Create the DD for the W state :math:`|W\rangle`.

        :math:`|W\rangle = \frac{1}{\sqrt{n}}(|100\ldots 0\rangle + |010\ldots 0\rangle + \ldots + |000\ldots 1\rangle)`

        Args:
            num_qubits: The number of qubits.
                        Must not be greater than the number of qubits the DDPackage is configured with.

        Returns:
            The DD for the W state.
            The resulting state is guaranteed to have its reference count increased.
        """

    def from_vector(self, state: npt.NDArray[(Any,), np.cdouble]) -> VectorDD:
        """Create a DD from a state vector.

        Args:
            state: The state vector.
                   Must have a length that is a power of 2.
                   Must not require more qubits than the DDPackage is configured with.

        Returns:
            The DD for the vector.
            The resulting state is guaranteed to have its reference count increased.
        """

    def apply_unitary_operation(self, vec: VectorDD, operation: Operation, permutation: Permutation = ...) -> VectorDD:
        """Apply a unitary operation to the DD.

        Args:
            vec: The input DD.
            operation: The operation.
                       Must be unitary.
            permutation: The permutation of the qubits.
                         Defaults to the identity permutation.

        Returns:
            The resulting DD.

        Notes:
            Automatically manages the reference count of the input and output DDs.
            The input DD must have a non-zero reference count.
        """

    def apply_measurement(
        self,
        vec: VectorDD,
        operation: NonUnitaryOperation,
        measurements: list[bool],
        permutation: Permutation = ...,
    ) -> tuple[VectorDD, list[bool]]:
        """Apply a measurement to the DD.

        Args:
            vec: The input DD.
            operation: The measurement operation.
            measurements: A list of bits with existing measurement outcomes.
            permutation: The permutation of the qubits.
                         Defaults to the identity permutation.

        Returns:
            The resulting DD after the measurement as well as the updated measurement outcomes.

        Notes:
            Automatically manages the reference count of the input and output DDs.
            The input DD must have a non-zero reference count.
        """

    def apply_reset(self, vec: VectorDD, operation: NonUnitaryOperation, permutation: Permutation = ...) -> VectorDD:
        """Apply a reset to the DD.

        Args:
            vec: The input DD.
            operation: The reset operation.
            permutation: The permutation of the qubits.
                         Defaults to the identity permutation.

        Returns:
            The resulting DD after the reset.

        Notes:
            Automatically manages the reference count of the input and output DDs.
            The input DD must have a non-zero reference count.
        """

    def apply_if_else_operation(
        self,
        vec: VectorDD,
        operation: IfElseOperation,
        measurements: list[bool],
        permutation: Permutation = ...,
    ) -> VectorDD:
        """Apply a classically controlled operation to the DD.

        Args:
            vec: The input DD.
            operation: The classically controlled operation.
            measurements: A list of bits with stored measurement outcomes.
            permutation: The permutation of the qubits. Defaults to the identity permutation.

        Returns:
            The resulting DD after the operation.

        Notes:
            Automatically manages the reference count of the input and output DDs.
            The input DD must have a non-zero reference count.
        """

    def measure_collapsing(self, vec: VectorDD, qubit: int) -> str:
        """Measure a qubit and collapse the DD.

        Args:
            vec: The input DD.
            qubit: The qubit to measure.

        Returns:
            The measurement outcome.

        Notes:
            Automatically manages the reference count of the input and output DDs.
            The input DD must have a non-zero reference count.
        """

    def measure_all(self, vec: VectorDD, collapse: bool = False) -> str:
        """Measure all qubits.

        Args:
            vec: The input DD.
            collapse: Whether to collapse the DD.

        Returns:
            The measurement outcome.

        Notes:
            Automatically manages the reference count of the input and output DDs.
            The input DD must have a non-zero reference count.
        """

    @staticmethod
    def identity() -> MatrixDD:
        r"""Create the DD for the identity matrix :math:`I`.

        Returns:
            The DD for the identity matrix.
        """

    def single_qubit_gate(
        self,
        matrix: npt.NDArray[np.cdouble],
        target: int,
    ) -> MatrixDD:
        r"""Create the DD for a single-qubit gate.

        Args:
            matrix: The :math:`2\times 2` matrix representing the single-qubit gate.
            target: The target qubit.

        Returns:
            The DD for the single-qubit gate.
        """

    def controlled_single_qubit_gate(
        self, matrix: npt.NDArray[np.cdouble], control: Control | int, target: int
    ) -> MatrixDD:
        r"""Create the DD for a controlled single-qubit gate.

        Args:
            matrix: The :math:`2\times 2` matrix representing the single-qubit gate.
            control: The control qubit.
            target: The target qubit.

        Returns:
            The DD for the controlled single-qubit gate.
        """

    def multi_controlled_single_qubit_gate(
        self,
        matrix: npt.NDArray[np.cdouble],
        controls: set[Control | int],
        target: int,
    ) -> MatrixDD:
        r"""Create the DD for a multi-controlled single-qubit gate.

        Args:
            matrix: The :math:`2\times 2` matrix representing the single-qubit gate.
            controls: The control qubits.
            target: The target qubit.

        Returns:
            The DD for the multi-controlled single-qubit gate.
        """

    def two_qubit_gate(
        self,
        matrix: npt.NDArray[np.cdouble],
        target0: int,
        target1: int,
    ) -> MatrixDD:
        r"""Create the DD for a two-qubit gate.

        Args:
            matrix: The :math:`4\times 4` matrix representing the two-qubit gate.
            target0: The first target qubit.
            target1: The second target qubit.

        Returns:
            The DD for the two-qubit gate.
        """

    def controlled_two_qubit_gate(
        self,
        matrix: npt.NDArray[np.cdouble],
        control: Control | int,
        target0: int,
        target1: int,
    ) -> MatrixDD:
        r"""Create the DD for a controlled two-qubit gate.

        Args:
            matrix: The :math:`4\times 4` matrix representing the two-qubit gate.
            control: The control qubit.
            target0: The first target qubit.
            target1: The second target qubit.

        Returns:
            The DD for the controlled two-qubit gate.
        """

    def multi_controlled_two_qubit_gate(
        self,
        matrix: npt.NDArray[np.cdouble],
        controls: set[Control | int],
        target0: int,
        target1: int,
    ) -> MatrixDD:
        r"""Create the DD for a multi-controlled two-qubit gate.

        Args:
            matrix: The :math:`4\times 4` matrix representing the two-qubit gate.
            controls: The control qubits.
            target0: The first target qubit.
            target1: The second target qubit.

        Returns:
            The DD for the multi-controlled two-qubit gate.
        """

    def from_matrix(self, matrix: npt.NDArray[np.cdouble]) -> MatrixDD:
        """Create a DD from a matrix.

        Args:
            matrix: The matrix.
                    Must be square and have a size that is a power of 2.

        Returns:
            The DD for the matrix.
        """

    def from_operation(self, operation: Operation, invert: bool = False) -> MatrixDD:
        """Create a DD from an operation.

        Args:
            operation: The operation.
                       Must be unitary.
            invert: Whether to get the inverse of the operation.

        Returns:
            The DD for the operation.
        """

    def inc_ref_vec(self, vec: VectorDD) -> None:
        """Increment the reference count of a vector."""

    def dec_ref_vec(self, vec: VectorDD) -> None:
        """Decrement the reference count of a vector."""

    def inc_ref_mat(self, mat: MatrixDD) -> None:
        """Increment the reference count of a matrix."""

    def dec_ref_mat(self, mat: MatrixDD) -> None:
        """Decrement the reference count of a matrix."""

    def garbage_collect(self, force: bool = False) -> bool:
        """Perform garbage collection on the DDPackage.

        Args:
            force: Whether to force garbage collection.
                   If set to True, garbage collection is performed regardless
                   of the current memory usage. If set to False, garbage collection
                   is only performed if the memory usage exceeds a certain threshold.

        Returns:
            Whether any nodes were collected during garbage collection.
        """

    def vector_add(self, lhs: VectorDD, rhs: VectorDD) -> VectorDD:
        """Add two vectors.

        Args:
            lhs: The left vector.
            rhs: The right vector.

        Returns:
            The sum of the two vectors.

        Notes:
            It is the caller's responsibility to update the reference count of the
            input and output vectors after the operation.

            Both vectors must have the same number of qubits.
        """

    def matrix_add(self, lhs: MatrixDD, rhs: MatrixDD) -> MatrixDD:
        """Add two matrices.

        Args:
            lhs: The left matrix.
            rhs: The right matrix.

        Returns:
            The sum of the two matrices.

        Notes:
            It is the caller's responsibility to update the reference count of the
            input and output matrices after the operation.

            Both matrices must have the same number of qubits.
        """

    def conjugate(self, vec: VectorDD) -> VectorDD:
        """Conjugate a vector.

        Args:
            vec: The vector.

        Returns:
            The conjugated vector.

        Notes:
            It is the caller's responsibility to update the reference count of the
            input and output vectors after the operation.
        """

    def conjugate_transpose(self, mat: MatrixDD) -> MatrixDD:
        """Conjugate transpose a matrix.

        Args:
            mat: The matrix.

        Returns:
            The conjugate transposed matrix.

        Notes:
            It is the caller's responsibility to update the reference count of the
            input and output matrices after the operation.
        """

    def matrix_vector_multiply(self, mat: MatrixDD, vec: VectorDD) -> VectorDD:
        """Multiply a matrix with a vector.

        Args:
            mat: The matrix.
            vec: The vector.

        Returns:
            The product of the matrix and the vector.

        Notes:
            It is the caller's responsibility to update the reference count of the
            input and output matrices after the operation.

            The vector must have at least as many qubits as the matrix non-trivially acts on.
        """

    def matrix_multiply(self, lhs: MatrixDD, rhs: MatrixDD) -> MatrixDD:
        """Multiply two matrices.

        Args:
            lhs: The left matrix.
            rhs: The right matrix.

        Returns:
            The product of the two matrices.

        Notes:
            It is the caller's responsibility to update the reference count of the
            input and output matrices after the operation.
        """

    def inner_product(self, lhs: VectorDD, rhs: VectorDD) -> complex:
        """Compute the inner product of two vectors.

        Args:
            lhs: The left vector.
            rhs: The right vector.

        Returns:
            The inner product of the two vectors.

        Notes:
            Both vectors must have the same number of qubits.
        """

    def fidelity(self, lhs: VectorDD, rhs: VectorDD) -> float:
        """Compute the fidelity of two vectors.

        Args:
            lhs: The left vector.
            rhs: The right vector.

        Returns:
            The fidelity of the two vectors.

        Notes:
            Both vectors must have the same number of qubits.
        """

    def expectation_value(self, observable: MatrixDD, state: VectorDD) -> float:
        r"""Compute the expectation value of an observable.

        Args:
            observable: The observable.
            state: The state.

        Returns:
            The expectation value of the observable.

        Notes:
            The state must have at least as many qubits as the observable non-trivially acts on.

            The method computes :math:`\langle \psi | O | \psi \rangle` as
            :math:`\langle \psi | (O | \psi \rangle)`.
        """

    def vector_kronecker(
        self, top: VectorDD, bottom: VectorDD, bottom_num_qubits: int, increment_index: bool = True
    ) -> VectorDD:
        """Compute the Kronecker product of two vectors.

        Args:
            top: The top vector.
            bottom: The bottom vector.
            bottom_num_qubits: The number of qubits of the bottom vector.
            increment_index: Whether to increment the indexes of the top vector.

        Returns:
            The Kronecker product of the two vectors.

        Notes:
            It is the caller's responsibility to update the reference count of the
            input and output vectors after the operation.
        """

    def matrix_kronecker(
        self, top: MatrixDD, bottom: MatrixDD, bottom_num_qubits: int, increment_index: bool = True
    ) -> MatrixDD:
        """Compute the Kronecker product of two matrices.

        Args:
            top: The top matrix.
            bottom: The bottom matrix.
            bottom_num_qubits: The number of qubits of the bottom matrix.
            increment_index: Whether to increment the indexes of the top matrix.

        Returns:
            The Kronecker product of the two matrices.

        Notes:
            It is the caller's responsibility to update the reference count of the
            input and output matrices after the operation.
        """

    def partial_trace(self, mat: MatrixDD, eliminate: list[bool]) -> MatrixDD:
        """Compute the partial trace of a matrix.

        Args:
            mat: The matrix.
            eliminate: The qubits to eliminate.
                       Must be at least as long as the number of qubits of the matrix.

        Returns:
            The partial trace of the matrix.
        """

    def trace(self, mat: MatrixDD, num_qubits: int) -> complex:
        """Compute the trace of a matrix.

        Args:
            mat: The matrix.
            num_qubits: The number of qubits of the matrix.

        Returns:
            The trace of the matrix.
        """

class VectorDD:
    """A class representing a vector decision diagram (DD)."""

    def is_terminal(self) -> bool:
        """Check if the DD is a terminal node."""

    def is_zero_terminal(self) -> bool:
        """Check if the DD is a zero terminal node."""

    def size(self) -> int:
        """Get the size of the DD by traversing it once."""

    def __getitem__(self, index: int) -> complex:
        """Get the amplitude of a basis state by index."""

    def get_amplitude(self, num_qubits: int, decisions: str) -> complex:
        """Get the amplitude of a basis state by decisions.

        Args:
            num_qubits: The number of qubits.
            decisions: The decisions as a string of bits (`0` or `1`), where
                       `decisions[i]` corresponds to the successor to follow at level `i` of the DD.
                       Must be at least `num_qubits` long.

        Returns:
            The amplitude of the basis state.
        """

    def get_vector(self, threshold: float = 0.0) -> Vector:
        """Get the state vector represented by the DD.

        Args:
            threshold: The threshold for not including amplitudes in the state vector.
                       Defaults to 0.0.

        Returns:
            The state vector.
        """

    def to_dot(
        self,
        colored: bool = True,
        edge_labels: bool = False,
        classic: bool = False,
        memory: bool = False,
        format_as_polar: bool = True,
    ) -> str:
        """Convert the DD to a DOT graph that can be plotted via Graphviz.

        Args:
            colored: Whether to use colored edge weights
            edge_labels: Whether to include edge weights as labels.
            classic: Whether to use the classic DD visualization style.
            memory: Whether to include memory information.
                    For debugging purposes only.
            format_as_polar: Whether to format the edge weights in polar coordinates.

        Returns:
            The DOT graph.
        """

    def to_svg(
        self,
        filename: str,
        colored: bool = True,
        edge_labels: bool = False,
        classic: bool = False,
        memory: bool = False,
        format_as_polar: bool = True,
    ) -> None:
        """Convert the DD to an SVG file that can be viewed in a browser.

        Requires the `dot` command from Graphviz to be installed and available in the PATH.

        Args:
            filename: The filename of the SVG file.
                      Any file extension will be replaced by `.dot` and then `.svg`.
            colored: Whether to use colored edge weights.
            edge_labels: Whether to include edge weights as labels.
            classic: Whether to use the classic DD visualization style.
            memory: Whether to include memory information.
                    For debugging purposes only.
            show: Whether to open the SVG file in the default browser.
            format_as_polar: Whether to format the edge weights in polar coordinates.
        """

class MatrixDD:
    """A class representing a matrix decision diagram (DD)."""

    def is_terminal(self) -> bool:
        """Check if the DD is a terminal node."""

    def is_zero_terminal(self) -> bool:
        """Check if the DD is a zero terminal node."""

    def is_identity(self, up_to_global_phase: bool = True) -> bool:
        """Check if the DD represents the identity matrix.

        Args:
            up_to_global_phase: Whether to ignore global phase.

        Returns:
            Whether the DD represents the identity matrix.
        """

    def size(self) -> int:
        """Get the size of the DD by traversing it once."""

    def get_entry(self, num_qubits: int, row: int, col: int) -> complex:
        """Get the entry of the matrix by row and column index."""

    def get_entry_by_path(self, num_qubits: int, decisions: str) -> complex:
        """Get the entry of the matrix by decisions.

        Args:
            num_qubits: The number of qubits.
            decisions: The decisions as a string of `0`, `1`, `2`, or `3`, where
                       `decisions[i]` corresponds to the successor to follow at level `i` of the DD.
                       Must be at least `num_qubits` long.

        Returns:
            The entry of the matrix.
        """

    def get_matrix(self, num_qubits: int, threshold: float = 0.0) -> Matrix:
        """Get the matrix represented by the DD.

        Args:
            num_qubits: The number of qubits.
            threshold: The threshold for not including entries in the matrix.
                       Defaults to 0.0.

        Returns:
            The matrix.
        """

    def to_dot(
        self,
        colored: bool = True,
        edge_labels: bool = False,
        classic: bool = False,
        memory: bool = False,
        format_as_polar: bool = True,
    ) -> str:
        """Convert the DD to a DOT graph that can be plotted via Graphviz.

        Args:
            colored: Whether to use colored edge weights
            edge_labels: Whether to include edge weights as labels.
            classic: Whether to use the classic DD visualization style.
            memory: Whether to include memory information.
                    For debugging purposes only.
            format_as_polar: Whether to format the edge weights in polar coordinates.

        Returns:
            The DOT graph.
        """

    def to_svg(
        self,
        filename: str,
        colored: bool = True,
        edge_labels: bool = False,
        classic: bool = False,
        memory: bool = False,
        format_as_polar: bool = True,
    ) -> None:
        """Convert the DD to an SVG file that can be viewed in a browser.

        Requires the `dot` command from Graphviz to be installed and available in the PATH.

        Args:
            filename: The filename of the SVG file.
                      Any file extension will be replaced by `.dot` and then `.svg`.
            colored: Whether to use colored edge weights.
            edge_labels: Whether to include edge weights as labels.
            classic: Whether to use the classic DD visualization style.
            memory: Whether to include memory information.
                    For debugging purposes only.
            show: Whether to open the SVG file in the default browser.
            format_as_polar: Whether to format the edge weights in polar coordinates.
        """

class BasisStates(Enum):
    """Enumeration of basis states."""

    zero = ...
    r"""The computational basis state :math:`|0\rangle`."""
    one = ...
    r"""The computational basis state :math:`|1\rangle`."""
    plus = ...
    r"""The superposition state :math:`|+\rangle = \frac{1}{\sqrt{2}}(|0\rangle + |1\rangle)`."""
    minus = ...
    r"""The superposition state :math:`|-\rangle = \frac{1}{\sqrt{2}}(|0\rangle - |1\rangle)`."""
    left = ...
    r"""The rotational superposition state :math:`|L\rangle = \frac{1}{\sqrt{2}}(|0\rangle + i|1\rangle)`."""
    right = ...
    r"""The rotational superposition state :math:`|R\rangle = \frac{1}{\sqrt{2}}(|0\rangle - i|1\rangle)`."""

class Vector:
    """A class representing a vector of complex numbers.

    Implements the buffer protocol so that the underlying memory can be accessed
    and easily converted to a NumPy array without copying.

    Examples:
        >>> from mqt.core.dd import DDPackage
        ... import numpy as np
        ...
        ... zero_state = DDPackage(2).zero_state(2)
        ... vec = np.array(zero_state.get_vector(), copy=False)
        ... print(vec)
        [1.+0.j 0.+0.j 0.+0.j 0.+0.j]

    """

    def __buffer__(self, flags: int, /) -> memoryview:
        """Return a buffer object that exposes the underlying memory of the object."""

    def __release_buffer__(self, buffer: memoryview, /) -> None:
        """Release the buffer object that exposes the underlying memory of the object."""

class Matrix:
    """A class representing a matrix of complex numbers.

    Implements the buffer protocol so that the underlying memory can be accessed
    and easily converted to a NumPy array without copying.

    Examples:
        >>> from mqt.core.dd import DDPackage
        ... import numpy as np
        ...
        ... identity = DDPackage(1).identity()
        ... mat = np.array(identity.get_matrix(1), copy=False)
        ... print(mat)
        [[1.+0.j 0.+0.j]
         [0.+0.j 1.+0.j]]
    """

    def __buffer__(self, flags: int, /) -> memoryview:
        """Return a buffer object that exposes the underlying memory of the object."""

    def __release_buffer__(self, buffer: memoryview, /) -> None:
        """Release the buffer object that exposes the underlying memory of the object."""
