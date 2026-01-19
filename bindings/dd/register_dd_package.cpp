/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "dd/CachedEdge.hpp"
#include "dd/DDDefinitions.hpp"
#include "dd/Node.hpp"
#include "dd/Operations.hpp"
#include "dd/Package.hpp"
#include "dd/StateGeneration.hpp"
#include "ir/Permutation.hpp"
#include "ir/operations/Control.hpp"
#include "ir/operations/IfElseOperation.hpp"
#include "ir/operations/NonUnitaryOperation.hpp"
#include "ir/operations/Operation.hpp"

#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/complex.h> // NOLINT(misc-include-cleaner)
#include <nanobind/stl/pair.h>    // NOLINT(misc-include-cleaner)
#include <nanobind/stl/set.h>     // NOLINT(misc-include-cleaner)
#include <nanobind/stl/string.h>  // NOLINT(misc-include-cleaner)
#include <nanobind/stl/vector.h>  // NOLINT(misc-include-cleaner)
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

namespace mqt {

namespace nb = nanobind;
using namespace nb::literals;

using Vector = nb::ndarray<nb::numpy, std::complex<dd::fp>, nb::ndim<1>>;
using Matrix = nb::ndarray<nb::numpy, std::complex<dd::fp>, nb::ndim<2>>;
using SingleQubitMatrix =
    nb::ndarray<nb::numpy, std::complex<dd::fp>, nb::shape<2, 2>>;
using TwoQubitMatrix =
    nb::ndarray<nb::numpy, std::complex<dd::fp>, nb::shape<4, 4>>;

namespace {

/// Recursive helper function to create a vector DD from a numpy array
dd::vCachedEdge makeDDFromVector(dd::Package& p, const Vector& v,
                                 const size_t startIdx, const size_t endIdx,
                                 const dd::Qubit level) {
  if (level == 0U) {
    const auto zeroSuccessor = dd::vCachedEdge::terminal(v(startIdx));
    const auto oneSuccessor = dd::vCachedEdge::terminal(v(startIdx + 1));
    return p.makeDDNode<dd::vNode, dd::CachedEdge>(
        0, {zeroSuccessor, oneSuccessor});
  }

  const auto half = startIdx + ((endIdx - startIdx) / 2);
  const auto zeroSuccessor = makeDDFromVector(p, v, startIdx, half, level - 1);
  const auto oneSuccessor = makeDDFromVector(p, v, half, endIdx, level - 1);
  return p.makeDDNode<dd::vNode, dd::CachedEdge>(level,
                                                 {zeroSuccessor, oneSuccessor});
}

/// Recursive helper function to create a matrix DD from a numpy array
dd::mCachedEdge makeDDFromMatrix(dd::Package& p, const Matrix& m,
                                 const size_t rowStart, const size_t rowEnd,
                                 const size_t colStart, const size_t colEnd,
                                 const dd::Qubit level) {
  if (level == 0U) {
    const auto zeroSuccessor = dd::mCachedEdge::terminal(m(rowStart, colStart));
    const auto oneSuccessor =
        dd::mCachedEdge::terminal(m(rowStart, colStart + 1));
    const auto twoSuccessor =
        dd::mCachedEdge::terminal(m(rowStart + 1, colStart));
    const auto threeSuccessor =
        dd::mCachedEdge::terminal(m(rowStart + 1, colStart + 1));
    return p.makeDDNode<dd::mNode, dd::CachedEdge>(
        0, {zeroSuccessor, oneSuccessor, twoSuccessor, threeSuccessor});
  }

  const auto rowHalf = rowStart + ((rowEnd - rowStart) / 2);
  const auto colHalf = colStart + ((colEnd - colStart) / 2);
  return p.makeDDNode<dd::mNode, dd::CachedEdge>(
      level,
      {makeDDFromMatrix(p, m, rowStart, rowHalf, colStart, colHalf, level - 1),
       makeDDFromMatrix(p, m, rowStart, rowHalf, colHalf, colEnd, level - 1),
       makeDDFromMatrix(p, m, rowHalf, rowEnd, colStart, colHalf, level - 1),
       makeDDFromMatrix(p, m, rowHalf, rowEnd, colHalf, colEnd, level - 1)});
}
} // namespace

// NOLINTNEXTLINE(misc-use-internal-linkage)
void registerDDPackage(const nb::module_& m) {
  auto dd = nb::class_<dd::Package>(
      m, "DDPackage",
      R"pb(The central manager for performing computations on decision diagrams.

It drives all computation on decision diagrams and maintains the necessary data structures for this purpose.
Specifically, it

- manages the memory for the decision diagram nodes (Memory Manager),
- ensures the canonical representation of decision diagrams (Unique Table),
- ensures the efficiency of decision diagram operations (Compute Table),
- provides methods for creating quantum states and operations from various sources,
- provides methods for various operations on quantum states and operations, and
- provides means for reference counting and garbage collection.

Notes:
    It is undefined behavior to pass VectorDD or MatrixDD objects that were created with a different DDPackage to the methods of the DDPackage.
    The only exception is the identity DD returned by identity(), which represents the global one-terminal and can be used with any DDPackage instance.

Args:
    num_qubits: The maximum number of qubits that the DDPackage can handle.
        Mainly influences the size of the unique tables.
        Can be adjusted dynamically using the `resize` method.
        Since resizing the DDPackage can be expensive, it is recommended to choose a value that is large enough for the quantum computations that are to be performed, but not unnecessarily large.
        Default is 32.)pb");

  // Constructor
  dd.def(nb::init<size_t>(), "num_qubits"_a = dd::Package::DEFAULT_QUBITS);

  // Resizing the package
  dd.def("resize", &dd::Package::resize, "num_qubits"_a,
         R"pb(Resize the DDPackage to accommodate a different number of qubits.

Args:
    num_qubits: The new number of qubits.
        Must be greater than zero.
        It is undefined behavior to resize the DDPackage to a smaller number of qubits and then perform operations on decision diagrams that are associated with qubits that are no longer present.)pb");

  // Getting the number of qubits the package is configured for
  dd.def_prop_ro("max_qubits", &dd::Package::qubits,
                 "The maximum number of qubits that the DDPackage can handle.");

  ///------------------------------------------------------------------------///
  /// Vector DD Generation
  ///------------------------------------------------------------------------///

  dd.def(
      "zero_state",
      [](dd::Package& p, const size_t numQubits) {
        return makeZeroState(numQubits, p);
      },
      "num_qubits"_a,
      // keep the DD package alive while the returned vector DD is alive.
      nb::keep_alive<0, 1>(),
      R"pb(Create the DD for the zero state :math:`| 0 \ldots 0 \rangle`.

Args:
    num_qubits: The number of qubits.
        Must not be greater than the number of qubits the DDPackage is configured with.

Returns:
    The DD for the zero state.
    The resulting state is guaranteed to have its reference count increased.)pb");

  dd.def(
      "computational_basis_state",
      [](dd::Package& p, const size_t numQubits,
         const std::vector<bool>& state) {
        return makeBasisState(numQubits, state, p);
      },
      "num_qubits"_a, "state"_a,
      // keep the DD package alive while the returned vector DD is alive.
      nb::keep_alive<0, 1>(),
      R"pb(Create the DD for the computational basis state :math:`| b_{n - 1} \ldots b_0 \rangle`.

Args:
    num_qubits: The number of qubits.
        Must not be greater than the number of qubits the DDPackage is configured with.
    state: The state as a list of booleans.
        Must be at least `num_qubits` long.

Returns:
    The DD for the computational basis state.
    The resulting state is guaranteed to have its reference count increased.)pb");

  nb::enum_<dd::BasisStates>(m, "BasisStates", "Enumeration of basis states.")
      .value("zero", dd::BasisStates::zero,
             R"pb(The computational basis state :math:`|0\rangle`.)pb")
      .value("one", dd::BasisStates::one,
             R"pb(The computational basis state :math:`|1\rangle`.)pb")
      .value(
          "plus", dd::BasisStates::plus,
          R"pb(The superposition state :math:`|+\rangle = \frac{1}{\sqrt{2}} (|0\rangle + |1\rangle)`.)pb")
      .value(
          "minus", dd::BasisStates::minus,
          R"pb(The superposition state :math:`|-\rangle = \frac{1}{\sqrt{2}} (|0\rangle - |1\rangle)`.)pb")
      .value(
          "right", dd::BasisStates::right,
          R"pb(The superposition state :math:`|R\rangle = \frac{1}{\sqrt{2}} (|0\rangle - i |1\rangle)`.)pb")
      .value(
          "left", dd::BasisStates::left,
          R"pb(The superposition state :math:`|L\rangle = \frac{1}{\sqrt{2}} (|0\rangle + i |1\rangle)`.)pb");

  dd.def(
      "basis_state",
      [](dd::Package& p, const size_t numQubits,
         const std::vector<dd::BasisStates>& state) {
        return makeBasisState(numQubits, state, p);
      },
      "num_qubits"_a, "state"_a,
      // keep the DD package alive while the returned vector DD is alive.
      nb::keep_alive<0, 1>(),
      R"pb(Create the DD for the basis state :math:`| B_{n - 1} \ldots B_0 \rangle`, where :math:`B_i \in \{0, 1, +\, -\, L, R\}`.

Args:
    num_qubits: The number of qubits.
        Must not be greater than the number of qubits the DDPackage is configured with.
    state: The state as an iterable of :class:`BasisStates`.
        Must be at least `num_qubits` long.

Returns:
    The DD for the basis state.
    The resulting state is guaranteed to have its reference count increased.)pb");

  dd.def(
      "ghz_state",
      [](dd::Package& p, const size_t numQubits) {
        return makeGHZState(numQubits, p);
      },
      "num_qubits"_a,
      // keep the DD package alive while the returned vector DD is alive.
      nb::keep_alive<0, 1>(),
      R"pb(Create the DD for the GHZ state :math:`\frac{1}{\sqrt{2}} (| 0 \ldots 0 \rangle + |1 \ldots 1 \rangle)`.

Args:
    num_qubits: The number of qubits.
        Must not be greater than the number of qubits the DDPackage is configured with.

Returns:
    The DD for the GHZ state.
    The resulting state is guaranteed to have its reference count increased.)pb");

  dd.def(
      "w_state",
      [](dd::Package& p, const size_t numQubits) {
        return makeWState(numQubits, p);
      },
      "num_qubits"_a,
      // keep the DD package alive while the returned vector DD is alive.
      nb::keep_alive<0, 1>(),
      R"pb(Create the DD for the W state :math:`|W\rangle`.

.. math::
    |W\rangle = \frac{1}{\sqrt{n}} (| 100 \ldots 0 \rangle + | 010 \ldots 0 \rangle + \ldots + | 000 \ldots 1 \rangle)

Args:
    num_qubits: The number of qubits.
        Must not be greater than the number of qubits the DDPackage is configured with.

Returns:
    The DD for the W state.
    The resulting state is guaranteed to have its reference count increased.)pb");

  dd.def(
      "from_vector",
      [](dd::Package& p, const Vector& v) {
        const auto length = v.shape(0);
        if (length == 0) {
          return dd::vEdge::one();
        }
        if ((length & (length - 1)) != 0) {
          throw std::invalid_argument(
              "State vector must have a length of a power of two.");
        }
        if (length == 1) {
          const auto state = dd::vEdge::terminal(p.cn.lookup(v(0)));
          p.incRef(state);
          return state;
        }
        const auto level = static_cast<dd::Qubit>(std::log2(length) - 1);
        const auto state = makeDDFromVector(p, v, 0, length, level);
        const dd::vEdge e{.p = state.p, .w = p.cn.lookup(state.w)};
        p.incRef(e);
        return e;
      },
      "state"_a,
      // keep the DD package alive while the returned vector DD is alive.
      nb::keep_alive<0, 1>(), R"pb(Create a DD from a state vector.

Args:
    state: The state vector.
        Must have a length that is a power of 2.
        Must not require more qubits than the DDPackage is configured with.

Returns:
    The DD for the vector.
    The resulting state is guaranteed to have its reference count increased.)pb");

  dd.def(
      "apply_unitary_operation",
      [](dd::Package& p, const dd::vEdge& v, const qc::Operation& op,
         const qc::Permutation& perm = {}) {
        return applyUnitaryOperation(op, v, p, perm);
      },
      "vec"_a, "operation"_a, "permutation"_a = qc::Permutation{},
      // keep the DD package alive while the returned vector DD is alive.
      nb::keep_alive<0, 1>(), R"pb(Apply a unitary operation to the DD.

Notes:
    Automatically manages the reference count of the input and output DDs.
    The input DD must have a non-zero reference count.

Args:
    vec: The input DD.
    operation: The operation. Must be unitary.
    permutation: The permutation of the qubits. Defaults to the identity permutation.

Returns:
    The resulting DD.)pb");

  dd.def(
      "apply_measurement",
      [](dd::Package& p, const dd::vEdge& v, const qc::NonUnitaryOperation& op,
         const std::vector<bool>& measurements,
         const qc::Permutation& perm = {}) {
        static std::mt19937_64 rng(std::random_device{}());
        auto measurementsCopy = measurements;
        return std::pair{
            applyMeasurement(op, v, p, rng, measurementsCopy, perm),
            measurementsCopy};
      },
      "vec"_a, "operation"_a, "measurements"_a,
      "permutation"_a = qc::Permutation{},
      // keep the DD package alive while the returned vector DD is alive.
      nb::keep_alive<0, 1>(), R"pb(Apply a measurement to the DD.

Notes:
    Automatically manages the reference count of the input and output DDs.
    The input DD must have a non-zero reference count

Args:
    vec: The input DD.
    operation: The measurement operation.
    measurements: A list of bits with existing measurement outcomes.
    permutation: The permutation of the qubits. Defaults to the identity permutation.

Returns:
    The resulting DD after the measurement as well as the updated measurement outcomes.)pb");

  dd.def(
      "apply_reset",
      [](dd::Package& p, const dd::vEdge& v, const qc::NonUnitaryOperation& op,
         const qc::Permutation& perm = {}) {
        static std::mt19937_64 rng(std::random_device{}());
        return applyReset(op, v, p, rng, perm);
      },
      "vec"_a, "operation"_a, "permutation"_a = qc::Permutation{},
      // keep the DD package alive while the returned vector DD is alive.
      nb::keep_alive<0, 1>(), R"pb(Apply a reset to the DD.

Notes:
    Automatically manages the reference count of the input and output DDs.
    The input DD must have a non-zero reference count.

Args:
    vec: The input DD.
    operation: The reset operation.
    permutation: The permutation of the qubits. Defaults to the identity permutation.

Returns:
    The resulting DD after the reset.)pb");

  dd.def(
      "apply_if_else_operation",
      [](dd::Package& p, const dd::vEdge& v, const qc::IfElseOperation& op,
         const std::vector<bool>& measurements,
         const qc::Permutation& perm = {}) {
        return applyIfElseOperation(op, v, p, measurements, perm);
      },
      "vec"_a, "operation"_a, "measurements"_a,
      "permutation"_a = qc::Permutation{},
      // keep the DD package alive while the returned vector DD is alive.
      nb::keep_alive<0, 1>(),
      R"pb(Apply a classically controlled operation to the DD.

Notes:
    Automatically manages the reference count of the input and output DDs.
    The input DD must have a non-zero reference count.

Args:
    vec: The input DD.
    operation: The classically controlled operation.
    measurements: A list of bits with stored measurement outcomes.
    permutation: The permutation of the qubits. Defaults to the identity permutation.

Returns:
    The resulting DD after the operation.)pb");

  dd.def(
      "measure_collapsing",
      [](dd::Package& p, dd::vEdge& v, const dd::Qubit q) {
        static std::mt19937_64 rng(std::random_device{}());
        return p.measureOneCollapsing(v, q, rng);
      },
      "vec"_a, "qubit"_a, R"pb(Measure a qubit and collapse the DD.

Notes:
    Automatically manages the reference count of the input and output DDs.
    The input DD must have a non-zero reference count.

Args:
    vec: The input DD.
    qubit: The qubit to measure.

Returns:
    The measurement outcome.)pb");

  dd.def(
      "measure_all",
      [](dd::Package& p, dd::vEdge& v, const bool collapse = false) {
        static std::mt19937_64 rng(std::random_device{}());
        return p.measureAll(v, collapse, rng);
      },
      "vec"_a, "collapse"_a = false, R"pb(Measure all qubits.

Notes:
    Automatically manages the reference count of the input and output DDs.
    The input DD must have a non-zero reference count.

Args:
    vec: The input DD.
    collapse: Whether to collapse the DD.

Returns:
    The measurement outcome.)pb");

  dd.def_static("identity", &dd::Package::makeIdent,
                R"pb(Create the DD for the identity matrix :math:`I`.

Notes:
    Returns the global one-terminal (identity matrix), which is package-agnostic and safe to use across DDPackage instances.

Returns:
    The DD for the identity matrix.)pb");

  dd.def(
      "single_qubit_gate",
      [](dd::Package& p, const SingleQubitMatrix& mat, const dd::Qubit target) {
        return p.makeGateDD({mat(0, 0), mat(0, 1), mat(1, 0), mat(1, 1)},
                            target);
      },
      "matrix"_a, "target"_a,
      // keep the DD package alive while the returned matrix DD is alive.
      nb::keep_alive<0, 1>(), R"pb(Create the DD for a single-qubit gate.

Args:
    matrix: The :math:`2 \times 2` matrix representing the single-qubit gate.
    target: The target qubit.

Returns:
    The DD for the single-qubit gate.)pb");

  dd.def(
      "controlled_single_qubit_gate",
      [](dd::Package& p, const SingleQubitMatrix& mat,
         const qc::Control& control, const dd::Qubit target) {
        return p.makeGateDD({mat(0, 0), mat(0, 1), mat(1, 0), mat(1, 1)},
                            control, target);
      },
      "matrix"_a, "control"_a, "target"_a,
      // keep the DD package alive while the returned matrix DD is alive.
      nb::keep_alive<0, 1>(),
      nb::sig(
          "def controlled_single_qubit_gate(self, "
          "matrix: Annotated[NDArray[numpy.complex128], {\"shape\": (2, 2)}],"
          "control: mqt.core.ir.operations.Control | int,"
          "target: int) -> mqt.core.dd.MatrixDD"),
      R"pb(Create the DD for a controlled single-qubit gate.

Args:
    matrix: The :math:`2 \times 2` matrix representing the single-qubit gate.
    control: The control qubit.
    target: The target qubit.

Returns:
    The DD for the controlled single-qubit gate.)pb");

  dd.def(
      "multi_controlled_single_qubit_gate",
      [](dd::Package& p, const SingleQubitMatrix& mat,
         const qc::Controls& controls, const dd::Qubit target) {
        return p.makeGateDD({mat(0, 0), mat(0, 1), mat(1, 0), mat(1, 1)},
                            controls, target);
      },
      "matrix"_a, "controls"_a, "target"_a,
      // keep the DD package alive while the returned matrix DD is alive.
      nb::keep_alive<0, 1>(),
      nb::sig(
          "def multi_controlled_single_qubit_gate(self, "
          "matrix: Annotated[NDArray[numpy.complex128], {\"shape\": (2, 2)}],"
          "controls: collections.abc.Set[mqt.core.ir.operations.Control | int],"
          "target: int) -> mqt.core.dd.MatrixDD"),
      R"pb(Create the DD for a multi-controlled single-qubit gate.

Args:
    matrix: The :math:`2 \times 2` matrix representing the single-qubit gate.
    controls: The control qubits.
    target: The target qubit.

Returns:
    The DD for the multi-controlled single-qubit gate.)pb");

  dd.def(
      "two_qubit_gate",
      [](dd::Package& p, const TwoQubitMatrix& mat, const dd::Qubit target0,
         const dd::Qubit target1) {
        return p.makeTwoQubitGateDD(
            {std::array{mat(0, 0), mat(0, 1), mat(0, 2), mat(0, 3)},
             {mat(1, 0), mat(1, 1), mat(1, 2), mat(1, 3)},
             {mat(2, 0), mat(2, 1), mat(2, 2), mat(2, 3)},
             {mat(3, 0), mat(3, 1), mat(3, 2), mat(3, 3)}},
            target0, target1);
      },
      "matrix"_a, "target0"_a, "target1"_a,
      // keep the DD package alive while the returned matrix DD is alive.
      nb::keep_alive<0, 1>(), R"pb(Create the DD for a two-qubit gate.

Args:
    matrix: The :math:`4 \times 4` matrix representing the two-qubit gate.
    target0: The first target qubit.
    target1: The second target qubit.

Returns:
    The DD for the two-qubit gate.)pb");

  dd.def(
      "controlled_two_qubit_gate",
      [](dd::Package& p, const TwoQubitMatrix& mat, const qc::Control& control,
         const dd::Qubit target0, const dd::Qubit target1) {
        return p.makeTwoQubitGateDD(
            {std::array{mat(0, 0), mat(0, 1), mat(0, 2), mat(0, 3)},
             {mat(1, 0), mat(1, 1), mat(1, 2), mat(1, 3)},
             {mat(2, 0), mat(2, 1), mat(2, 2), mat(2, 3)},
             {mat(3, 0), mat(3, 1), mat(3, 2), mat(3, 3)}},
            control, target0, target1);
      },
      "matrix"_a, "control"_a, "target0"_a, "target1"_a,
      // keep the DD package alive while the returned matrix DD is alive.
      nb::keep_alive<0, 1>(),
      nb::sig(
          "def controlled_two_qubit_gate(self, "
          "matrix: Annotated[NDArray[numpy.complex128], {\"shape\": (4, 4)}],"
          "control: mqt.core.ir.operations.Control | int,"
          "target0: int, target1: int) -> mqt.core.dd.MatrixDD"),
      R"pb(Create the DD for a controlled two-qubit gate.

Args:
    matrix: The :math:`4 \times 4` matrix representing the two-qubit gate.
    control: The control qubit.
    target0: The first target qubit.
    target1: The second target qubit.

Returns:
    The DD for the controlled two-qubit gate.)pb");

  dd.def(
      "multi_controlled_two_qubit_gate",
      [](dd::Package& p, const TwoQubitMatrix& mat,
         const qc::Controls& controls, const dd::Qubit target0,
         const dd::Qubit target1) {
        return p.makeTwoQubitGateDD(
            {std::array{mat(0, 0), mat(0, 1), mat(0, 2), mat(0, 3)},
             {mat(1, 0), mat(1, 1), mat(1, 2), mat(1, 3)},
             {mat(2, 0), mat(2, 1), mat(2, 2), mat(2, 3)},
             {mat(3, 0), mat(3, 1), mat(3, 2), mat(3, 3)}},
            controls, target0, target1);
      },
      "matrix"_a, "controls"_a, "target0"_a, "target1"_a,
      // keep the DD package alive while the returned matrix DD is alive.
      nb::keep_alive<0, 1>(),
      nb::sig(
          "def multi_controlled_two_qubit_gate(self, "
          "matrix: Annotated[NDArray[numpy.complex128], {\"shape\": (4, 4)}],"
          "controls: collections.abc.Set[mqt.core.ir.operations.Control | int],"
          "target0: int, target1: int) -> mqt.core.dd.MatrixDD"),
      R"pb(Create the DD for a multi-controlled two-qubit gate.

Args:
    matrix: The :math:`4 \times 4` matrix representing the two-qubit gate.
    controls: The control qubits.
    target0: The first target qubit.
    target1: The second target qubit.

Returns:
    The DD for the multi-controlled two-qubit gate.)pb");

  dd.def(
      "from_matrix",
      [](dd::Package& p, const Matrix& mat) {
        const auto rows = mat.shape(0);
        const auto cols = mat.shape(1);
        if (rows != cols) {
          throw std::invalid_argument("Matrix must be square.");
        }
        if (rows == 0) {
          return dd::mEdge::one();
        }
        if ((rows & (rows - 1)) != 0) {
          throw std::invalid_argument(
              "Matrix must have a size of a power of two.");
        }
        if (rows == 1) {
          return dd::mEdge::terminal(p.cn.lookup(mat(0, 0)));
        }
        const auto level = static_cast<dd::Qubit>(std::log2(rows) - 1);
        const auto matrixDD = makeDDFromMatrix(p, mat, 0, rows, 0, cols, level);
        return dd::mEdge{.p = matrixDD.p, .w = p.cn.lookup(matrixDD.w)};
      },
      "matrix"_a,
      // keep the DD package alive while the returned matrix DD is alive.
      nb::keep_alive<0, 1>(), R"pb(Create a DD from a matrix.

Args:
    matrix: The matrix. Must be square and have a size that is a power of 2.

Returns:
    The DD for the matrix.)pb");

  dd.def(
      "from_operation",
      [](dd::Package& p, const qc::Operation& op, const bool invert = false) {
        if (invert) {
          return getInverseDD(op, p);
        }
        return getDD(op, p);
      },
      "operation"_a, "invert"_a = false,
      // keep the DD package alive while the returned matrix DD is alive.
      nb::keep_alive<0, 1>(), R"pb(Create a DD from an operation.

Args:
    operation: The operation. Must be unitary.
    invert: Whether to get the inverse of the operation.

Returns:
    The DD for the operation.)pb");

  // Reference counting and garbage collection
  dd.def("inc_ref_vec", &dd::Package::incRef<dd::vNode>, "vec"_a,
         "Increment the reference count of a vector.");
  dd.def("inc_ref_mat", &dd::Package::incRef<dd::mNode>, "mat"_a,
         "Increment the reference count of a matrix.");
  dd.def("dec_ref_vec", &dd::Package::decRef<dd::vNode>, "vec"_a,
         "Decrement the reference count of a vector.");
  dd.def("dec_ref_mat", &dd::Package::decRef<dd::mNode>, "mat"_a,
         "Decrement the reference count of a matrix.");
  dd.def("garbage_collect", &dd::Package::garbageCollect, "force"_a = false,
         R"pb(Perform garbage collection on the DDPackage.

Args:
    force: Whether to force garbage collection.
        If set to True, garbage collection is performed regardless of the current memory usage.
        If set to False, garbage collection is only performed if the memory usage exceeds a certain threshold.

Returns:
    Whether any nodes were collected during garbage collection.)pb");

  // Operations on DDs
  dd.def("vector_add",
         static_cast<dd::vEdge (dd::Package::*)(
             const dd::vEdge&, const dd::vEdge&)>(&dd::Package::add),
         "lhs"_a, "rhs"_a,
         // keep the DD package alive while the returned vector DD is alive.
         nb::keep_alive<0, 1>(), R"pb(Add two vectors.

Notes:
    It is the caller's responsibility to update the reference count of the input and output vectors after the operation.

    Both vectors must have the same number of qubits.

Args:
    lhs: The left vector.
    rhs: The right vector.

Returns:
    The sum of the two vectors.)pb");

  dd.def("matrix_add",
         static_cast<dd::mEdge (dd::Package::*)(
             const dd::mEdge&, const dd::mEdge&)>(&dd::Package::add),
         "lhs"_a, "rhs"_a,
         // keep the DD package alive while the returned matrix DD is alive.
         nb::keep_alive<0, 1>(), R"pb(Add two matrices.

Notes:
    It is the caller's responsibility to update the reference count of the input and output matrices after the operation.

    Both matrices must have the same number of qubits.

Args:
    lhs: The left matrix.
    rhs: The right matrix.

Returns:
    The sum of the two matrices.)pb");

  dd.def("conjugate", &dd::Package::conjugate, "vec"_a,
         // keep the DD package alive while the returned vector DD is alive.
         nb::keep_alive<0, 1>(), R"pb(Conjugate a vector.

Notes:
    It is the caller's responsibility to update the reference count of the input and output vectors after the operation.

Args:
    vec: The vector.

Returns:
    The conjugated vector.)pb");

  dd.def("conjugate_transpose", &dd::Package::conjugateTranspose, "mat"_a,
         // keep the DD package alive while the returned matrix DD is alive.
         nb::keep_alive<0, 1>(), R"pb(Conjugate transpose a matrix.

Notes:
    It is the caller's responsibility to update the reference count of the input and output matrices after the operation.

Args:
    mat: The matrix.

Returns:
    The conjugate transposed matrix.)pb");

  dd.def(
      "matrix_vector_multiply",
      [](dd::Package& p, const dd::mEdge& mat, const dd::vEdge& vec) {
        return p.multiply(mat, vec);
      },
      "mat"_a, "vec"_a,
      // keep the DD package alive while the returned vector DD is alive.
      nb::keep_alive<0, 1>(), R"pb(Multiply a matrix with a vector.

Notes:
    It is the caller's responsibility to update the reference count of the input and output matrices after the operation.

    The vector must have at least as many qubits as the matrix non-trivially acts on.

Args:
    mat: The matrix.
    vec: The vector.

Returns:
    The product of the matrix and the vector.)pb");

  dd.def(
      "matrix_multiply",
      [](dd::Package& p, const dd::mEdge& lhs, const dd::mEdge& rhs) {
        return p.multiply(lhs, rhs);
      },
      "lhs"_a, "rhs"_a,
      // keep the DD package alive while the returned matrix DD is alive.
      nb::keep_alive<0, 1>(), R"pb(Multiply two matrices.

Notes:
    It is the caller's responsibility to update the reference count of the input and output matrices after the operation.

Args:
    lhs: The left matrix.
    rhs: The right matrix.

Returns:
    The product of the two matrices.)pb");

  dd.def(
      "inner_product",
      [](dd::Package& p, const dd::vEdge& lhs, const dd::vEdge& rhs) {
        return std::complex<dd::fp>{p.innerProduct(lhs, rhs)};
      },
      "lhs"_a, "rhs"_a, R"pb(Compute the inner product of two vectors.

Notes:
    Both vectors must have the same number of qubits.

Args:
    lhs: The left vector.
    rhs: The right vector.

Returns:
    The inner product of the two vectors.)pb");

  dd.def("fidelity", &dd::Package::fidelity, "lhs"_a, "rhs"_a,
         R"pb(Compute the fidelity of two vectors.

Notes:
    Both vectors must have the same number of qubits.

Args:
    lhs: The left vector.
    rhs: The right vector.

Returns:
    The fidelity of the two vectors.)pb");

  dd.def("expectation_value", &dd::Package::expectationValue, "observable"_a,
         "state"_a, R"pb(Compute the expectation value of an observable.

Notes:
    The state must have at least as many qubits as the observable non-trivially acts on.

    The method computes :math:`\langle \psi | O | \psi \rangle` as :math:`\langle \psi | (O | \psi \rangle)`.

Args:
    observable: The observable.
    state: The state.

Returns:
    The expectation value of the observable.)pb");

  dd.def("vector_kronecker",
         static_cast<dd::vEdge (dd::Package::*)(
             const dd::vEdge&, const dd::vEdge&, size_t, bool)>(
             &dd::Package::kronecker),
         "top"_a, "bottom"_a, "bottom_num_qubits"_a, "increment_index"_a = true,
         // keep the DD package alive while the returned vector DD is alive.
         nb::keep_alive<0, 1>(),
         R"pb(Compute the Kronecker product of two vectors.

Notes:
    It is the caller's responsibility to update the reference count of the input and output vectors after the operation.

Args:
    top: The top vector.
    bottom: The bottom vector.
    bottom_num_qubits: The number of qubits of the bottom vector.
    increment_index: Whether to increment the indexes of the top vector.

Returns:
    The Kronecker product of the two vectors.)pb");

  dd.def("matrix_kronecker",
         static_cast<dd::mEdge (dd::Package::*)(
             const dd::mEdge&, const dd::mEdge&, size_t, bool)>(
             &dd::Package::kronecker),
         "top"_a, "bottom"_a, "bottom_num_qubits"_a, "increment_index"_a = true,
         // keep the DD package alive while the returned matrix DD is alive.
         nb::keep_alive<0, 1>(),
         R"pb(Compute the Kronecker product of two matrices.

Notes:
    It is the caller's responsibility to update the reference count of the input and output matrices after the operation.

Args:
    top: The top matrix.
    bottom: The bottom matrix.
    bottom_num_qubits: The number of qubits of the bottom matrix.
    increment_index: Whether to increment the indexes of the top matrix.

Returns:
    The Kronecker product of the two matrices.)pb");

  dd.def("partial_trace", &dd::Package::partialTrace, "mat"_a, "eliminate"_a,
         // keep the DD package alive while the returned matrix DD is alive.
         nb::keep_alive<0, 1>(), R"pb(Compute the partial trace of a matrix.

Args:
    mat: The matrix.
    eliminate: The qubits to eliminate. Must be at least as long as the number of qubits of the matrix.

Returns:
    The partial trace of the matrix.)pb");

  dd.def(
      "trace",
      [](dd::Package& p, const dd::mEdge& mat, const size_t numQubits) {
        return std::complex<dd::fp>{p.trace(mat, numQubits)};
      },
      "mat"_a, "num_qubits"_a, R"pb(Compute the trace of a matrix.

Args:
    mat: The matrix.
    num_qubits: The number of qubits of the matrix.

Returns:
    The trace of the matrix.)pb");
}

} // namespace mqt
