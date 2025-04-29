/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
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
#include "ir/Permutation.hpp"
#include "ir/operations/ClassicControlledOperation.hpp"
#include "ir/operations/Control.hpp"
#include "ir/operations/NonUnitaryOperation.hpp"
#include "ir/operations/Operation.hpp"
#include "python/pybind11.hpp"

#include <cmath>
#include <complex>
#include <cstddef>
#include <memory>
#include <pybind11/numpy.h>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

namespace mqt {

namespace py = pybind11;
using namespace pybind11::literals;

namespace {
/// Recursive helper function to create a vector DD from a numpy array
dd::vCachedEdge makeDDFromVector(
    dd::Package& p,
    const py::detail::unchecked_reference<std::complex<dd::fp>, 1>& v,
    const size_t startIdx, const size_t endIdx, const dd::Qubit level) {
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
dd::mCachedEdge makeDDFromMatrix(
    dd::Package& p,
    const py::detail::unchecked_reference<std::complex<dd::fp>, 2>& m,
    const size_t rowStart, const size_t rowEnd, const size_t colStart,
    const size_t colEnd, const dd::Qubit level) {
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

void registerDDPackage(const py::module& mod) {
  auto dd =
      py::class_<dd::Package, std::unique_ptr<dd::Package>>(mod, "DDPackage");

  // Constructor
  dd.def(py::init<size_t>(), "num_qubits"_a = dd::Package::DEFAULT_QUBITS);

  // Resizing the package
  dd.def("resize", &dd::Package::resize, "num_qubits"_a);

  // Getting the number of qubits the package is configured for
  dd.def_property_readonly("max_qubits", &dd::Package::qubits);

  ///------------------------------------------------------------------------///
  /// Vector DD Generation
  ///------------------------------------------------------------------------///

  dd.def(
      "zero_state",
      [](dd::Package& p, const size_t numQubits) {
        return p.makeZeroState(numQubits);
      },
      "num_qubits"_a,
      // keep the DD package alive while the returned vector DD is alive.
      py::keep_alive<0, 1>());

  dd.def(
      "computational_basis_state",
      [](dd::Package& p, const size_t numQubits,
         const std::vector<bool>& state) {
        return p.makeBasisState(numQubits, state);
      },
      "num_qubits"_a, "state"_a,
      // keep the DD package alive while the returned vector DD is alive.
      py::keep_alive<0, 1>());

  py::enum_<dd::BasisStates>(mod, "BasisStates")
      .value("zero", dd::BasisStates::zero)
      .value("one", dd::BasisStates::one)
      .value("plus", dd::BasisStates::plus)
      .value("minus", dd::BasisStates::minus)
      .value("right", dd::BasisStates::right)
      .value("left", dd::BasisStates::left);

  dd.def(
      "basis_state",
      [](dd::Package& p, const size_t numQubits,
         const std::vector<dd::BasisStates>& state) {
        return p.makeBasisState(numQubits, state);
      },
      "num_qubits"_a, "state"_a,
      // keep the DD package alive while the returned vector DD is alive.
      py::keep_alive<0, 1>());

  dd.def("ghz_state", &dd::Package::makeGHZState, "num_qubits"_a,
         // keep the DD package alive while the returned vector DD is alive.
         py::keep_alive<0, 1>());

  dd.def("w_state", &dd::Package::makeWState, "num_qubits"_a,
         // keep the DD package alive while the returned vector DD is alive.
         py::keep_alive<0, 1>());

  dd.def(
      "from_vector",
      [](dd::Package& p, const py::array_t<std::complex<dd::fp>>& v) {
        const auto data = v.unchecked<1>();
        const auto length = static_cast<size_t>(data.shape(0));
        if (length == 0) {
          return dd::vEdge::one();
        }
        if ((length & (length - 1)) != 0) {
          throw std::invalid_argument(
              "State vector must have a length of a power of two.");
        }
        if (length == 1) {
          return dd::vEdge::terminal(p.cn.lookup(data(0)));
        }

        const auto level = static_cast<dd::Qubit>(std::log2(length) - 1);
        const auto state = makeDDFromVector(p, data, 0, length, level);
        const dd::vEdge e{state.p, p.cn.lookup(state.w)};
        p.incRef(e);
        return e;
      },
      "state"_a,
      // keep the DD package alive while the returned vector DD is alive.
      py::keep_alive<0, 1>());

  dd.def(
      "apply_unitary_operation",
      [](dd::Package& p, const dd::vEdge& v, const qc::Operation& op,
         const qc::Permutation& perm = {}) {
        return dd::applyUnitaryOperation(op, v, p, perm);
      },
      "vec"_a, "operation"_a, "permutation"_a = qc::Permutation{},
      // keep the DD package alive while the returned vector DD is alive.
      py::keep_alive<0, 1>());

  dd.def(
      "apply_measurement",
      [](dd::Package& p, const dd::vEdge& v, const qc::NonUnitaryOperation& op,
         const std::vector<bool>& measurements,
         const qc::Permutation& perm = {}) {
        static std::mt19937_64 rng(std::random_device{}());
        auto measurementsCopy = measurements;
        return std::pair{
            dd::applyMeasurement(op, v, p, rng, measurementsCopy, perm),
            measurementsCopy};
      },
      "vec"_a, "operation"_a, "measurements"_a,
      "permutation"_a = qc::Permutation{},
      // keep the DD package alive while the returned vector DD is alive.
      py::keep_alive<0, 1>());

  dd.def(
      "apply_reset",
      [](dd::Package& p, const dd::vEdge& v, const qc::NonUnitaryOperation& op,
         const qc::Permutation& perm = {}) {
        static std::mt19937_64 rng(std::random_device{}());
        return dd::applyReset(op, v, p, rng, perm);
      },
      "vec"_a, "operation"_a, "permutation"_a = qc::Permutation{},
      // keep the DD package alive while the returned vector DD is alive.
      py::keep_alive<0, 1>());

  dd.def(
      "apply_classic_controlled_operation",
      [](dd::Package& p, const dd::vEdge& v,
         const qc::ClassicControlledOperation& op,
         const std::vector<bool>& measurements,
         const qc::Permutation& perm = {}) {
        return dd::applyClassicControlledOperation(op, v, p, measurements,
                                                   perm);
      },
      "vec"_a, "operation"_a, "measurements"_a,
      "permutation"_a = qc::Permutation{},
      // keep the DD package alive while the returned vector DD is alive.
      py::keep_alive<0, 1>());

  dd.def(
      "measure_collapsing",
      [](dd::Package& p, dd::vEdge& v, const dd::Qubit q) {
        static std::mt19937_64 rng(std::random_device{}());
        return p.measureOneCollapsing(v, q, rng);
      },
      "vec"_a, "qubit"_a);

  dd.def(
      "measure_all",
      [](dd::Package& p, dd::vEdge& v, const bool collapse = false) {
        static std::mt19937_64 rng(std::random_device{}());
        return p.measureAll(v, collapse, rng);
      },
      "vec"_a, "collapse"_a = false);

  dd.def_static("identity", &dd::Package::makeIdent);

  using NumPyMatrix = py::array_t<std::complex<dd::fp>,
                                  py::array::c_style | py::array::forcecast>;
  dd.def(
      "single_qubit_gate",
      [](dd::Package& p, const NumPyMatrix& m, const dd::Qubit target) {
        if (m.ndim() != 2 || m.shape(0) != 2 || m.shape(1) != 2) {
          throw std::invalid_argument("Matrix must be 2x2.");
        }
        const auto data = m.unchecked<2>();
        return p.makeGateDD({data(0, 0), data(0, 1), data(1, 0), data(1, 1)},
                            target);
      },
      "matrix"_a, "target"_a,
      // keep the DD package alive while the returned matrix DD is alive.
      py::keep_alive<0, 1>());

  dd.def(
      "controlled_single_qubit_gate",
      [](dd::Package& p, const NumPyMatrix& m, const qc::Control& control,
         const dd::Qubit target) {
        if (m.ndim() != 2 || m.shape(0) != 2 || m.shape(1) != 2) {
          throw std::invalid_argument("Matrix must be 2x2.");
        }
        const auto data = m.unchecked<2>();
        return p.makeGateDD({data(0, 0), data(0, 1), data(1, 0), data(1, 1)},
                            control, target);
      },
      "matrix"_a, "control"_a, "target"_a,
      // keep the DD package alive while the returned matrix DD is alive.
      py::keep_alive<0, 1>());

  dd.def(
      "multi_controlled_single_qubit_gate",
      [](dd::Package& p, const NumPyMatrix& m, const qc::Controls& controls,
         const dd::Qubit target) {
        if (m.ndim() != 2 || m.shape(0) != 2 || m.shape(1) != 2) {
          throw std::invalid_argument("Matrix must be 2x2.");
        }
        const auto data = m.unchecked<2>();
        return p.makeGateDD({data(0, 0), data(0, 1), data(1, 0), data(1, 1)},
                            controls, target);
      },
      "matrix"_a, "controls"_a, "target"_a,
      // keep the DD package alive while the returned matrix DD is alive.
      py::keep_alive<0, 1>());

  dd.def(
      "two_qubit_gate",
      [](dd::Package& p, const NumPyMatrix& m, const dd::Qubit target0,
         const dd::Qubit target1) {
        if (m.ndim() != 2 || m.shape(0) != 4 || m.shape(1) != 4) {
          throw std::invalid_argument("Matrix must be 4x4.");
        }
        const auto data = m.unchecked<2>();
        return p.makeTwoQubitGateDD(
            {std::array{data(0, 0), data(0, 1), data(0, 2), data(0, 3)},
             {data(1, 0), data(1, 1), data(1, 2), data(1, 3)},
             {data(2, 0), data(2, 1), data(2, 2), data(2, 3)},
             {data(3, 0), data(3, 1), data(3, 2), data(3, 3)}},
            target0, target1);
      },
      "matrix"_a, "target0"_a, "target1"_a,
      // keep the DD package alive while the returned matrix DD is alive.
      py::keep_alive<0, 1>());

  dd.def(
      "controlled_two_qubit_gate",
      [](dd::Package& p, const NumPyMatrix& m, const qc::Control& control,
         const dd::Qubit target0, const dd::Qubit target1) {
        if (m.ndim() != 2 || m.shape(0) != 4 || m.shape(1) != 4) {
          throw std::invalid_argument("Matrix must be 4x4.");
        }
        const auto data = m.unchecked<2>();
        return p.makeTwoQubitGateDD(
            {std::array{data(0, 0), data(0, 1), data(0, 2), data(0, 3)},
             {data(1, 0), data(1, 1), data(1, 2), data(1, 3)},
             {data(2, 0), data(2, 1), data(2, 2), data(2, 3)},
             {data(3, 0), data(3, 1), data(3, 2), data(3, 3)}},
            control, target0, target1);
      },
      "matrix"_a, "control"_a, "target0"_a, "target1"_a,
      // keep the DD package alive while the returned matrix DD is alive.
      py::keep_alive<0, 1>());

  dd.def(
      "multi_controlled_two_qubit_gate",
      [](dd::Package& p, const NumPyMatrix& m, const qc::Controls& controls,
         const dd::Qubit target0, const dd::Qubit target1) {
        if (m.ndim() != 2 || m.shape(0) != 4 || m.shape(1) != 4) {
          throw std::invalid_argument("Matrix must be 4x4.");
        }
        const auto data = m.unchecked<2>();
        return p.makeTwoQubitGateDD(
            {std::array{data(0, 0), data(0, 1), data(0, 2), data(0, 3)},
             {data(1, 0), data(1, 1), data(1, 2), data(1, 3)},
             {data(2, 0), data(2, 1), data(2, 2), data(2, 3)},
             {data(3, 0), data(3, 1), data(3, 2), data(3, 3)}},
            controls, target0, target1);
      },
      "matrix"_a, "controls"_a, "target0"_a, "target1"_a,
      // keep the DD package alive while the returned matrix DD is alive.
      py::keep_alive<0, 1>());

  dd.def(
      "from_matrix",
      [](dd::Package& p, const NumPyMatrix& m) {
        const auto data = m.unchecked<2>();
        const auto rows = static_cast<size_t>(data.shape(0));
        const auto cols = static_cast<size_t>(data.shape(1));
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
          return dd::mEdge::terminal(p.cn.lookup(data(0, 0)));
        }
        const auto level = static_cast<dd::Qubit>(std::log2(rows) - 1);
        const auto matrixDD =
            makeDDFromMatrix(p, data, 0, rows, 0, cols, level);
        return dd::mEdge{matrixDD.p, p.cn.lookup(matrixDD.w)};
      },
      "matrix"_a,
      // keep the DD package alive while the returned matrix DD is alive.
      py::keep_alive<0, 1>());

  dd.def(
      "from_operation",
      [](dd::Package& p, const qc::Operation& op, const bool invert = false) {
        if (invert) {
          return dd::getInverseDD(op, p);
        }
        return dd::getDD(op, p);
      },
      "operation"_a, "invert"_a = false,
      // keep the DD package alive while the returned matrix DD is alive.
      py::keep_alive<0, 1>());

  // Reference counting and garbage collection
  dd.def("inc_ref_vec", &dd::Package::incRef<dd::vNode>, "vec"_a);
  dd.def("inc_ref_mat", &dd::Package::incRef<dd::mNode>, "mat"_a);
  dd.def("dec_ref_vec", &dd::Package::decRef<dd::vNode>, "vec"_a);
  dd.def("dec_ref_mat", &dd::Package::decRef<dd::mNode>, "mat"_a);
  dd.def("garbage_collect", &dd::Package::garbageCollect, "force"_a = false);

  // Operations on DDs
  dd.def("vector_add",
         static_cast<dd::vEdge (dd::Package::*)(
             const dd::vEdge&, const dd::vEdge&)>(&dd::Package::add),
         "lhs"_a, "rhs"_a,
         // keep the DD package alive while the returned vector DD is alive.
         py::keep_alive<0, 1>());

  dd.def("matrix_add",
         static_cast<dd::mEdge (dd::Package::*)(
             const dd::mEdge&, const dd::mEdge&)>(&dd::Package::add),
         "lhs"_a, "rhs"_a,
         // keep the DD package alive while the returned matrix DD is alive.
         py::keep_alive<0, 1>());

  dd.def("conjugate", &dd::Package::conjugate, "vec"_a,
         // keep the DD package alive while the returned vector DD is alive.
         py::keep_alive<0, 1>());

  dd.def("conjugate_transpose", &dd::Package::conjugateTranspose, "mat"_a,
         // keep the DD package alive while the returned matrix DD is alive.
         py::keep_alive<0, 1>());

  dd.def(
      "matrix_vector_multiply",
      [](dd::Package& p, const dd::mEdge& mat, const dd::vEdge& vec) {
        return p.multiply(mat, vec);
      },
      "mat"_a, "vec"_a,
      // keep the DD package alive while the returned vector DD is alive.
      py::keep_alive<0, 1>());

  dd.def(
      "matrix_multiply",
      [](dd::Package& p, const dd::mEdge& lhs, const dd::mEdge& rhs) {
        return p.multiply(lhs, rhs);
      },
      "lhs"_a, "rhs"_a,
      // keep the DD package alive while the returned matrix DD is alive.
      py::keep_alive<0, 1>());

  dd.def(
      "inner_product",
      [](dd::Package& p, const dd::vEdge& lhs, const dd::vEdge& rhs) {
        return std::complex<dd::fp>{p.innerProduct(lhs, rhs)};
      },
      "lhs"_a, "rhs"_a);

  dd.def("fidelity", &dd::Package::fidelity, "lhs"_a, "rhs"_a);

  dd.def("expectation_value", &dd::Package::expectationValue, "observable"_a,
         "state"_a);

  dd.def("vector_kronecker",
         static_cast<dd::vEdge (dd::Package::*)(
             const dd::vEdge&, const dd::vEdge&, size_t, bool)>(
             &dd::Package::kronecker),
         "top"_a, "bottom"_a, "bottom_num_qubits"_a, "increment_index"_a = true,
         // keep the DD package alive while the returned vector DD is alive.
         py::keep_alive<0, 1>());

  dd.def("matrix_kronecker",
         static_cast<dd::mEdge (dd::Package::*)(
             const dd::mEdge&, const dd::mEdge&, size_t, bool)>(
             &dd::Package::kronecker),
         "top"_a, "bottom"_a, "bottom_num_qubits"_a, "increment_index"_a = true,
         // keep the DD package alive while the returned matrix DD is alive.
         py::keep_alive<0, 1>());

  dd.def("partial_trace", &dd::Package::partialTrace, "mat"_a, "eliminate"_a,
         // keep the DD package alive while the returned matrix DD is alive.
         py::keep_alive<0, 1>());

  dd.def(
      "trace",
      [](dd::Package& p, const dd::mEdge& mat, const size_t numQubits) {
        return std::complex<dd::fp>{p.trace(mat, numQubits)};
      },
      "mat"_a, "num_qubits"_a);
}
} // namespace mqt
