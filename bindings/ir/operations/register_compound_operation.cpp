/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "ir/operations/CompoundOperation.hpp"
#include "ir/operations/Operation.hpp"

// These includes must be the first includes for any bindings code
// clang-format off
#include <pybind11/pybind11.h>
<<<<<<< HEAD
#include <pybind11/stl.h> // NOLINT(misc-include-cleaner)

#include <pybind11/cast.h>
#include <pybind11/pytypes.h>
=======
#include <pybind11/stl.h>
>>>>>>> 59e09070 (Remove pybind11.hpp)
// clang-format on

#include <cstddef>
#include <cstdint>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace mqt {

namespace py = pybind11;
<<<<<<< HEAD
using namespace pybind11::literals;
=======
using namespace py::literals;
>>>>>>> 59e09070 (Remove pybind11.hpp)

using DiffType = std::vector<std::unique_ptr<qc::Operation>>::difference_type;
using SizeType = std::vector<std::unique_ptr<qc::Operation>>::size_type;

// NOLINTNEXTLINE(misc-use-internal-linkage)
void registerCompoundOperation(const py::module& m) {
  auto wrap = [](DiffType i, const SizeType size) {
    if (i < 0) {
      i += static_cast<DiffType>(size);
    }
    if (i < 0 || static_cast<SizeType>(i) >= size) {
      throw py::index_error();
    }
    return i;
  };

  py::class_<qc::CompoundOperation, qc::Operation>(m, "CompoundOperation")
      .def(py::init<>())
      .def(py::init([](const std::vector<qc::Operation*>& ops) {
             std::vector<std::unique_ptr<qc::Operation>> uniqueOps;
             uniqueOps.reserve(ops.size());
             for (const auto& op : ops) {
               uniqueOps.emplace_back(op->clone());
             }
             return qc::CompoundOperation(std::move(uniqueOps));
           }),
           "ops"_a)
      .def("__len__", &qc::CompoundOperation::size)
      .def(
          "__getitem__",
          [&wrap](const qc::CompoundOperation& op, DiffType i) {
            i = wrap(i, op.size());
            return op.at(static_cast<SizeType>(i)).get();
          },
          py::return_value_policy::reference_internal, "i"_a)
      .def(
          "__getitem__",
          [](const qc::CompoundOperation& op, const py::slice& slice) {
            std::size_t start{};
            std::size_t stop{};
            std::size_t step{};
            std::size_t sliceLength{};
            if (!slice.compute(op.size(), &start, &stop, &step, &sliceLength)) {
              throw py::error_already_set();
            }
            auto ops = std::vector<qc::Operation*>();
            ops.reserve(sliceLength);
            for (std::size_t i = start; i < stop; i += step) {
              ops.emplace_back(op.at(i).get());
            }
            return ops;
          },
          py::return_value_policy::reference_internal, "slice"_a)
      .def(
          "__setitem__",
          [&wrap](qc::CompoundOperation& compOp, DiffType i,
                  const qc::Operation& op) {
            i = wrap(i, compOp.size());
            compOp[static_cast<SizeType>(i)] = op.clone();
          },
          "idx"_a, "op"_a)
      .def(
          "__setitem__",
          [](qc::CompoundOperation& compOp, const py::slice& slice,
             const std::vector<qc::Operation*>& ops) {
            std::size_t start{};
            std::size_t stop{};
            std::size_t step{};
            std::size_t sliceLength{};
            if (!slice.compute(compOp.size(), &start, &stop, &step,
                               &sliceLength)) {
              throw py::error_already_set();
            }
            if (sliceLength != ops.size()) {
              throw std::runtime_error(
                  "Length of slice and number of operations do not match.");
            }
            for (std::size_t i = 0; i < sliceLength; ++i) {
              compOp[start] = ops[i]->clone();
              start += step;
            }
          },
          "slice"_a, "ops"_a)
      .def(
          "__delitem__",
          [&wrap](qc::CompoundOperation& compOp, DiffType i) {
            i = wrap(i, compOp.size());
            compOp.erase(compOp.begin() + i);
          },
          "idx"_a)
      .def(
          "__delitem__",
          [](qc::CompoundOperation& compOp, const py::slice& slice) {
            std::size_t start{};
            std::size_t stop{};
            std::size_t step{};
            std::size_t sliceLength{};
            if (!slice.compute(compOp.size(), &start, &stop, &step,
                               &sliceLength)) {
              throw py::error_already_set();
            }
            // delete in reverse order to not invalidate indices
            for (std::size_t i = sliceLength; i > 0; --i) {
              compOp.erase(compOp.begin() +
                           static_cast<int64_t>(start + ((i - 1) * step)));
            }
          },
          "slice"_a)
      .def(
          "append",
          [](qc::CompoundOperation& compOp, const qc::Operation& op) {
            compOp.emplace_back(op.clone());
          },
          "op"_a)
      .def(
          "insert",
          [](qc::CompoundOperation& compOp, const std::size_t idx,
             const qc::Operation& op) {
            compOp.insert(compOp.begin() + static_cast<int64_t>(idx),
                          op.clone());
          },
          "idx"_a, "op"_a)
      .def("empty", &qc::CompoundOperation::empty)
      .def("clear", &qc::CompoundOperation::clear)
      .def("__repr__", [](const qc::CompoundOperation& op) {
        std::stringstream ss;
        ss << "CompoundOperation([..." << op.size() << " ops...])";
        return ss.str();
      });
}
} // namespace mqt
