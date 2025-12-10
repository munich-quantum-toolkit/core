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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>     // NOLINT(misc-include-cleaner)
#include <nanobind/stl/unique_ptr.h> // NOLINT(misc-include-cleaner)
#include <nanobind/stl/vector.h>     // NOLINT(misc-include-cleaner)
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace mqt {

namespace nb = nanobind;
using namespace nb::literals;

using DiffType = std::vector<std::unique_ptr<qc::Operation>>::difference_type;
using SizeType = std::vector<std::unique_ptr<qc::Operation>>::size_type;

// NOLINTNEXTLINE(misc-use-internal-linkage)
void registerCompoundOperation(const nb::module_& m) {
  auto wrap = [](DiffType i, const SizeType size) {
    if (i < 0) {
      i += static_cast<DiffType>(size);
    }
    if (i < 0 || std::cmp_greater_equal(i, size)) {
      throw nb::index_error();
    }
    return i;
  };

  nb::class_<qc::CompoundOperation, qc::Operation>(m, "CompoundOperation")
      .def(nb::init<>())
      .def(
          "__init__",
          [](qc::CompoundOperation* self,
             const std::vector<qc::Operation*>& ops) {
            std::vector<std::unique_ptr<qc::Operation>> uniqueOps;
            uniqueOps.reserve(ops.size());
            for (const auto& op : ops) {
              uniqueOps.emplace_back(op->clone());
            }
            new (self) qc::CompoundOperation(std::move(uniqueOps));
          },
          "ops"_a)
      .def("__len__", &qc::CompoundOperation::size)
      .def(
          "__getitem__",
          [&wrap](const qc::CompoundOperation& op, DiffType i) {
            i = wrap(i, op.size());
            return op.at(static_cast<SizeType>(i)).get();
          },
          nb::rv_policy::reference_internal, "index"_a)
      .def(
          "__getitem__",
          [](const qc::CompoundOperation& op, const nb::slice& slice) {
            auto [start, stop, step, sliceLength] = slice.compute(op.size());
            auto ops = std::vector<qc::Operation*>();
            ops.reserve(sliceLength);
            for (auto i = start; i < stop; i += step) {
              ops.emplace_back(op.at(static_cast<SizeType>(i)).get());
            }
            return ops;
          },
          nb::rv_policy::reference_internal, "index"_a)
      .def(
          "__setitem__",
          [&wrap](qc::CompoundOperation& compOp, DiffType i,
                  const qc::Operation& op) {
            i = wrap(i, compOp.size());
            compOp[static_cast<SizeType>(i)] = op.clone();
          },
          "index"_a, "value"_a)
      .def(
          "__setitem__",
          [](qc::CompoundOperation& compOp, const nb::slice& slice,
             const std::vector<qc::Operation*>& ops) {
            auto [start, stop, step, sliceLength] =
                slice.compute(compOp.size());
            if (sliceLength != ops.size()) {
              throw std::runtime_error(
                  "Length of slice and number of operations do not match.");
            }
            for (std::size_t i = 0; i < sliceLength; ++i) {
              compOp[static_cast<SizeType>(start)] = ops[i]->clone();
              start += step;
            }
          },
          "index"_a, "value"_a)
      .def(
          "__delitem__",
          [&wrap](qc::CompoundOperation& op, DiffType i) {
            i = wrap(i, op.size());
            op.erase(op.begin() + i);
          },
          "index"_a)
      .def(
          "__delitem__",
          [](qc::CompoundOperation& op, const nb::slice& slice) {
            auto [start, stop, step, sliceLength] = slice.compute(op.size());
            // delete in reverse order to not invalidate indices
            for (std::size_t i = sliceLength; i > 0; --i) {
              const auto offset = static_cast<DiffType>(
                  static_cast<SizeType>(start) +
                  ((i - 1) * static_cast<SizeType>(step)));
              op.erase(op.begin() + offset);
            }
          },
          "index"_a)
      .def(
          "append",
          [](qc::CompoundOperation& compOp, const qc::Operation& op) {
            compOp.emplace_back(op.clone());
          },
          "value"_a)
      .def(
          "insert",
          [](qc::CompoundOperation& compOp, const std::size_t idx,
             const qc::Operation& op) {
            compOp.insert(compOp.begin() + static_cast<int64_t>(idx),
                          op.clone());
          },
          "index"_a, "value"_a)
      .def("empty", &qc::CompoundOperation::empty)
      .def("clear", &qc::CompoundOperation::clear)
      .def("__repr__", [](const qc::CompoundOperation& op) {
        std::stringstream ss;
        ss << "CompoundOperation([..." << op.size() << " ops...])";
        return ss.str();
      });
}
} // namespace mqt
