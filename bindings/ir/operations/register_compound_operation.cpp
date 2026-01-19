/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "ir/operations/CompoundOperation.hpp"
#include "ir/operations/Operation.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <functional>
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

  nb::class_<qc::CompoundOperation, qc::Operation>(
      m, "CompoundOperation",
      nb::sig(
          "class CompoundOperation(mqt.core.ir.operations.Operation, "
          "collections.abc.MutableSequence[mqt.core.ir.operations.Operation])"),
      R"pb(Compound quantum operation.

This class is used to aggregate and group multiple operations into a single object.
This is useful for optimizations and for representing complex quantum functionality.
A :class:`CompoundOperation` can contain any number of operations, including other :class:`CompoundOperation`'s.

Args:
    ops: The operations that are part of the compound operation.)pb")

      .def(nb::init<>())
      .def(
          "__init__",
          [](qc::CompoundOperation* self,
             const std::vector<qc::Operation*>& ops) {
            std::vector<std::unique_ptr<qc::Operation>> uniqueOps;
            uniqueOps.reserve(ops.size());
            for (const auto& op : ops) {
              assert(op != nullptr && "ops must not contain nullptr");
              uniqueOps.emplace_back(op->clone());
            }
            new (self) qc::CompoundOperation(std::move(uniqueOps));
          },
          "ops"_a)

      .def("__len__", &qc::CompoundOperation::size,
           "The number of operations in the compound operation.")

      .def(
          "__getitem__",
          [wrap](const qc::CompoundOperation& op, DiffType i) {
            i = wrap(i, op.size());
            return op.at(static_cast<SizeType>(i)).get();
          },
          nb::rv_policy::reference_internal, "index"_a,
          R"pb(Get the operation at the given index.

Note:
    This gives direct access to the operations in the compound operation

Args:
    index: The index of the operation to get.

Returns:
    The operation at the given index.)pb")

      .def(
          "__getitem__",
          [](const qc::CompoundOperation& op, const nb::slice& slice) {
            auto [start, stop, step, sliceLength] = slice.compute(op.size());
            auto ops = std::vector<qc::Operation*>();
            ops.reserve(sliceLength);
            for (std::size_t i = 0; i < sliceLength; ++i) {
              auto idx = static_cast<DiffType>(start) +
                         (static_cast<DiffType>(i) * step);
              ops.emplace_back(op.at(static_cast<SizeType>(idx)).get());
            }
            return ops;
          },
          nb::rv_policy::reference_internal, "index"_a,
          R"pb(Get the operations in the given slice.

Note:
    This gives direct access to the operations in the compound operation.

Args:
    index: The slice of the operations to get.

Returns:
    The operations in the given slice.)pb")

      .def(
          "__setitem__",
          [wrap](qc::CompoundOperation& compOp, DiffType i,
                 const qc::Operation& op) {
            i = wrap(i, compOp.size());
            compOp[static_cast<SizeType>(i)] = op.clone();
          },
          "index"_a, "value"_a, R"pb(Set the operation at the given index.

Args:
    index: The index of the operation to set.
    value: The operation to set at the given index.)pb")

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
              assert(ops[i] != nullptr && "ops must not contain nullptr");
              compOp[static_cast<SizeType>(start)] = ops[i]->clone();
              start += step;
            }
          },
          nb::sig("def __setitem__(self, index: slice, value: "
                  "collections.abc.Iterable[mqt.core.ir.operations.Operation]) "
                  "-> None"),
          R"pb(Set the operations in the given slice.

Args:
    index: The slice of operations to set.
    value: The operations to set in the given slice.)pb")

      .def(
          "__delitem__",
          [wrap](qc::CompoundOperation& op, DiffType i) {
            i = wrap(i, op.size());
            op.erase(op.begin() + i);
          },
          "index"_a, R"pb(Delete the operation at the given index.

Args:
    index: The index of the operation to delete.)pb")

      .def(
          "__delitem__",
          [](qc::CompoundOperation& op, const nb::slice& slice) {
            auto [start, stop, step, sliceLength] = slice.compute(op.size());
            // Delete in reverse order to not invalidate indices
            std::vector<DiffType> indices;
            indices.reserve(sliceLength);
            for (std::size_t i = 0; i < sliceLength; ++i) {
              indices.emplace_back(static_cast<DiffType>(start) +
                                   (static_cast<DiffType>(i) * step));
            }
            std::ranges::sort(indices, std::greater<>());
            for (const auto idx : indices) {
              op.erase(op.begin() + idx);
            }
          },
          "index"_a, R"pb(Delete the operations in the given slice.

Args:
    index: The slice of operations to delete.)pb")

      .def(
          "append",
          [](qc::CompoundOperation& compOp, const qc::Operation& op) {
            compOp.emplace_back(op.clone());
          },
          "value"_a, "Append an operation to the compound operation.")

      .def(
          "insert",
          [](qc::CompoundOperation& compOp, const std::size_t idx,
             const qc::Operation& op) {
            compOp.insert(compOp.begin() + static_cast<int64_t>(idx),
                          op.clone());
          },
          "index"_a, "value"_a, R"pb(Insert an operation at the given index.

Args:
    index: The index to insert the operation at.
    value: The operation to insert.)pb")

      .def("empty", &qc::CompoundOperation::empty,
           "Check if the compound operation is empty.")

      .def("clear", &qc::CompoundOperation::clear,
           "Clear all operations in the compound operation.")

      .def("__repr__", [](const qc::CompoundOperation& op) {
        std::stringstream ss;
        ss << "CompoundOperation([..." << op.size() << " ops...])";
        return ss.str();
      });
}
} // namespace mqt
