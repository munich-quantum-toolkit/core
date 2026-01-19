/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "ir/Definitions.hpp"
#include "ir/operations/NonUnitaryOperation.hpp"
#include "ir/operations/OpType.hpp"
#include "ir/operations/Operation.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h> // NOLINT(misc-include-cleaner)
#include <nanobind/stl/vector.h> // NOLINT(misc-include-cleaner)
#include <sstream>
#include <vector>

namespace mqt {

namespace nb = nanobind;
using namespace nb::literals;

// NOLINTNEXTLINE(misc-use-internal-linkage)
void registerNonUnitaryOperation(const nb::module_& m) {
  nb::class_<qc::NonUnitaryOperation, qc::Operation>(m, "NonUnitaryOperation",
                                                     R"pb(Non-unitary operation.

This class is used to represent all non-unitary operations, i.e., operations that are not reversible.
This includes measurements and resets.

Args:
    targets: The target qubit(s) of the operation.
    classics: The classical bit(s) that are associated with the operation (only relevant for measurements).
    op_type: The type of the operation.)pb")

      .def(nb::init<std::vector<qc::Qubit>, std::vector<qc::Bit>>(),
           "targets"_a, "classics"_a)
      .def(nb::init<qc::Qubit, qc::Bit>(), "target"_a, "classic"_a)
      .def(nb::init<std::vector<qc::Qubit>, qc::OpType>(), "targets"_a,
           "op_type"_a = qc::OpType::Reset)

      .def_prop_ro("classics",
                   nb::overload_cast<>(&qc::NonUnitaryOperation::getClassics,
                                       nb::const_),
                   "The classical bits that are associated with the operation.")

      .def("__repr__", [](const qc::NonUnitaryOperation& op) {
        std::stringstream ss;
        ss << "NonUnitaryOperation(";
        const auto& targets = op.getTargets();
        if (targets.size() == 1U) {
          ss << "target=" << targets[0];
        } else {
          ss << "targets=[";
          for (const auto& target : targets) {
            ss << target << ", ";
          }
          ss << "]";
        }
        const auto& classics = op.getClassics();
        if (!classics.empty()) {
          ss << ", ";
          if (classics.size() == 1U) {
            ss << "classic=" << classics[0];
          } else {
            ss << "classics=[";
            for (const auto& classic : classics) {
              ss << classic << ", ";
            }
            ss << "]";
          }
        }
        ss << ")";
        return ss.str();
      });
}

} // namespace mqt
