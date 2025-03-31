/*
 * Copyright (c) 2025 Chair for Design Automation, TUM
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
#include "python/pybind11.hpp"

#include <sstream>
#include <vector>

namespace mqt {

void registerNonUnitaryOperation(const py::module& m) {
  py::class_<qc::NonUnitaryOperation, qc::Operation>(m, "NonUnitaryOperation")
      .def(py::init<std::vector<qc::Qubit>, std::vector<qc::Bit>>(),
           "targets"_a, "classics"_a)
      .def(py::init<qc::Qubit, qc::Bit>(), "target"_a, "classic"_a)
      .def(py::init<std::vector<qc::Qubit>, qc::OpType>(), "targets"_a,
           "op_type"_a = qc::OpType::Reset)
      .def_property_readonly(
          "classics", py::overload_cast<>(&qc::NonUnitaryOperation::getClassics,
                                          py::const_))
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
