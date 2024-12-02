/*
 * Copyright (c) 2024 Chair for Design Automation, TUM
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "Definitions.hpp"
#include "ir/operations/Control.hpp"
#include "ir/operations/Expression.hpp"
#include "ir/operations/OpType.hpp"
#include "ir/operations/StandardOperation.hpp"
#include "ir/operations/SymbolicOperation.hpp"
#include "python/pybind11.hpp"

#include <vector>

namespace mqt {

void registerSymbolicOperation(py::module& m) {
  py::class_<qc::SymbolicOperation, qc::StandardOperation>(
      m, "SymbolicOperation",
      "Class representing a symbolic operation."
      "This encompasses all symbolic versions of `StandardOperation` that "
      "involve (float) angle parameters.")
      .def(py::init<>(), "Create an empty symbolic operation.")
      .def(py::init<qc::Qubit, qc::OpType,
                    const std::vector<qc::SymbolOrNumber>&>(),
           "target"_a, "op_type"_a,
           "params"_a = std::vector<qc::SymbolOrNumber>{})
      .def(py::init<const qc::Targets&, qc::OpType,
                    const std::vector<qc::SymbolOrNumber>&>(),
           "targets"_a, "op_type"_a,
           "params"_a = std::vector<qc::SymbolOrNumber>{})
      .def(py::init<qc::Control, qc::Qubit, qc::OpType,
                    const std::vector<qc::SymbolOrNumber>&>(),
           "control"_a, "target"_a, "op_type"_a,
           "params"_a = std::vector<qc::SymbolOrNumber>{})
      .def(py::init<qc::Control, const qc::Targets&, qc::OpType,
                    const std::vector<qc::SymbolOrNumber>&>(),
           "control"_a, "targets"_a, "op_type"_a,
           "params"_a = std::vector<qc::SymbolOrNumber>{})
      .def(py::init<const qc::Controls&, qc::Qubit, qc::OpType,
                    const std::vector<qc::SymbolOrNumber>&>(),
           "controls"_a, "target"_a, "op_type"_a,
           "params"_a = std::vector<qc::SymbolOrNumber>{})
      .def(py::init<const qc::Controls&, const qc::Targets&, qc::OpType,
                    const std::vector<qc::SymbolOrNumber>&>(),
           "controls"_a, "targets"_a, "op_type"_a,
           "params"_a = std::vector<qc::SymbolOrNumber>{})
      .def(py::init<const qc::Controls&, qc::Qubit, qc::Qubit, qc::OpType,
                    const std::vector<qc::SymbolOrNumber>&>(),
           "controls"_a, "target0"_a, "target1"_a, "op_type"_a,
           "params"_a = std::vector<qc::SymbolOrNumber>{})
      .def("get_parameter", &qc::SymbolicOperation::getParameter)
      .def("get_parameters", &qc::SymbolicOperation::getParameters)
      .def("get_instantiated_operation",
           &qc::SymbolicOperation::getInstantiatedOperation, "assignment"_a)
      .def("instantiate", &qc::SymbolicOperation::instantiate, "assignment"_a);
}

} // namespace mqt
