/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "ir/Definitions.hpp"
#include "ir/operations/Control.hpp"
#include "ir/operations/Expression.hpp"
#include "ir/operations/OpType.hpp"
#include "ir/operations/StandardOperation.hpp"
#include "ir/operations/SymbolicOperation.hpp"

// These includes must be the first includes for any bindings code
// clang-format off
#include <pybind11/pybind11.h>
<<<<<<< HEAD
#include <pybind11/stl.h> // NOLINT(misc-include-cleaner)

#include <pybind11/cast.h>
=======
#include <pybind11/stl.h>
>>>>>>> 59e09070 (Remove pybind11.hpp)
// clang-format on

#include <vector>

namespace mqt {

namespace py = pybind11;
<<<<<<< HEAD
using namespace pybind11::literals;

// NOLINTNEXTLINE(misc-use-internal-linkage)
=======
using namespace py::literals;

>>>>>>> 59e09070 (Remove pybind11.hpp)
void registerSymbolicOperation(const py::module& m) {
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
