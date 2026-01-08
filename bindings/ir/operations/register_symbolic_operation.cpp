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
#include "ir/operations/Control.hpp"
#include "ir/operations/Expression.hpp"
#include "ir/operations/OpType.hpp"
#include "ir/operations/StandardOperation.hpp"
#include "ir/operations/SymbolicOperation.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/set.h>           // NOLINT(misc-include-cleaner)
#include <nanobind/stl/string.h>        // NOLINT(misc-include-cleaner)
#include <nanobind/stl/unordered_map.h> // NOLINT(misc-include-cleaner)
#include <nanobind/stl/variant.h>       // NOLINT(misc-include-cleaner)
#include <nanobind/stl/vector.h>        // NOLINT(misc-include-cleaner)
#include <vector>

namespace mqt {

namespace nb = nanobind;
using namespace nb::literals;

// NOLINTNEXTLINE(misc-use-internal-linkage)
void registerSymbolicOperation(const nb::module_& m) {
  nb::class_<qc::SymbolicOperation, qc::StandardOperation>(
      m, "SymbolicOperation",
      R"pb(Symbolic quantum operation.

This class is used to represent quantum operations that are not yet fully defined.
This can be useful for representing operations that depend on parameters that are not yet known.
A :class:`SymbolicOperation` is defined by its :class:`OpType`, the qubits (controls and targets) it acts on, and its parameters.
The parameters can be either fixed values or symbolic expressions.

Args:
     controls: The control qubit(s) of the operation (if any).
     targets: The target qubit(s) of the operation.
     op_type: The type of the operation.
     params: The parameters of the operation (if any).)pb")

      .def(nb::init<>())
      .def(nb::init<qc::Qubit, qc::OpType,
                    const std::vector<qc::SymbolOrNumber>&>(),
           "target"_a, "op_type"_a,
           "params"_a.sig("...") = std::vector<qc::SymbolOrNumber>{})
      .def(nb::init<const qc::Targets&, qc::OpType,
                    const std::vector<qc::SymbolOrNumber>&>(),
           "targets"_a, "op_type"_a,
           "params"_a.sig("...") = std::vector<qc::SymbolOrNumber>{})
      .def(nb::init<qc::Control, qc::Qubit, qc::OpType,
                    const std::vector<qc::SymbolOrNumber>&>(),
           "control"_a, "target"_a, "op_type"_a,
           "params"_a.sig("...") = std::vector<qc::SymbolOrNumber>{})
      .def(nb::init<qc::Control, const qc::Targets&, qc::OpType,
                    const std::vector<qc::SymbolOrNumber>&>(),
           "control"_a, "targets"_a, "op_type"_a,
           "params"_a.sig("...") = std::vector<qc::SymbolOrNumber>{})
      .def(nb::init<const qc::Controls&, qc::Qubit, qc::OpType,
                    const std::vector<qc::SymbolOrNumber>&>(),
           "controls"_a, "target"_a, "op_type"_a,
           "params"_a.sig("...") = std::vector<qc::SymbolOrNumber>{})
      .def(nb::init<const qc::Controls&, const qc::Targets&, qc::OpType,
                    const std::vector<qc::SymbolOrNumber>&>(),
           "controls"_a, "targets"_a, "op_type"_a,
           "params"_a.sig("...") = std::vector<qc::SymbolOrNumber>{})
      .def(nb::init<const qc::Controls&, qc::Qubit, qc::Qubit, qc::OpType,
                    const std::vector<qc::SymbolOrNumber>&>(),
           "controls"_a, "target0"_a, "target1"_a, "op_type"_a,
           "params"_a.sig("...") = std::vector<qc::SymbolOrNumber>{})
      .def("get_parameter", &qc::SymbolicOperation::getParameter, "index"_a,
           R"pb(Get the parameter at the given index.

Args:
     index: The index of the parameter to get.

Returns:
     The parameter at the given index.)pb")

      .def("get_parameters", &qc::SymbolicOperation::getParameters,
           R"pb(Get all parameters of the operation.

Returns:
     The parameters of the operation.)pb")

      .def("get_instantiated_operation",
           &qc::SymbolicOperation::getInstantiatedOperation, "assignment"_a,
           R"pb(Get the instantiated operation.

Args:
     assignment: The assignment of the symbolic parameters.

Returns:
     The instantiated operation.)pb")

      .def("instantiate", &qc::SymbolicOperation::instantiate, "assignment"_a,
           R"pb(Instantiate the operation (in-place).

Args:
     assignment: The assignment of the symbolic parameters.)pb");
}

} // namespace mqt
