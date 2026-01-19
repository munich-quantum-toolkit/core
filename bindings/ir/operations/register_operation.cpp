/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "ir/operations/Control.hpp"
#include "ir/operations/Operation.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/operators.h>
#include <nanobind/stl/set.h>        // NOLINT(misc-include-cleaner)
#include <nanobind/stl/string.h>     // NOLINT(misc-include-cleaner)
#include <nanobind/stl/unique_ptr.h> // NOLINT(misc-include-cleaner)
#include <nanobind/stl/vector.h>     // NOLINT(misc-include-cleaner)
#include <sstream>

namespace mqt {

namespace nb = nanobind;
using namespace nb::literals;

// NOLINTNEXTLINE(misc-use-internal-linkage)
void registerOperation(const nb::module_& m) {
  nb::class_<qc::Operation>(m, "Operation")
      .def_prop_ro("name", &qc::Operation::getName,
                   "The name of the operation.")

      .def_prop_rw("type_", &qc::Operation::getType, &qc::Operation::setGate,
                   "The type of the operation.")

      .def_prop_rw(
          "targets", [](const qc::Operation& op) { return op.getTargets(); },
          &qc::Operation::setTargets, R"pb(The targets of the operation.

Note:
    The notion of a target might not make sense for all types of operations.)pb")

      .def_prop_ro("num_targets", &qc::Operation::getNtargets,
                   "The number of targets of the operation.")

      .def_prop_rw(
          "controls", [](const qc::Operation& op) { return op.getControls(); },
          &qc::Operation::setControls, R"pb(The controls of the operation.

Note:
    The notion of a control might not make sense for all types of operations.)pb")

      .def_prop_ro("num_controls", &qc::Operation::getNcontrols,
                   "The number of controls of the operation.")

      .def("add_control", &qc::Operation::addControl, "control"_a,
           R"pb(Add a control to the operation.

Args:
    control: The control to add.)pb")

      .def("add_controls", &qc::Operation::addControls, "controls"_a,
           R"pb(Add multiple controls to the operation.

Args:
    controls: The controls to add.)pb")

      .def("clear_controls", &qc::Operation::clearControls,
           "Clear all controls of the operation.")

      .def(
          "remove_control",
          [](qc::Operation& op, const qc::Control& c) { op.removeControl(c); },
          "control"_a, R"pb(Remove a control from the operation.

Args:
    control: The control to remove.)pb")

      .def("remove_controls", &qc::Operation::removeControls, "controls"_a,
           R"pb(Remove multiple controls from the operation.

Args:
    controls: The controls to remove.)pb")

      .def("get_used_qubits", &qc::Operation::getUsedQubits,
           R"pb(Get the qubits that are used by the operation.

Returns:
    The set of qubits that are used by the operation.)pb")

      .def("acts_on", &qc::Operation::actsOn, "qubit"_a,
           R"pb(Check if the operation acts on a specific qubit.

Args:
    qubit: The qubit to check.

Returns:
    True if the operation acts on the qubit, False otherwise.)pb")

      .def_prop_rw(
          "parameter",
          [](const qc::Operation& op) { return op.getParameter(); },
          &qc::Operation::setParameter, R"pb(The parameters of the operation.

Note:
    The notion of a parameter might not make sense for all types of operations.)pb")

      .def("is_unitary", &qc::Operation::isUnitary,
           R"pb(Check if the operation is unitary.

Returns:
    True if the operation is unitary, False otherwise.)pb")

      .def("is_standard_operation", &qc::Operation::isStandardOperation,
           R"pb(Check if the operation is a :class:`StandardOperation`.

Returns:
    True if the operation is a :class:`StandardOperation`, False otherwise.)pb")

      .def("is_compound_operation", &qc::Operation::isCompoundOperation,
           R"pb(Check if the operation is a :class:`CompoundOperation`.

Returns:
    True if the operation is a :class:`CompoundOperation`, False otherwise.)pb")

      .def("is_non_unitary_operation", &qc::Operation::isNonUnitaryOperation,
           R"pb(Check if the operation is a :class:`NonUnitaryOperation`.

Returns:
    True if the operation is a :class:`NonUnitaryOperation`, False otherwise.)pb")

      .def("is_if_else_operation", &qc::Operation::isIfElseOperation,
           R"pb(Check if the operation is a :class:`IfElseOperation`.

Returns:
    True if the operation is a :class:`IfElseOperation`, False otherwise.)pb")

      .def("is_symbolic_operation", &qc::Operation::isSymbolicOperation,
           R"pb(Check if the operation is a :class:`SymbolicOperation`.

Returns:
    True if the operation is a :class:`SymbolicOperation`, False otherwise.)pb")

      .def("is_controlled", &qc::Operation::isControlled,
           R"pb(Check if the operation is controlled.

Returns:
    True if the operation is controlled, False otherwise.)pb")

      .def("get_inverted", &qc::Operation::getInverted,
           R"pb(Get the inverse of the operation.

Returns:
    The inverse of the operation.)pb")

      .def("invert", &qc::Operation::invert, "Invert the operation (in-place).")

      .def(nb::self == nb::self,
           nb::sig("def __eq__(self, arg: object, /) -> bool"))
      .def(nb::self != nb::self,
           nb::sig("def __ne__(self, arg: object, /) -> bool"))
      .def(nb::hash(nb::self))

      .def("__repr__", [](const qc::Operation& op) {
        std::ostringstream oss;
        oss << "Operation(type=" << op.getType() << ", ...)";
        return oss.str();
      });
}
} // namespace mqt
