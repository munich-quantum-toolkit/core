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

#include <nanobind/nanobind.h>
#include <nanobind/operators.h>
#include <nanobind/stl/set.h>    // NOLINT(misc-include-cleaner)
#include <nanobind/stl/string.h> // NOLINT(misc-include-cleaner)

namespace mqt {

namespace nb = nanobind;
using namespace nb::literals;

// NOLINTNEXTLINE(misc-use-internal-linkage)
void registerControl(const nb::module_& m) {
  auto control = nb::class_<qc::Control>(
      m, "Control",
      R"pb(A control is a pair of a qubit and a type. The type can be either positive or negative.

Args:
    qubit: The qubit that is the control.
    type_: The type of the control.)pb");

  nb::enum_<qc::Control::Type>(control, "Type", "Enumeration of control types.")
      .value("Pos", qc::Control::Type::Pos)
      .value("Neg", qc::Control::Type::Neg);

  control.def(nb::init<qc::Qubit, qc::Control::Type>(), "qubit"_a,
              "type_"_a.sig("...") = qc::Control::Type::Pos);

  control.def_ro("qubit", &qc::Control::qubit,
                 "The qubit that is the control.");

  control.def_ro("type_", &qc::Control::type, "The type of the control.");

  control.def("__str__", [](const qc::Control& c) { return c.toString(); });
  control.def("__repr__", [](const qc::Control& c) { return c.toString(); });

  control.def(nb::self == nb::self,
              nb::sig("def __eq__(self, arg: object, /) -> bool"));
  control.def(nb::self != nb::self,
              nb::sig("def __ne__(self, arg: object, /) -> bool"));
  control.def(nb::hash(nb::self));

  nb::implicitly_convertible<nb::int_, qc::Control>();
}

} // namespace mqt
