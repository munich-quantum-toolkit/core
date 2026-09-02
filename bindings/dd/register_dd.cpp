/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "dd/DDDefinitions.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/operators.h>
#include <nanobind/stl/string.h> // NOLINT(misc-include-cleaner)

namespace mqt {

namespace nb = nanobind;
using namespace nb::literals;

// forward declarations
void registerVectorDDs(const nb::module_& m);
void registerMatrixDDs(const nb::module_& m);
void registerDDPackage(const nb::module_& m);

namespace {
void registerControl(const nb::module_& m) {
  auto control = nb::class_<dd::Control>(
      m, "Control",
      R"pb(Control a raw matrix DD operation with one qubit.

Args:
    qubit: Control qubit index.
    type_: Control polarity.)pb");

  nb::enum_<dd::Control::Type>(control, "Type", "Control polarity.")
      .value("Pos", dd::Control::Type::Pos)
      .value("Neg", dd::Control::Type::Neg);

  control.def(nb::init<dd::Qubit, dd::Control::Type>(), "qubit"_a,
              "type_"_a.sig("...") = dd::Control::Type::Pos);
  control.def_ro("qubit", &dd::Control::qubit, "Control qubit index.");
  control.def_ro("type_", &dd::Control::type, "Control polarity.");
  control.def("__str__", &dd::Control::toString);
  control.def("__repr__", &dd::Control::toString);
  control.def(nb::self == nb::self,
              nb::sig("def __eq__(self, arg: object, /) -> bool"));
  control.def(nb::self != nb::self,
              nb::sig("def __ne__(self, arg: object, /) -> bool"));
  control.def(nb::hash(nb::self));

  nb::implicitly_convertible<nb::int_, dd::Control>();
}
} // namespace

/// NOLINTNEXTLINE(performance-unnecessary-value-param)
NB_MODULE(MQT_CORE_MODULE_NAME, m) {
  m.doc() = "MQT Core decision diagram module.";

  /// Controls for raw matrix DD construction.
  registerControl(m);

  // Vector Decision Diagrams
  registerVectorDDs(m);

  // Matrix Decision Diagrams
  registerMatrixDDs(m);

  // DD Package
  registerDDPackage(m);
}

} // namespace mqt
