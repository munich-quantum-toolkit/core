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

#include <nanobind/nanobind.h>
#include <nanobind/operators.h>
#include <nanobind/stl/string.h>

namespace mqt {

namespace nb = nanobind;
using namespace nb::literals;

// NOLINTNEXTLINE(misc-use-internal-linkage)
void registerControl(const nb::module_& m) {
  auto control = nb::class_<qc::Control>(m, "Control");

  nb::enum_<qc::Control::Type>(control, "Type", "enum.Enum",
                               "Enumeration of control types.")
      .value("Pos", qc::Control::Type::Pos)
      .value("Neg", qc::Control::Type::Neg);

  control.def(nb::init<qc::Qubit, qc::Control::Type>(), "qubit"_a,
              "type_"_a = qc::Control::Type::Pos);
  control.def_rw("type_", &qc::Control::type);
  control.def_rw("qubit", &qc::Control::qubit);
  control.def("__str__", [](const qc::Control& c) { return c.toString(); });
  control.def("__repr__", [](const qc::Control& c) { return c.toString(); });
  control.def(nb::self == nb::self);
  control.def(nb::self != nb::self);
  control.def(nb::hash(nb::self));
  nb::implicitly_convertible<nb::int_, qc::Control>();
}

} // namespace mqt
