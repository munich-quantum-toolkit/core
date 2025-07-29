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

// These includes must be the first includes for any bindings code
// clang-format off
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // NOLINT(misc-include-cleaner)

#include <pybind11/cast.h>
#include <pybind11/native_enum.h>
#include <pybind11/operators.h>
#include <pybind11/pytypes.h>
// clang-format on

namespace mqt {

namespace py = pybind11;
using namespace pybind11::literals;

// NOLINTNEXTLINE(misc-use-internal-linkage)
void registerControl(const py::module& m) {

  auto control = py::class_<qc::Control>(m, "Control");

  py::native_enum<qc::Control::Type>(control, "Type", "enum.Enum",
                                     "Test string.")
      .value("Pos", qc::Control::Type::Pos)
      .value("Neg", qc::Control::Type::Neg)
      .finalize();

  control.def(py::init<qc::Qubit, qc::Control::Type>(), "qubit"_a,
              "type_"_a = qc::Control::Type::Pos);
  control.def_readwrite("type_", &qc::Control::type);
  control.def_readwrite("qubit", &qc::Control::qubit);
  control.def("__str__", [](const qc::Control& c) { return c.toString(); });
  control.def("__repr__", [](const qc::Control& c) { return c.toString(); });
  control.def(py::self == py::self); // NOLINT(misc-redundant-expression)
  control.def(py::self != py::self); // NOLINT(misc-redundant-expression)
  control.def(hash(py::self));
  py::implicitly_convertible<py::int_, qc::Control>();
}

} // namespace mqt
