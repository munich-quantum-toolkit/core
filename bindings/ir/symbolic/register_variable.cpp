/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "ir/operations/Expression.hpp"

// These includes must be the first includes for any bindings code
// clang-format off
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // NOLINT(misc-include-cleaner)

#include <pybind11/cast.h>
#include <pybind11/operators.h>
// clang-format on

#include <string>

namespace mqt {

namespace py = pybind11;
using namespace pybind11::literals;

// NOLINTNEXTLINE(misc-use-internal-linkage)
void registerVariable(py::module& m) {
  py::class_<sym::Variable>(m, "Variable")
      .def(py::init<std::string>(), "name"_a = "")
      .def_property_readonly("name", &sym::Variable::getName)
      .def("__str__", &sym::Variable::getName)
      .def("__repr__", &sym::Variable::getName)
      .def(py::self == py::self) // NOLINT(misc-redundant-expression)
      .def(py::self != py::self) // NOLINT(misc-redundant-expression)
      .def(hash(py::self))
      .def(py::self < py::self)  // NOLINT(misc-redundant-expression)
      .def(py::self > py::self); // NOLINT(misc-redundant-expression)
}
} // namespace mqt
