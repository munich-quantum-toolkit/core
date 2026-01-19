/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "ir/operations/Expression.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/operators.h>
#include <nanobind/stl/string.h> // NOLINT(misc-include-cleaner)
#include <string>

namespace mqt {

namespace nb = nanobind;
using namespace nb::literals;

// NOLINTNEXTLINE(misc-use-internal-linkage)
void registerVariable(const nb::module_& m) {
  nb::class_<sym::Variable>(m, "Variable", R"pb(A symbolic variable.
Note:
    Variables are uniquely identified by their name, so if a variable with the same name already exists, the existing variable will be returned.

Args:
    name: The name of the variable.)pb")

      .def(nb::init<std::string>(), "name"_a = "")

      .def_prop_ro("name", &sym::Variable::getName, "The name of the variable.")

      .def("__str__", &sym::Variable::getName)
      .def("__repr__", &sym::Variable::getName)

      .def(nb::self == nb::self,
           nb::sig("def __eq__(self, arg: object, /) -> bool"))
      .def(nb::self != nb::self,
           nb::sig("def __ne__(self, arg: object, /) -> bool"))
      .def(nb::hash(nb::self))
      .def(nb::self < nb::self)
      .def(nb::self > nb::self);
}
} // namespace mqt
