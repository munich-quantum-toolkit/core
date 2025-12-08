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

#include <nanobind/nanobind.h>
#include <nanobind/operators.h>
#include <nanobind/stl/string.h>
#include <string>

namespace mqt {

namespace nb = nanobind;
using namespace nb::literals;

// NOLINTNEXTLINE(misc-use-internal-linkage)
void registerVariable(nb::module_& m) {
  nb::class_<sym::Variable>(m, "Variable")
      .def(nb::init<std::string>(), "name"_a = "")
      .def_prop_ro("name", &sym::Variable::getName)
      .def("__str__", &sym::Variable::getName)
      .def("__repr__", &sym::Variable::getName)
      .def(nb::self == nb::self)
      .def(nb::self != nb::self)
      .def(nb::hash(nb::self))
      .def(nb::self < nb::self)
      .def(nb::self > nb::self);
}
} // namespace mqt
