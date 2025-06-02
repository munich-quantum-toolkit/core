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
#include "python/pybind11.hpp"

#include <pybind11/operators.h>
#include <string>

namespace mqt {

void registerVariable(py::module& m) {
  py::class_<sym::Variable>(m, "Variable")
      .def(py::init<std::string>(), "name"_a = "")
      .def_property_readonly("name", &sym::Variable::getName)
      .def("__str__", &sym::Variable::getName)
      .def("__repr__", &sym::Variable::getName)
      .def(py::self == py::self)
      .def(py::self != py::self)
      .def(hash(py::self))
      .def(py::self < py::self)
      .def(py::self > py::self);
}
} // namespace mqt
