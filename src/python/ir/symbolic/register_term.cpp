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
#include <sstream>

namespace mqt {

void registerTerm(py::module& m) {
  py::class_<sym::Term<double>>(m, "Term")
      .def(py::init<sym::Variable, double>(), "variable"_a,
           "coefficient"_a = 1.0)
      .def_property_readonly("variable", &sym::Term<double>::getVar)
      .def_property_readonly("coefficient", &sym::Term<double>::getCoeff)
      .def("has_zero_coefficient", &sym::Term<double>::hasZeroCoeff)
      .def("add_coefficient", &sym::Term<double>::addCoeff, "coeff"_a)
      .def("evaluate", &sym::Term<double>::evaluate, "assignment"_a)
      .def(py::self * double())
      .def(double() * py::self)
      .def(py::self / double())
      .def(double() / py::self)
      .def(py::self == py::self)
      .def(py::self != py::self)
      .def(hash(py::self))
      .def("__str__",
           [](const sym::Term<double>& term) {
             std::stringstream ss;
             ss << term;
             return ss.str();
           })
      .def("__repr__", [](const sym::Term<double>& term) {
        std::stringstream ss;
        ss << term;
        return ss.str();
      });
}
} // namespace mqt
