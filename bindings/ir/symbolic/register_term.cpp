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
<<<<<<< HEAD
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // NOLINT(misc-include-cleaner)

#include <pybind11/cast.h>
#include <pybind11/operators.h>
=======
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
>>>>>>> 59e09070 (Remove pybind11.hpp)
// clang-format on

#include <sstream>

namespace mqt {

namespace py = pybind11;
<<<<<<< HEAD
using namespace pybind11::literals;

// NOLINTNEXTLINE(misc-use-internal-linkage)
=======
using namespace py::literals;

>>>>>>> 59e09070 (Remove pybind11.hpp)
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
      .def(py::self == py::self) // NOLINT(misc-redundant-expression)
      .def(py::self != py::self) // NOLINT(misc-redundant-expression)
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
