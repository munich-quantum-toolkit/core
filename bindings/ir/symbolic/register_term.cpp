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
#include <nanobind/stl/string.h> // NOLINT(misc-include-cleaner)
#include <sstream>

namespace mqt {

namespace nb = nanobind;
using namespace nb::literals;

// NOLINTNEXTLINE(misc-use-internal-linkage)
void registerTerm(nb::module_& m) {
  nb::class_<sym::Term<double>>(m, "Term")
      .def(nb::init<sym::Variable, double>(), "variable"_a,
           "coefficient"_a = 1.0)
      .def_prop_ro("variable", &sym::Term<double>::getVar)
      .def_prop_ro("coefficient", &sym::Term<double>::getCoeff)
      .def("has_zero_coefficient", &sym::Term<double>::hasZeroCoeff)
      .def("add_coefficient", &sym::Term<double>::addCoeff, "coeff"_a)
      .def("evaluate", &sym::Term<double>::evaluate, "assignment"_a)
      .def(nb::self * double())
      .def(double() * nb::self)
      .def(nb::self / double())
      .def(double() / nb::self)
      .def(nb::self == nb::self)
      .def(nb::self != nb::self)
      .def(nb::hash(nb::self))
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
