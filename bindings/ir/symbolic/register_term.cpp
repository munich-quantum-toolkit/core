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
#include <nanobind/stl/string.h>        // NOLINT(misc-include-cleaner)
#include <nanobind/stl/unordered_map.h> // NOLINT(misc-include-cleaner)
#include <sstream>

namespace mqt {

namespace nb = nanobind;
using namespace nb::literals;

// NOLINTNEXTLINE(misc-use-internal-linkage)
void registerTerm(const nb::module_& m) {
  nb::class_<sym::Term<double>>(
      m, "Term",
      R"pb(A symbolic term which consists of a variable with a given coefficient.

Args:
    variable: The variable of the term.
    coefficient: The coefficient of the term.)pb")

      .def(nb::init<sym::Variable, double>(), "variable"_a,
           "coefficient"_a = 1.0)

      .def_prop_ro("variable", &sym::Term<double>::getVar,
                   "The variable of the term.")

      .def_prop_ro("coefficient", &sym::Term<double>::getCoeff,
                   "The coefficient of the term.")

      .def("has_zero_coefficient", &sym::Term<double>::hasZeroCoeff,
           "Check if the coefficient of the term is zero.")

      .def("add_coefficient", &sym::Term<double>::addCoeff, "coeff"_a,
           R"pb(Add a coefficient to the coefficient of this term.

Args:
    coeff: The coefficient to add.)pb")

      .def("evaluate", &sym::Term<double>::evaluate, "assignment"_a,
           R"pb(Evaluate the term with a given variable assignment.

Args:
    assignment: The variable assignment.

Returns:
    The evaluated value of the term.)pb")

      .def(nb::self * double(), nb::is_operator())
      .def(double() * nb::self, nb::is_operator())
      .def(nb::self / double(), nb::is_operator())

      .def(nb::self == nb::self,
           nb::sig("def __eq__(self, arg: object, /) -> bool"))
      .def(nb::self != nb::self,
           nb::sig("def __ne__(self, arg: object, /) -> bool"))
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
