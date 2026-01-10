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

#include <cstddef>
#include <nanobind/make_iterator.h>
#include <nanobind/nanobind.h>
#include <nanobind/operators.h>
#include <nanobind/stl/string.h>        // NOLINT(misc-include-cleaner)
#include <nanobind/stl/unordered_map.h> // NOLINT(misc-include-cleaner)
#include <nanobind/stl/unordered_set.h> // NOLINT(misc-include-cleaner)
#include <nanobind/stl/vector.h>        // NOLINT(misc-include-cleaner)
#include <sstream>
#include <vector>

namespace mqt {

namespace nb = nanobind;
using namespace nb::literals;

// NOLINTNEXTLINE(misc-use-internal-linkage)
void registerExpression(const nb::module_& m) {
  nb::class_<sym::Expression<double, double>>(
      m, "Expression",
      R"pb(A symbolic expression which consists of a sum of terms and a constant.

The expression is of the form :math:`constant + term_1 + term_2 + \dots + term_n`.
Alternatively, an expression can be created with a single term and a constant or just a constant.

Args:
    terms: The list of terms.
    constant: The constant.)pb")

      .def(nb::init<double>(), "constant"_a = 0.0)
      .def(nb::init<const std::vector<sym::Term<double>>&, double>(), "terms"_a,
           "constant"_a = 0.0)
      .def(
          "__init__",
          [](sym::Expression<double, double>* self,
             const sym::Term<double>& term, double constant) {
            new (self) sym::Expression<double, double>(
                std::vector<sym::Term<double>>{term}, constant);
          },
          "term"_a, "constant"_a = 0.0)

      .def_prop_rw("constant", &sym::Expression<double, double>::getConst,
                   &sym::Expression<double, double>::setConst,
                   "The constant of the expression.")
      .def(
          "__iter__",
          [](const sym::Expression<double, double>& expr) {
            return make_iterator(nb::type<sym::Expression<double, double>>(),
                                 "iterator", expr.begin(), expr.end());
          },
          nb::keep_alive<0, 1>())

      .def(
          "__getitem__",
          [](const sym::Expression<double, double>& expr, nb::ssize_t idx) {
            const auto n = static_cast<nb::ssize_t>(expr.numTerms());
            if (idx < 0) {
              idx += n;
            }
            if (idx < 0 || idx >= n) {
              throw nb::index_error();
            }
            // NOLINTNEXTLINE(*-pro-bounds-avoid-unchecked-container-access)
            return expr.getTerms()[static_cast<std::size_t>(idx)];
          },
          "index"_a)

      .def("is_zero", &sym::Expression<double, double>::isZero,
           "Check if the expression is zero.")

      .def("is_constant", &sym::Expression<double, double>::isConstant,
           "Check if the expression is a constant.")

      .def("num_terms", &sym::Expression<double, double>::numTerms,
           "The number of terms in the expression.")

      .def("__len__", &sym::Expression<double, double>::numTerms)

      .def_prop_ro("terms", &sym::Expression<double, double>::getTerms,
                   "The terms of the expression.")

      .def_prop_ro("variables", &sym::Expression<double, double>::getVariables,
                   "The variables in the expression.")

      .def("evaluate", &sym::Expression<double, double>::evaluate,
           "assignment"_a,
           R"pb(Evaluate the expression with a given variable assignment.

Args:
    assignment: The variable assignment.

Returns:
    The evaluated value of the expression.)pb")

      // addition operators
      .def(nb::self + nb::self, nb::is_operator())
      .def(nb::self + double(), nb::is_operator())
      .def(
          "__add__",
          [](const sym::Expression<double, double>& lhs,
             const sym::Term<double>& rhs) { return lhs + rhs; },
          nb::is_operator())
      .def(
          "__radd__",
          [](const sym::Expression<double, double>& rhs,
             const sym::Term<double>& lhs) { return lhs + rhs; },
          nb::is_operator())
      .def(
          "__radd__",
          [](const sym::Expression<double, double>& rhs, const double lhs) {
            return rhs + lhs;
          },
          nb::is_operator())
      // subtraction operators
      // NOLINTNEXTLINE(misc-redundant-expression)
      .def(nb::self - nb::self, nb::is_operator())
      .def(nb::self - double(), nb::is_operator())
      .def(double() - nb::self, nb::is_operator())
      .def(
          "__sub__",
          [](const sym::Expression<double, double>& lhs,
             const sym::Term<double>& rhs) { return lhs - rhs; },
          nb::is_operator())
      .def(
          "__rsub__",
          [](const sym::Expression<double, double>& rhs,
             const sym::Term<double>& lhs) { return lhs - rhs; },
          nb::is_operator())
      // multiplication operators
      .def(nb::self * double(), nb::is_operator())
      .def(double() * nb::self, nb::is_operator())
      // division operators
      .def(nb::self / double(), nb::is_operator())
      // comparison operators
      .def(nb::self == nb::self,
           nb::sig("def __eq__(self, arg: object, /) -> bool"))
      .def(nb::self != nb::self,
           nb::sig("def __ne__(self, arg: object, /) -> bool"))
      .def(nb::hash(nb::self))
      .def("__str__",
           [](const sym::Expression<double, double>& expr) {
             std::stringstream ss;
             ss << expr;
             return ss.str();
           })
      .def("__repr__", [](const sym::Expression<double, double>& expr) {
        std::stringstream ss;
        ss << expr;
        return ss.str();
      });
}
} // namespace mqt
