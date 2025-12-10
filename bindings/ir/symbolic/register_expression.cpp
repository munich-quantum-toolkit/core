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

#include <cstddef>
#include <nanobind/make_iterator.h>
#include <nanobind/nanobind.h>
#include <nanobind/operators.h>
#include <nanobind/stl/string.h> // NOLINT(misc-include-cleaner)
#include <nanobind/stl/vector.h> // NOLINT(misc-include-cleaner)
#include <sstream>
#include <vector>

namespace mqt {

namespace nb = nanobind;
using namespace nb::literals;

// NOLINTNEXTLINE(misc-use-internal-linkage)
void registerExpression(nb::module_& m) {
  nb::class_<sym::Expression<double, double>>(m, "Expression")
      .def(
          "__init__",
          [](sym::Expression<double, double>* self,
             const std::vector<sym::Term<double>>& terms, double constant) {
            new (self) sym::Expression<double, double>(terms, constant);
          },
          "terms"_a, "constant"_a = 0.0)
      .def(
          "__init__",
          [](sym::Expression<double, double>* self,
             const sym::Term<double>& term, double constant) {
            new (self) sym::Expression<double, double>(
                std::vector<sym::Term<double>>{term}, constant);
          },
          "term"_a, "constant"_a = 0.0)
      .def(nb::init<double>(), "constant"_a = 0.0)
      .def_prop_rw("constant", &sym::Expression<double, double>::getConst,
                   &sym::Expression<double, double>::setConst)
      .def(
          "__iter__",
          [](const sym::Expression<double, double>& expr) {
            return nb::make_iterator(
                nb::type<sym::Expression<double, double>>(), "iterator",
                expr.begin(), expr.end());
          },
          nb::keep_alive<0, 1>())
      .def("__getitem__",
           [](const sym::Expression<double, double>& expr,
              const std::size_t idx) {
             if (idx >= expr.numTerms()) {
               throw nb::index_error();
             }
             return expr.getTerms()[idx];
           })
      .def("is_zero", &sym::Expression<double, double>::isZero)
      .def("is_constant", &sym::Expression<double, double>::isConstant)
      .def("num_terms", &sym::Expression<double, double>::numTerms)
      .def("__len__", &sym::Expression<double, double>::numTerms)
      .def_prop_ro("terms", &sym::Expression<double, double>::getTerms)
      .def_prop_ro("variables", &sym::Expression<double, double>::getVariables)
      .def("evaluate", &sym::Expression<double, double>::evaluate,
           "assignment"_a)
      // addition operators
      .def(nb::self + nb::self)
      .def(nb::self + double())
      .def("__add__", [](const sym::Expression<double, double>& lhs,
                         const sym::Term<double>& rhs) { return lhs + rhs; })
      .def("__radd__", [](const sym::Expression<double, double>& rhs,
                          const sym::Term<double>& lhs) { return lhs + rhs; })
      .def("__radd__", [](const sym::Expression<double, double>& rhs,
                          const double lhs) { return rhs + lhs; })
      // subtraction operators
      .def(nb::self - nb::self) // NOLINT(misc-redundant-expression)
      .def(nb::self - double())
      .def(double() - nb::self)
      .def("__sub__", [](const sym::Expression<double, double>& lhs,
                         const sym::Term<double>& rhs) { return lhs - rhs; })
      .def("__rsub__", [](const sym::Expression<double, double>& rhs,
                          const sym::Term<double>& lhs) { return lhs - rhs; })
      // multiplication operators
      .def(nb::self * double())
      .def(double() * nb::self)
      // division operators
      .def(nb::self / double())
      .def("__rtruediv__", [](const sym::Expression<double, double>& rhs,
                              double lhs) { return rhs / lhs; })
      // comparison operators
      .def(nb::self == nb::self)
      .def(nb::self != nb::self)
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
