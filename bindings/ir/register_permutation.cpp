/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "ir/Definitions.hpp"
#include "ir/Permutation.hpp"
#include "ir/operations/Control.hpp"

#include <iterator>
#include <nanobind/make_iterator.h>
#include <nanobind/nanobind.h>
#include <nanobind/operators.h>
#include <nanobind/stl/string.h> // NOLINT(misc-include-cleaner)
#include <sstream>

namespace mqt {

namespace nb = nanobind;
using namespace nb::literals;

// NOLINTNEXTLINE(misc-use-internal-linkage)
void registerPermutation(nb::module_& m) {
  nb::class_<qc::Permutation>(m, "Permutation")
      .def(nb::init<>())
      .def(
          "__init__",
          [](qc::Permutation* self, const nb::dict& p) {
            qc::Permutation perm;
            for (const auto& [key, value] : p) {
              perm[nb::cast<qc::Qubit>(key)] = nb::cast<qc::Qubit>(value);
            }
            new (self) qc::Permutation(std::move(perm));
          },
          "perm"_a, "Create a permutation from a dictionary.")
      .def("apply",
           nb::overload_cast<const qc::Controls&>(&qc::Permutation::apply,
                                                  nb::const_),
           "controls"_a)
      .def("apply",
           nb::overload_cast<const qc::Targets&>(&qc::Permutation::apply,
                                                 nb::const_),
           "targets"_a)
      .def("clear", [](qc::Permutation& p) { p.clear(); })
      .def("__getitem__",
           [](const qc::Permutation& p, const qc::Qubit q) { return p.at(q); })
      .def("__setitem__", [](qc::Permutation& p, const qc::Qubit q,
                             const qc::Qubit r) { p[q] = r; })
      .def("__delitem__",
           [](qc::Permutation& p, const qc::Qubit q) { p.erase(q); })
      .def("__len__", &qc::Permutation::size)
      .def("__iter__",
           [](const qc::Permutation& p) {
             return nb::make_key_iterator(nb::type<qc::Permutation>(),
                                          "key_iterator", p.begin(), p.end());
           })
      .def(
          "items",
          [](const qc::Permutation& p) {
            return nb::make_iterator(nb::type<qc::Permutation>(),
                                     "item_iterator", p.begin(), p.end());
          },
          nb::keep_alive<0, 1>())
      .def(nb::self == nb::self)
      .def(nb::self != nb::self)
      .def(nb::hash(nb::self))
      .def("__str__",
           [](const qc::Permutation& p) {
             std::stringstream ss;
             ss << "{";
             for (auto it = p.cbegin(); it != p.cend(); ++it) {
               ss << it->first << ": " << it->second;
               if (std::next(it) != p.cend()) {
                 ss << ", ";
               }
             }
             ss << "}";
             return ss.str();
           })
      .def("__repr__", [](const qc::Permutation& p) {
        std::stringstream ss;
        ss << "Permutation({";
        for (auto it = p.cbegin(); it != p.cend(); ++it) {
          ss << it->first << ": " << it->second;
          if (std::next(it) != p.cend()) {
            ss << ", ";
          }
        }
        ss << "})";
        return ss.str();
      });
  nb::implicitly_convertible<nb::dict, qc::Permutation>();
}

} // namespace mqt
