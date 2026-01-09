/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "ir/Definitions.hpp"
#include "ir/Permutation.hpp"
#include "ir/operations/Control.hpp"

#include <cstdint>
#include <iterator>
#include <limits>
#include <nanobind/make_iterator.h>
#include <nanobind/nanobind.h>
#include <nanobind/operators.h>
#include <nanobind/stl/set.h>     // NOLINT(misc-include-cleaner)
#include <nanobind/stl/string.h>  // NOLINT(misc-include-cleaner)
#include <nanobind/stl/variant.h> // NOLINT(misc-include-cleaner)
#include <nanobind/stl/vector.h>  // NOLINT(misc-include-cleaner)
#include <sstream>
#include <string>
#include <utility>

namespace mqt {

namespace nb = nanobind;
using namespace nb::literals;

namespace {

qc::Qubit nbIntToQubit(const nb::int_& value) {
  const auto valueInt = static_cast<std::int64_t>(value);
  if (valueInt < 0) {
    throw nb::value_error("Qubit index cannot be negative");
  }
  const auto valueUint = static_cast<std::uint64_t>(valueInt);
  if (valueUint > std::numeric_limits<qc::Qubit>::max()) {
    throw nb::value_error("Qubit index exceeds maximum value");
  }
  return static_cast<qc::Qubit>(valueUint);
}

} // namespace

// NOLINTNEXTLINE(misc-use-internal-linkage)
void registerPermutation(const nb::module_& m) {
  nb::class_<qc::Permutation>(
      m, "Permutation",
      nb::sig("class Permutation(collections.abc.MutableMapping[int, int])"),
      R"pb(A class to represent a permutation of the qubits in a quantum circuit.

Args:
    permutation: The permutation to initialize the object with.)pb")

      .def(nb::init<>())

      .def(
          "__init__",
          [](qc::Permutation* self,
             const nb::typed<nb::dict, nb::int_, nb::int_>& p) {
            qc::Permutation perm;
            for (const auto& [key, value] : p) {
              const auto keyQubit = nbIntToQubit(static_cast<nb::int_>(key));
              const auto valueQubit =
                  nbIntToQubit(static_cast<nb::int_>(value));
              perm[keyQubit] = valueQubit;
            }
            new (self) qc::Permutation(std::move(perm));
          },
          "permutation"_a, "Create a permutation from a dictionary.")

      .def("apply",
           nb::overload_cast<const qc::Controls&>(&qc::Permutation::apply,
                                                  nb::const_),
           "controls"_a, R"pb(Apply the permutation to a set of controls.

Args:
    controls: The set of controls to apply the permutation to.

Returns:
    The set of controls with the permutation applied.)pb")

      .def("apply",
           nb::overload_cast<const qc::Targets&>(&qc::Permutation::apply,
                                                 nb::const_),
           "targets"_a, R"pb(Apply the permutation to a list of targets.

Args:
    targets: The list of targets to apply the permutation to.

Returns:
    The list of targets with the permutation applied.)pb")

      .def(
          "clear", [](qc::Permutation& p) { p.clear(); },
          "Clear the permutation of all indices and values.")

      .def(
          "__getitem__",
          [](const qc::Permutation& p, const nb::int_& index) {
            const auto q = nbIntToQubit(index);
            const auto it = p.find(q);
            if (it == p.end()) {
              const auto msg =
                  std::string("Permutation does not contain index ") +
                  std::to_string(q);
              throw nb::key_error(msg.c_str());
            }
            return it->second;
          },
          "index"_a, R"pb(Get the value of the permutation at the given index.

Args:
    index: The index to get the value of the permutation at.

Returns:
    The value of the permutation at the given index.)pb")

      .def(
          "__setitem__",
          [](qc::Permutation& p, const nb::int_& index, const nb::int_& value) {
            const auto q = nbIntToQubit(index);
            const auto r = nbIntToQubit(value);
            p[q] = r;
          },
          "index"_a, "value"_a,
          R"pb(Set the value of the permutation at the given index.

Args:
    index: The index to set the value of the permutation at.
    value: The value to set the permutation at the given index to.)pb")

      .def(
          "__delitem__",
          [](qc::Permutation& p, const nb::int_& index) {
            const auto q = nbIntToQubit(index);
            const auto it = p.find(q);
            if (it == p.end()) {
              // Match Python's KeyError semantics for missing keys.
              const auto msg =
                  std::string("Permutation does not contain index ") +
                  std::to_string(q);
              throw nb::key_error(msg.c_str());
            }
            p.erase(it);
          },
          "index"_a,
          R"pb(Delete the value of the permutation at the given index.

Args:
    index: The index to delete the value of the permutation at.)pb")

      .def("__len__", &qc::Permutation::size,
           "Return the number of indices in the permutation.")

      .def(
          "__iter__",
          [](const qc::Permutation& p) {
            return make_key_iterator(
                nb::type<qc::Permutation>(), "key_iterator", p.begin(), p.end(),
                "Return an iterator over the indices of the permutation.");
          },
          nb::keep_alive<0, 1>())

      .def(
          "items",
          [](const qc::Permutation& p) {
            return make_iterator(
                nb::type<qc::Permutation>(), "item_iterator", p.begin(),
                p.end(),
                "Return an iterable over the items of the permutation.");
          },
          nb::sig("def items(self) -> collections.abc.ItemsView[int, int]"),
          nb::keep_alive<0, 1>())

      .def(nb::self == nb::self,
           nb::sig("def __eq__(self, arg: object, /) -> bool"))
      .def(nb::self != nb::self,
           nb::sig("def __ne__(self, arg: object, /) -> bool"))

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
