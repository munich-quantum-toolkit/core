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

#include <cstdint>
#include <iterator>
#include <nanobind/make_iterator.h>
#include <nanobind/nanobind.h>
#include <nanobind/operators.h>
#include <nanobind/stl/set.h>     // NOLINT(misc-include-cleaner)
#include <nanobind/stl/string.h>  // NOLINT(misc-include-cleaner)
#include <nanobind/stl/variant.h> // NOLINT(misc-include-cleaner)
#include <nanobind/stl/vector.h>  // NOLINT(misc-include-cleaner)
#include <set>
#include <sstream>
#include <utility>
#include <variant>

namespace mqt {

namespace nb = nanobind;
using namespace nb::literals;

using Control = std::variant<qc::Control, std::uint32_t>;
using Controls = std::set<Control>;

namespace {

/// Helper function to convert Control variant to qc::Control
qc::Control getControl(const Control& control) {
  if (std::holds_alternative<qc::Control>(control)) {
    return std::get<qc::Control>(control);
  }
  return static_cast<qc::Control>(std::get<std::uint32_t>(control));
}

/// Helper function to convert Controls variant to qc::Controls
qc::Controls getControls(const Controls& controls) {
  qc::Controls result;
  for (const auto& control : controls) {
    result.insert(getControl(control));
  }
  return result;
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
              perm[nb::cast<qc::Qubit>(key)] = nb::cast<qc::Qubit>(value);
            }
            new (self) qc::Permutation(std::move(perm));
          },
          "permutation"_a, "Create a permutation from a dictionary.")

      .def(
          "apply",
          [](const qc::Permutation& p, const Controls& controls) {
            return p.apply(getControls(controls));
          },
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
          [](const qc::Permutation& p, const qc::Qubit q) { return p.at(q); },
          "index"_a, R"pb(Get the value of the permutation at the given index.

Args:
    index: The index to get the value of the permutation at.

Returns:
    The value of the permutation at the given index.)pb")

      .def(
          "__setitem__",
          [](qc::Permutation& p, const qc::Qubit q, const qc::Qubit r) {
            p[q] = r;
          },
          "index"_a, "value"_a,
          R"pb(Set the value of the permutation at the given index.

Args:
    index: The index to set the value of the permutation at.
    value: The value to set the permutation at the given index to.)pb")

      .def(
          "__delitem__",
          [](qc::Permutation& p, const qc::Qubit q) { p.erase(q); }, "index"_a,
          R"pb(Delete the value of the permutation at the given index.

Args:
    index: The index to delete the value of the permutation at.)pb")

      .def("__len__", &qc::Permutation::size,
           "Return the number of indices in the permutation.")

      .def(
          "__iter__",
          [](const qc::Permutation& p) {
            return nb::make_key_iterator(
                nb::type<qc::Permutation>(), "key_iterator", p.begin(), p.end(),
                "Return an iterator over the indices of the permutation.");
          },
          nb::keep_alive<0, 1>())

      .def(
          "items",
          [](const qc::Permutation& p) {
            return nb::make_iterator(
                nb::type<qc::Permutation>(), "item_iterator", p.begin(),
                p.end(),
                "Return an iterable over the items of the permutation.");
          },
          nb::sig("def items(self) -> collections.abc.ItemsView[int, int]"),
          nb::keep_alive<0, 1>())

      .def(nb::self == nb::self, // NOLINT(misc-redundant-expression)
           nb::sig("def __eq__(self, arg: object, /) -> bool"))
      .def(nb::self != nb::self, // NOLINT(misc-redundant-expression)
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
