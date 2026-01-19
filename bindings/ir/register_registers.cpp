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
#include "ir/Register.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <nanobind/nanobind.h>
#include <nanobind/operators.h>
#include <nanobind/stl/string.h> // NOLINT(misc-include-cleaner)
#include <string>

namespace mqt {

namespace nb = nanobind;
using namespace nb::literals;

// NOLINTNEXTLINE(misc-use-internal-linkage)
void registerRegisters(const nb::module_& m) {
  nb::class_<qc::QuantumRegister>(
      m, "QuantumRegister", R"pb(A class to represent a collection of qubits.

Args:
    start: The starting index of the quantum register.
    size: The number of qubits in the quantum register.
    name: The name of the quantum register. A name will be generated if not provided.)pb")

      .def(nb::init<const qc::Qubit, const std::size_t, const std::string&>(),
           "start"_a, "size"_a, "name"_a = "")

      .def_prop_ro(
          "name", [](const qc::QuantumRegister& reg) { return reg.getName(); },
          "The name of the quantum register.")

      .def_prop_rw(
          "start",
          [](const qc::QuantumRegister& reg) { return reg.getStartIndex(); },
          [](qc::QuantumRegister& reg, const nb::int_& start) {
            const auto startInt = static_cast<std::int64_t>(start);
            if (startInt < 0) {
              throw nb::value_error("Start index cannot be negative");
            }
            const auto startUint = static_cast<std::uint64_t>(startInt);
            if (startUint > std::numeric_limits<qc::Qubit>::max()) {
              throw nb::value_error("Start index exceeds maximum value");
            }
            reg.getStartIndex() = static_cast<qc::Qubit>(startUint);
          },
          "The index of the first qubit in the quantum register.")
      .def_prop_rw(
          "size", [](const qc::QuantumRegister& reg) { return reg.getSize(); },
          [](qc::QuantumRegister& reg, const nb::int_& size) {
            const auto sizeInt = static_cast<std::int64_t>(size);
            if (sizeInt < 0) {
              throw nb::value_error("Size cannot be negative");
            }
            const auto sizeUint = static_cast<std::uint64_t>(sizeInt);
            if (sizeUint > std::numeric_limits<std::size_t>::max()) {
              throw nb::value_error("Size exceeds maximum value");
            }
            reg.getSize() = static_cast<std::size_t>(sizeUint);
          },
          "The number of qubits in the quantum register.")
      .def_prop_ro(
          "end",
          [](const qc::QuantumRegister& reg) { return reg.getEndIndex(); },
          "Index of the last qubit in the quantum register.")

      .def(nb::self == nb::self,
           nb::sig("def __eq__(self, arg: object, /) -> bool"))
      .def(nb::self != nb::self,
           nb::sig("def __ne__(self, arg: object, /) -> bool"))
      .def(nb::hash(nb::self))

      .def(
          "__getitem__",
          [](const qc::QuantumRegister& reg, nb::ssize_t idx) {
            const auto n = static_cast<nb::ssize_t>(reg.getSize());
            if (idx < 0) {
              idx += n;
            }
            if (idx < 0 || idx >= n) {
              throw nb::index_error();
            }
            return reg.getGlobalIndex(static_cast<qc::Qubit>(idx));
          },
          "key"_a, "Get the qubit at the specified index.")

      .def("__contains__", &qc::QuantumRegister::contains, "item"_a,
           "Check if the quantum register contains a qubit.")

      .def("__repr__", [](const qc::QuantumRegister& reg) {
        return "QuantumRegister(name=" + reg.getName() +
               ", start=" + std::to_string(reg.getStartIndex()) +
               ", size=" + std::to_string(reg.getSize()) + ")";
      });

  nb::class_<qc::ClassicalRegister>(
      m, "ClassicalRegister",
      R"pb(A class to represent a collection of classical bits.

Args:
    start: The starting index of the classical register.
    size: The number of bits in the classical register.
    name: The name of the classical register. A name will be generated if not provided.)pb")

      .def(nb::init<const qc::Bit, const std::size_t, const std::string&>(),
           "start"_a, "size"_a, "name"_a = "")

      .def_prop_ro(
          "name",
          [](const qc::ClassicalRegister& reg) { return reg.getName(); },
          "The name of the classical register.")

      .def_prop_rw(
          "start",
          [](const qc::ClassicalRegister& reg) { return reg.getStartIndex(); },
          [](qc::ClassicalRegister& reg, const nb::int_& start) {
            const auto startInt = static_cast<std::int64_t>(start);
            if (startInt < 0) {
              throw nb::value_error("Start index cannot be negative");
            }
            const auto startUint = static_cast<std::uint64_t>(startInt);
            if (startUint > std::numeric_limits<qc::Bit>::max()) {
              throw nb::value_error("Start index exceeds maximum value");
            }
            reg.getStartIndex() = static_cast<qc::Bit>(startUint);
          },
          "The index of the first bit in the classical register.")

      .def_prop_rw(
          "size",
          [](const qc::ClassicalRegister& reg) { return reg.getSize(); },
          [](qc::ClassicalRegister& reg, const nb::int_& size) {
            const auto sizeInt = static_cast<std::int64_t>(size);
            if (sizeInt < 0) {
              throw nb::value_error("Size cannot be negative");
            }
            const auto sizeUint = static_cast<std::uint64_t>(sizeInt);
            if (sizeUint > std::numeric_limits<std::size_t>::max()) {
              throw nb::value_error("Size exceeds maximum value");
            }
            reg.getSize() = static_cast<std::size_t>(sizeUint);
          },
          "The number of bits in the classical register.")

      .def_prop_ro(
          "end",
          [](const qc::ClassicalRegister& reg) { return reg.getEndIndex(); },
          "Index of the last bit in the classical register.")

      // NOLINTNEXTLINE(misc-redundant-expression)
      .def(nb::self == nb::self,
           nb::sig("def __eq__(self, arg: object, /) -> bool"))
      // NOLINTNEXTLINE(misc-redundant-expression)
      .def(nb::self != nb::self,
           nb::sig("def __ne__(self, arg: object, /) -> bool"))
      .def(nb::hash(nb::self))

      .def(
          "__getitem__",
          [](const qc::ClassicalRegister& reg, nb::ssize_t idx) {
            const auto n = static_cast<nb::ssize_t>(reg.getSize());
            if (idx < 0) {
              idx += n;
            }
            if (idx < 0 || idx >= n) {
              throw nb::index_error();
            }
            return reg.getGlobalIndex(static_cast<qc::Bit>(idx));
          },
          "key"_a, "Get the bit at the specified index.")

      .def("__contains__", &qc::ClassicalRegister::contains, "item"_a,
           "Check if the classical register contains a bit.")

      .def("__repr__", [](const qc::ClassicalRegister& reg) {
        return "ClassicalRegister(name=" + reg.getName() +
               ", start=" + std::to_string(reg.getStartIndex()) +
               ", size=" + std::to_string(reg.getSize()) + ")";
      });
}

} // namespace mqt
