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
#include "ir/Register.hpp"

#include <cstddef>
#include <nanobind/nanobind.h>
#include <nanobind/operators.h>
#include <nanobind/stl/string.h> // NOLINT(misc-include-cleaner)
#include <string>

namespace mqt {

namespace nb = nanobind;
using namespace nb::literals;

// NOLINTNEXTLINE(misc-use-internal-linkage)
void registerRegisters(nb::module_& m) {
  nb::class_<qc::QuantumRegister>(m, "QuantumRegister")
      .def(nb::init<const qc::Qubit, const std::size_t, const std::string&>(),
           "start"_a, "size"_a, "name"_a = "")
      .def_prop_ro("name",
                   [](const qc::QuantumRegister& reg) { return reg.getName(); })
      .def_prop_rw(
          "start",
          [](const qc::QuantumRegister& reg) { return reg.getStartIndex(); },
          [](qc::QuantumRegister& reg, const qc::Qubit start) {
            reg.getStartIndex() = start;
          })
      .def_prop_rw(
          "size", [](const qc::QuantumRegister& reg) { return reg.getSize(); },
          [](qc::QuantumRegister& reg, const std::size_t size) {
            reg.getSize() = size;
          })
      .def_prop_ro(
          "end",
          [](const qc::QuantumRegister& reg) { return reg.getEndIndex(); })
      .def(nb::self == nb::self) // NOLINT(misc-redundant-expression)
      .def(nb::self != nb::self) // NOLINT(misc-redundant-expression)
      .def(hash(nb::self))
      .def("__getitem__", &qc::QuantumRegister::getGlobalIndex, "key"_a)
      .def("__contains__", &qc::QuantumRegister::contains)
      .def("__repr__", [](const qc::QuantumRegister& reg) {
        return "QuantumRegister(name=" + reg.getName() +
               ", start=" + std::to_string(reg.getStartIndex()) +
               ", size=" + std::to_string(reg.getSize()) + ")";
      });

  nb::class_<qc::ClassicalRegister>(m, "ClassicalRegister")
      .def(nb::init<const qc::Bit, const std::size_t, const std::string&>(),
           "start"_a, "size"_a, "name"_a = "")
      .def_prop_ro(
          "name",
          [](const qc::ClassicalRegister& reg) { return reg.getName(); })
      .def_prop_rw(
          "start",
          [](const qc::ClassicalRegister& reg) { return reg.getStartIndex(); },
          [](qc::ClassicalRegister& reg, const qc::Bit start) {
            reg.getStartIndex() = start;
          })
      .def_prop_rw(
          "size",
          [](const qc::ClassicalRegister& reg) { return reg.getSize(); },
          [](qc::ClassicalRegister& reg, const std::size_t size) {
            reg.getSize() = size;
          })
      .def_prop_ro(
          "end",
          [](const qc::ClassicalRegister& reg) { return reg.getEndIndex(); })
      .def(nb::self == nb::self) // NOLINT(misc-redundant-expression)
      .def(nb::self != nb::self) // NOLINT(misc-redundant-expression)
      .def(nb::hash(nb::self))
      .def("__getitem__", &qc::ClassicalRegister::getGlobalIndex, "key"_a)
      .def("__contains__", &qc::ClassicalRegister::contains)
      .def("__repr__", [](const qc::ClassicalRegister& reg) {
        return "ClassicalRegister(name=" + reg.getName() +
               ", start=" + std::to_string(reg.getStartIndex()) +
               ", size=" + std::to_string(reg.getSize()) + ")";
      });
}

} // namespace mqt
