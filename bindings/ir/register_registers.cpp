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

// These includes must be the first includes for any bindings code
// clang-format off
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // NOLINT(misc-include-cleaner)

#include <pybind11/cast.h>
#include <pybind11/operators.h>
// clang-format on

#include <cstddef>
#include <string>

namespace mqt {

namespace py = pybind11;
using namespace pybind11::literals;

// NOLINTNEXTLINE(misc-use-internal-linkage)
void registerRegisters(py::module& m) {
  py::class_<qc::QuantumRegister>(m, "QuantumRegister")
      .def(py::init<const qc::Qubit, const std::size_t, const std::string&>(),
           "start"_a, "size"_a, "name"_a = "")
      .def_property_readonly(
          "name", [](const qc::QuantumRegister& reg) { return reg.getName(); })
      .def_property(
          "start",
          [](const qc::QuantumRegister& reg) { return reg.getStartIndex(); },
          [](qc::QuantumRegister& reg, const qc::Qubit start) {
            reg.getStartIndex() = start;
          })
      .def_property(
          "size", [](const qc::QuantumRegister& reg) { return reg.getSize(); },
          [](qc::QuantumRegister& reg, const std::size_t size) {
            reg.getSize() = size;
          })
      .def_property_readonly(
          "end",
          [](const qc::QuantumRegister& reg) { return reg.getEndIndex(); })
      .def(py::self == py::self) // NOLINT(misc-redundant-expression)
      .def(py::self != py::self) // NOLINT(misc-redundant-expression)
      .def(hash(py::self))
      .def("__getitem__", &qc::QuantumRegister::getGlobalIndex, "key"_a)
      .def("__contains__", &qc::QuantumRegister::contains)
      .def("__repr__", [](const qc::QuantumRegister& reg) {
        return "QuantumRegister(name=" + reg.getName() +
               ", start=" + std::to_string(reg.getStartIndex()) +
               ", size=" + std::to_string(reg.getSize()) + ")";
      });

  py::class_<qc::ClassicalRegister>(m, "ClassicalRegister")
      .def(py::init<const qc::Bit, const std::size_t, const std::string&>(),
           "start"_a, "size"_a, "name"_a = "")
      .def_property_readonly(
          "name",
          [](const qc::ClassicalRegister& reg) { return reg.getName(); })
      .def_property(
          "start",
          [](const qc::ClassicalRegister& reg) { return reg.getStartIndex(); },
          [](qc::ClassicalRegister& reg, const qc::Bit start) {
            reg.getStartIndex() = start;
          })
      .def_property(
          "size",
          [](const qc::ClassicalRegister& reg) { return reg.getSize(); },
          [](qc::ClassicalRegister& reg, const std::size_t size) {
            reg.getSize() = size;
          })
      .def_property_readonly(
          "end",
          [](const qc::ClassicalRegister& reg) { return reg.getEndIndex(); })
      .def(py::self == py::self) // NOLINT(misc-redundant-expression)
      .def(py::self != py::self) // NOLINT(misc-redundant-expression)
      .def(hash(py::self))
      .def("__getitem__", &qc::ClassicalRegister::getGlobalIndex, "key"_a)
      .def("__contains__", &qc::ClassicalRegister::contains)
      .def("__repr__", [](const qc::ClassicalRegister& reg) {
        return "ClassicalRegister(name=" + reg.getName() +
               ", start=" + std::to_string(reg.getStartIndex()) +
               ", size=" + std::to_string(reg.getSize()) + ")";
      });
}

} // namespace mqt
