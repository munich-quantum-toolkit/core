/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

// These includes must be the first includes for any bindings code
// clang-format off
#include "fomac/FoMaC.hpp"
#include "na/fomac/Device.hpp"
#include "na/device/Generator.hpp"

#include <pybind11/cast.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h> // NOLINT(misc-include-cleaner)
#include <string>
// clang-format on

namespace mqt {

namespace py = pybind11;
using namespace py::literals;

namespace {
template <typename T>
concept pyClass = requires(T t) { py::cast(t); };
template <pyClass T> [[nodiscard]] auto repr(T c) -> std::string {
  return py::repr(py::cast(c)).template cast<std::string>();
}
} // namespace

// The definition of the (in-)equality operators produces warnings in clang-tidy
// which are ignored by the following comment
// NOLINTBEGIN(misc-redundant-expression)
PYBIND11_MODULE(MQT_CORE_MODULE_NAME, m, py::mod_gil_not_used()) {
  pybind11::module_::import("mqt.core.fomac");

  auto device =
      py::class_<na::FoMaC::Device, fomac::FoMaC::Device>(m, "Device");

  auto lattice = py::class_<na::Device::Lattice>(device, "Lattice");

  auto vector = py::class_<na::Device::Vector>(lattice, "Vector");
  vector.def_readonly("x", &na::Device::Vector::x);
  vector.def_readonly("y", &na::Device::Vector::y);
  vector.def("__repr__", [](const na::Device::Vector& v) {
    return "<Vector x=" + std::to_string(v.x) + " y=" + std::to_string(v.y) +
           ">";
  });
  vector.def(py::self == py::self);
  vector.def(py::self != py::self);

  auto region = py::class_<na::Device::Region>(lattice, "Region");

  auto size = py::class_<na::Device::Region::Size>(region, "Size");
  size.def_readonly("width", &na::Device::Region::Size::width);
  size.def_readonly("height", &na::Device::Region::Size::height);
  size.def("__repr__", [](const na::Device::Region::Size& s) {
    return "<Size width=" + std::to_string(s.width) +
           " height=" + std::to_string(s.height) + ">";
  });
  size.def(py::self == py::self);
  size.def(py::self != py::self);

  region.def_readonly("origin", &na::Device::Region::origin);
  region.def_readonly("size", &na::Device::Region::size);
  region.def("__repr__", [](const na::Device::Region& r) {
    return "<Region origin=" + repr(r.origin) + " size=" + repr(r.size) + ">";
  });
  region.def(py::self == py::self);
  region.def(py::self != py::self);

  lattice.def_readonly("lattice_origin", &na::Device::Lattice::latticeOrigin);
  lattice.def_readonly("lattice_vector_1",
                       &na::Device::Lattice::latticeVector1);
  lattice.def_readonly("lattice_vector_2",
                       &na::Device::Lattice::latticeVector2);
  lattice.def_readonly("sublattice_offsets",
                       &na::Device::Lattice::sublatticeOffsets);
  lattice.def_readonly("extent", &na::Device::Lattice::extent);
  lattice.def("__repr__", [](const na::Device::Lattice& l) {
    return "<Lattice origin=" +
           py::repr(py::cast(l.latticeOrigin)).cast<std::string>() + ">";
  });
  lattice.def(py::self == py::self);
  lattice.def(py::self != py::self);

  device.def_property_readonly("traps", &na::FoMaC::Device::getTraps);
  device.def_property_readonly("t1", [](const na::FoMaC::Device& dev) {
    return dev.getDecoherenceTimes().t1;
  });
  device.def_property_readonly("t2", [](const na::FoMaC::Device& dev) {
    return dev.getDecoherenceTimes().t2;
  });
  device.def("__repr__", [](const fomac::FoMaC::Device& dev) {
    return "<Device name=\"" + dev.getName() + "\">";
  });
  device.def(py::self == py::self);
  device.def(py::self != py::self);

  m.def("devices", &na::FoMaC::getDevices);
}
// NOLINTEND(misc-redundant-expression)
} // namespace mqt
