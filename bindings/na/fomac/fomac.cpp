/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "fomac/FoMaC.hpp"

#include "na/fomac/Device.hpp"
#include "qdmi/na/Generator.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/operators.h>
#include <nanobind/stl/string.h> // NOLINT(misc-include-cleaner)
#include <nanobind/stl/vector.h> // NOLINT(misc-include-cleaner)
#include <string>

namespace mqt {

namespace nb = nanobind;
using namespace nb::literals;

namespace {

template <typename T>
concept pyClass = requires(T t) { nb::cast(t); };
template <pyClass T> [[nodiscard]] auto repr(T c) -> std::string {
  return nb::repr(nb::cast(c)).c_str();
}

} // namespace

// NOLINTNEXTLINE
NB_MODULE(MQT_CORE_MODULE_NAME, m) {
  nb::module_::import_("mqt.core.fomac");

  auto device =
      nb::class_<na::Session::Device, fomac::Session::Device>(m, "Device");

  auto lattice = nb::class_<na::Device::Lattice>(device, "Lattice");

  auto vector = nb::class_<na::Device::Vector>(lattice, "Vector");
  vector.def_ro("x", &na::Device::Vector::x);
  vector.def_ro("y", &na::Device::Vector::y);
  vector.def("__repr__", [](const na::Device::Vector& v) {
    return "<Vector x=" + std::to_string(v.x) + " y=" + std::to_string(v.y) +
           ">";
  });
  vector.def(nb::self == nb::self); // NOLINT(misc-redundant-expression)
  vector.def(nb::self != nb::self); // NOLINT(misc-redundant-expression)

  auto region = nb::class_<na::Device::Region>(lattice, "Region");

  auto size = nb::class_<na::Device::Region::Size>(region, "Size");
  size.def_ro("width", &na::Device::Region::Size::width);
  size.def_ro("height", &na::Device::Region::Size::height);
  size.def("__repr__", [](const na::Device::Region::Size& s) {
    return "<Size width=" + std::to_string(s.width) +
           " height=" + std::to_string(s.height) + ">";
  });
  size.def(nb::self == nb::self); // NOLINT(misc-redundant-expression)
  size.def(nb::self != nb::self); // NOLINT(misc-redundant-expression)

  region.def_ro("origin", &na::Device::Region::origin);
  region.def_ro("size", &na::Device::Region::size);
  region.def("__repr__", [](const na::Device::Region& r) {
    return "<Region origin=" + repr(r.origin) + " size=" + repr(r.size) + ">";
  });
  region.def(nb::self == nb::self); // NOLINT(misc-redundant-expression)
  region.def(nb::self != nb::self); // NOLINT(misc-redundant-expression)

  lattice.def_ro("lattice_origin", &na::Device::Lattice::latticeOrigin);
  lattice.def_ro("lattice_vector_1", &na::Device::Lattice::latticeVector1);
  lattice.def_ro("lattice_vector_2", &na::Device::Lattice::latticeVector2);
  lattice.def_ro("sublattice_offsets", &na::Device::Lattice::sublatticeOffsets);
  lattice.def_ro("extent", &na::Device::Lattice::extent);
  lattice.def("__repr__", [](const na::Device::Lattice& l) {
    return "<Lattice origin=" + repr(l.latticeOrigin) + ">";
  });
  lattice.def(nb::self == nb::self); // NOLINT(misc-redundant-expression)
  lattice.def(nb::self != nb::self); // NOLINT(misc-redundant-expression)

  device.def_prop_ro("traps", &na::Session::Device::getTraps);
  device.def_prop_ro("t1", [](const na::Session::Device& dev) {
    return dev.getDecoherenceTimes().t1;
  });
  device.def_prop_ro("t2", [](const na::Session::Device& dev) {
    return dev.getDecoherenceTimes().t2;
  });
  device.def("__repr__", [](const fomac::Session::Device& dev) {
    return "<Device name=\"" + dev.getName() + "\">";
  });
  device.def(nb::self == nb::self); // NOLINT(misc-redundant-expression)
  device.def(nb::self != nb::self); // NOLINT(misc-redundant-expression)

  m.def("devices", &na::Session::getDevices);
  device.def_static("try_create_from_device",
                    &na::Session::Device::tryCreateFromDevice, "device"_a);
}

} // namespace mqt
