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
#include "qdmi/FoMaC.hpp"
#include <pybind11/cast.h>
#include <pybind11/operators.h>
#include <qdmi/client.h>
#include <pybind11/native_enum.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // NOLINT(misc-include-cleaner)
#include <string>
#include <vector>
// clang-format on

namespace mqt {

namespace py = pybind11;
using namespace py::literals;

PYBIND11_MODULE(MQT_CORE_MODULE_NAME, m, py::mod_gil_not_used()) {
  py::native_enum<QDMI_Device_Status>(m, "DeviceStatus", "enum.Enum",
                                      "Enumeration of device status.")
      .value("offline", QDMI_DEVICE_STATUS_OFFLINE)
      .value("idle", QDMI_DEVICE_STATUS_IDLE)
      .value("busy", QDMI_DEVICE_STATUS_BUSY)
      .value("error", QDMI_DEVICE_STATUS_ERROR)
      .value("maintenance", QDMI_DEVICE_STATUS_MAINTENANCE)
      .value("calibration", QDMI_DEVICE_STATUS_CALIBRATION)
      .export_values()
      .finalize();
  auto site = py::class_<qdmi::FoMaC::Device::Site>(m, "Site");
  site.def("index", &qdmi::FoMaC::Device::Site::getIndex);
  site.def("t1", &qdmi::FoMaC::Device::Site::getT1);
  site.def("t2", &qdmi::FoMaC::Device::Site::getT2);
  site.def("name", &qdmi::FoMaC::Device::Site::getName);
  site.def("x_coordinate", &qdmi::FoMaC::Device::Site::getXCoordinate);
  site.def("y_coordinate", &qdmi::FoMaC::Device::Site::getYCoordinate);
  site.def("z_coordinate", &qdmi::FoMaC::Device::Site::getZCoordinate);
  site.def("is_zone", &qdmi::FoMaC::Device::Site::isZone);
  site.def("x_extent", &qdmi::FoMaC::Device::Site::getXExtent);
  site.def("y_extent", &qdmi::FoMaC::Device::Site::getYExtent);
  site.def("z_extent", &qdmi::FoMaC::Device::Site::getZExtent);
  site.def("module_index", &qdmi::FoMaC::Device::Site::getModuleIndex);
  site.def("submodule_index", &qdmi::FoMaC::Device::Site::getSubmoduleIndex);
  site.def("__repr__", [](const qdmi::FoMaC::Device::Site& s) {
    return "<Site index=" + std::to_string(s.getIndex()) + ">";
  });
  site.def(py::self == py::self); // NOLINT(misc-redundant-expression)
  site.def(py::self != py::self); // NOLINT(misc-redundant-expression)

  auto operation = py::class_<qdmi::FoMaC::Device::Operation>(m, "Operation");
  operation.def("name", &qdmi::FoMaC::Device::Operation::getName,
                "sites"_a = std::vector<qdmi::FoMaC::Device::Site>{},
                "params"_a = std::vector<double>{});
  operation.def("qubits_num", &qdmi::FoMaC::Device::Operation::getQubitsNum,
                "sites"_a = std::vector<qdmi::FoMaC::Device::Site>{},
                "params"_a = std::vector<double>{});
  operation.def("parameters_num",
                &qdmi::FoMaC::Device::Operation::getParametersNum,
                "sites"_a = std::vector<qdmi::FoMaC::Device::Site>{},
                "params"_a = std::vector<double>{});
  operation.def("duration", &qdmi::FoMaC::Device::Operation::getDuration,
                "sites"_a = std::vector<qdmi::FoMaC::Device::Site>{},
                "params"_a = std::vector<double>{});
  operation.def("fidelity", &qdmi::FoMaC::Device::Operation::getFidelity,
                "sites"_a = std::vector<qdmi::FoMaC::Device::Site>{},
                "params"_a = std::vector<double>{});
  operation.def("interaction_radius",
                &qdmi::FoMaC::Device::Operation::getInteractionRadius,
                "sites"_a = std::vector<qdmi::FoMaC::Device::Site>{},
                "params"_a = std::vector<double>{});
  operation.def("blocking_radius",
                &qdmi::FoMaC::Device::Operation::getBlockingRadius,
                "sites"_a = std::vector<qdmi::FoMaC::Device::Site>{},
                "params"_a = std::vector<double>{});
  operation.def("idling_fidelity",
                &qdmi::FoMaC::Device::Operation::getIdlingFidelity,
                "sites"_a = std::vector<qdmi::FoMaC::Device::Site>{},
                "params"_a = std::vector<double>{});
  operation.def("is_zoned", &qdmi::FoMaC::Device::Operation::isZoned,
                "sites"_a = std::vector<qdmi::FoMaC::Device::Site>{},
                "params"_a = std::vector<double>{});
  operation.def("sites", &qdmi::FoMaC::Device::Operation::getSites,
                "sites"_a = std::vector<qdmi::FoMaC::Device::Site>{},
                "params"_a = std::vector<double>{});
  operation.def("mean_shuttling_speed",
                &qdmi::FoMaC::Device::Operation::getMeanShuttlingSpeed,
                "sites"_a = std::vector<qdmi::FoMaC::Device::Site>{},
                "params"_a = std::vector<double>{});
  operation.def("__repr__", [](const qdmi::FoMaC::Device::Operation& op) {
    return "<Operation name=\"" + op.getName() + "\">";
  });
  operation.def(py::self == py::self); // NOLINT(misc-redundant-expression)
  operation.def(py::self != py::self); // NOLINT(misc-redundant-expression)

  auto device = py::class_<qdmi::FoMaC::Device>(m, "Device");
  device.def("name", &qdmi::FoMaC::Device::getName);
  device.def("version", &qdmi::FoMaC::Device::getVersion);
  device.def("status", &qdmi::FoMaC::Device::getStatus);
  device.def("library_version", &qdmi::FoMaC::Device::getLibraryVersion);
  device.def("qubits_num", &qdmi::FoMaC::Device::getQubitsNum);
  device.def("sites", &qdmi::FoMaC::Device::getSites);
  device.def("operations", &qdmi::FoMaC::Device::getOperations);
  device.def("coupling_map", &qdmi::FoMaC::Device::getCouplingMap);
  device.def("needs_calibration", &qdmi::FoMaC::Device::getNeedsCalibration);
  device.def("length_unit", &qdmi::FoMaC::Device::getLengthUnit);
  device.def("length_scale_factor", &qdmi::FoMaC::Device::getLengthScaleFactor);
  device.def("duration_unit", &qdmi::FoMaC::Device::getDurationUnit);
  device.def("duration_scale_factor",
             &qdmi::FoMaC::Device::getDurationScaleFactor);
  device.def("min_atom_distance", &qdmi::FoMaC::Device::getMinAtomDistance);
  device.def("__repr__", [](const qdmi::FoMaC::Device& dev) {
    return "<Device name=\"" + dev.getName() + "\">";
  });
  device.def(py::self == py::self); // NOLINT(misc-redundant-expression)
  device.def(py::self != py::self); // NOLINT(misc-redundant-expression)

  m.def("devices", &qdmi::FoMaC::getDevices);
}

} // namespace mqt
