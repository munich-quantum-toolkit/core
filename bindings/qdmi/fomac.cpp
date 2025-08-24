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
#include <qdmi/client.h>
#include <pybind11/native_enum.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // NOLINT(misc-include-cleaner)
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
  auto site = py::class_<fomac::Site>(m, "Site");
  site.def("index", &fomac::Site::getIndex);
  site.def("t1", &fomac::Site::getT1);
  site.def("t2", &fomac::Site::getT2);
  site.def("name", &fomac::Site::getName);
  site.def("x_coordinate", &fomac::Site::getXCoordinate);
  site.def("y_coordinate", &fomac::Site::getYCoordinate);
  site.def("z_coordinate", &fomac::Site::getZCoordinate);
  site.def("is_zone", &fomac::Site::isZone);
  site.def("x_extent", &fomac::Site::getXExtent);
  site.def("y_extent", &fomac::Site::getYExtent);
  site.def("z_extent", &fomac::Site::getZExtent);
  site.def("module_index", &fomac::Site::getModuleIndex);
  site.def("submodule_index", &fomac::Site::getSubmoduleIndex);
  site.def("__repr__", [](const fomac::Site& s) {
    return "<Site index=" + std::to_string(s.getIndex()) + "\">";
  });

  auto operation = py::class_<fomac::Operation>(m, "Operation");
  operation.def("name", &fomac::Operation::getName,
                "sites"_a = std::vector<fomac::Site>{},
                "params"_a = std::vector<double>{});
  operation.def("qubits_num", &fomac::Operation::getQubitsNum,
                "sites"_a = std::vector<fomac::Site>{},
                "params"_a = std::vector<double>{});
  operation.def("parameters_num", &fomac::Operation::getParametersNum,
                "sites"_a = std::vector<fomac::Site>{},
                "params"_a = std::vector<double>{});
  operation.def("duration", &fomac::Operation::getDuration,
                "sites"_a = std::vector<fomac::Site>{},
                "params"_a = std::vector<double>{});
  operation.def("fidelity", &fomac::Operation::getFidelity,
                "sites"_a = std::vector<fomac::Site>{},
                "params"_a = std::vector<double>{});
  operation.def("interaction_radius", &fomac::Operation::getInteractionRadius,
                "sites"_a = std::vector<fomac::Site>{},
                "params"_a = std::vector<double>{});
  operation.def("blocking_radius", &fomac::Operation::getBlockingRadius,
                "sites"_a = std::vector<fomac::Site>{},
                "params"_a = std::vector<double>{});
  operation.def("idling_fidelity", &fomac::Operation::getIdlingFidelity,
                "sites"_a = std::vector<fomac::Site>{},
                "params"_a = std::vector<double>{});
  operation.def("is_zoned", &fomac::Operation::isZoned,
                "sites"_a = std::vector<fomac::Site>{},
                "params"_a = std::vector<double>{});
  operation.def("sites", &fomac::Operation::getSites,
                "sites"_a = std::vector<fomac::Site>{},
                "params"_a = std::vector<double>{});
  operation.def("mean_shuttling_speed",
                &fomac::Operation::getMeanShuttlingSpeed,
                "sites"_a = std::vector<fomac::Site>{},
                "params"_a = std::vector<double>{});
  operation.def("__repr__", [](const fomac::Operation& op) {
    return "<Operation name=\"" + op.getName() + "\">";
  });

  auto device = py::class_<fomac::Device>(m, "Device");
  device.def("name", &fomac::Device::getName);
  device.def("version", &fomac::Device::getVersion);
  device.def("status", &fomac::Device::getStatus);
  device.def("library_version", &fomac::Device::getLibraryVersion);
  device.def("qubits_num", &fomac::Device::getQubitsNum);
  device.def("sites", &fomac::Device::getSites);
  device.def("operations", &fomac::Device::getOperations);
  device.def("coupling_map", &fomac::Device::getCouplingMap);
  device.def("needs_calibration", &fomac::Device::getNeedsCalibration);
  device.def("length_unit", &fomac::Device::getLengthUnit);
  device.def("length_scale_factor", &fomac::Device::getLengthScaleFactor);
  device.def("duration_unit", &fomac::Device::getDurationUnit);
  device.def("duration_scale_factor", &fomac::Device::getDurationScaleFactor);
  device.def("min_atom_distance", &fomac::Device::getMinAtomDistance);
  device.def("__repr__", [](const fomac::Device& dev) {
    return "<Device name=\"" + dev.getName() + "\">";
  });

  m.def("devices", &fomac::FoMaC::queryDevices);
}

} // namespace mqt
