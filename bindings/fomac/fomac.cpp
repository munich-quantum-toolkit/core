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
  auto device = py::class_<fomac::FoMaC::Device>(m, "Device");

  py::native_enum<QDMI_Device_Status>(device, "Status", "enum.Enum",
                                      "Enumeration of device status.")
      .value("offline", QDMI_DEVICE_STATUS_OFFLINE)
      .value("idle", QDMI_DEVICE_STATUS_IDLE)
      .value("busy", QDMI_DEVICE_STATUS_BUSY)
      .value("error", QDMI_DEVICE_STATUS_ERROR)
      .value("maintenance", QDMI_DEVICE_STATUS_MAINTENANCE)
      .value("calibration", QDMI_DEVICE_STATUS_CALIBRATION)
      .export_values()
      .finalize();

  auto site = py::class_<fomac::FoMaC::Device::Site>(device, "Site");
  site.def("index", &fomac::FoMaC::Device::Site::getIndex);
  site.def("t1", &fomac::FoMaC::Device::Site::getT1);
  site.def("t2", &fomac::FoMaC::Device::Site::getT2);
  site.def("name", &fomac::FoMaC::Device::Site::getName);
  site.def("x_coordinate", &fomac::FoMaC::Device::Site::getXCoordinate);
  site.def("y_coordinate", &fomac::FoMaC::Device::Site::getYCoordinate);
  site.def("z_coordinate", &fomac::FoMaC::Device::Site::getZCoordinate);
  site.def("is_zone", &fomac::FoMaC::Device::Site::isZone);
  site.def("x_extent", &fomac::FoMaC::Device::Site::getXExtent);
  site.def("y_extent", &fomac::FoMaC::Device::Site::getYExtent);
  site.def("z_extent", &fomac::FoMaC::Device::Site::getZExtent);
  site.def("module_index", &fomac::FoMaC::Device::Site::getModuleIndex);
  site.def("submodule_index", &fomac::FoMaC::Device::Site::getSubmoduleIndex);
  site.def("__repr__", [](const fomac::FoMaC::Device::Site& s) {
    return "<Site index=" + std::to_string(s.getIndex()) + ">";
  });
  site.def(py::self == py::self); // NOLINT(misc-redundant-expression)
  site.def(py::self != py::self); // NOLINT(misc-redundant-expression)

  auto operation =
      py::class_<fomac::FoMaC::Device::Operation>(device, "Operation");
  operation.def("name", &fomac::FoMaC::Device::Operation::getName,
                "sites"_a = std::vector<fomac::FoMaC::Device::Site>{},
                "params"_a = std::vector<double>{});
  operation.def("qubits_num", &fomac::FoMaC::Device::Operation::getQubitsNum,
                "sites"_a = std::vector<fomac::FoMaC::Device::Site>{},
                "params"_a = std::vector<double>{});
  operation.def("parameters_num",
                &fomac::FoMaC::Device::Operation::getParametersNum,
                "sites"_a = std::vector<fomac::FoMaC::Device::Site>{},
                "params"_a = std::vector<double>{});
  operation.def("duration", &fomac::FoMaC::Device::Operation::getDuration,
                "sites"_a = std::vector<fomac::FoMaC::Device::Site>{},
                "params"_a = std::vector<double>{});
  operation.def("fidelity", &fomac::FoMaC::Device::Operation::getFidelity,
                "sites"_a = std::vector<fomac::FoMaC::Device::Site>{},
                "params"_a = std::vector<double>{});
  operation.def("interaction_radius",
                &fomac::FoMaC::Device::Operation::getInteractionRadius,
                "sites"_a = std::vector<fomac::FoMaC::Device::Site>{},
                "params"_a = std::vector<double>{});
  operation.def("blocking_radius",
                &fomac::FoMaC::Device::Operation::getBlockingRadius,
                "sites"_a = std::vector<fomac::FoMaC::Device::Site>{},
                "params"_a = std::vector<double>{});
  operation.def("idling_fidelity",
                &fomac::FoMaC::Device::Operation::getIdlingFidelity,
                "sites"_a = std::vector<fomac::FoMaC::Device::Site>{},
                "params"_a = std::vector<double>{});
  operation.def("is_zoned", &fomac::FoMaC::Device::Operation::isZoned,
                "sites"_a = std::vector<fomac::FoMaC::Device::Site>{},
                "params"_a = std::vector<double>{});
  operation.def("sites", &fomac::FoMaC::Device::Operation::getSites,
                "sites"_a = std::vector<fomac::FoMaC::Device::Site>{},
                "params"_a = std::vector<double>{});
  operation.def("mean_shuttling_speed",
                &fomac::FoMaC::Device::Operation::getMeanShuttlingSpeed,
                "sites"_a = std::vector<fomac::FoMaC::Device::Site>{},
                "params"_a = std::vector<double>{});
  operation.def("__repr__", [](const fomac::FoMaC::Device::Operation& op) {
    return "<Operation name=\"" + op.getName() + "\">";
  });
  operation.def(py::self == py::self); // NOLINT(misc-redundant-expression)
  operation.def(py::self != py::self); // NOLINT(misc-redundant-expression)

  device.def("name", &fomac::FoMaC::Device::getName);
  device.def("version", &fomac::FoMaC::Device::getVersion);
  device.def("status", &fomac::FoMaC::Device::getStatus);
  device.def("library_version", &fomac::FoMaC::Device::getLibraryVersion);
  device.def("qubits_num", &fomac::FoMaC::Device::getQubitsNum);
  device.def("sites", &fomac::FoMaC::Device::getSites);
  device.def("operations", &fomac::FoMaC::Device::getOperations);
  device.def("coupling_map", &fomac::FoMaC::Device::getCouplingMap);
  device.def("needs_calibration", &fomac::FoMaC::Device::getNeedsCalibration);
  device.def("length_unit", &fomac::FoMaC::Device::getLengthUnit);
  device.def("length_scale_factor",
             &fomac::FoMaC::Device::getLengthScaleFactor);
  device.def("duration_unit", &fomac::FoMaC::Device::getDurationUnit);
  device.def("duration_scale_factor",
             &fomac::FoMaC::Device::getDurationScaleFactor);
  device.def("min_atom_distance", &fomac::FoMaC::Device::getMinAtomDistance);
  device.def("__repr__", [](const fomac::FoMaC::Device& dev) {
    return "<Device name=\"" + dev.getName() + "\">";
  });

  m.def("devices", &fomac::FoMaC::getDevices);
}

} // namespace mqt
