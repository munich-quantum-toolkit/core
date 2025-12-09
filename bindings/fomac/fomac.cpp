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

#include "qdmi/Driver.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/operators.h>
#include <nanobind/stl/complex.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <optional>
#include <qdmi/client.h>
#include <string>
#include <vector>

namespace mqt {

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(MQT_CORE_MODULE_NAME, m) {
  // Session class
  auto session = nb::class_<fomac::Session>(m, "Session");
  session.def(
      "__init__",
      [](fomac::Session* self, std::optional<std::string> token,
         std::optional<std::string> authFile,
         std::optional<std::string> authUrl,
         std::optional<std::string> username,
         std::optional<std::string> password,
         std::optional<std::string> projectId,
         std::optional<std::string> custom1, std::optional<std::string> custom2,
         std::optional<std::string> custom3, std::optional<std::string> custom4,
         std::optional<std::string> custom5) {
        fomac::SessionConfig config{token,    authFile,  authUrl, username,
                                    password, projectId, custom1, custom2,
                                    custom3,  custom4,   custom5};
        new (self) fomac::Session(config);
      },
      "token"_a = std::nullopt, "auth_file"_a = std::nullopt,
      "auth_url"_a = std::nullopt, "username"_a = std::nullopt,
      "password"_a = std::nullopt, "project_id"_a = std::nullopt,
      "custom1"_a = std::nullopt, "custom2"_a = std::nullopt,
      "custom3"_a = std::nullopt, "custom4"_a = std::nullopt,
      "custom5"_a = std::nullopt);
  session.def("get_devices", &fomac::Session::getDevices);

  // Job class
  auto job = nb::class_<fomac::Session::Job>(m, "Job");
  job.def("check", &fomac::Session::Job::check);
  job.def("wait", &fomac::Session::Job::wait, "timeout"_a = 0);
  job.def("cancel", &fomac::Session::Job::cancel);
  job.def("get_shots", &fomac::Session::Job::getShots);
  job.def("get_counts", &fomac::Session::Job::getCounts);
  job.def("get_dense_statevector", &fomac::Session::Job::getDenseStateVector);
  job.def("get_dense_probabilities",
          &fomac::Session::Job::getDenseProbabilities);
  job.def("get_sparse_statevector", &fomac::Session::Job::getSparseStateVector);
  job.def("get_sparse_probabilities",
          &fomac::Session::Job::getSparseProbabilities);
  job.def_prop_ro("id", &fomac::Session::Job::getId);
  job.def_prop_ro("program_format", &fomac::Session::Job::getProgramFormat);
  job.def_prop_ro("program", &fomac::Session::Job::getProgram);
  job.def_prop_ro("num_shots", &fomac::Session::Job::getNumShots);
  job.def(nb::self == nb::self);
  job.def(nb::self != nb::self);

  // JobStatus enum
  nb::enum_<QDMI_Job_Status>(job, "Status", "enum.Enum",
                             "Enumeration of job status.")
      .value("CREATED", QDMI_JOB_STATUS_CREATED)
      .value("SUBMITTED", QDMI_JOB_STATUS_SUBMITTED)
      .value("QUEUED", QDMI_JOB_STATUS_QUEUED)
      .value("RUNNING", QDMI_JOB_STATUS_RUNNING)
      .value("DONE", QDMI_JOB_STATUS_DONE)
      .value("CANCELED", QDMI_JOB_STATUS_CANCELED)
      .value("FAILED", QDMI_JOB_STATUS_FAILED)
      .export_values();

  // ProgramFormat enum
  nb::enum_<QDMI_Program_Format>(m, "ProgramFormat", "enum.Enum",
                                 "Enumeration of program formats.")
      .value("QASM2", QDMI_PROGRAM_FORMAT_QASM2)
      .value("QASM3", QDMI_PROGRAM_FORMAT_QASM3)
      .value("QIR_BASE_STRING", QDMI_PROGRAM_FORMAT_QIRBASESTRING)
      .value("QIR_BASE_MODULE", QDMI_PROGRAM_FORMAT_QIRBASEMODULE)
      .value("QIR_ADAPTIVE_STRING", QDMI_PROGRAM_FORMAT_QIRADAPTIVESTRING)
      .value("QIR_ADAPTIVE_MODULE", QDMI_PROGRAM_FORMAT_QIRADAPTIVEMODULE)
      .value("CALIBRATION", QDMI_PROGRAM_FORMAT_CALIBRATION)
      .value("QPY", QDMI_PROGRAM_FORMAT_QPY)
      .value("IQM_JSON", QDMI_PROGRAM_FORMAT_IQMJSON)
      .value("CUSTOM1", QDMI_PROGRAM_FORMAT_CUSTOM1)
      .value("CUSTOM2", QDMI_PROGRAM_FORMAT_CUSTOM2)
      .value("CUSTOM3", QDMI_PROGRAM_FORMAT_CUSTOM3)
      .value("CUSTOM4", QDMI_PROGRAM_FORMAT_CUSTOM4)
      .value("CUSTOM5", QDMI_PROGRAM_FORMAT_CUSTOM5)
      .export_values();

  // Device class
  auto device = nb::class_<fomac::Session::Device>(m, "Device");
  nb::enum_<QDMI_Device_Status>(device, "Status", "enum.Enum",
                                "Enumeration of device status.")
      .value("offline", QDMI_DEVICE_STATUS_OFFLINE)
      .value("idle", QDMI_DEVICE_STATUS_IDLE)
      .value("busy", QDMI_DEVICE_STATUS_BUSY)
      .value("error", QDMI_DEVICE_STATUS_ERROR)
      .value("maintenance", QDMI_DEVICE_STATUS_MAINTENANCE)
      .value("calibration", QDMI_DEVICE_STATUS_CALIBRATION)
      .export_values();
  device.def("name", &fomac::Session::Device::getName);
  device.def("version", &fomac::Session::Device::getVersion);
  device.def("status", &fomac::Session::Device::getStatus);
  device.def("library_version", &fomac::Session::Device::getLibraryVersion);
  device.def("qubits_num", &fomac::Session::Device::getQubitsNum);
  device.def("sites", &fomac::Session::Device::getSites);
  device.def("regular_sites", &fomac::Session::Device::getRegularSites);
  device.def("zones", &fomac::Session::Device::getZones);
  device.def("operations", &fomac::Session::Device::getOperations);
  device.def("coupling_map", &fomac::Session::Device::getCouplingMap);
  device.def("needs_calibration", &fomac::Session::Device::getNeedsCalibration);
  device.def("length_unit", &fomac::Session::Device::getLengthUnit);
  device.def("length_scale_factor",
             &fomac::Session::Device::getLengthScaleFactor);
  device.def("duration_unit", &fomac::Session::Device::getDurationUnit);
  device.def("duration_scale_factor",
             &fomac::Session::Device::getDurationScaleFactor);
  device.def("min_atom_distance", &fomac::Session::Device::getMinAtomDistance);
  device.def("supported_program_formats",
             &fomac::Session::Device::getSupportedProgramFormats);
  device.def("submit_job", &fomac::Session::Device::submitJob, "program"_a,
             "program_format"_a, "num_shots"_a);
  device.def("__repr__", [](const fomac::Session::Device& dev) {
    return "<Device name=\"" + dev.getName() + "\">";
  });
  device.def(nb::self == nb::self);
  device.def(nb::self != nb::self);

  // Site class
  auto site = nb::class_<fomac::Session::Device::Site>(device, "Site");
  site.def("index", &fomac::Session::Device::Site::getIndex);
  site.def("t1", &fomac::Session::Device::Site::getT1);
  site.def("t2", &fomac::Session::Device::Site::getT2);
  site.def("name", &fomac::Session::Device::Site::getName);
  site.def("x_coordinate", &fomac::Session::Device::Site::getXCoordinate);
  site.def("y_coordinate", &fomac::Session::Device::Site::getYCoordinate);
  site.def("z_coordinate", &fomac::Session::Device::Site::getZCoordinate);
  site.def("is_zone", &fomac::Session::Device::Site::isZone);
  site.def("x_extent", &fomac::Session::Device::Site::getXExtent);
  site.def("y_extent", &fomac::Session::Device::Site::getYExtent);
  site.def("z_extent", &fomac::Session::Device::Site::getZExtent);
  site.def("module_index", &fomac::Session::Device::Site::getModuleIndex);
  site.def("submodule_index", &fomac::Session::Device::Site::getSubmoduleIndex);
  site.def("__repr__", [](const fomac::Session::Device::Site& s) {
    return "<Site index=" + std::to_string(s.getIndex()) + ">";
  });
  site.def(nb::self == nb::self);
  site.def(nb::self != nb::self);

  // Operation class
  auto operation =
      nb::class_<fomac::Session::Device::Operation>(device, "Operation");
  operation.def("name", &fomac::Session::Device::Operation::getName,
                "sites"_a = std::vector<fomac::Session::Device::Site>{},
                "params"_a = std::vector<double>{});
  operation.def("qubits_num", &fomac::Session::Device::Operation::getQubitsNum,
                "sites"_a = std::vector<fomac::Session::Device::Site>{},
                "params"_a = std::vector<double>{});
  operation.def("parameters_num",
                &fomac::Session::Device::Operation::getParametersNum,
                "sites"_a = std::vector<fomac::Session::Device::Site>{},
                "params"_a = std::vector<double>{});
  operation.def("duration", &fomac::Session::Device::Operation::getDuration,
                "sites"_a = std::vector<fomac::Session::Device::Site>{},
                "params"_a = std::vector<double>{});
  operation.def("fidelity", &fomac::Session::Device::Operation::getFidelity,
                "sites"_a = std::vector<fomac::Session::Device::Site>{},
                "params"_a = std::vector<double>{});
  operation.def("interaction_radius",
                &fomac::Session::Device::Operation::getInteractionRadius,
                "sites"_a = std::vector<fomac::Session::Device::Site>{},
                "params"_a = std::vector<double>{});
  operation.def("blocking_radius",
                &fomac::Session::Device::Operation::getBlockingRadius,
                "sites"_a = std::vector<fomac::Session::Device::Site>{},
                "params"_a = std::vector<double>{});
  operation.def("idling_fidelity",
                &fomac::Session::Device::Operation::getIdlingFidelity,
                "sites"_a = std::vector<fomac::Session::Device::Site>{},
                "params"_a = std::vector<double>{});
  operation.def("is_zoned", &fomac::Session::Device::Operation::isZoned);
  operation.def("sites", &fomac::Session::Device::Operation::getSites);
  operation.def("site_pairs", &fomac::Session::Device::Operation::getSitePairs);
  operation.def("mean_shuttling_speed",
                &fomac::Session::Device::Operation::getMeanShuttlingSpeed,
                "sites"_a = std::vector<fomac::Session::Device::Site>{},
                "params"_a = std::vector<double>{});
  operation.def("__repr__", [](const fomac::Session::Device::Operation& op) {
    return "<Operation name=\"" + op.getName() + "\">";
  });
  operation.def(nb::self == nb::self);
  operation.def(nb::self != nb::self);

#ifndef _WIN32
  // Module-level function to add dynamic device libraries on non-Windows
  // systems
  m.def(
      "add_dynamic_device_library",
      [](const std::string& libraryPath, const std::string& prefix,
         const std::optional<std::string>& baseUrl = std::nullopt,
         const std::optional<std::string>& token = std::nullopt,
         const std::optional<std::string>& authFile = std::nullopt,
         const std::optional<std::string>& authUrl = std::nullopt,
         const std::optional<std::string>& username = std::nullopt,
         const std::optional<std::string>& password = std::nullopt,
         const std::optional<std::string>& custom1 = std::nullopt,
         const std::optional<std::string>& custom2 = std::nullopt,
         const std::optional<std::string>& custom3 = std::nullopt,
         const std::optional<std::string>& custom4 = std::nullopt,
         const std::optional<std::string>& custom5 =
             std::nullopt) -> fomac::Session::Device {
        const qdmi::DeviceSessionConfig config{.baseUrl = baseUrl,
                                               .token = token,
                                               .authFile = authFile,
                                               .authUrl = authUrl,
                                               .username = username,
                                               .password = password,
                                               .custom1 = custom1,
                                               .custom2 = custom2,
                                               .custom3 = custom3,
                                               .custom4 = custom4,
                                               .custom5 = custom5};
        auto* const qdmiDevice = qdmi::Driver::get().addDynamicDeviceLibrary(
            libraryPath, prefix, config);
        return fomac::Session::Device::fromQDMIDevice(qdmiDevice);
      },
      "library_path"_a, "prefix"_a, "base_url"_a = std::nullopt,
      "token"_a = std::nullopt, "auth_file"_a = std::nullopt,
      "auth_url"_a = std::nullopt, "username"_a = std::nullopt,
      "password"_a = std::nullopt, "custom1"_a = std::nullopt,
      "custom2"_a = std::nullopt, "custom3"_a = std::nullopt,
      "custom4"_a = std::nullopt, "custom5"_a = std::nullopt);
#endif // _WIN32
}

} // namespace mqt
