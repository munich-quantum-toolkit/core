/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "qdmi/Client.hpp"
#include "qdmi/common/Common.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/operators.h>
#include <nanobind/stl/complex.h>    // NOLINT(misc-include-cleaner)
#include <nanobind/stl/filesystem.h> // NOLINT(misc-include-cleaner)
#include <nanobind/stl/map.h>        // NOLINT(misc-include-cleaner)
#include <nanobind/stl/optional.h>   // NOLINT(misc-include-cleaner)
#include <nanobind/stl/pair.h>       // NOLINT(misc-include-cleaner)
#include <nanobind/stl/string.h>     // NOLINT(misc-include-cleaner)
#include <nanobind/stl/variant.h>    // NOLINT(misc-include-cleaner)
#include <nanobind/stl/vector.h>     // NOLINT(misc-include-cleaner)
#include <qdmi/client.h>

#include <cstddef>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace mqt {

namespace nb = nanobind;
using namespace nb::literals;

namespace bindings {
void registerSlurm(nb::module_& qdmiModule);
}

namespace {
qdmi::SessionConfig makeClientSessionConfig(
    std::optional<std::filesystem::path> driverPath,
    std::optional<std::string> token,
    std::optional<std::filesystem::path> authFile,
    std::optional<std::string> authUrl, std::optional<std::string> username,
    std::optional<std::string> password, std::optional<std::string> projectId,
    std::optional<std::string> custom1, std::optional<std::string> custom2,
    std::optional<std::string> custom3, std::optional<std::string> custom4,
    std::optional<std::string> custom5) {
  return {
      .driverPath = std::move(driverPath),
      .token = std::move(token),
      .authFile = std::move(authFile),
      .authUrl = std::move(authUrl),
      .username = std::move(username),
      .password = std::move(password),
      .projectId = std::move(projectId),
      .custom1 = std::move(custom1),
      .custom2 = std::move(custom2),
      .custom3 = std::move(custom3),
      .custom4 = std::move(custom4),
      .custom5 = std::move(custom5),
  };
}

[[nodiscard]] auto makeDeviceSessionJson(
    const std::optional<std::string>& baseUrl,
    const std::optional<std::string>& token,
    const std::optional<std::filesystem::path>& authFile,
    const std::optional<std::string>& authUrl,
    const std::optional<std::string>& username,
    const std::optional<std::string>& password,
    const std::optional<std::string>& deviceConfig,
    const std::optional<std::filesystem::path>& deviceConfigFile,
    const std::optional<std::string>& custom1,
    const std::optional<std::string>& custom2,
    const std::optional<std::string>& custom3,
    const std::optional<std::string>& custom4,
    const std::optional<std::string>& custom5) -> std::string {
  if (deviceConfig && deviceConfigFile) {
    throw nb::value_error(
        "device_config and device_config_file are mutually exclusive");
  }
  nb::dict session;
  const auto setString = [&session](const char* key,
                                    const std::optional<std::string>& value) {
    if (value) {
      session[key] = *value;
    }
  };
  setString("base-url", baseUrl);
  setString("token", token);
  if (authFile) {
    session["auth-file"] = qdmi::detail::pathToUtf8(*authFile);
  }
  setString("auth-url", authUrl);
  setString("username", username);
  setString("password", password);
  setString("custom1", custom1);
  setString("custom2", custom2);
  setString("custom3", custom3);
  setString("custom4", custom4);
  setString("custom5", custom5);
  if (deviceConfig) {
    nb::dict source;
    source["inline"] =
        nb::module_::import_("json").attr("loads")(*deviceConfig);
    session["device-config"] = std::move(source);
  } else if (deviceConfigFile) {
    nb::dict source;
    source["file"] = qdmi::detail::pathToUtf8(*deviceConfigFile);
    session["device-config"] = std::move(source);
  }
  if (session.empty()) {
    return {};
  }
  return nb::cast<std::string>(
      nb::module_::import_("json").attr("dumps")(session));
}

template <typename Query>
[[nodiscard]] nb::object queryCustomValue(Query query,
                                          const nb::handle valueType) {
  const auto returnValue =
      []<typename T>(std::optional<T> value) -> nb::object {
    if (!value.has_value()) {
      return nb::none();
    }
    return nb::cast(std::move(*value));
  };

  const auto builtins = nb::builtins();
  if (valueType.is(builtins["str"])) {
    return returnValue(query.template operator()<std::string>());
  }
  if (valueType.is(builtins["bool"])) {
    return returnValue(query.template operator()<bool>());
  }
  if (valueType.is(builtins["int"])) {
    return returnValue(query.template operator()<int>());
  }
  if (valueType.is(builtins["float"])) {
    return returnValue(query.template operator()<double>());
  }
  if (valueType.is(builtins["bytes"])) {
    const auto value = query.template operator()<std::vector<std::byte>>();
    if (!value.has_value()) {
      return nb::none();
    }
    return nb::bytes(reinterpret_cast<const char*>(value->data()),
                     value->size());
  }
  throw nb::type_error(
      "value_type must be exactly str, bool, int, float, or bytes");
}

} // namespace

NB_MODULE(MQT_CORE_MODULE_NAME, qdmiModule) {
  qdmiModule.doc() = "QDMI Client entities and MQT Core's default driver.";
  auto defaultDriver = qdmiModule.def_submodule(
      "default_driver", "Configure MQT Core's packaged QDMI Client driver.");
  bindings::registerSlurm(qdmiModule);

  nb::class_<qdmi::Session>(qdmiModule, "ClientSession",
                            "One initialized QDMI Client session.")
      .def(
          "__init__",
          [](qdmi::Session* self,
             std::optional<std::filesystem::path> driverPath,
             std::optional<std::string> token,
             std::optional<std::filesystem::path> authFile,
             std::optional<std::string> authUrl,
             std::optional<std::string> username,
             std::optional<std::string> password,
             std::optional<std::string> projectId,
             std::optional<std::string> custom1,
             std::optional<std::string> custom2,
             std::optional<std::string> custom3,
             std::optional<std::string> custom4,
             std::optional<std::string> custom5) {
            new (self) qdmi::Session(makeClientSessionConfig(
                std::move(driverPath), std::move(token), std::move(authFile),
                std::move(authUrl), std::move(username), std::move(password),
                std::move(projectId), std::move(custom1), std::move(custom2),
                std::move(custom3), std::move(custom4), std::move(custom5)));
          },
          nb::kw_only(), "driver_path"_a = std::nullopt,
          "token"_a = std::nullopt, "auth_file"_a = std::nullopt,
          "auth_url"_a = std::nullopt, "username"_a = std::nullopt,
          "password"_a = std::nullopt, "project_id"_a = std::nullopt,
          "custom1"_a = std::nullopt, "custom2"_a = std::nullopt,
          "custom3"_a = std::nullopt, "custom4"_a = std::nullopt,
          "custom5"_a = std::nullopt)
      .def_prop_ro("devices", &qdmi::Session::getDevices,
                   "The devices visible to this authenticated session.");

  qdmiModule.def(
      "open_device",
      [](const std::string& deviceId,
         std::optional<std::filesystem::path> driverPath,
         std::optional<std::string> token,
         std::optional<std::filesystem::path> authFile,
         std::optional<std::string> authUrl,
         std::optional<std::string> username,
         std::optional<std::string> password,
         std::optional<std::string> projectId,
         std::optional<std::string> custom1, std::optional<std::string> custom2,
         std::optional<std::string> custom3, std::optional<std::string> custom4,
         std::optional<std::string> custom5) {
        return qdmi::Session::openDevice(
            deviceId,
            makeClientSessionConfig(
                std::move(driverPath), std::move(token), std::move(authFile),
                std::move(authUrl), std::move(username), std::move(password),
                std::move(projectId), std::move(custom1), std::move(custom2),
                std::move(custom3), std::move(custom4), std::move(custom5)));
      },
      "device_id"_a, nb::kw_only(), "driver_path"_a = std::nullopt,
      "token"_a = std::nullopt, "auth_file"_a = std::nullopt,
      "auth_url"_a = std::nullopt, "username"_a = std::nullopt,
      "password"_a = std::nullopt, "project_id"_a = std::nullopt,
      "custom1"_a = std::nullopt, "custom2"_a = std::nullopt,
      "custom3"_a = std::nullopt, "custom4"_a = std::nullopt,
      "custom5"_a = std::nullopt,
      "Open a Client-visible device by stable ID in a fresh session.");

  // Job class
  auto job = nb::class_<qdmi::Job>(
      qdmiModule, "Job",
      "A job represents a submitted quantum program execution.");

  job.def("check", &qdmi::Job::check, "Returns the current status of the job.");

  job.def("wait", &qdmi::Job::wait, "timeout"_a = 0,
          nb::call_guard<nb::gil_scoped_release>(),
          R"pb(Waits for the job to complete.

Args:
    timeout: The maximum time to wait in seconds. If 0, waits indefinitely.

Returns:
    True if the job completed within the timeout, False otherwise.)pb");

  job.def("cancel", &qdmi::Job::cancel, "Cancels the job.");

  job.def("get_shots", &qdmi::Job::getShots,
          nb::call_guard<nb::gil_scoped_release>(),
          "Returns the raw shot results from the job.");

  job.def("get_counts", &qdmi::Job::getCounts,
          nb::call_guard<nb::gil_scoped_release>(),
          "Returns the measurement counts from the job.");

  job.def("get_dense_statevector", &qdmi::Job::getDenseStateVector,
          "Returns the dense statevector from the job (typically only "
          "available from simulator devices).");

  job.def("get_dense_probabilities", &qdmi::Job::getDenseProbabilities,
          "Returns the dense probabilities from the job (typically only "
          "available from simulator devices).");

  job.def("get_sparse_statevector", &qdmi::Job::getSparseStateVector,
          "Returns the sparse statevector from the job (typically only "
          "available from simulator devices).");

  job.def("get_sparse_probabilities", &qdmi::Job::getSparseProbabilities,
          "Returns the sparse probabilities from the job (typically only "
          "available from simulator devices).");

  job.def(
      "query_custom_property",
      [](const qdmi::Job& self, const qdmi::CustomProperty customProperty,
         const nb::handle valueType) {
        return queryCustomValue(
            [&self, customProperty]<qdmi::custom_property_value T> {
              return self.queryCustomProperty<T>(customProperty);
            },
            valueType);
      },
      "custom_property"_a, "value_type"_a,
      nb::sig("def query_custom_property(self, custom_property: "
              "CustomProperty, "
              "value_type: type[str] | type[bool] | type[int] | type[float] | "
              "type[bytes]) -> str | bool | int | float | bytes | None"),
      R"pb(Query an implementation-defined custom job property.

The caller must provide the type documented by the device implementation.
Use ``bytes`` to retrieve the value without interpretation. Returns ``None``
when the custom slot is unsupported.)pb");

  job.def(
      "get_custom_result",
      [](const qdmi::Job& self, const qdmi::CustomProperty customProperty,
         const nb::handle valueType) {
        return queryCustomValue(
            [&self, customProperty]<qdmi::custom_property_value T> {
              return self.getCustomResult<T>(customProperty);
            },
            valueType);
      },
      "custom_property"_a, "value_type"_a,
      nb::sig("def get_custom_result(self, custom_property: CustomProperty, "
              "value_type: type[str] | type[bool] | type[int] | type[float] | "
              "type[bytes]) -> str | bool | int | float | bytes | None"),
      R"pb(Return an implementation-defined custom job result.

The caller must provide the type documented by the device implementation.
Use ``bytes`` to retrieve the value without interpretation. Returns ``None``
when the custom slot is unsupported.)pb");

  job.def_prop_ro("id", &qdmi::Job::getId, "The job ID.");

  job.def_prop_ro("program_format", &qdmi::Job::getProgramFormat,
                  "The format of the submitted program.");

  job.def_prop_ro("program", &qdmi::Job::getProgram, "The submitted program.");

  job.def_prop_ro(
      "program_bytes",
      [](const qdmi::Job& self) {
        const auto program = self.getProgramBytes();
        return nb::bytes(program.data(), program.size());
      },
      "The exact bytes of the submitted program.");

  job.def_prop_ro("num_shots", &qdmi::Job::getNumShots, "The number of shots.");

  job.def_prop_ro(
      "queue_position", &qdmi::Job::getQueuePosition,
      "The number of jobs ahead in the queue, or None if unavailable or not "
      "applicable in the current state.");

  job.def(nb::self == nb::self,
          nb::sig("def __eq__(self, arg: object, /) -> bool"));
  job.def(nb::self != nb::self,
          nb::sig("def __ne__(self, arg: object, /) -> bool"));

  // JobStatus enum
  nb::enum_<QDMI_Job_Status>(job, "Status", "Enumeration of job status.")
      .value("CREATED", QDMI_JOB_STATUS_CREATED)
      .value("SUBMITTED", QDMI_JOB_STATUS_SUBMITTED)
      .value("QUEUED", QDMI_JOB_STATUS_QUEUED)
      .value("RUNNING", QDMI_JOB_STATUS_RUNNING)
      .value("DONE", QDMI_JOB_STATUS_DONE)
      .value("CANCELED", QDMI_JOB_STATUS_CANCELED)
      .value("FAILED", QDMI_JOB_STATUS_FAILED);

  // ProgramFormat enum
  nb::enum_<QDMI_Program_Format>(qdmiModule, "ProgramFormat",
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
      .value("BATCH_JOB", QDMI_PROGRAM_FORMAT_BATCHJOB)
      .value("CUSTOM1", QDMI_PROGRAM_FORMAT_CUSTOM1)
      .value("CUSTOM2", QDMI_PROGRAM_FORMAT_CUSTOM2)
      .value("CUSTOM3", QDMI_PROGRAM_FORMAT_CUSTOM3)
      .value("CUSTOM4", QDMI_PROGRAM_FORMAT_CUSTOM4)
      .value("CUSTOM5", QDMI_PROGRAM_FORMAT_CUSTOM5);

  qdmiModule.def("is_binary_program_format", &qdmi::isBinaryProgramFormat,
                 "program_format"_a,
                 R"pb(Returns whether a program format carries a binary payload.

``QIR_BASE_MODULE``, ``QIR_ADAPTIVE_MODULE``, and ``QPY`` hold bitcode or
another serialized object. Such a payload may contain a null byte and is not
text, so the device must receive it as exact bytes. Pass ``bytes`` to
:meth:`Device.submit_job` for these formats and ``str`` for the others.

Args:
    program_format: The program format to classify.

Returns:
    True if the format requires exact-byte submission.)pb");

  nb::enum_<qdmi::CustomProperty>(
      qdmiModule, "CustomProperty",
      "An implementation-defined custom property or result slot.")
      .value("CUSTOM1", qdmi::CustomProperty::Custom1)
      .value("CUSTOM2", qdmi::CustomProperty::Custom2)
      .value("CUSTOM3", qdmi::CustomProperty::Custom3)
      .value("CUSTOM4", qdmi::CustomProperty::Custom4)
      .value("CUSTOM5", qdmi::CustomProperty::Custom5);

  // Device class
  auto device = nb::class_<qdmi::Device>(
      qdmiModule, "Device",
      "A device represents a quantum device with its properties and "
      "capabilities.");

  nb::enum_<QDMI_Device_Status>(device, "Status",
                                "Enumeration of device status.")
      .value("OFFLINE", QDMI_DEVICE_STATUS_OFFLINE)
      .value("IDLE", QDMI_DEVICE_STATUS_IDLE)
      .value("BUSY", QDMI_DEVICE_STATUS_BUSY)
      .value("ERROR", QDMI_DEVICE_STATUS_ERROR)
      .value("MAINTENANCE", QDMI_DEVICE_STATUS_MAINTENANCE)
      .value("CALIBRATION", QDMI_DEVICE_STATUS_CALIBRATION);

  device.def("name", &qdmi::Device::getName, "Returns the name of the device.");

  device.def_prop_ro("id", &qdmi::Device::getId,
                     "The stable Client-visible device ID.");

  device.def("version", &qdmi::Device::getVersion,
             "Returns the version of the device.");

  device.def("status", &qdmi::Device::getStatus,
             "Returns the current status of the device.");

  device.def("library_version", &qdmi::Device::getLibraryVersion,
             "Returns the version of the library used to define the device.");

  device.def("qubits_num", &qdmi::Device::getQubitsNum,
             "Returns the number of qubits available on the device.");

  device.def("sites", &qdmi::Device::getSites,
             "Returns the list of all sites (zone and regular sites) available "
             "on the device.");

  device.def("regular_sites", &qdmi::Device::getRegularSites,
             "Returns the list of regular sites (without zone sites) available "
             "on the device.");

  device.def("zones", &qdmi::Device::getZones,
             "Returns the list of zone sites (without regular sites) available "
             "on the device.");

  device.def("operations", &qdmi::Device::getOperations,
             "Returns the list of operations supported by the device.");

  device.def("coupling_map", &qdmi::Device::getCouplingMap,
             "Returns the coupling map of the device as a list of site pairs.");

  device.def("needs_calibration", &qdmi::Device::getNeedsCalibration,
             "Returns whether the device needs calibration.");

  device.def("queue_length", &qdmi::Device::getQueueLength,
             "Returns the current queue length, or None if unavailable.");

  device.def("length_unit", &qdmi::Device::getLengthUnit,
             "Returns the unit of length used by the device.");

  device.def("length_scale_factor", &qdmi::Device::getLengthScaleFactor,
             "Returns the scale factor for length used by the device.");

  device.def("duration_unit", &qdmi::Device::getDurationUnit,
             "Returns the unit of duration used by the device.");

  device.def("duration_scale_factor", &qdmi::Device::getDurationScaleFactor,
             "Returns the scale factor for duration used by the device.");

  device.def("min_atom_distance", &qdmi::Device::getMinAtomDistance,
             "Returns the minimum atom distance on the device.");

  device.def("supported_program_formats",
             &qdmi::Device::getSupportedProgramFormats,
             "Returns the list of program formats supported by the device.");

  device.def("child_devices", &qdmi::Device::getChildDevices,
             "Returns the direct child devices managed by this device.");

  device.def(
      "query_custom_operations", &qdmi::Device::queryCustomOperations,
      "custom_property"_a,
      R"pb(Query a custom device property that contains operation handles.

Returns normal :class:`Device.Operation` objects, or ``None`` when the custom
slot is unsupported. A supported empty list is returned as an empty list.)pb");

  device.def(
      "query_custom_property",
      [](const qdmi::Device& self, const qdmi::CustomProperty customProperty,
         const nb::handle valueType) {
        return queryCustomValue(
            [&self, customProperty]<qdmi::custom_property_value T> {
              return self.queryCustomProperty<T>(customProperty);
            },
            valueType);
      },
      "custom_property"_a, "value_type"_a,
      nb::sig("def query_custom_property(self, custom_property: "
              "CustomProperty, "
              "value_type: type[str] | type[bool] | type[int] | type[float] | "
              "type[bytes]) -> str | bool | int | float | bytes | None"),
      R"pb(Query an implementation-defined custom device property.

The caller must provide the type documented by the device implementation.
Use ``bytes`` to retrieve the value without interpretation. Returns ``None``
when the custom slot is unsupported.)pb");

  device.def(
      "submit_job",
      [](const qdmi::Device& self, const std::string& program,
         const QDMI_Program_Format format, const std::optional<size_t> numShots,
         const std::optional<qdmi::CustomJobParameter>& custom1,
         const std::optional<qdmi::CustomJobParameter>& custom2,
         const std::optional<qdmi::CustomJobParameter>& custom3,
         const std::optional<qdmi::CustomJobParameter>& custom4,
         const std::optional<qdmi::CustomJobParameter>& custom5) {
        if (numShots.has_value()) {
          return self.submitJob(program, format, *numShots, custom1, custom2,
                                custom3, custom4, custom5);
        }
        return self.submitJob(program, format, custom1, custom2, custom3,
                              custom4, custom5);
      },
      "program"_a, "program_format"_a, "num_shots"_a = nb::none(),
      nb::kw_only(), "custom1"_a = nb::none(), "custom2"_a = nb::none(),
      "custom3"_a = nb::none(), "custom4"_a = nb::none(),
      "custom5"_a = nb::none(), nb::rv_policy::reference_internal,
      "Submits a text job to the device.");

  device.def(
      "submit_job",
      [](const qdmi::Device& self, const nb::bytes& program,
         const QDMI_Program_Format format, const std::optional<size_t> numShots,
         const std::optional<qdmi::CustomJobParameter>& custom1,
         const std::optional<qdmi::CustomJobParameter>& custom2,
         const std::optional<qdmi::CustomJobParameter>& custom3,
         const std::optional<qdmi::CustomJobParameter>& custom4,
         const std::optional<qdmi::CustomJobParameter>& custom5) {
        const auto bytes = std::span{
            static_cast<const std::byte*>(program.data()), program.size()};
        if (numShots.has_value()) {
          return self.submitJob(bytes, format, *numShots, custom1, custom2,
                                custom3, custom4, custom5);
        }
        return self.submitJob(bytes, format, custom1, custom2, custom3, custom4,
                              custom5);
      },
      "program"_a, "program_format"_a, "num_shots"_a = nb::none(),
      nb::kw_only(), "custom1"_a = nb::none(), "custom2"_a = nb::none(),
      "custom3"_a = nb::none(), "custom4"_a = nb::none(),
      "custom5"_a = nb::none(), nb::rv_policy::reference_internal,
      "Submits an exact byte payload to the device.");

  device.def(
      "submit_calibration_job",
      [](const qdmi::Device& self,
         const std::optional<std::variant<std::string, nb::bytes>>& program,
         const std::optional<qdmi::CustomJobParameter>& custom1,
         const std::optional<qdmi::CustomJobParameter>& custom2,
         const std::optional<qdmi::CustomJobParameter>& custom3,
         const std::optional<qdmi::CustomJobParameter>& custom4,
         const std::optional<qdmi::CustomJobParameter>& custom5) {
        if (!program.has_value()) {
          return self.submitCalibrationJob(std::nullopt, custom1, custom2,
                                           custom3, custom4, custom5);
        }
        if (const auto* text = std::get_if<std::string>(&*program);
            text != nullptr) {
          return self.submitCalibrationJob(*text, custom1, custom2, custom3,
                                           custom4, custom5);
        }
        const auto& payload = std::get<nb::bytes>(*program);
        const auto bytes = std::span{
            static_cast<const std::byte*>(payload.data()), payload.size()};
        return self.submitCalibrationJob(bytes, custom1, custom2, custom3,
                                         custom4, custom5);
      },
      "program"_a = nb::none(), nb::kw_only(), "custom1"_a = nb::none(),
      "custom2"_a = nb::none(), "custom3"_a = nb::none(),
      "custom4"_a = nb::none(), "custom5"_a = nb::none(),
      nb::rv_policy::reference_internal,
      R"pb(Triggers a calibration run on the device.

QDMI does not require a program for a calibration run, so ``program`` is
optional and may be a string or bytes. When it is given, the device defines
what it means, which is usually a configuration for the run. A calibration run
executes no circuit, so it takes no shot count.)pb");

  device.def(
      "retrieve_job_by_id",
      [](const qdmi::Device& self, const std::string& jobId) {
        return self.retrieveJobById(jobId);
      },
      "job_id"_a, nb::rv_policy::reference_internal,
      "Retrieves an existing job by its device-provided ID.");

  device.def("__repr__", [](const qdmi::Device& dev) {
    return "<Device name=\"" + dev.getName() + "\">";
  });

  device.def(nb::self == nb::self,
             nb::sig("def __eq__(self, arg: object, /) -> bool"));
  device.def(nb::self != nb::self,
             nb::sig("def __ne__(self, arg: object, /) -> bool"));

  // Site class
  auto site = nb::class_<qdmi::Site>(
      device, "Site",
      "A site represents a potential qubit location on a quantum device.");

  site.def("index", &qdmi::Site::getIndex, "Returns the index of the site.");

  site.def("t1", &qdmi::Site::getT1,
           "Returns the T1 coherence time of the site.");

  site.def("t2", &qdmi::Site::getT2,
           "Returns the T2 coherence time of the site.");

  site.def("name", &qdmi::Site::getName, "Returns the name of the site.");

  site.def("x_coordinate", &qdmi::Site::getXCoordinate,
           "Returns the x coordinate of the site.");

  site.def("y_coordinate", &qdmi::Site::getYCoordinate,
           "Returns the y coordinate of the site.");

  site.def("z_coordinate", &qdmi::Site::getZCoordinate,
           "Returns the z coordinate of the site.");

  site.def("is_zone", &qdmi::Site::isZone,
           "Returns whether the site is a zone.");

  site.def("x_extent", &qdmi::Site::getXExtent,
           "Returns the x extent of the site.");

  site.def("y_extent", &qdmi::Site::getYExtent,
           "Returns the y extent of the site.");

  site.def("z_extent", &qdmi::Site::getZExtent,
           "Returns the z extent of the site.");

  site.def("module_index", &qdmi::Site::getModuleIndex,
           "Returns the index of the module the site belongs to.");

  site.def("submodule_index", &qdmi::Site::getSubmoduleIndex,
           "Returns the index of the submodule the site belongs to.");

  site.def(
      "query_custom_property",
      [](const qdmi::Site& self, const qdmi::CustomProperty customProperty,
         const nb::handle valueType) {
        return queryCustomValue(
            [&self, customProperty]<qdmi::custom_property_value T> {
              return self.queryCustomProperty<T>(customProperty);
            },
            valueType);
      },
      "custom_property"_a, "value_type"_a,
      nb::sig("def query_custom_property(self, custom_property: "
              "CustomProperty, "
              "value_type: type[str] | type[bool] | type[int] | type[float] | "
              "type[bytes]) -> str | bool | int | float | bytes | None"),
      R"pb(Query an implementation-defined custom site property.

The caller must provide the type documented by the device implementation.
Use ``bytes`` to retrieve the value without interpretation. Returns ``None``
when the custom slot is unsupported.)pb");

  site.def("__repr__", [](const qdmi::Site& s) {
    return "<Site index=" + std::to_string(s.getIndex()) + ">";
  });

  site.def(nb::self == nb::self,
           nb::sig("def __eq__(self, arg: object, /) -> bool"));
  site.def(nb::self != nb::self,
           nb::sig("def __ne__(self, arg: object, /) -> bool"));
  // Operation class
  auto operation = nb::class_<qdmi::Operation>(
      device, "Operation",
      "An operation represents a quantum operation that can be performed on a "
      "quantum device.");

  operation.def("name", &qdmi::Operation::getName,
                "sites"_a.sig("...") = std::vector<qdmi::Site>{},
                "params"_a.sig("...") = std::vector<double>{},
                "Returns the name of the operation.");

  operation.def("qubits_num", &qdmi::Operation::getQubitsNum,
                "sites"_a.sig("...") = std::vector<qdmi::Site>{},
                "params"_a.sig("...") = std::vector<double>{},
                "Returns the number of qubits the operation acts on.");

  operation.def("parameters_num", &qdmi::Operation::getParametersNum,
                "sites"_a.sig("...") = std::vector<qdmi::Site>{},
                "params"_a.sig("...") = std::vector<double>{},
                "Returns the number of parameters the operation has.");

  operation.def("duration", &qdmi::Operation::getDuration,
                "sites"_a.sig("...") = std::vector<qdmi::Site>{},
                "params"_a.sig("...") = std::vector<double>{},
                "Returns the duration of the operation.");

  operation.def("fidelity", &qdmi::Operation::getFidelity,
                "sites"_a.sig("...") = std::vector<qdmi::Site>{},
                "params"_a.sig("...") = std::vector<double>{},
                "Returns the fidelity of the operation.");

  operation.def("interaction_radius", &qdmi::Operation::getInteractionRadius,
                "sites"_a.sig("...") = std::vector<qdmi::Site>{},
                "params"_a.sig("...") = std::vector<double>{},
                "Returns the interaction radius of the operation.");

  operation.def("blocking_radius", &qdmi::Operation::getBlockingRadius,
                "sites"_a.sig("...") = std::vector<qdmi::Site>{},
                "params"_a.sig("...") = std::vector<double>{},
                "Returns the blocking radius of the operation.");

  operation.def("idling_fidelity", &qdmi::Operation::getIdlingFidelity,
                "sites"_a.sig("...") = std::vector<qdmi::Site>{},
                "params"_a.sig("...") = std::vector<double>{},
                "Returns the idling fidelity of the operation.");

  operation.def("is_zoned", &qdmi::Operation::isZoned,
                "Returns whether the operation is zoned.");

  operation.def("sites", &qdmi::Operation::getSites,
                "Returns the list of sites the operation can be performed on.");

  operation.def("site_pairs", &qdmi::Operation::getSitePairs,
                "Returns the list of site pairs the local 2-qubit operation "
                "can be performed on.");

  operation.def("mean_shuttling_speed", &qdmi::Operation::getMeanShuttlingSpeed,
                "sites"_a.sig("...") = std::vector<qdmi::Site>{},
                "params"_a.sig("...") = std::vector<double>{},
                "Returns the mean shuttling speed of the operation.");

  operation.def(
      "query_custom_property",
      [](const qdmi::Operation& self, const qdmi::CustomProperty customProperty,
         const nb::handle valueType, const std::vector<qdmi::Site>& sites,
         const std::vector<double>& params) {
        return queryCustomValue(
            [&self, customProperty, &sites,
             &params]<qdmi::custom_property_value T> {
              return self.queryCustomProperty<T>(customProperty, sites, params);
            },
            valueType);
      },
      "custom_property"_a, "value_type"_a,
      "sites"_a.sig("...") = std::vector<qdmi::Site>{},
      "params"_a.sig("...") = std::vector<double>{},
      nb::sig("def query_custom_property(self, custom_property: "
              "CustomProperty, "
              "value_type: type[str] | type[bool] | type[int] | type[float] | "
              "type[bytes], sites: Sequence[mqt.core.qdmi.Device.Site] = "
              "..., params: Sequence[float] = ...) -> str | bool | int | "
              "float | bytes | None"),
      R"pb(Query an implementation-defined custom operation property.

The caller must provide the type documented by the device implementation.
Use ``bytes`` to retrieve the value without interpretation. Returns ``None``
when the custom slot is unsupported.)pb");

  operation.def("__repr__", [](const qdmi::Operation& op) {
    return "<Operation name=\"" + op.getName() + "\">";
  });

  operation.def(nb::self == nb::self,
                nb::sig("def __eq__(self, arg: object, /) -> bool"));
  operation.def(nb::self != nb::self,
                nb::sig("def __ne__(self, arg: object, /) -> bool"));

  defaultDriver.def("add_manifest", &qdmi::default_driver::addManifest,
                    "manifest_path"_a,
                    "Stage one installed package manifest before the default "
                    "driver freezes.");

  defaultDriver.def(
      "open_device",
      [](const std::string& deviceId,
         const std::optional<std::filesystem::path>& driverPath,
         const std::optional<std::string>& baseUrl,
         const std::optional<std::string>& token,
         const std::optional<std::filesystem::path>& authFile,
         const std::optional<std::string>& authUrl,
         const std::optional<std::string>& username,
         const std::optional<std::string>& password,
         const std::optional<std::string>& deviceConfig,
         const std::optional<std::filesystem::path>& deviceConfigFile,
         const std::optional<std::string>& custom1,
         const std::optional<std::string>& custom2,
         const std::optional<std::string>& custom3,
         const std::optional<std::string>& custom4,
         const std::optional<std::string>& custom5) {
        return qdmi::default_driver::openDevice(
            deviceId,
            makeDeviceSessionJson(baseUrl, token, authFile, authUrl, username,
                                  password, deviceConfig, deviceConfigFile,
                                  custom1, custom2, custom3, custom4, custom5),
            driverPath);
      },
      "device_id"_a, nb::kw_only(), "driver_path"_a = std::nullopt,
      "base_url"_a = std::nullopt, "token"_a = std::nullopt,
      "auth_file"_a = std::nullopt, "auth_url"_a = std::nullopt,
      "username"_a = std::nullopt, "password"_a = std::nullopt,
      "device_config"_a = std::nullopt, "device_config_file"_a = std::nullopt,
      "custom1"_a = std::nullopt, "custom2"_a = std::nullopt,
      "custom3"_a = std::nullopt, "custom4"_a = std::nullopt,
      "custom5"_a = std::nullopt,
      "Open one device through MQT Core's strict private driver extension.");

  nb::module_::import_("mqt.core._qdmi_discovery")
      .attr("discover_qdmi_manifests")(defaultDriver.attr("add_manifest"));
}

} // namespace mqt
