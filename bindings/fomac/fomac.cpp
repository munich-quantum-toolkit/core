/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "fomac/FoMaC.hpp"

#include "qdmi/driver/Driver.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/operators.h>
#include <nanobind/stl/complex.h>  // NOLINT(misc-include-cleaner)
#include <nanobind/stl/map.h>      // NOLINT(misc-include-cleaner)
#include <nanobind/stl/optional.h> // NOLINT(misc-include-cleaner)
#include <nanobind/stl/pair.h>     // NOLINT(misc-include-cleaner)
#include <nanobind/stl/string.h>   // NOLINT(misc-include-cleaner)
#include <nanobind/stl/variant.h>  // NOLINT(misc-include-cleaner)
#include <nanobind/stl/vector.h>   // NOLINT(misc-include-cleaner)
#include <qdmi/client.h>

#include <cstddef>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace mqt {

namespace nb = nanobind;
using namespace nb::literals;

namespace {
template <typename Query>
[[nodiscard]] nb::object queryCustomValue(Query&& query,
                                          const nb::handle valueType) {
  const auto returnValue =
      []<typename T>(std::optional<T> value) -> nb::object {
    if (!value.has_value()) {
      return nb::none();
    }
    return nb::cast(std::move(*value));
  };

  if (valueType.ptr() == reinterpret_cast<PyObject*>(&PyUnicode_Type)) {
    return returnValue(query.template operator()<std::string>());
  }
  if (valueType.ptr() == reinterpret_cast<PyObject*>(&PyBool_Type)) {
    return returnValue(query.template operator()<bool>());
  }
  if (valueType.ptr() == reinterpret_cast<PyObject*>(&PyLong_Type)) {
    return returnValue(query.template operator()<int>());
  }
  if (valueType.ptr() == reinterpret_cast<PyObject*>(&PyFloat_Type)) {
    return returnValue(query.template operator()<double>());
  }
  if (valueType.ptr() == reinterpret_cast<PyObject*>(&PyBytes_Type)) {
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

NB_MODULE(MQT_CORE_MODULE_NAME, m) {
  // Session class
  auto session = nb::class_<fomac::Session>(
      m, "Session", R"pb(A FoMaC session for managing QDMI devices.

Allows creating isolated sessions with independent authentication settings.
All authentication parameters are optional and can be provided as keyword arguments to the constructor.)pb");

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
        const fomac::SessionConfig config{.token = std::move(token),
                                          .authFile = std::move(authFile),
                                          .authUrl = std::move(authUrl),
                                          .username = std::move(username),
                                          .password = std::move(password),
                                          .projectId = std::move(projectId),
                                          .custom1 = std::move(custom1),
                                          .custom2 = std::move(custom2),
                                          .custom3 = std::move(custom3),
                                          .custom4 = std::move(custom4),
                                          .custom5 = std::move(custom5)};
        new (self) fomac::Session(config);
      },
      nb::kw_only(), "token"_a = std::nullopt, "auth_file"_a = std::nullopt,
      "auth_url"_a = std::nullopt, "username"_a = std::nullopt,
      "password"_a = std::nullopt, "project_id"_a = std::nullopt,
      "custom1"_a = std::nullopt, "custom2"_a = std::nullopt,
      "custom3"_a = std::nullopt, "custom4"_a = std::nullopt,
      "custom5"_a = std::nullopt,
      R"pb(Create a new FoMaC session with optional authentication.

Args:
    token: Authentication token
    auth_file: Path to file containing authentication information
    auth_url: URL to authentication server
    username: Username for authentication
    password: Password for authentication
    project_id: Project ID for session
    custom1: Custom configuration parameter 1
    custom2: Custom configuration parameter 2
    custom3: Custom configuration parameter 3
    custom4: Custom configuration parameter 4
    custom5: Custom configuration parameter 5

Raises:
    RuntimeError: If auth_file does not exist
    RuntimeError: If auth_url has invalid format

Example:
    >>> from mqt.core.fomac import Session
    >>> # Session without authentication
    >>> session = Session()
    >>> devices = session.get_devices()
    >>>
    >>> # Session with token authentication
    >>> session = Session(token="my_secret_token")
    >>> devices = session.get_devices()
    >>>
    >>> # Session with file-based authentication
    >>> session = Session(auth_file="/path/to/auth.json")
    >>> devices = session.get_devices()
    >>>
    >>> # Session with multiple parameters
    >>> session = Session(
    ...     auth_url="https://auth.example.com", username="user", password="pass", project_id="project-123"
    ... )
    >>> devices = session.get_devices())pb");

  session.def("get_devices", &fomac::Session::getDevices,
              nb::rv_policy::reference_internal,
              R"pb(Get available devices from this session.

Returns:
    List of available devices.)pb");

  // Job class
  auto job = nb::class_<fomac::Job>(
      m, "Job", "A job represents a submitted quantum program execution.");

  job.def("check", &fomac::Job::check,
          "Returns the current status of the job.");

  job.def("wait", &fomac::Job::wait, "timeout"_a = 0,
          R"pb(Waits for the job to complete.

Args:
    timeout: The maximum time to wait in seconds. If 0, waits indefinitely.

Returns:
    True if the job completed within the timeout, False otherwise.)pb");

  job.def("cancel", &fomac::Job::cancel, "Cancels the job.");

  job.def("get_shots", &fomac::Job::getShots,
          "Returns the raw shot results from the job.");

  job.def("get_counts", &fomac::Job::getCounts,
          "Returns the measurement counts from the job.");

  job.def("get_dense_statevector", &fomac::Job::getDenseStateVector,
          "Returns the dense statevector from the job (typically only "
          "available from simulator devices).");

  job.def("get_dense_probabilities", &fomac::Job::getDenseProbabilities,
          "Returns the dense probabilities from the job (typically only "
          "available from simulator devices).");

  job.def("get_sparse_statevector", &fomac::Job::getSparseStateVector,
          "Returns the sparse statevector from the job (typically only "
          "available from simulator devices).");

  job.def("get_sparse_probabilities", &fomac::Job::getSparseProbabilities,
          "Returns the sparse probabilities from the job (typically only "
          "available from simulator devices).");

  job.def(
      "query_custom_property",
      [](const fomac::Job& self, const fomac::CustomProperty customProperty,
         const nb::handle valueType) {
        return queryCustomValue(
            [&self, customProperty]<fomac::custom_property_value T>() {
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
      [](const fomac::Job& self, const fomac::CustomProperty customProperty,
         const nb::handle valueType) {
        return queryCustomValue(
            [&self, customProperty]<fomac::custom_property_value T>() {
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

  job.def_prop_ro("id", &fomac::Job::getId, "The job ID.");

  job.def_prop_ro("program_format", &fomac::Job::getProgramFormat,
                  "The format of the submitted program.");

  job.def_prop_ro("program", &fomac::Job::getProgram, "The submitted program.");

  job.def_prop_ro("num_shots", &fomac::Job::getNumShots,
                  "The number of shots.");

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
  nb::enum_<QDMI_Program_Format>(m, "ProgramFormat",
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
      .value("CUSTOM5", QDMI_PROGRAM_FORMAT_CUSTOM5);

  nb::enum_<fomac::CustomProperty>(
      m, "CustomProperty",
      "An implementation-defined custom property or result slot.")
      .value("CUSTOM1", fomac::CustomProperty::Custom1)
      .value("CUSTOM2", fomac::CustomProperty::Custom2)
      .value("CUSTOM3", fomac::CustomProperty::Custom3)
      .value("CUSTOM4", fomac::CustomProperty::Custom4)
      .value("CUSTOM5", fomac::CustomProperty::Custom5);

  // Device class
  auto device = nb::class_<fomac::Device>(
      m, "Device",
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

  device.def("name", &fomac::Device::getName,
             "Returns the name of the device.");

  device.def("version", &fomac::Device::getVersion,
             "Returns the version of the device.");

  device.def("status", &fomac::Device::getStatus,
             "Returns the current status of the device.");

  device.def("library_version", &fomac::Device::getLibraryVersion,
             "Returns the version of the library used to define the device.");

  device.def("qubits_num", &fomac::Device::getQubitsNum,
             "Returns the number of qubits available on the device.");

  device.def("sites", &fomac::Device::getSites,
             "Returns the list of all sites (zone and regular sites) available "
             "on the device.");

  device.def("regular_sites", &fomac::Device::getRegularSites,
             "Returns the list of regular sites (without zone sites) available "
             "on the device.");

  device.def("zones", &fomac::Device::getZones,
             "Returns the list of zone sites (without regular sites) available "
             "on the device.");

  device.def("operations", &fomac::Device::getOperations,
             "Returns the list of operations supported by the device.");

  device.def("coupling_map", &fomac::Device::getCouplingMap,
             "Returns the coupling map of the device as a list of site pairs.");

  device.def("needs_calibration", &fomac::Device::getNeedsCalibration,
             "Returns whether the device needs calibration.");

  device.def("length_unit", &fomac::Device::getLengthUnit,
             "Returns the unit of length used by the device.");

  device.def("length_scale_factor", &fomac::Device::getLengthScaleFactor,
             "Returns the scale factor for length used by the device.");

  device.def("duration_unit", &fomac::Device::getDurationUnit,
             "Returns the unit of duration used by the device.");

  device.def("duration_scale_factor", &fomac::Device::getDurationScaleFactor,
             "Returns the scale factor for duration used by the device.");

  device.def("min_atom_distance", &fomac::Device::getMinAtomDistance,
             "Returns the minimum atom distance on the device.");

  device.def("supported_program_formats",
             &fomac::Device::getSupportedProgramFormats,
             "Returns the list of program formats supported by the device.");

  device.def(
      "query_custom_property",
      [](const fomac::Device& self, const fomac::CustomProperty customProperty,
         const nb::handle valueType) {
        return queryCustomValue(
            [&self, customProperty]<fomac::custom_property_value T>() {
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

  device.def("submit_job", &fomac::Device::submitJob, "program"_a,
             "program_format"_a, "num_shots"_a, nb::kw_only(),
             "custom1"_a = nb::none(), "custom2"_a = nb::none(),
             "custom3"_a = nb::none(), "custom4"_a = nb::none(),
             "custom5"_a = nb::none(), nb::rv_policy::reference_internal,
             "Submits a job to the device.");

  device.def("__repr__", [](const fomac::Device& dev) {
    return "<Device name=\"" + dev.getName() + "\">";
  });

  device.def(nb::self == nb::self,
             nb::sig("def __eq__(self, arg: object, /) -> bool"));
  device.def(nb::self != nb::self,
             nb::sig("def __ne__(self, arg: object, /) -> bool"));

  // Site class
  auto site = nb::class_<fomac::Site>(
      device, "Site",
      "A site represents a potential qubit location on a quantum device.");

  site.def("index", &fomac::Site::getIndex, "Returns the index of the site.");

  site.def("t1", &fomac::Site::getT1,
           "Returns the T1 coherence time of the site.");

  site.def("t2", &fomac::Site::getT2,
           "Returns the T2 coherence time of the site.");

  site.def("name", &fomac::Site::getName, "Returns the name of the site.");

  site.def("x_coordinate", &fomac::Site::getXCoordinate,
           "Returns the x coordinate of the site.");

  site.def("y_coordinate", &fomac::Site::getYCoordinate,
           "Returns the y coordinate of the site.");

  site.def("z_coordinate", &fomac::Site::getZCoordinate,
           "Returns the z coordinate of the site.");

  site.def("is_zone", &fomac::Site::isZone,
           "Returns whether the site is a zone.");

  site.def("x_extent", &fomac::Site::getXExtent,
           "Returns the x extent of the site.");

  site.def("y_extent", &fomac::Site::getYExtent,
           "Returns the y extent of the site.");

  site.def("z_extent", &fomac::Site::getZExtent,
           "Returns the z extent of the site.");

  site.def("module_index", &fomac::Site::getModuleIndex,
           "Returns the index of the module the site belongs to.");

  site.def("submodule_index", &fomac::Site::getSubmoduleIndex,
           "Returns the index of the submodule the site belongs to.");

  site.def(
      "query_custom_property",
      [](const fomac::Site& self, const fomac::CustomProperty customProperty,
         const nb::handle valueType) {
        return queryCustomValue(
            [&self, customProperty]<fomac::custom_property_value T>() {
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

  site.def("__repr__", [](const fomac::Site& s) {
    return "<Site index=" + std::to_string(s.getIndex()) + ">";
  });

  site.def(nb::self == nb::self,
           nb::sig("def __eq__(self, arg: object, /) -> bool"));
  site.def(nb::self != nb::self,
           nb::sig("def __ne__(self, arg: object, /) -> bool"));

  // Operation class
  auto operation = nb::class_<fomac::Operation>(
      device, "Operation",
      "An operation represents a quantum operation that can be performed on a "
      "quantum device.");

  operation.def("name", &fomac::Operation::getName,
                "sites"_a.sig("...") = std::vector<fomac::Site>{},
                "params"_a.sig("...") = std::vector<double>{},
                "Returns the name of the operation.");

  operation.def("qubits_num", &fomac::Operation::getQubitsNum,
                "sites"_a.sig("...") = std::vector<fomac::Site>{},
                "params"_a.sig("...") = std::vector<double>{},
                "Returns the number of qubits the operation acts on.");

  operation.def("parameters_num", &fomac::Operation::getParametersNum,
                "sites"_a.sig("...") = std::vector<fomac::Site>{},
                "params"_a.sig("...") = std::vector<double>{},
                "Returns the number of parameters the operation has.");

  operation.def("duration", &fomac::Operation::getDuration,
                "sites"_a.sig("...") = std::vector<fomac::Site>{},
                "params"_a.sig("...") = std::vector<double>{},
                "Returns the duration of the operation.");

  operation.def("fidelity", &fomac::Operation::getFidelity,
                "sites"_a.sig("...") = std::vector<fomac::Site>{},
                "params"_a.sig("...") = std::vector<double>{},
                "Returns the fidelity of the operation.");

  operation.def("interaction_radius", &fomac::Operation::getInteractionRadius,
                "sites"_a.sig("...") = std::vector<fomac::Site>{},
                "params"_a.sig("...") = std::vector<double>{},
                "Returns the interaction radius of the operation.");

  operation.def("blocking_radius", &fomac::Operation::getBlockingRadius,
                "sites"_a.sig("...") = std::vector<fomac::Site>{},
                "params"_a.sig("...") = std::vector<double>{},
                "Returns the blocking radius of the operation.");

  operation.def("idling_fidelity", &fomac::Operation::getIdlingFidelity,
                "sites"_a.sig("...") = std::vector<fomac::Site>{},
                "params"_a.sig("...") = std::vector<double>{},
                "Returns the idling fidelity of the operation.");

  operation.def("is_zoned", &fomac::Operation::isZoned,
                "Returns whether the operation is zoned.");

  operation.def("sites", &fomac::Operation::getSites,
                "Returns the list of sites the operation can be performed on.");

  operation.def("site_pairs", &fomac::Operation::getSitePairs,
                "Returns the list of site pairs the local 2-qubit operation "
                "can be performed on.");

  operation.def("mean_shuttling_speed",
                &fomac::Operation::getMeanShuttlingSpeed,
                "sites"_a.sig("...") = std::vector<fomac::Site>{},
                "params"_a.sig("...") = std::vector<double>{},
                "Returns the mean shuttling speed of the operation.");

  operation.def(
      "query_custom_property",
      [](const fomac::Operation& self,
         const fomac::CustomProperty customProperty, const nb::handle valueType,
         const std::vector<fomac::Site>& sites,
         const std::vector<double>& params) {
        return queryCustomValue(
            [&self, customProperty, &sites,
             &params]<fomac::custom_property_value T>() {
              return self.queryCustomProperty<T>(customProperty, sites, params);
            },
            valueType);
      },
      "custom_property"_a, "value_type"_a,
      "sites"_a.sig("...") = std::vector<fomac::Site>{},
      "params"_a.sig("...") = std::vector<double>{},
      nb::sig("def query_custom_property(self, custom_property: "
              "CustomProperty, "
              "value_type: type[str] | type[bool] | type[int] | type[float] | "
              "type[bytes], sites: Sequence[mqt.core.fomac.Device.Site] = "
              "..., params: Sequence[float] = ...) -> str | bool | int | "
              "float | bytes | None"),
      R"pb(Query an implementation-defined custom operation property.

The caller must provide the type documented by the device implementation.
Use ``bytes`` to retrieve the value without interpretation. Returns ``None``
when the custom slot is unsupported.)pb");

  operation.def("__repr__", [](const fomac::Operation& op) {
    return "<Operation name=\"" + op.getName() + "\">";
  });

  operation.def(nb::self == nb::self,
                nb::sig("def __eq__(self, arg: object, /) -> bool"));
  operation.def(nb::self != nb::self,
                nb::sig("def __ne__(self, arg: object, /) -> bool"));

  // Module-level function to add dynamic device libraries
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
             std::nullopt) -> fomac::Device {
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
        return fomac::Session::createSessionlessDevice(qdmiDevice);
      },
      "library_path"_a, "prefix"_a, nb::kw_only(), "base_url"_a = std::nullopt,
      "token"_a = std::nullopt, "auth_file"_a = std::nullopt,
      "auth_url"_a = std::nullopt, "username"_a = std::nullopt,
      "password"_a = std::nullopt, "custom1"_a = std::nullopt,
      "custom2"_a = std::nullopt, "custom3"_a = std::nullopt,
      "custom4"_a = std::nullopt, "custom5"_a = std::nullopt,
      R"pb(Load a dynamic device library into the QDMI driver.

This function loads a shared library (.so, .dll, or .dylib) that implements a QDMI device interface and makes it available for use in sessions.

Args:
    library_path: Path to the shared library file to load.
    prefix: Function prefix used by the library (e.g., "MY_DEVICE").
    base_url: Optional base URL for the device API endpoint.
    token: Optional authentication token.
    auth_file: Optional path to authentication file.
    auth_url: Optional authentication server URL.
    username: Optional username for authentication.
    password: Optional password for authentication.
    custom1: Optional custom configuration parameter 1.
    custom2: Optional custom configuration parameter 2.
    custom3: Optional custom configuration parameter 3.
    custom4: Optional custom configuration parameter 4.
    custom5: Optional custom configuration parameter 5.

Returns:
    Device: The newly loaded device that can be used to create backends.

Raises:
    RuntimeError: If library loading fails or configuration is invalid.

Examples:
    Load a device library with configuration:

    >>> import mqt.core.fomac as fomac
    >>> device = fomac.add_dynamic_device_library(
    ...     "/path/to/libmy_device.so", "MY_DEVICE", base_url="http://localhost:8080", custom1="API_V2"
    ... )

    Now the device can be used directly:

    >>> from mqt.core.plugins.qiskit import QDMIBackend
    >>> backend = QDMIBackend(device=device))pb");
}

} // namespace mqt
