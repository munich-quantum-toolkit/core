/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "qdmi/Device.hpp"
#include "qdmi/DeviceManager.hpp"
#include "qdmi/DeviceRegistry.hpp"

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
#include <nlohmann/json.hpp>         // NOLINT(misc-include-cleaner)

#include <cstddef>
#include <filesystem>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace mqt {

namespace nb = nanobind;
using namespace nb::literals;
using Json = nlohmann::json; // NOLINT(misc-include-cleaner)

namespace {
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

NB_MODULE(MQT_CORE_MODULE_NAME,
          m) { // NOLINT(performance-unnecessary-value-param)
  auto sessionParameters = nb::class_<qdmi::SessionParameters>(
      m, "SessionParameters", R"pb(Parameters for one QDMI device session.)pb");
  sessionParameters.def(nb::init<>(), R"pb(Create empty session parameters.)pb")
      .def_rw("base_url", &qdmi::SessionParameters::baseUrl,
              R"pb(Base URL of the device service.)pb")
      .def_rw("token", &qdmi::SessionParameters::token,
              R"pb(Authentication token.)pb")
      .def_rw("auth_file", &qdmi::SessionParameters::authFile,
              R"pb(Path to an authentication file.)pb")
      .def_rw("auth_url", &qdmi::SessionParameters::authUrl,
              R"pb(URL of the authentication service.)pb")
      .def_rw("username", &qdmi::SessionParameters::username,
              R"pb(Authentication username.)pb")
      .def_rw("password", &qdmi::SessionParameters::password,
              R"pb(Authentication password.)pb")
      .def_rw("custom1", &qdmi::SessionParameters::custom1,
              R"pb(First implementation-defined session parameter.)pb")
      .def_rw("custom2", &qdmi::SessionParameters::custom2,
              R"pb(Second implementation-defined session parameter.)pb")
      .def_rw("custom3", &qdmi::SessionParameters::custom3,
              R"pb(Third implementation-defined session parameter.)pb")
      .def_rw("custom4", &qdmi::SessionParameters::custom4,
              R"pb(Fourth implementation-defined session parameter.)pb")
      .def_rw("custom5", &qdmi::SessionParameters::custom5,
              R"pb(Fifth implementation-defined session parameter.)pb");

  auto definition = nb::class_<qdmi::DeviceDefinition>(
      m, "DeviceDefinition",
      R"pb(A side-effect-free QDMI device registration.)pb");
  definition
      .def(
          "__init__",
          [](qdmi::DeviceDefinition* self, std::string deviceId,
             std::filesystem::path library, std::string prefix, bool enabled,
             qdmi::SessionParameters session) {
            new (self) qdmi::DeviceDefinition{.id = std::move(deviceId),
                                              .library = std::move(library),
                                              .prefix = std::move(prefix),
                                              .session = std::move(session),
                                              .enabled = enabled};
          },
          "device_id"_a, "library"_a, "prefix"_a, nb::kw_only(),
          "enabled"_a = true, "session"_a = qdmi::SessionParameters{},
          R"pb(Create a device definition without loading its library.

Args:
    device_id: Stable identifier used for discovery and opening.
    library: Path to the native QDMI device library.
    prefix: Symbol prefix exported by the QDMI implementation.
    enabled: Whether the definition participates in discovery.
    session: Default parameters for sessions opened from this definition.)pb")
      .def_rw("device_id", &qdmi::DeviceDefinition::id,
              R"pb(Stable device identifier.)pb")
      .def_rw("library", &qdmi::DeviceDefinition::library,
              R"pb(Path to the native QDMI device library.)pb")
      .def_rw("prefix", &qdmi::DeviceDefinition::prefix,
              R"pb(Symbol prefix exported by the device library.)pb")
      .def_rw("enabled", &qdmi::DeviceDefinition::enabled,
              R"pb(Whether this definition is enabled.)pb")
      .def_rw("session", &qdmi::DeviceDefinition::session,
              R"pb(Default parameters for newly opened sessions.)pb")
      .def_ro("source", &qdmi::DeviceDefinition::source,
              R"pb(Configuration source that declared the definition.)pb");

  auto configOptions = nb::class_<qdmi::ConfigOptions>(
      m, "ConfigOptions", R"pb(Controls QDMI configuration discovery.)pb");
  configOptions.def(
      "__init__",
      [](qdmi::ConfigOptions* self,
         std::optional<std::filesystem::path> configRoot,
         std::optional<std::filesystem::path> explicitFile,
         std::optional<std::filesystem::path> baseDirectory, bool isolated,
         std::optional<std::string> inlineJson,
         std::vector<qdmi::DeviceDefinition> runtimeOverrides) {
        qdmi::ConfigOptions options{.configRoot = std::move(configRoot),
                                    .explicitFile = std::move(explicitFile),
                                    .baseDirectory = std::move(baseDirectory),
                                    .runtimeOverrides =
                                        std::move(runtimeOverrides),
                                    .isolated = isolated};
        if (inlineJson) {
          options.inlineOverrides = Json::parse(*inlineJson);
        }
        new (self) qdmi::ConfigOptions(std::move(options));
      },
      nb::kw_only(), "config_root"_a = std::nullopt,
      "explicit_file"_a = std::nullopt, "base_directory"_a = std::nullopt,
      "isolated"_a = false, "inline_json"_a = std::nullopt,
      "runtime_overrides"_a = std::vector<qdmi::DeviceDefinition>{},
      R"pb(Create configuration discovery options.

Args:
    config_root: Root containing relocatable built-in manifest fragments.
    explicit_file: Configuration file replacing system, user, and project discovery.
    base_directory: Base for relative paths in inline configuration.
    isolated: Exclude packaged built-in definitions when true.
    inline_json: JSON configuration layered above discovered files.
    runtime_overrides: Device definitions applied at highest precedence.)pb");

  // Job class
  auto job = nb::class_<qdmi::Job>(
      m, "Job",
      R"pb(A submitted quantum program execution retaining its device session.)pb");

  job.def("check", &qdmi::Job::check,
          R"pb(Return the current QDMI job status.)pb");

  job.def("wait", &qdmi::Job::wait, "timeout"_a = 0,
          R"pb(Waits for the job to complete.

Args:
    timeout: The maximum time to wait in seconds. If 0, waits indefinitely.

Returns:
    True if the job completed within the timeout, False otherwise.)pb");

  job.def("cancel", &qdmi::Job::cancel,
          R"pb(Request cancellation of the job.)pb");

  job.def("get_shots", &qdmi::Job::getShots,
          R"pb(Return the raw shot results.)pb");

  job.def("get_counts", &qdmi::Job::getCounts,
          R"pb(Return measurement counts keyed by bit string.)pb");

  job.def("get_dense_statevector", &qdmi::Job::getDenseStateVector,
          R"pb(Return the dense state vector.

This result is typically available only from simulator devices.)pb");

  job.def("get_dense_probabilities", &qdmi::Job::getDenseProbabilities,
          R"pb(Return the dense probability vector.

This result is typically available only from simulator devices.)pb");

  job.def("get_sparse_statevector", &qdmi::Job::getSparseStateVector,
          R"pb(Return the sparse state vector keyed by basis state.

This result is typically available only from simulator devices.)pb");

  job.def("get_sparse_probabilities", &qdmi::Job::getSparseProbabilities,
          R"pb(Return sparse probabilities keyed by basis state.

This result is typically available only from simulator devices.)pb");

  job.def(
      "query_custom_property",
      [](const qdmi::Job& self, const qdmi::CustomProperty customProperty,
         const nb::handle valueType) {
        return queryCustomValue(
            [&self, customProperty]<qdmi::custom_property_value T>() {
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
Use ``bytes`` to retrieve the value without interpretation.

Args:
    custom_property: Custom property slot to query.
    value_type: Expected Python type of the property value.

Returns:
    The typed property value, or ``None`` when the slot is unsupported.)pb");

  job.def(
      "get_custom_result",
      [](const qdmi::Job& self, const qdmi::CustomProperty customProperty,
         const nb::handle valueType) {
        return queryCustomValue(
            [&self, customProperty]<qdmi::custom_property_value T>() {
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
Use ``bytes`` to retrieve the value without interpretation.

Args:
    custom_property: Custom result slot to retrieve.
    value_type: Expected Python type of the result value.

Returns:
    The typed result value, or ``None`` when the slot is unsupported.)pb");

  job.def_prop_ro("id", &qdmi::Job::getId,
                  R"pb(The provider-assigned job identifier.)pb");

  job.def_prop_ro("program_format", &qdmi::Job::getProgramFormat,
                  R"pb(The QDMI format of the submitted program.)pb");

  job.def_prop_ro("program", &qdmi::Job::getProgram,
                  R"pb(The submitted program.)pb");

  job.def_prop_ro("num_shots", &qdmi::Job::getNumShots,
                  R"pb(The requested number of shots.)pb");

  job.def(nb::self == nb::self,
          nb::sig("def __eq__(self, arg: object, /) -> bool"),
          R"pb(Return whether two objects refer to the same job.)pb");
  job.def(nb::self != nb::self,
          nb::sig("def __ne__(self, arg: object, /) -> bool"),
          R"pb(Return whether two objects refer to different jobs.)pb");

  // JobStatus enum
  nb::enum_<QDMI_Job_Status>(job, "Status",
                             R"pb(Status values defined by QDMI.)pb")
      .value("CREATED", QDMI_JOB_STATUS_CREATED)
      .value("SUBMITTED", QDMI_JOB_STATUS_SUBMITTED)
      .value("QUEUED", QDMI_JOB_STATUS_QUEUED)
      .value("RUNNING", QDMI_JOB_STATUS_RUNNING)
      .value("DONE", QDMI_JOB_STATUS_DONE)
      .value("CANCELED", QDMI_JOB_STATUS_CANCELED)
      .value("FAILED", QDMI_JOB_STATUS_FAILED);

  // ProgramFormat enum
  nb::enum_<QDMI_Program_Format>(m, "ProgramFormat",
                                 R"pb(Program formats defined by QDMI.)pb")
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

  nb::enum_<qdmi::CustomProperty>(
      m, "CustomProperty",
      R"pb(An implementation-defined custom property or result slot.)pb")
      .value("CUSTOM1", qdmi::CustomProperty::Custom1)
      .value("CUSTOM2", qdmi::CustomProperty::Custom2)
      .value("CUSTOM3", qdmi::CustomProperty::Custom3)
      .value("CUSTOM4", qdmi::CustomProperty::Custom4)
      .value("CUSTOM5", qdmi::CustomProperty::Custom5);

  // Device class
  auto device = nb::class_<qdmi::Device>(
      m, "Device", R"pb(One initialized QDMI device session.

The object owns the native library and session state required by its sites,
operations, child devices, and jobs.)pb");

  nb::enum_<QDMI_Device_Status>(device, "Status",
                                R"pb(Status values defined by QDMI.)pb")
      .value("OFFLINE", QDMI_DEVICE_STATUS_OFFLINE)
      .value("IDLE", QDMI_DEVICE_STATUS_IDLE)
      .value("BUSY", QDMI_DEVICE_STATUS_BUSY)
      .value("ERROR", QDMI_DEVICE_STATUS_ERROR)
      .value("MAINTENANCE", QDMI_DEVICE_STATUS_MAINTENANCE)
      .value("CALIBRATION", QDMI_DEVICE_STATUS_CALIBRATION);

  device.def("name", &qdmi::Device::getName,
             R"pb(Return the provider-reported device name.)pb");

  device.def("version", &qdmi::Device::getVersion,
             R"pb(Return the provider-reported device version.)pb");

  device.def("status", &qdmi::Device::getStatus,
             R"pb(Return the current QDMI device status.)pb");

  device.def("library_version", &qdmi::Device::getLibraryVersion,
             R"pb(Return the provider library version.)pb");

  device.def("qubits_num", &qdmi::Device::getQubitsNum,
             R"pb(Return the number of qubits available on the device.)pb");

  device.def("sites", &qdmi::Device::getSites,
             R"pb(Return all regular sites and zones.)pb");

  device.def("regular_sites", &qdmi::Device::getRegularSites,
             R"pb(Return sites that are not zones.)pb");

  device.def("zones", &qdmi::Device::getZones,
             R"pb(Return sites that represent zones.)pb");

  device.def("operations", &qdmi::Device::getOperations,
             R"pb(Return operations supported by the device.)pb");

  device.def("coupling_map", &qdmi::Device::getCouplingMap,
             R"pb(Return the optional coupling map as site pairs.)pb");

  device.def("needs_calibration", &qdmi::Device::getNeedsCalibration,
             R"pb(Return the optional calibration requirement.)pb");

  device.def("length_unit", &qdmi::Device::getLengthUnit,
             R"pb(Return the optional device length unit.)pb");

  device.def("length_scale_factor", &qdmi::Device::getLengthScaleFactor,
             R"pb(Return the optional length scale factor.)pb");

  device.def("duration_unit", &qdmi::Device::getDurationUnit,
             R"pb(Return the optional device duration unit.)pb");

  device.def("duration_scale_factor", &qdmi::Device::getDurationScaleFactor,
             R"pb(Return the optional duration scale factor.)pb");

  device.def("min_atom_distance", &qdmi::Device::getMinAtomDistance,
             R"pb(Return the optional minimum atom distance.)pb");

  device.def("supported_program_formats",
             &qdmi::Device::getSupportedProgramFormats,
             R"pb(Return the QDMI program formats accepted by the device.)pb");

  device.def("child_devices", &qdmi::Device::getChildDevices,
             R"pb(Return directly managed child devices.)pb");

  device.def(
      "query_custom_property",
      [](const qdmi::Device& self, const qdmi::CustomProperty customProperty,
         const nb::handle valueType) {
        return queryCustomValue(
            [&self, customProperty]<qdmi::custom_property_value T>() {
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
Use ``bytes`` to retrieve the value without interpretation.

Args:
    custom_property: Custom property slot to query.
    value_type: Expected Python type of the property value.

Returns:
    The typed property value, or ``None`` when the slot is unsupported.)pb");

  device.def("submit_job", &qdmi::Device::submitJob, "program"_a,
             "program_format"_a, "num_shots"_a, nb::kw_only(),
             "custom1"_a = nb::none(), "custom2"_a = nb::none(),
             "custom3"_a = nb::none(), "custom4"_a = nb::none(),
             "custom5"_a = nb::none(),
             nb::sig("def submit_job(self, program: str, program_format: "
                     "ProgramFormat, num_shots: int, *, custom1: str | bool | "
                     "int | float | None = None, custom2: str | bool | int | "
                     "float | None = None, custom3: str | bool | int | float | "
                     "None = None, custom4: str | bool | int | float | None = "
                     "None, custom5: str | bool | int | float | None = None) "
                     "-> Job"),
             nb::rv_policy::reference_internal,
             R"pb(Submit a quantum program to the device.

Args:
    program: Serialized program data.
    program_format: QDMI format of ``program``.
    num_shots: Number of requested executions.
    custom1: First implementation-defined job parameter.
    custom2: Second implementation-defined job parameter.
    custom3: Third implementation-defined job parameter.
    custom4: Fourth implementation-defined job parameter.
    custom5: Fifth implementation-defined job parameter.

Returns:
    A job retaining the device session.)pb");

  device.def(
      "__repr__",
      [](const qdmi::Device& dev) {
        return "<Device name=\"" + dev.getName() + "\">";
      },
      R"pb(Return a concise device representation.)pb");

  device.def(nb::self == nb::self,
             nb::sig("def __eq__(self, arg: object, /) -> bool"),
             R"pb(Return whether two objects refer to the same device.)pb");
  device.def(nb::self != nb::self,
             nb::sig("def __ne__(self, arg: object, /) -> bool"),
             R"pb(Return whether two objects refer to different devices.)pb");

  // Site class
  auto site = nb::class_<qdmi::Site>(
      device, "Site", R"pb(A physical site or zone belonging to a device.)pb");

  site.def("index", &qdmi::Site::getIndex,
           R"pb(Return the provider-assigned site index.)pb");

  site.def("t1", &qdmi::Site::getT1,
           R"pb(Return the optional T1 coherence time.)pb");

  site.def("t2", &qdmi::Site::getT2,
           R"pb(Return the optional T2 coherence time.)pb");

  site.def("name", &qdmi::Site::getName,
           R"pb(Return the optional site name.)pb");

  site.def("x_coordinate", &qdmi::Site::getXCoordinate,
           R"pb(Return the optional x coordinate.)pb");

  site.def("y_coordinate", &qdmi::Site::getYCoordinate,
           R"pb(Return the optional y coordinate.)pb");

  site.def("z_coordinate", &qdmi::Site::getZCoordinate,
           R"pb(Return the optional z coordinate.)pb");

  site.def("is_zone", &qdmi::Site::isZone,
           R"pb(Return whether this site represents a zone.)pb");

  site.def("x_extent", &qdmi::Site::getXExtent,
           R"pb(Return the optional x extent of the zone.)pb");

  site.def("y_extent", &qdmi::Site::getYExtent,
           R"pb(Return the optional y extent of the zone.)pb");

  site.def("z_extent", &qdmi::Site::getZExtent,
           R"pb(Return the optional z extent of the zone.)pb");

  site.def("module_index", &qdmi::Site::getModuleIndex,
           R"pb(Return the optional module index.)pb");

  site.def("submodule_index", &qdmi::Site::getSubmoduleIndex,
           R"pb(Return the optional submodule index.)pb");

  site.def(
      "query_custom_property",
      [](const qdmi::Site& self, const qdmi::CustomProperty customProperty,
         const nb::handle valueType) {
        return queryCustomValue(
            [&self, customProperty]<qdmi::custom_property_value T>() {
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
Use ``bytes`` to retrieve the value without interpretation.

Args:
    custom_property: Custom property slot to query.
    value_type: Expected Python type of the property value.

Returns:
    The typed property value, or ``None`` when the slot is unsupported.)pb");

  site.def(
      "__repr__",
      [](const qdmi::Site& s) {
        return "<Site index=" + std::to_string(s.getIndex()) + ">";
      },
      R"pb(Return a concise site representation.)pb");

  site.def(nb::self == nb::self,
           nb::sig("def __eq__(self, arg: object, /) -> bool"),
           R"pb(Return whether two objects refer to the same site.)pb");
  site.def(nb::self != nb::self,
           nb::sig("def __ne__(self, arg: object, /) -> bool"),
           R"pb(Return whether two objects refer to different sites.)pb");

  // Operation class
  auto operation = nb::class_<qdmi::Operation>(
      device, "Operation", R"pb(A quantum operation supported by a device.)pb");

  operation.def(
      "name", &qdmi::Operation::getName,
      "sites"_a.sig("...") = std::vector<qdmi::Site>{},
      "params"_a.sig("...") = std::vector<double>{},
      R"pb(Return the operation name for the given sites and parameters.)pb");

  operation.def("qubits_num", &qdmi::Operation::getQubitsNum,
                "sites"_a.sig("...") = std::vector<qdmi::Site>{},
                "params"_a.sig("...") = std::vector<double>{},
                R"pb(Return the optional operation arity.)pb");

  operation.def("parameters_num", &qdmi::Operation::getParametersNum,
                "sites"_a.sig("...") = std::vector<qdmi::Site>{},
                "params"_a.sig("...") = std::vector<double>{},
                R"pb(Return the number of operation parameters.)pb");

  operation.def(
      "duration", &qdmi::Operation::getDuration,
      "sites"_a.sig("...") = std::vector<qdmi::Site>{},
      "params"_a.sig("...") = std::vector<double>{},
      R"pb(Return the optional duration for this operation instance.)pb");

  operation.def(
      "fidelity", &qdmi::Operation::getFidelity,
      "sites"_a.sig("...") = std::vector<qdmi::Site>{},
      "params"_a.sig("...") = std::vector<double>{},
      R"pb(Return the optional fidelity for this operation instance.)pb");

  operation.def("interaction_radius", &qdmi::Operation::getInteractionRadius,
                "sites"_a.sig("...") = std::vector<qdmi::Site>{},
                "params"_a.sig("...") = std::vector<double>{},
                R"pb(Return the optional interaction radius.)pb");

  operation.def("blocking_radius", &qdmi::Operation::getBlockingRadius,
                "sites"_a.sig("...") = std::vector<qdmi::Site>{},
                "params"_a.sig("...") = std::vector<double>{},
                R"pb(Return the optional blocking radius.)pb");

  operation.def("idling_fidelity", &qdmi::Operation::getIdlingFidelity,
                "sites"_a.sig("...") = std::vector<qdmi::Site>{},
                "params"_a.sig("...") = std::vector<double>{},
                R"pb(Return the optional idling fidelity.)pb");

  operation.def("is_zoned", &qdmi::Operation::isZoned,
                R"pb(Return whether the operation is restricted to zones.)pb");

  operation.def("sites", &qdmi::Operation::getSites,
                R"pb(Return sites on which the operation is available.)pb");

  operation.def(
      "site_pairs", &qdmi::Operation::getSitePairs,
      R"pb(Return supported site pairs for a local two-site operation.)pb");

  operation.def("mean_shuttling_speed", &qdmi::Operation::getMeanShuttlingSpeed,
                "sites"_a.sig("...") = std::vector<qdmi::Site>{},
                "params"_a.sig("...") = std::vector<double>{},
                R"pb(Return the optional mean shuttling speed.)pb");

  operation.def(
      "query_custom_property",
      [](const qdmi::Operation& self, const qdmi::CustomProperty customProperty,
         const nb::handle valueType, const std::vector<qdmi::Site>& sites,
         const std::vector<double>& params) {
        return queryCustomValue(
            [&self, customProperty, &sites,
             &params]<qdmi::custom_property_value T>() {
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
Use ``bytes`` to retrieve the value without interpretation.

Args:
    custom_property: Custom property slot to query.
    value_type: Expected Python type of the property value.
    sites: Sites for the operation instance.
    params: Parameters for the operation instance.

Returns:
    The typed property value, or ``None`` when the slot is unsupported.)pb");

  operation.def(
      "__repr__",
      [](const qdmi::Operation& op) {
        return "<Operation name=\"" + op.getName() + "\">";
      },
      R"pb(Return a concise operation representation.)pb");

  operation.def(
      nb::self == nb::self, nb::sig("def __eq__(self, arg: object, /) -> bool"),
      R"pb(Return whether two objects refer to the same operation.)pb");
  operation.def(
      nb::self != nb::self, nb::sig("def __ne__(self, arg: object, /) -> bool"),
      R"pb(Return whether two objects refer to different operations.)pb");

  auto openAllResult = nb::class_<qdmi::OpenAllResult>(
      m, "OpenAllResult",
      R"pb(Devices and per-ID errors produced by bulk opening.)pb");
  openAllResult
      .def_ro("devices", &qdmi::OpenAllResult::devices,
              R"pb(Successfully opened devices keyed by stable ID.)pb")
      .def_ro(
          "errors", &qdmi::OpenAllResult::errors,
          R"pb(Error messages for failed definitions keyed by stable ID.)pb");

  auto manager = nb::class_<qdmi::DeviceManager>(
      m, "DeviceManager", R"pb(Discover and lazily open QDMI devices.

Definitions are discovered without loading native libraries. Opening a device
creates an independent session while compatible devices may share a loaded
library.)pb");
  manager
      .def(nb::init<const qdmi::ConfigOptions&>(),
           "options"_a = qdmi::ConfigOptions{},
           R"pb(Create a manager using the supplied discovery options.)pb")
      .def_prop_ro(
          "definitions",
          [](const qdmi::DeviceManager& self) { return self.definitions(); },
          R"pb(A snapshot of the currently registered device definitions.)pb")
      .def("register_device", &qdmi::DeviceManager::registerDevice,
           "definition"_a, nb::kw_only(), "replace"_a = false,
           R"pb(Register a complete device definition.

Args:
    definition: Definition to register.
    replace: Replace an existing definition with the same ID.)pb")
      .def(
          "unregister_device",
          [](qdmi::DeviceManager& self, const std::string& deviceId) {
            return self.unregisterDevice(deviceId);
          },
          "device_id"_a,
          R"pb(Remove a definition without invalidating opened devices.

Returns:
    Whether a definition with the requested ID existed.)pb")
      .def(
          "open",
          [](qdmi::DeviceManager& self, const std::string& deviceId,
             const qdmi::SessionParameters& sessionOverrides) {
            return self.open(deviceId, sessionOverrides);
          },
          "device_id"_a, nb::kw_only(),
          "session_overrides"_a = qdmi::SessionParameters{},
          R"pb(Open one device by stable ID.

The supplied session values override the definition defaults field by field.
The native library is loaded only when this method is called.)pb")
      .def("open_all", &qdmi::DeviceManager::openAll, nb::kw_only(),
           "session_overrides"_a = qdmi::SessionParameters{},
           R"pb(Open a snapshot of all definitions independently.

Failures are retained by device ID and do not prevent other definitions from
opening.)pb");
}

} // namespace mqt
