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
      m, "SessionParameters", "Parameters for one QDMI device session.");
  sessionParameters.def(nb::init<>())
      .def_rw("base_url", &qdmi::SessionParameters::baseUrl)
      .def_rw("token", &qdmi::SessionParameters::token)
      .def_rw("auth_file", &qdmi::SessionParameters::authFile)
      .def_rw("auth_url", &qdmi::SessionParameters::authUrl)
      .def_rw("username", &qdmi::SessionParameters::username)
      .def_rw("password", &qdmi::SessionParameters::password)
      .def_rw("custom1", &qdmi::SessionParameters::custom1)
      .def_rw("custom2", &qdmi::SessionParameters::custom2)
      .def_rw("custom3", &qdmi::SessionParameters::custom3)
      .def_rw("custom4", &qdmi::SessionParameters::custom4)
      .def_rw("custom5", &qdmi::SessionParameters::custom5);

  auto definition = nb::class_<qdmi::DeviceDefinition>(
      m, "DeviceDefinition", "A side-effect-free QDMI device registration.");
  definition
      .def(
          "__init__",
          [](qdmi::DeviceDefinition* self, std::string id,
             std::filesystem::path library, std::string prefix, std::string abi,
             bool enabled, qdmi::SessionParameters session) {
            new (self) qdmi::DeviceDefinition{.id = std::move(id),
                                              .library = std::move(library),
                                              .abi = std::move(abi),
                                              .prefix = std::move(prefix),
                                              .enabled = enabled,
                                              .session = std::move(session)};
          },
          "id"_a, "library"_a, "prefix"_a, nb::kw_only(), "abi"_a = "qdmi-v1",
          "enabled"_a = true, "session"_a = qdmi::SessionParameters{})
      .def_rw("id", &qdmi::DeviceDefinition::id)
      .def_rw("library", &qdmi::DeviceDefinition::library)
      .def_rw("abi", &qdmi::DeviceDefinition::abi)
      .def_rw("prefix", &qdmi::DeviceDefinition::prefix)
      .def_rw("enabled", &qdmi::DeviceDefinition::enabled)
      .def_rw("session", &qdmi::DeviceDefinition::session)
      .def_ro("source", &qdmi::DeviceDefinition::source);

  auto configOptions = nb::class_<qdmi::ConfigOptions>(
      m, "ConfigOptions", "Controls QDMI configuration discovery.");
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
                                    .isolated = isolated,
                                    .runtimeOverrides =
                                        std::move(runtimeOverrides)};
        if (inlineJson) {
          options.inlineOverrides = Json::parse(*inlineJson);
        }
        new (self) qdmi::ConfigOptions(std::move(options));
      },
      nb::kw_only(), "config_root"_a = std::nullopt,
      "explicit_file"_a = std::nullopt, "base_directory"_a = std::nullopt,
      "isolated"_a = false, "inline_json"_a = std::nullopt,
      "runtime_overrides"_a = std::vector<qdmi::DeviceDefinition>{});

  // Job class
  auto job = nb::class_<qdmi::Job>(
      m, "Job", "A job represents a submitted quantum program execution.");

  job.def("check", &qdmi::Job::check, "Returns the current status of the job.");

  job.def("wait", &qdmi::Job::wait, "timeout"_a = 0,
          R"pb(Waits for the job to complete.

Args:
    timeout: The maximum time to wait in seconds. If 0, waits indefinitely.

Returns:
    True if the job completed within the timeout, False otherwise.)pb");

  job.def("cancel", &qdmi::Job::cancel, "Cancels the job.");

  job.def("get_shots", &qdmi::Job::getShots,
          "Returns the raw shot results from the job.");

  job.def("get_counts", &qdmi::Job::getCounts,
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
Use ``bytes`` to retrieve the value without interpretation. Returns ``None``
when the custom slot is unsupported.)pb");

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
Use ``bytes`` to retrieve the value without interpretation. Returns ``None``
when the custom slot is unsupported.)pb");

  job.def_prop_ro("id", &qdmi::Job::getId, "The job ID.");

  job.def_prop_ro("program_format", &qdmi::Job::getProgramFormat,
                  "The format of the submitted program.");

  job.def_prop_ro("program", &qdmi::Job::getProgram, "The submitted program.");

  job.def_prop_ro("num_shots", &qdmi::Job::getNumShots, "The number of shots.");

  job.def(nb::self == nb::self,
          nb::sig("def __eq__(self, arg: object, /) -> bool"));
  job.def(nb::self != nb::self,
          nb::sig("def __ne__(self, arg: object, /) -> bool"));

  // JobStatus enum
  nb::enum_<qdmi::JobStatus>(job, "Status", "Enumeration of job status.")
      .value("CREATED", qdmi::JobStatus::Created)
      .value("SUBMITTED", qdmi::JobStatus::Submitted)
      .value("QUEUED", qdmi::JobStatus::Queued)
      .value("RUNNING", qdmi::JobStatus::Running)
      .value("DONE", qdmi::JobStatus::Done)
      .value("CANCELED", qdmi::JobStatus::Canceled)
      .value("FAILED", qdmi::JobStatus::Failed);

  // ProgramFormat enum
  nb::enum_<qdmi::ProgramFormat>(m, "ProgramFormat",
                                 "Enumeration of program formats.")
      .value("QASM2", qdmi::ProgramFormat::Qasm2)
      .value("QASM3", qdmi::ProgramFormat::Qasm3)
      .value("QIR_BASE_STRING", qdmi::ProgramFormat::QirBaseString)
      .value("QIR_BASE_MODULE", qdmi::ProgramFormat::QirBaseModule)
      .value("QIR_ADAPTIVE_STRING", qdmi::ProgramFormat::QirAdaptiveString)
      .value("QIR_ADAPTIVE_MODULE", qdmi::ProgramFormat::QirAdaptiveModule)
      .value("CALIBRATION", qdmi::ProgramFormat::Calibration)
      .value("QPY", qdmi::ProgramFormat::Qpy)
      .value("IQM_JSON", qdmi::ProgramFormat::IqmJson)
      .value("BATCH_JOB", qdmi::ProgramFormat::BatchJob)
      .value("CUSTOM1", qdmi::ProgramFormat::Custom1)
      .value("CUSTOM2", qdmi::ProgramFormat::Custom2)
      .value("CUSTOM3", qdmi::ProgramFormat::Custom3)
      .value("CUSTOM4", qdmi::ProgramFormat::Custom4)
      .value("CUSTOM5", qdmi::ProgramFormat::Custom5);

  nb::enum_<qdmi::CustomProperty>(
      m, "CustomProperty",
      "An implementation-defined custom property or result slot.")
      .value("CUSTOM1", qdmi::CustomProperty::Custom1)
      .value("CUSTOM2", qdmi::CustomProperty::Custom2)
      .value("CUSTOM3", qdmi::CustomProperty::Custom3)
      .value("CUSTOM4", qdmi::CustomProperty::Custom4)
      .value("CUSTOM5", qdmi::CustomProperty::Custom5);

  // Device class
  auto device = nb::class_<qdmi::Device>(
      m, "Device",
      "A device represents a quantum device with its properties and "
      "capabilities.");

  nb::enum_<qdmi::DeviceStatus>(device, "Status",
                                "Enumeration of device status.")
      .value("OFFLINE", qdmi::DeviceStatus::Offline)
      .value("IDLE", qdmi::DeviceStatus::Idle)
      .value("BUSY", qdmi::DeviceStatus::Busy)
      .value("ERROR", qdmi::DeviceStatus::Error)
      .value("MAINTENANCE", qdmi::DeviceStatus::Maintenance)
      .value("CALIBRATION", qdmi::DeviceStatus::Calibration);

  device.def("name", &qdmi::Device::getName, "Returns the name of the device.");

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
Use ``bytes`` to retrieve the value without interpretation. Returns ``None``
when the custom slot is unsupported.)pb");

  device.def("submit_job", &qdmi::Device::submitJob, "program"_a,
             "program_format"_a, "num_shots"_a, nb::kw_only(),
             "custom1"_a = nb::none(), "custom2"_a = nb::none(),
             "custom3"_a = nb::none(), "custom4"_a = nb::none(),
             "custom5"_a = nb::none(), nb::rv_policy::reference_internal,
             "Submits a job to the device.");

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
Use ``bytes`` to retrieve the value without interpretation. Returns ``None``
when the custom slot is unsupported.)pb");

  operation.def("__repr__", [](const qdmi::Operation& op) {
    return "<Operation name=\"" + op.getName() + "\">";
  });

  operation.def(nb::self == nb::self,
                nb::sig("def __eq__(self, arg: object, /) -> bool"));
  operation.def(nb::self != nb::self,
                nb::sig("def __ne__(self, arg: object, /) -> bool"));

  auto manager = nb::class_<qdmi::DeviceManager>(
      m, "DeviceManager", "Discovers and lazily opens QDMI devices.");
  manager
      .def(nb::init<const qdmi::ConfigOptions&>(),
           "options"_a = qdmi::ConfigOptions{})
      .def_prop_ro(
          "definitions",
          [](const qdmi::DeviceManager& self) { return self.definitions(); })
      .def("register_device", &qdmi::DeviceManager::registerDevice,
           "definition"_a, nb::kw_only(), "replace"_a = false)
      .def(
          "unregister_device",
          [](qdmi::DeviceManager& self, const std::string& id) {
            return self.unregisterDevice(id);
          },
          "id"_a)
      .def(
          "open",
          [](qdmi::DeviceManager& self, const std::string& id,
             const qdmi::SessionParameters& sessionOverrides) {
            return self.open(id, sessionOverrides);
          },
          "id"_a, nb::kw_only(),
          "session_overrides"_a = qdmi::SessionParameters{});
}

} // namespace mqt
