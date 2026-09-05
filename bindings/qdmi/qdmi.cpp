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
#include "qdmi/driver/Driver.hpp"
#include "qdmi/driver/SessionConfig.hpp"

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
  qdmiModule.doc() = "QDMI entities and access to MQT Core's QDMI driver.";
  auto driver = qdmiModule.def_submodule(
      "driver", "Register, discover, and open QDMI devices through MQT Core.");
  bindings::registerSlurm(qdmiModule);

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
  nb::class_<qdmi::DeviceDefinition>(
      driver, "DeviceDefinition",
      R"pb(A stable QDMI device registration that can be stored before loading.)pb")
      .def(
          "__init__",
          [](qdmi::DeviceDefinition* self, std::string deviceId,
             std::filesystem::path libraryPath, std::string prefix,
             const std::optional<std::string>& baseUrl = std::nullopt,
             const std::optional<std::string>& token = std::nullopt,
             const std::optional<std::filesystem::path>& authFile =
                 std::nullopt,
             const std::optional<std::string>& authUrl = std::nullopt,
             const std::optional<std::string>& username = std::nullopt,
             const std::optional<std::string>& password = std::nullopt,
             const std::optional<std::string>& deviceConfig = std::nullopt,
             const std::optional<std::filesystem::path>& deviceConfigFile =
                 std::nullopt,
             const std::optional<std::string>& custom1 = std::nullopt,
             const std::optional<std::string>& custom2 = std::nullopt,
             const std::optional<std::string>& custom3 = std::nullopt,
             const std::optional<std::string>& custom4 = std::nullopt,
             const std::optional<std::string>& custom5 = std::nullopt) {
            new (self) qdmi::DeviceDefinition{
                .id = std::move(deviceId),
                .library = std::move(libraryPath),
                .prefix = std::move(prefix),
                .session = qdmi::makeDeviceSessionConfig(
                    baseUrl, token, authFile, authUrl, username, password,
                    deviceConfig, deviceConfigFile, custom1, custom2, custom3,
                    custom4, custom5),
            };
          },
          "device_id"_a, "library_path"_a, "prefix"_a, nb::kw_only(),
          "base_url"_a = std::nullopt, "token"_a = std::nullopt,
          "auth_file"_a = std::nullopt, "auth_url"_a = std::nullopt,
          "username"_a = std::nullopt, "password"_a = std::nullopt,
          "device_config"_a = std::nullopt,
          "device_config_file"_a = std::nullopt, "custom1"_a = std::nullopt,
          "custom2"_a = std::nullopt, "custom3"_a = std::nullopt,
          "custom4"_a = std::nullopt, "custom5"_a = std::nullopt,
          R"pb(Create a device definition without loading its native library.

Args:
    device_id: Stable identifier used by :func:`open_device`.
    library_path: Path to the shared QDMI device library.
    prefix: Function prefix used by the library (for example, ``MY_DEVICE``).
    base_url: Optional base URL for the device API endpoint.
    token: Optional authentication token.
    auth_file: Optional path to an authentication file.
    auth_url: Optional authentication server URL.
    username: Optional authentication username.
    password: Optional authentication password.
    device_config: Optional inline JSON device description.
    device_config_file: Optional device-description JSON file.
    custom1: Optional custom configuration parameter 1.
    custom2: Optional custom configuration parameter 2.
    custom3: Optional custom configuration parameter 3.
    custom4: Optional custom configuration parameter 4.
    custom5: Optional custom configuration parameter 5.)pb")
      .def_ro("device_id", &qdmi::DeviceDefinition::id,
              R"pb(Stable identifier used to open the device.)pb")
      .def_ro("library_path", &qdmi::DeviceDefinition::library,
              R"pb(Path to the native QDMI device library.)pb")
      .def_ro("prefix", &qdmi::DeviceDefinition::prefix,
              R"pb(Prefix used for the QDMI device interface functions.)pb");

  driver.def(
      "register_device",
      [](qdmi::DeviceDefinition definition, const bool replace) {
        qdmi::Driver::get().registerDevice(std::move(definition), replace);
      },
      "definition"_a, nb::kw_only(), "replace"_a = false,
      R"pb(Register a QDMI device definition without loading its library.

Args:
    definition: Definition to validate and store.
    replace: Replace an existing definition if it has not been opened.

Raises:
    ValueError: If the definition is invalid or its ID is already registered.
    RuntimeError: If replacing an already opened ID.)pb");

  driver.def(
      "register_device_if_absent",
      [](qdmi::DeviceDefinition definition) {
        return qdmi::Driver::get().registerDeviceIfAbsent(
            std::move(definition));
      },
      "definition"_a,
      R"pb(Register a valid QDMI device definition if its ID is absent.

Existing and explicitly disabled IDs are not inserted. Invalid definitions
still raise.

Args:
    definition: Definition to validate and store.

Returns:
    bool: Whether the definition was inserted.

Raises:
    ValueError: If the definition is invalid.)pb");

  driver.def(
      "registered_device_ids",
      [] { return qdmi::Driver::get().registeredDeviceIds(); },
      R"pb(Return registered, enabled QDMI device IDs in registration order.

This includes devices registered at runtime and does not load native device
libraries or expose their definitions.)pb");

  driver.def(
      "open_device",
      [](const std::string& deviceId, std::optional<std::string> baseUrl,
         std::optional<std::string> token,
         std::optional<std::filesystem::path> authFile,
         std::optional<std::string> authUrl,
         std::optional<std::string> username,
         std::optional<std::string> password,
         std::optional<std::string> deviceConfig,
         std::optional<std::filesystem::path> deviceConfigFile,
         std::optional<std::string> custom1, std::optional<std::string> custom2,
         std::optional<std::string> custom3, std::optional<std::string> custom4,
         std::optional<std::string> custom5) {
        const auto overrides = qdmi::makeDeviceSessionConfig(
            std::move(baseUrl), std::move(token), std::move(authFile),
            std::move(authUrl), std::move(username), std::move(password),
            std::move(deviceConfig), std::move(deviceConfigFile),
            std::move(custom1), std::move(custom2), std::move(custom3),
            std::move(custom4), std::move(custom5));
        return qdmi::Session::openDevice(deviceId, overrides);
      },
      "device_id"_a, nb::kw_only(), "base_url"_a = std::nullopt,
      "token"_a = std::nullopt, "auth_file"_a = std::nullopt,
      "auth_url"_a = std::nullopt, "username"_a = std::nullopt,
      "password"_a = std::nullopt, "device_config"_a = std::nullopt,
      "device_config_file"_a = std::nullopt, "custom1"_a = std::nullopt,
      "custom2"_a = std::nullopt, "custom3"_a = std::nullopt,
      "custom4"_a = std::nullopt, "custom5"_a = std::nullopt,
      R"pb(Open a registered QDMI device by stable ID.

Every call creates a fresh device session while keeping the stable registration
unchanged. Opening the device loads trusted native device code.

Args:
    device_id: Stable ID of a registered device.
    base_url: Optional base URL override for the device API endpoint.
    token: Optional authentication token override.
    auth_file: Optional authentication-file override.
    auth_url: Optional authentication server URL override.
    username: Optional authentication username override.
    password: Optional authentication password override.
    device_config: Optional inline JSON device-description override.
    device_config_file: Optional device-description JSON file override.
    custom1: Optional custom configuration parameter 1 override.
    custom2: Optional custom configuration parameter 2 override.
    custom3: Optional custom configuration parameter 3 override.
    custom4: Optional custom configuration parameter 4 override.
    custom5: Optional custom configuration parameter 5 override.

Returns:
    mqt.core.qdmi.Device: The opened device, ready for direct backend construction.

Raises:
    IndexError: If the ID is not registered.
    RuntimeError: If the device library cannot be loaded or initialized.)pb");
}

} // namespace mqt
