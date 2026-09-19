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
#include "qdmi/common/Diagnostics.hpp"
#include "qdmi/driver/Driver.hpp"

#include "qdmi/client.h"

#include <algorithm>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iterator>
#include <map>
#include <memory>
#include <optional>
#include <regex>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace qdmi {
namespace detail {
Result<std::vector<std::string>> parseShots(const std::string_view shots,
                                            const size_t numShots) {
  if (numShots == 0) {
    if (!shots.empty()) {
      return Error{
          .status = QDMI_ERROR_FATAL,
          .message = "Number of shots mismatch",
      };
    }
    return {};
  }

  std::vector<std::string> parsed;
  parsed.reserve(numShots);
  size_t start = 0;
  while (true) {
    const auto end = shots.find(',', start);
    parsed.emplace_back(shots.substr(start, end - start));
    if (end == std::string_view::npos) {
      break;
    }
    start = end + 1;
  }
  if (parsed.size() != numShots) {
    return Error{
        .status = QDMI_ERROR_FATAL,
        .message = "Number of shots mismatch",
    };
  }
  return parsed;
}

} // namespace detail

namespace {
/// Rejects the formats that `submitJob` cannot carry.
/// A batch job's program is a list of job handles rather than a byte blob, so
/// this API cannot express it at all. A calibration run has its own entry
/// point, because its payload is optional and it takes no shot count.
std::optional<Error>
rejectUnsupportedProgramFormat(const QDMI_Program_Format format) {
  if (format == QDMI_PROGRAM_FORMAT_BATCHJOB) {
    return Error{
        .status = QDMI_ERROR_INVALIDARGUMENT,
        .message = "MQT Core does not support batch jobs. A batch "
                   "job's program is a list "
                   "of job handles, which this API cannot express",
    };
  }
  if (format == QDMI_PROGRAM_FORMAT_CALIBRATION) {
    return Error{
        .status = QDMI_ERROR_INVALIDARGUMENT,
        .message =
            "Use submitCalibrationJob (submit_calibration_job in Python) to "
            "trigger a calibration run",
    };
  }
  return std::nullopt;
}
template <typename T>
Result<std::map<std::string, T>>
getSparseResult(QDMI_Job job, const QDMI_Job_Result keysResult,
                const QDMI_Job_Result valuesResult,
                const std::string& description, const std::string& valueType,
                const std::string& mismatch) {
  size_t keysSize = 0;
  if (auto error = checkError(
          QDMI_job_get_results(job, keysResult, 0, nullptr, &keysSize),
          "Querying " + description + " keys size")) {
    return std::move(*error);
  }

  if (keysSize == 0) {
    return {};
  }

  std::string keys(keysSize, '\0');
  if (auto error = checkError(
          QDMI_job_get_results(job, keysResult, keysSize, keys.data(), nullptr),
          "Querying " + description + " keys")) {
    return std::move(*error);
  }
  keys.pop_back();

  size_t valuesSize = 0;
  if (auto error = checkError(
          QDMI_job_get_results(job, valuesResult, 0, nullptr, &valuesSize),
          "Querying " + description + " values size")) {
    return std::move(*error);
  }

  if (valuesSize % sizeof(T) != 0) {
    return Error{
        .status = QDMI_ERROR_FATAL,
        .message = "Invalid " + description +
                   " values size: not a multiple of " + valueType,
    };
  }

  std::vector<T> values(valuesSize / sizeof(T));
  if (auto error =
          checkError(QDMI_job_get_results(job, valuesResult, valuesSize,
                                          values.data(), nullptr),
                     "Querying " + description + " values")) {
    return std::move(*error);
  }

  /// Parse the comma-separated keys.
  std::map<std::string, T> result;
  if (keys.empty() && values.size() == 1) {
    result[""] = values.front();
    return result;
  }
  std::istringstream keysStream(keys);
  std::string key;
  size_t idx = 0;
  while (std::getline(keysStream, key, ',')) {
    if (idx >= values.size()) {
      return Error{.status = QDMI_ERROR_FATAL, .message = mismatch};
    }
    result[key] = values[idx];
    ++idx;
  }

  if (idx != values.size()) {
    return Error{.status = QDMI_ERROR_FATAL, .message = mismatch};
  }
  return result;
}
} // namespace

Result<size_t> Site::getIndex() const {
  return queryProperty<size_t>(QDMI_SITE_PROPERTY_INDEX);
}
Result<std::optional<uint64_t>> Site::getT1() const {
  return queryProperty<std::optional<uint64_t>>(QDMI_SITE_PROPERTY_T1);
}
Result<std::optional<uint64_t>> Site::getT2() const {
  return queryProperty<std::optional<uint64_t>>(QDMI_SITE_PROPERTY_T2);
}
Result<std::optional<std::string>> Site::getName() const {
  return queryProperty<std::optional<std::string>>(QDMI_SITE_PROPERTY_NAME);
}
Result<std::optional<int64_t>> Site::getXCoordinate() const {
  return queryProperty<std::optional<int64_t>>(QDMI_SITE_PROPERTY_XCOORDINATE);
}
Result<std::optional<int64_t>> Site::getYCoordinate() const {
  return queryProperty<std::optional<int64_t>>(QDMI_SITE_PROPERTY_YCOORDINATE);
}
Result<std::optional<int64_t>> Site::getZCoordinate() const {
  return queryProperty<std::optional<int64_t>>(QDMI_SITE_PROPERTY_ZCOORDINATE);
}
Result<bool> Site::isZone() const {
  auto result = queryProperty<std::optional<bool>>(QDMI_SITE_PROPERTY_ISZONE);
  if (auto* error = std::get_if<Error>(&result)) {
    return std::move(*error);
  }
  return std::get<0>(result).value_or(false);
}
Result<std::optional<uint64_t>> Site::getXExtent() const {
  return queryProperty<std::optional<uint64_t>>(QDMI_SITE_PROPERTY_XEXTENT);
}
Result<std::optional<uint64_t>> Site::getYExtent() const {
  return queryProperty<std::optional<uint64_t>>(QDMI_SITE_PROPERTY_YEXTENT);
}
Result<std::optional<uint64_t>> Site::getZExtent() const {
  return queryProperty<std::optional<uint64_t>>(QDMI_SITE_PROPERTY_ZEXTENT);
}
Result<std::optional<uint64_t>> Site::getModuleIndex() const {
  return queryProperty<std::optional<uint64_t>>(QDMI_SITE_PROPERTY_MODULEINDEX);
}
Result<std::optional<uint64_t>> Site::getSubmoduleIndex() const {
  return queryProperty<std::optional<uint64_t>>(
      QDMI_SITE_PROPERTY_SUBMODULEINDEX);
}
Result<std::string>
Operation::getName(const std::vector<Site>& sites,
                   const std::vector<double>& params) const {
  return queryProperty<std::string>(QDMI_OPERATION_PROPERTY_NAME, sites,
                                    params);
}
Result<std::optional<size_t>>
Operation::getQubitsNum(const std::vector<Site>& sites,
                        const std::vector<double>& params) const {
  return queryProperty<std::optional<size_t>>(QDMI_OPERATION_PROPERTY_QUBITSNUM,
                                              sites, params);
}
Result<size_t>
Operation::getParametersNum(const std::vector<Site>& sites,
                            const std::vector<double>& params) const {
  return queryProperty<size_t>(QDMI_OPERATION_PROPERTY_PARAMETERSNUM, sites,
                               params);
}
Result<std::optional<uint64_t>>
Operation::getDuration(const std::vector<Site>& sites,
                       const std::vector<double>& params) const {
  return queryProperty<std::optional<uint64_t>>(
      QDMI_OPERATION_PROPERTY_DURATION, sites, params);
}
Result<std::optional<double>>
Operation::getFidelity(const std::vector<Site>& sites,
                       const std::vector<double>& params) const {
  return queryProperty<std::optional<double>>(QDMI_OPERATION_PROPERTY_FIDELITY,
                                              sites, params);
}
Result<std::optional<uint64_t>>
Operation::getInteractionRadius(const std::vector<Site>& sites,
                                const std::vector<double>& params) const {
  return queryProperty<std::optional<uint64_t>>(
      QDMI_OPERATION_PROPERTY_INTERACTIONRADIUS, sites, params);
}
Result<std::optional<uint64_t>>
Operation::getBlockingRadius(const std::vector<Site>& sites,
                             const std::vector<double>& params) const {
  return queryProperty<std::optional<uint64_t>>(
      QDMI_OPERATION_PROPERTY_BLOCKINGRADIUS, sites, params);
}
Result<std::optional<double>>
Operation::getIdlingFidelity(const std::vector<Site>& sites,
                             const std::vector<double>& params) const {
  return queryProperty<std::optional<double>>(
      QDMI_OPERATION_PROPERTY_IDLINGFIDELITY, sites, params);
}
Result<bool> Operation::isZoned() const {
  auto result = queryProperty<std::optional<bool>>(
      QDMI_OPERATION_PROPERTY_ISZONED, {}, {});
  if (auto* error = std::get_if<Error>(&result)) {
    return std::move(*error);
  }
  return std::get<0>(result).value_or(false);
}
Result<std::optional<std::vector<Site>>> Operation::getSites() const {
  auto qdmiSitesResult = queryProperty<std::optional<std::vector<QDMI_Site>>>(
      QDMI_OPERATION_PROPERTY_SITES, {}, {});
  if (auto* error = std::get_if<Error>(&qdmiSitesResult)) {
    return std::move(*error);
  }
  auto& qdmiSites = std::get<0>(qdmiSitesResult);
  if (!qdmiSites.has_value()) {
    return std::nullopt;
  }
  std::vector<Site> returnedSites;
  returnedSites.reserve(qdmiSites->size());
  std::ranges::transform(
      *qdmiSites, std::back_inserter(returnedSites),
      [this](const QDMI_Site& site) -> Site { return {device_, site}; });
  return returnedSites;
}
Result<std::optional<std::vector<std::pair<Site, Site>>>>
Operation::getSitePairs() const {
  auto qubitsNum = getQubitsNum({}, {});
  if (auto* error = std::get_if<Error>(&qubitsNum)) {
    return std::move(*error);
  }
  if (std::get<0>(qubitsNum) != 2) {
    return std::nullopt;
  }
  auto zoned = isZoned();
  if (auto* error = std::get_if<Error>(&zoned)) {
    return std::move(*error);
  }
  if (std::get<0>(zoned)) {
    return std::nullopt;
  }
  auto sites = getSites();
  if (auto* error = std::get_if<Error>(&sites)) {
    return std::move(*error);
  }
  const auto& sitesOpt = std::get<0>(sites);
  if (!sitesOpt || sitesOpt->empty() || sitesOpt->size() % 2 != 0) {
    return std::nullopt;
  }
  std::vector<std::pair<Site, Site>> pairs;
  pairs.reserve(sitesOpt->size() / 2);
  for (size_t i = 0; i < sitesOpt->size(); i += 2) {
    pairs.emplace_back((*sitesOpt)[i], (*sitesOpt)[i + 1]);
  }
  return pairs;
}
Result<std::optional<uint64_t>>
Operation::getMeanShuttlingSpeed(const std::vector<Site>& sites,
                                 const std::vector<double>& params) const {
  return queryProperty<std::optional<uint64_t>>(
      QDMI_OPERATION_PROPERTY_MEANSHUTTLINGSPEED, sites, params);
}
Result<std::string> Device::getName() const {
  return queryProperty<std::string>(QDMI_DEVICE_PROPERTY_NAME);
}

Result<std::string> Device::getVersion() const {
  return queryProperty<std::string>(QDMI_DEVICE_PROPERTY_VERSION);
}

Result<QDMI_Device_Status> Device::getStatus() const {
  return queryProperty<QDMI_Device_Status>(QDMI_DEVICE_PROPERTY_STATUS);
}

Result<std::string> Device::getLibraryVersion() const {
  return queryProperty<std::string>(QDMI_DEVICE_PROPERTY_LIBRARYVERSION);
}

Result<size_t> Device::getQubitsNum() const {
  return queryProperty<size_t>(QDMI_DEVICE_PROPERTY_QUBITSNUM);
}

Result<std::vector<Site>> Device::getSites() const {
  auto qdmiSitesResult =
      queryProperty<std::vector<QDMI_Site>>(QDMI_DEVICE_PROPERTY_SITES);
  if (auto* error = std::get_if<Error>(&qdmiSitesResult)) {
    return std::move(*error);
  }
  auto& qdmiSites = std::get<0>(qdmiSitesResult);
  std::vector<Site> sites;
  sites.reserve(qdmiSites.size());
  std::ranges::transform(
      qdmiSites, std::back_inserter(sites),
      [this](const QDMI_Site& site) -> Site { return {device_, site}; });
  return sites;
}

Result<std::vector<Site>> Device::getRegularSites() const {
  auto result = getSites();
  if (auto* error = std::get_if<Error>(&result)) {
    return std::move(*error);
  }
  std::vector<Site> sites;
  for (auto& site : std::get<0>(result)) {
    auto zone = site.isZone();
    if (auto* error = std::get_if<Error>(&zone)) {
      return std::move(*error);
    }
    if (!std::get<0>(zone)) {
      sites.emplace_back(std::move(site));
    }
  }
  return sites;
}

Result<std::vector<Site>> Device::getZones() const {
  auto result = getSites();
  if (auto* error = std::get_if<Error>(&result)) {
    return std::move(*error);
  }
  std::vector<Site> sites;
  for (auto& site : std::get<0>(result)) {
    auto zone = site.isZone();
    if (auto* error = std::get_if<Error>(&zone)) {
      return std::move(*error);
    }
    if (std::get<0>(zone)) {
      sites.emplace_back(std::move(site));
    }
  }
  return sites;
}

Result<std::vector<Operation>> Device::getOperations() const {
  auto qdmiOperationsResult = queryProperty<std::vector<QDMI_Operation>>(
      QDMI_DEVICE_PROPERTY_OPERATIONS);
  if (auto* error = std::get_if<Error>(&qdmiOperationsResult)) {
    return std::move(*error);
  }
  auto& qdmiOperations = std::get<0>(qdmiOperationsResult);
  return wrapOperations(qdmiOperations);
}

Result<std::optional<std::vector<Operation>>>
Device::queryCustomOperations(const CustomProperty property) const {
  auto propertyResult = detail::toDeviceProperty(property);
  if (auto* error = std::get_if<Error>(&propertyResult)) {
    return std::move(*error);
  }
  const auto qdmiProperty = std::get<0>(propertyResult);
  auto handlesResult = detail::queryHandleArray<QDMI_Operation>(
      [this, qdmiProperty](const size_t size, void* value, size_t* sizeRet) {
        return QDMI_device_query_device_property(device_.get(), qdmiProperty,
                                                 size, value, sizeRet);
      },
      "custom operation list " +
          std::to_string(static_cast<unsigned>(property)));
  if (auto* error = std::get_if<Error>(&handlesResult)) {
    return std::move(*error);
  }
  auto& handles = std::get<0>(handlesResult);
  if (!handles.has_value()) {
    return std::nullopt;
  }
  return wrapOperations(*handles);
}

std::vector<Operation>
Device::wrapOperations(const std::span<const QDMI_Operation> operations) const {
  std::vector<Operation> wrappedOperations;
  wrappedOperations.reserve(operations.size());
  std::ranges::transform(
      operations, std::back_inserter(wrappedOperations),
      [this](const QDMI_Operation& op) -> Operation { return {device_, op}; });
  return wrappedOperations;
}

Result<std::optional<std::vector<std::pair<Site, Site>>>>
Device::getCouplingMap() const {
  auto qdmiCouplingMapResult = queryProperty<
      std::optional<std::vector<std::pair<QDMI_Site, QDMI_Site>>>>(
      QDMI_DEVICE_PROPERTY_COUPLINGMAP);
  if (auto* error = std::get_if<Error>(&qdmiCouplingMapResult)) {
    return std::move(*error);
  }
  auto& qdmiCouplingMap = std::get<0>(qdmiCouplingMapResult);
  if (!qdmiCouplingMap.has_value()) {
    return std::nullopt;
  }

  std::vector<std::pair<Site, Site>> couplingMap;
  couplingMap.reserve(qdmiCouplingMap->size());
  std::ranges::transform(*qdmiCouplingMap, std::back_inserter(couplingMap),
                         [this](const std::pair<QDMI_Site, QDMI_Site>& pair)
                             -> std::pair<Site, Site> {
                           return {
                               Site{device_, pair.first},
                               Site{device_, pair.second},
                           };
                         });
  return couplingMap;
}

Result<std::optional<size_t>> Device::getNeedsCalibration() const {
  return queryProperty<std::optional<size_t>>(
      QDMI_DEVICE_PROPERTY_NEEDSCALIBRATION);
}

Result<std::optional<size_t>> Device::getQueueLength() const {
  return queryProperty<std::optional<size_t>>(QDMI_DEVICE_PROPERTY_QUEUELENGTH);
}

Result<std::optional<std::string>> Device::getLengthUnit() const {
  return queryProperty<std::optional<std::string>>(
      QDMI_DEVICE_PROPERTY_LENGTHUNIT);
}

Result<std::optional<double>> Device::getLengthScaleFactor() const {
  return queryProperty<std::optional<double>>(
      QDMI_DEVICE_PROPERTY_LENGTHSCALEFACTOR);
}

Result<std::optional<std::string>> Device::getDurationUnit() const {
  return queryProperty<std::optional<std::string>>(
      QDMI_DEVICE_PROPERTY_DURATIONUNIT);
}

Result<std::optional<double>> Device::getDurationScaleFactor() const {
  return queryProperty<std::optional<double>>(
      QDMI_DEVICE_PROPERTY_DURATIONSCALEFACTOR);
}

Result<std::optional<uint64_t>> Device::getMinAtomDistance() const {
  return queryProperty<std::optional<uint64_t>>(
      QDMI_DEVICE_PROPERTY_MINATOMDISTANCE);
}

Result<std::vector<QDMI_Program_Format>>
Device::getSupportedProgramFormats() const {
  return queryProperty<std::vector<QDMI_Program_Format>>(
      QDMI_DEVICE_PROPERTY_SUPPORTEDPROGRAMFORMATS);
}

Result<std::vector<Device>> Device::getChildDevices() const {
  size_t size = 0;
  auto result = QDMI_device_query_device_property(
      device_.get(), QDMI_DEVICE_PROPERTY_CHILDDEVICES, 0, nullptr, &size);
  if (result == QDMI_ERROR_NOTSUPPORTED) {
    return {};
  }
  if (auto error = checkError(result, "Querying child devices size")) {
    return std::move(*error);
  }
  if (size % sizeof(QDMI_Device) != 0) {
    return Error{
        .status = QDMI_ERROR_FATAL,
        .message = "Invalid child device list size",
    };
  }

  std::vector<QDMI_Device> handles(size / sizeof(QDMI_Device));
  if (size != 0) {
    result = QDMI_device_query_device_property(
        device_.get(), QDMI_DEVICE_PROPERTY_CHILDDEVICES, size,
        static_cast<void*>(handles.data()), nullptr);
    if (auto error = checkError(result, "Querying child devices")) {
      return std::move(*error);
    }
  }

  std::vector<Device> devices;
  devices.reserve(handles.size());
  std::ranges::transform(
      handles, std::back_inserter(devices),
      [this](QDMI_Device_impl_d* const handle) {
        return Device(std::shared_ptr<QDMI_Device_impl_d>(device_, handle));
      });
  return devices;
}

Result<Job>
Device::submitJob(const std::string& program, const QDMI_Program_Format format,
                  const size_t numShots,
                  const std::optional<CustomJobParameter>& custom1,
                  const std::optional<CustomJobParameter>& custom2,
                  const std::optional<CustomJobParameter>& custom3,
                  const std::optional<CustomJobParameter>& custom4,
                  const std::optional<CustomJobParameter>& custom5) const {
  if (isBinaryProgramFormat(format)) {
    return Error{
        .status = QDMI_ERROR_INVALIDARGUMENT,
        .message = "Binary program formats require exact-byte submission",
    };
  }
  if (auto error = rejectUnsupportedProgramFormat(format)) {
    return std::move(*error);
  }

  const auto bytes = std::as_bytes(
      std::span(program.c_str(), static_cast<size_t>(program.size() + 1)));
  return submitJob(bytes, format, numShots, custom1, custom2, custom3, custom4,
                   custom5);
}

Result<Job>
Device::submitJob(const std::string& program, const QDMI_Program_Format format,
                  const std::optional<CustomJobParameter>& custom1,
                  const std::optional<CustomJobParameter>& custom2,
                  const std::optional<CustomJobParameter>& custom3,
                  const std::optional<CustomJobParameter>& custom4,
                  const std::optional<CustomJobParameter>& custom5) const {
  if (isBinaryProgramFormat(format)) {
    return Error{
        .status = QDMI_ERROR_INVALIDARGUMENT,
        .message = "Binary program formats require exact-byte submission",
    };
  }
  if (auto error = rejectUnsupportedProgramFormat(format)) {
    return std::move(*error);
  }

  const auto bytes = std::as_bytes(
      std::span(program.c_str(), static_cast<size_t>(program.size() + 1)));
  return submitJob(bytes, format, custom1, custom2, custom3, custom4, custom5);
}

Result<Job>
Device::submitJob(const std::span<const std::byte> program,
                  const QDMI_Program_Format format, const size_t numShots,
                  const std::optional<CustomJobParameter>& custom1,
                  const std::optional<CustomJobParameter>& custom2,
                  const std::optional<CustomJobParameter>& custom3,
                  const std::optional<CustomJobParameter>& custom4,
                  const std::optional<CustomJobParameter>& custom5) const {
  if (auto error = rejectUnsupportedProgramFormat(format)) {
    return std::move(*error);
  }

  return submitJobImpl(format, program, numShots, custom1, custom2, custom3,
                       custom4, custom5);
}

Result<Job>
Device::submitJob(const std::span<const std::byte> program,
                  const QDMI_Program_Format format,
                  const std::optional<CustomJobParameter>& custom1,
                  const std::optional<CustomJobParameter>& custom2,
                  const std::optional<CustomJobParameter>& custom3,
                  const std::optional<CustomJobParameter>& custom4,
                  const std::optional<CustomJobParameter>& custom5) const {
  if (auto error = rejectUnsupportedProgramFormat(format)) {
    return std::move(*error);
  }

  return submitJobImpl(format, program, std::nullopt, custom1, custom2, custom3,
                       custom4, custom5);
}

Result<Job>
Device::submitJobImpl(const QDMI_Program_Format format,
                      const std::optional<std::span<const std::byte>> program,
                      const std::optional<size_t> numShots,
                      const std::optional<CustomJobParameter>& custom1,
                      const std::optional<CustomJobParameter>& custom2,
                      const std::optional<CustomJobParameter>& custom3,
                      const std::optional<CustomJobParameter>& custom4,
                      const std::optional<CustomJobParameter>& custom5) const {
  QDMI_Job job = nullptr;
  if (auto error = checkError(QDMI_device_create_job(device_.get(), &job),
                              "Creating job")) {
    return std::move(*error);
  }
  Job jobWrapper{job, device_};

  if (auto error = checkError(
          QDMI_job_set_parameter(jobWrapper, QDMI_JOB_PARAMETER_PROGRAMFORMAT,
                                 sizeof(format), &format),
          "Setting program format")) {
    return std::move(*error);
  }
  if (program.has_value()) {
    if (auto error = checkError(
            QDMI_job_set_parameter(jobWrapper, QDMI_JOB_PARAMETER_PROGRAM,
                                   program->size(), program->data()),
            "Setting program")) {
      return std::move(*error);
    }
  }
  if (numShots.has_value()) {
    if (auto error = checkError(
            QDMI_job_set_parameter(jobWrapper, QDMI_JOB_PARAMETER_SHOTSNUM,
                                   sizeof(*numShots), &*numShots),
            "Setting number of shots")) {
      return std::move(*error);
    }
  }

  if (custom1.has_value()) {
    if (auto error = setCustomJobParam(jobWrapper, QDMI_JOB_PARAMETER_CUSTOM1,
                                       *custom1)) {
      return std::move(*error);
    }
  }
  if (custom2.has_value()) {
    if (auto error = setCustomJobParam(jobWrapper, QDMI_JOB_PARAMETER_CUSTOM2,
                                       *custom2)) {
      return std::move(*error);
    }
  }
  if (custom3.has_value()) {
    if (auto error = setCustomJobParam(jobWrapper, QDMI_JOB_PARAMETER_CUSTOM3,
                                       *custom3)) {
      return std::move(*error);
    }
  }
  if (custom4.has_value()) {
    if (auto error = setCustomJobParam(jobWrapper, QDMI_JOB_PARAMETER_CUSTOM4,
                                       *custom4)) {
      return std::move(*error);
    }
  }
  if (custom5.has_value()) {
    if (auto error = setCustomJobParam(jobWrapper, QDMI_JOB_PARAMETER_CUSTOM5,
                                       *custom5)) {
      return std::move(*error);
    }
  }

  if (auto error = checkError(QDMI_job_submit(jobWrapper), "Submitting job")) {
    return std::move(*error);
  }
  return jobWrapper;
}

Result<Job> Device::submitCalibrationJob(
    const std::optional<std::span<const std::byte>> program,
    const std::optional<CustomJobParameter>& custom1,
    const std::optional<CustomJobParameter>& custom2,
    const std::optional<CustomJobParameter>& custom3,
    const std::optional<CustomJobParameter>& custom4,
    const std::optional<CustomJobParameter>& custom5) const {
  const auto payload =
      program.has_value() && !program->empty() ? program : std::nullopt;
  return submitJobImpl(QDMI_PROGRAM_FORMAT_CALIBRATION, payload, std::nullopt,
                       custom1, custom2, custom3, custom4, custom5);
}

Result<Job> Device::submitCalibrationJob(
    const std::string& program,
    const std::optional<CustomJobParameter>& custom1,
    const std::optional<CustomJobParameter>& custom2,
    const std::optional<CustomJobParameter>& custom3,
    const std::optional<CustomJobParameter>& custom4,
    const std::optional<CustomJobParameter>& custom5) const {
  const auto bytes = std::as_bytes(
      std::span(program.c_str(), static_cast<size_t>(program.size() + 1)));
  return submitCalibrationJob(bytes, custom1, custom2, custom3, custom4,
                              custom5);
}

Result<Job> Device::retrieveJobById(const std::string_view jobId) const {
  const std::string id{jobId};
  QDMI_Job job = nullptr;
  if (auto error = checkError(
          QDMI_session_retrieve_job_by_id(device_.get(), id.c_str(), &job),
          "Retrieving job")) {
    return std::move(*error);
  }
  return Job{job, device_};
}

std::optional<Error>
Device::setCustomJobParam(QDMI_Job job, const QDMI_Job_Parameter param,
                          const CustomJobParameter& value) {
  return std::visit(
      [&]<typename CustomValue>(
          const CustomValue& customValue) -> std::optional<Error> {
        using T = std::decay_t<CustomValue>;
        if constexpr (std::is_same_v<T, std::string>) {
          if (auto error = checkError(
                  QDMI_job_set_parameter(job, param, customValue.size() + 1,
                                         customValue.c_str()),
                  "Setting custom parameter")) {
            return std::move(error);
          }
        } else {
          static_assert(std::is_trivially_copyable_v<T>,
                        "Custom job parameters must be trivially copyable");
          if (auto error = checkError(
                  QDMI_job_set_parameter(job, param, sizeof(T), &customValue),
                  "Setting custom parameter")) {
            return std::move(error);
          }
        }
        return std::nullopt;
      },
      value);
}

Result<QDMI_Job_Status> Job::check() const {
  QDMI_Job_Status status{};
  if (auto error = checkError(QDMI_job_check(job_.get(), &status),
                              "Checking job status")) {
    return std::move(*error);
  }
  return status;
}

Result<bool> Job::wait(const size_t timeout) const {
  const auto ret = QDMI_job_wait(job_.get(), timeout);
  if (ret == QDMI_SUCCESS) {
    return true;
  }
  if (ret == QDMI_ERROR_TIMEOUT) {
    return false;
  }
  if (auto error = checkError(ret, "Waiting for job")) {
    return std::move(*error);
  }
  return true;
}

std::optional<Error> Job::cancel() const {
  return checkError(QDMI_job_cancel(job_.get()), "Cancelling job");
}

auto Job::operator=(Job&& other) noexcept -> Job& {
  if (this != &other) {
    // Release the current job while its owning device session is still alive.
    job_.reset();
    device_ = std::move(other.device_);
    job_ = std::move(other.job_);
  }
  return *this;
}

Result<std::string> Job::getId() const {
  return detail::queryProperty<std::string>(
      [this](const size_t size, void* value, size_t* sizeRet) {
        return QDMI_job_query_property(job_.get(), QDMI_JOB_PROPERTY_ID, size,
                                       value, sizeRet);
      },
      "Querying job ID", "Querying job ID size");
}

Result<QDMI_Program_Format> Job::getProgramFormat() const {
  QDMI_Program_Format format{};
  if (auto error = checkError(
          QDMI_job_query_property(job_.get(), QDMI_JOB_PROPERTY_PROGRAMFORMAT,
                                  sizeof(format), &format, nullptr),
          "Querying program format")) {
    return std::move(*error);
  }
  return format;
}

Result<std::vector<std::byte>> Job::getProgramBytes() const {
  size_t size = 0;
  if (auto error = checkError(QDMI_job_query_property(job_.get(),
                                                      QDMI_JOB_PROPERTY_PROGRAM,
                                                      0, nullptr, &size),
                              "Querying program size")) {
    return std::move(*error);
  }

  std::vector<std::byte> program(size);
  if (size != 0) {
    if (auto error = checkError(
            QDMI_job_query_property(job_.get(), QDMI_JOB_PROPERTY_PROGRAM, size,
                                    program.data(), nullptr),
            "Querying program")) {
      return std::move(*error);
    }
  }
  return program;
}

Result<std::string> Job::getProgram() const {
  auto formatResult = getProgramFormat();
  if (auto* error = std::get_if<Error>(&formatResult)) {
    return std::move(*error);
  }
  auto const& format = std::get<0>(formatResult);
  if (isBinaryProgramFormat(format)) {
    return Error{
        .status = QDMI_ERROR_INVALIDARGUMENT,
        .message = "Cannot decode a binary program as a string; use "
                   "getProgramBytes()",
    };
  }

  auto programResult = getProgramBytes();
  if (auto* error = std::get_if<Error>(&programResult)) {
    return std::move(*error);
  }
  auto& program = std::get<0>(programResult);
  if (program.empty() || program.back() != std::byte{0}) {
    return Error{
        .status = QDMI_ERROR_INVALIDARGUMENT,
        .message = "Cannot decode program as a null-terminated string; use "
                   "getProgramBytes() for binary payloads",
    };
  }
  return std::string(reinterpret_cast<const char*>(program.data()),
                     program.size() - 1);
}

Result<size_t> Job::getNumShots() const {
  size_t numShots = 0;
  if (auto error = checkError(
          QDMI_job_query_property(job_.get(), QDMI_JOB_PROPERTY_SHOTSNUM,
                                  sizeof(numShots), &numShots, nullptr),
          "Querying number of shots")) {
    return std::move(*error);
  }
  return numShots;
}

Result<std::optional<size_t>> Job::getQueuePosition() const {
  size_t queuePosition = 0;
  const auto result =
      QDMI_job_query_property(job_.get(), QDMI_JOB_PROPERTY_QUEUEPOSITION,
                              sizeof(queuePosition), &queuePosition, nullptr);
  return detail::queuePositionFromResult(result, queuePosition);
}

Result<std::vector<std::string>> Job::getShots() const {
  size_t shotsSize = 0;
  if (auto error =
          checkError(QDMI_job_get_results(job_.get(), QDMI_JOB_RESULT_SHOTS, 0,
                                          nullptr, &shotsSize),
                     "Querying shots size")) {
    return std::move(*error);
  }

  if (shotsSize == 0) {
    return {};
  }

  std::string shots(shotsSize, '\0');
  if (auto error =
          checkError(QDMI_job_get_results(job_.get(), QDMI_JOB_RESULT_SHOTS,
                                          shotsSize, shots.data(), nullptr),
                     "Querying shots")) {
    return std::move(*error);
  }
  shots.pop_back();

  auto count = getNumShots();
  if (auto* error = std::get_if<Error>(&count)) {
    return std::move(*error);
  }
  return detail::parseShots(shots, std::get<0>(count));
}

Result<std::map<std::string, size_t>> Job::getCounts() const {
  return getSparseResult<size_t>(
      job_.get(), QDMI_JOB_RESULT_HIST_KEYS, QDMI_JOB_RESULT_HIST_VALUES,
      "histogram", "size_t", "Histogram key/value count mismatch");
}

Result<std::vector<std::complex<double>>> Job::getDenseStateVector() const {
  size_t size = 0;
  if (auto error = checkError(
          QDMI_job_get_results(job_.get(), QDMI_JOB_RESULT_STATEVECTOR_DENSE, 0,
                               nullptr, &size),
          "Querying dense state vector size")) {
    return std::move(*error);
  }

  if (size % sizeof(std::complex<double>) != 0) {
    return Error{
        .status = QDMI_ERROR_FATAL,
        .message =
            "Invalid state vector size: not a multiple of complex<double>",
    };
  }

  std::vector<std::complex<double>> stateVector(size /
                                                sizeof(std::complex<double>));
  if (auto error = checkError(
          QDMI_job_get_results(job_.get(), QDMI_JOB_RESULT_STATEVECTOR_DENSE,
                               size, stateVector.data(), nullptr),
          "Querying dense state vector")) {
    return std::move(*error);
  }
  return stateVector;
}

Result<std::vector<double>> Job::getDenseProbabilities() const {
  size_t size = 0;
  if (auto error = checkError(
          QDMI_job_get_results(job_.get(), QDMI_JOB_RESULT_PROBABILITIES_DENSE,
                               0, nullptr, &size),
          "Querying dense probabilities size")) {
    return std::move(*error);
  }

  if (size % sizeof(double) != 0) {
    return Error{
        .status = QDMI_ERROR_FATAL,
        .message = "Invalid probabilities size: not a multiple of double",
    };
  }

  std::vector<double> probabilities(size / sizeof(double));
  if (auto error = checkError(
          QDMI_job_get_results(job_.get(), QDMI_JOB_RESULT_PROBABILITIES_DENSE,
                               size, probabilities.data(), nullptr),
          "Querying dense probabilities")) {
    return std::move(*error);
  }
  return probabilities;
}

Result<std::map<std::string, std::complex<double>>>
Job::getSparseStateVector() const {
  return getSparseResult<std::complex<double>>(
      job_.get(), QDMI_JOB_RESULT_STATEVECTOR_SPARSE_KEYS,
      QDMI_JOB_RESULT_STATEVECTOR_SPARSE_VALUES, "sparse state vector",
      "complex<double>", "Sparse state vector key/value count mismatch");
}

Result<std::map<std::string, double>> Job::getSparseProbabilities() const {
  return getSparseResult<double>(
      job_.get(), QDMI_JOB_RESULT_PROBABILITIES_SPARSE_KEYS,
      QDMI_JOB_RESULT_PROBABILITIES_SPARSE_VALUES, "sparse probabilities",
      "double", "Sparse probabilities key/value count mismatch");
}

Device Session::createSessionlessDevice(QDMI_Device device) {
  return Device(device);
}

Result<Device> Session::openDevice(const std::string_view id,
                                   const qdmi::DeviceSessionConfig& overrides) {
  auto device = Driver::get().openFresh(id, overrides);
  if (auto* error = std::get_if<Error>(&device)) {
    return std::move(*error);
  }
  return Device(std::get<0>(std::move(device)));
}

Result<std::vector<std::string>> Session::registeredDeviceIds() {
  return Driver::get().registeredDeviceIds();
}

Result<Session> Session::create(const SessionConfig& config) {
  Session session;
  QDMI_Session handle = nullptr;
  if (auto error =
          checkError(QDMI_session_alloc(&handle), "Allocating QDMI session")) {
    return std::move(*error);
  }
  session.session_.reset(handle);

  const auto setParameter =
      [&session](const std::optional<std::string>& value,
                 QDMI_Session_Parameter param) -> std::optional<Error> {
    if (value) {
      const auto status = static_cast<QDMI_STATUS>(QDMI_session_set_parameter(
          session.session_.get(), param, value->size() + 1, value->c_str()));
      if (status == QDMI_ERROR_NOTSUPPORTED) {
        // Optional parameter not supported by session - skip it
        qdmi::diagnostics::info("Session parameter {} not supported (skipped)",
                                qdmi::toString(param));
        return std::nullopt;
      }
      if (status == QDMI_SUCCESS) {
        return std::nullopt;
      }
      return checkError(status, std::string("Setting session parameter ") +
                                    qdmi::toString(param));
    }
    return std::nullopt;
  };

  if (config.authFile) {
    std::error_code error;
    if (!std::filesystem::exists(*config.authFile, error)) {
      return Error{
          .status = QDMI_ERROR_FATAL,
          .message = "Authentication file does not exist: " +
                     config.authFile->string(),
      };
    }
  }
  if (config.authUrl) {
    /// Match HTTP(S) URLs with bracketed IPv6, IPv4, localhost, or a domain.
    /// Apply the host word boundary only outside IPv6: ']' is not a word
    /// character.
    static const std::regex URL_PATTERN(
        R"(^https?://(?:\[[a-fA-F0-9:]+\]|(?:(?:\d{1,3}\.){3}\d{1,3}|localhost|(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6})\b)(?::\d+)?(?:[-a-zA-Z0-9()@:%_\+.~#?&/=]*)$)",
        std::regex::optimize);
    if (!std::regex_match(*config.authUrl, URL_PATTERN)) {
      return Error{
          .status = QDMI_ERROR_FATAL,
          .message = "Invalid URL format: " + *config.authUrl,
      };
    }
  }

  if (auto error = setParameter(config.token, QDMI_SESSION_PARAMETER_TOKEN)) {
    return std::move(*error);
  }
  if (config.authFile) {
    const std::optional authFile = config.authFile->string();
    if (auto error = setParameter(authFile, QDMI_SESSION_PARAMETER_AUTHFILE)) {
      return std::move(*error);
    }
  }
  if (auto error =
          setParameter(config.authUrl, QDMI_SESSION_PARAMETER_AUTHURL)) {
    return std::move(*error);
  }
  if (auto error =
          setParameter(config.username, QDMI_SESSION_PARAMETER_USERNAME)) {
    return std::move(*error);
  }
  if (auto error =
          setParameter(config.password, QDMI_SESSION_PARAMETER_PASSWORD)) {
    return std::move(*error);
  }
  if (auto error =
          setParameter(config.projectId, QDMI_SESSION_PARAMETER_PROJECTID)) {
    return std::move(*error);
  }
  if (auto error =
          setParameter(config.custom1, QDMI_SESSION_PARAMETER_CUSTOM1)) {
    return std::move(*error);
  }
  if (auto error =
          setParameter(config.custom2, QDMI_SESSION_PARAMETER_CUSTOM2)) {
    return std::move(*error);
  }
  if (auto error =
          setParameter(config.custom3, QDMI_SESSION_PARAMETER_CUSTOM3)) {
    return std::move(*error);
  }
  if (auto error =
          setParameter(config.custom4, QDMI_SESSION_PARAMETER_CUSTOM4)) {
    return std::move(*error);
  }
  if (auto error =
          setParameter(config.custom5, QDMI_SESSION_PARAMETER_CUSTOM5)) {
    return std::move(*error);
  }

  if (auto error = checkError(QDMI_session_init(session.session_.get()),
                              "Initializing session")) {
    return std::move(*error);
  }
  return session;
}

Result<std::vector<Device>> Session::getDevices() {
  auto qdmiDevicesResult =
      queryProperty<std::vector<QDMI_Device>>(QDMI_SESSION_PROPERTY_DEVICES);
  if (auto* error = std::get_if<Error>(&qdmiDevicesResult)) {
    return std::move(*error);
  }
  auto& qdmiDevices = std::get<0>(qdmiDevicesResult);
  std::vector<Device> devices;
  devices.reserve(qdmiDevices.size());
  std::ranges::transform(
      qdmiDevices, std::back_inserter(devices),
      [](QDMI_Device_impl_d* const& dev) -> Device { return Device(dev); });
  return devices;
}
} // namespace qdmi
