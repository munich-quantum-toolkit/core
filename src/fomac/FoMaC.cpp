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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <optional>
#include <qdmi/client.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace fomac {

namespace {
/**
 * @brief Function used to mark unreachable code
 * @details Uses compiler specific extensions if possible. Even if no extension
 * is used, undefined behavior is still raised by an empty function body and the
 * noreturn attribute.
 */
[[noreturn]] inline void unreachable() {
#ifdef __GNUC__ // GCC, Clang, ICC
  __builtin_unreachable();
#elif defined(_MSC_VER) // MSVC
  __assume(false);
#endif
}
} // namespace

auto toString(const QDMI_STATUS result) -> std::string {
  switch (result) {
  case QDMI_WARN_GENERAL:
    return "General warning";
  case QDMI_SUCCESS:
    return "Success";
  case QDMI_ERROR_FATAL:
    return "A fatal error";
  case QDMI_ERROR_OUTOFMEM:
    return "Out of memory";
  case QDMI_ERROR_NOTIMPLEMENTED:
    return "Not implemented";
  case QDMI_ERROR_LIBNOTFOUND:
    return "Library not found";
  case QDMI_ERROR_NOTFOUND:
    return "Element not found";
  case QDMI_ERROR_OUTOFRANGE:
    return "Out of range";
  case QDMI_ERROR_INVALIDARGUMENT:
    return "Invalid argument";
  case QDMI_ERROR_PERMISSIONDENIED:
    return "Permission denied";
  case QDMI_ERROR_NOTSUPPORTED:
    return "Not supported";
  case QDMI_ERROR_BADSTATE:
    return "Bad state";
  case QDMI_ERROR_TIMEOUT:
    return "Timeout";
  }
  unreachable();
}
auto toString(QDMI_Site_Property prop) -> std::string {
  switch (prop) {
  case QDMI_SITE_PROPERTY_INDEX:
    return "QDMI_SITE_PROPERTY_INDEX";
  case QDMI_SITE_PROPERTY_T1:
    return "QDMI_SITE_PROPERTY_T1";
  case QDMI_SITE_PROPERTY_T2:
    return "QDMI_SITE_PROPERTY_T2";
  case QDMI_SITE_PROPERTY_NAME:
    return "QDMI_SITE_PROPERTY_NAME";
  case QDMI_SITE_PROPERTY_XCOORDINATE:
    return "QDMI_SITE_PROPERTY_XCOORDINATE";
  case QDMI_SITE_PROPERTY_YCOORDINATE:
    return "QDMI_SITE_PROPERTY_YCOORDINATE";
  case QDMI_SITE_PROPERTY_ZCOORDINATE:
    return "QDMI_SITE_PROPERTY_ZCOORDINATE";
  case QDMI_SITE_PROPERTY_ISZONE:
    return "QDMI_SITE_PROPERTY_ISZONE";
  case QDMI_SITE_PROPERTY_XEXTENT:
    return "QDMI_SITE_PROPERTY_XEXTENT";
  case QDMI_SITE_PROPERTY_YEXTENT:
    return "QDMI_SITE_PROPERTY_YEXTENT";
  case QDMI_SITE_PROPERTY_ZEXTENT:
    return "QDMI_SITE_PROPERTY_ZEXTENT";
  case QDMI_SITE_PROPERTY_MODULEINDEX:
    return "QDMI_SITE_PROPERTY_MODULEINDEX";
  case QDMI_SITE_PROPERTY_SUBMODULEINDEX:
    return "QDMI_SITE_PROPERTY_SUBMODULEINDEX";
  case QDMI_SITE_PROPERTY_MAX:
  case QDMI_SITE_PROPERTY_CUSTOM1:
  case QDMI_SITE_PROPERTY_CUSTOM2:
  case QDMI_SITE_PROPERTY_CUSTOM3:
  case QDMI_SITE_PROPERTY_CUSTOM4:
  case QDMI_SITE_PROPERTY_CUSTOM5:
    return "QDMI_SITE_PROPERTY_UNKNOWN";
  }
  unreachable();
}
auto toString(QDMI_Operation_Property prop) -> std::string {
  switch (prop) {
  case QDMI_OPERATION_PROPERTY_NAME:
    return "QDMI_OPERATION_PROPERTY_NAME";
  case QDMI_OPERATION_PROPERTY_QUBITSNUM:
    return "QDMI_OPERATION_PROPERTY_QUBITSNUM";
  case QDMI_OPERATION_PROPERTY_PARAMETERSNUM:
    return "QDMI_OPERATION_PROPERTY_PARAMETERSNUM";
  case QDMI_OPERATION_PROPERTY_DURATION:
    return "QDMI_OPERATION_PROPERTY_DURATION";
  case QDMI_OPERATION_PROPERTY_FIDELITY:
    return "QDMI_OPERATION_PROPERTY_FIDELITY";
  case QDMI_OPERATION_PROPERTY_INTERACTIONRADIUS:
    return "QDMI_OPERATION_PROPERTY_INTERACTIONRADIUS";
  case QDMI_OPERATION_PROPERTY_BLOCKINGRADIUS:
    return "QDMI_OPERATION_PROPERTY_BLOCKINGRADIUS";
  case QDMI_OPERATION_PROPERTY_IDLINGFIDELITY:
    return "QDMI_OPERATION_PROPERTY_IDLINGFIDELITY";
  case QDMI_OPERATION_PROPERTY_ISZONED:
    return "QDMI_OPERATION_PROPERTY_ISZONED";
  case QDMI_OPERATION_PROPERTY_SITES:
    return "QDMI_OPERATION_PROPERTY_SITES";
  case QDMI_OPERATION_PROPERTY_MEANSHUTTLINGSPEED:
    return "QDMI_OPERATION_PROPERTY_MEANSHUTTLINGSPEED";
  case QDMI_OPERATION_PROPERTY_MAX:
  case QDMI_OPERATION_PROPERTY_CUSTOM1:
  case QDMI_OPERATION_PROPERTY_CUSTOM2:
  case QDMI_OPERATION_PROPERTY_CUSTOM3:
  case QDMI_OPERATION_PROPERTY_CUSTOM4:
  case QDMI_OPERATION_PROPERTY_CUSTOM5:
    return "QDMI_OPERATION_PROPERTY_UNKNOWN";
  }
  unreachable();
}
auto toString(QDMI_Device_Property prop) -> std::string {
  switch (prop) {
  case QDMI_DEVICE_PROPERTY_NAME:
    return "QDMI_DEVICE_PROPERTY_NAME";
  case QDMI_DEVICE_PROPERTY_VERSION:
    return "QDMI_DEVICE_PROPERTY_VERSION";
  case QDMI_DEVICE_PROPERTY_STATUS:
    return "QDMI_DEVICE_PROPERTY_STATUS";
  case QDMI_DEVICE_PROPERTY_LIBRARYVERSION:
    return "QDMI_DEVICE_PROPERTY_LIBRARYVERSION";
  case QDMI_DEVICE_PROPERTY_QUBITSNUM:
    return "QDMI_DEVICE_PROPERTY_QUBITSNUM";
  case QDMI_DEVICE_PROPERTY_SITES:
    return "QDMI_DEVICE_PROPERTY_SITES";
  case QDMI_DEVICE_PROPERTY_OPERATIONS:
    return "QDMI_DEVICE_PROPERTY_OPERATIONS";
  case QDMI_DEVICE_PROPERTY_COUPLINGMAP:
    return "QDMI_DEVICE_PROPERTY_COUPLINGMAP";
  case QDMI_DEVICE_PROPERTY_NEEDSCALIBRATION:
    return "QDMI_DEVICE_PROPERTY_NEEDSCALIBRATION";
  case QDMI_DEVICE_PROPERTY_LENGTHUNIT:
    return "QDMI_DEVICE_PROPERTY_LENGTHUNIT";
  case QDMI_DEVICE_PROPERTY_LENGTHSCALEFACTOR:
    return "QDMI_DEVICE_PROPERTY_LENGTHSCALEFACTOR";
  case QDMI_DEVICE_PROPERTY_DURATIONUNIT:
    return "QDMI_DEVICE_PROPERTY_DURATIONUNIT";
  case QDMI_DEVICE_PROPERTY_DURATIONSCALEFACTOR:
    return "QDMI_DEVICE_PROPERTY_DURATIONSCALEFACTOR";
  case QDMI_DEVICE_PROPERTY_MINATOMDISTANCE:
    return "QDMI_DEVICE_PROPERTY_MINATOMDISTANCE";
  case QDMI_DEVICE_PROPERTY_PULSESUPPORT:
    return "QDMI_DEVICE_PROPERTY_PULSESUPPORT";
  case QDMI_DEVICE_PROPERTY_MAX:
  case QDMI_DEVICE_PROPERTY_CUSTOM1:
  case QDMI_DEVICE_PROPERTY_CUSTOM2:
  case QDMI_DEVICE_PROPERTY_CUSTOM3:
  case QDMI_DEVICE_PROPERTY_CUSTOM4:
  case QDMI_DEVICE_PROPERTY_CUSTOM5:
    return "QDMI_DEVICE_PROPERTY_UNKNOWN";
  }
  unreachable();
}
auto throwError(int result, const std::string& msg) -> void {
  std::ostringstream ss;
  ss << msg << ": " << toString(static_cast<QDMI_STATUS>(result)) << ".";
  switch (result) {
  case QDMI_ERROR_OUTOFMEM:
    throw std::bad_alloc();
  case QDMI_ERROR_OUTOFRANGE:
    throw std::out_of_range(ss.str());
  case QDMI_ERROR_INVALIDARGUMENT:
    throw std::invalid_argument(ss.str());
  case QDMI_ERROR_FATAL:
  case QDMI_ERROR_NOTIMPLEMENTED:
  case QDMI_ERROR_LIBNOTFOUND:
  case QDMI_ERROR_NOTFOUND:
  case QDMI_ERROR_PERMISSIONDENIED:
  case QDMI_ERROR_NOTSUPPORTED:
  case QDMI_ERROR_BADSTATE:
  case QDMI_ERROR_TIMEOUT:
    throw std::runtime_error(ss.str());
  default:
    throw std::runtime_error("Unknown error code: " +
                             toString(static_cast<QDMI_STATUS>(result)) + ".");
  }
}
auto FoMaC::Device::Site::getIndex() const -> size_t {
  return queryProperty<size_t>(QDMI_SITE_PROPERTY_INDEX);
}
auto FoMaC::Device::Site::getT1() const -> std::optional<uint64_t> {
  return queryProperty<std::optional<uint64_t>>(QDMI_SITE_PROPERTY_T1);
}
auto FoMaC::Device::Site::getT2() const -> std::optional<uint64_t> {
  return queryProperty<std::optional<uint64_t>>(QDMI_SITE_PROPERTY_T2);
}
auto FoMaC::Device::Site::getName() const -> std::optional<std::string> {
  return queryProperty<std::optional<std::string>>(QDMI_SITE_PROPERTY_NAME);
}
auto FoMaC::Device::Site::getXCoordinate() const -> std::optional<int64_t> {
  return queryProperty<std::optional<int64_t>>(QDMI_SITE_PROPERTY_XCOORDINATE);
}
auto FoMaC::Device::Site::getYCoordinate() const -> std::optional<int64_t> {
  return queryProperty<std::optional<int64_t>>(QDMI_SITE_PROPERTY_YCOORDINATE);
}
auto FoMaC::Device::Site::getZCoordinate() const -> std::optional<int64_t> {
  return queryProperty<std::optional<int64_t>>(QDMI_SITE_PROPERTY_ZCOORDINATE);
}
auto FoMaC::Device::Site::isZone() const -> std::optional<bool> {
  return queryProperty<std::optional<bool>>(QDMI_SITE_PROPERTY_ISZONE);
}
auto FoMaC::Device::Site::getXExtent() const -> std::optional<uint64_t> {
  return queryProperty<std::optional<uint64_t>>(QDMI_SITE_PROPERTY_XEXTENT);
}
auto FoMaC::Device::Site::getYExtent() const -> std::optional<uint64_t> {
  return queryProperty<std::optional<uint64_t>>(QDMI_SITE_PROPERTY_YEXTENT);
}
auto FoMaC::Device::Site::getZExtent() const -> std::optional<uint64_t> {
  return queryProperty<std::optional<uint64_t>>(QDMI_SITE_PROPERTY_ZEXTENT);
}
auto FoMaC::Device::Site::getModuleIndex() const -> std::optional<uint64_t> {
  return queryProperty<std::optional<uint64_t>>(QDMI_SITE_PROPERTY_MODULEINDEX);
}
auto FoMaC::Device::Site::getSubmoduleIndex() const -> std::optional<uint64_t> {
  return queryProperty<std::optional<uint64_t>>(
      QDMI_SITE_PROPERTY_SUBMODULEINDEX);
}
auto FoMaC::Device::Operation::getName(const std::vector<Site>& sites,
                                       const std::vector<double>& params) const
    -> std::string {
  return queryProperty<std::string>(QDMI_OPERATION_PROPERTY_NAME, sites,
                                    params);
}
auto FoMaC::Device::Operation::getQubitsNum(
    const std::vector<Site>& sites, const std::vector<double>& params) const
    -> std::optional<size_t> {
  return queryProperty<std::optional<size_t>>(QDMI_OPERATION_PROPERTY_QUBITSNUM,
                                              sites, params);
}
auto FoMaC::Device::Operation::getParametersNum(
    const std::vector<Site>& sites, const std::vector<double>& params) const
    -> size_t {
  return queryProperty<size_t>(QDMI_OPERATION_PROPERTY_PARAMETERSNUM, sites,
                               params);
}
auto FoMaC::Device::Operation::getDuration(
    const std::vector<Site>& sites, const std::vector<double>& params) const
    -> std::optional<uint64_t> {
  return queryProperty<std::optional<uint64_t>>(
      QDMI_OPERATION_PROPERTY_DURATION, sites, params);
}
auto FoMaC::Device::Operation::getFidelity(
    const std::vector<Site>& sites, const std::vector<double>& params) const
    -> std::optional<double> {
  return queryProperty<std::optional<double>>(QDMI_OPERATION_PROPERTY_FIDELITY,
                                              sites, params);
}
auto FoMaC::Device::Operation::getInteractionRadius(
    const std::vector<Site>& sites, const std::vector<double>& params) const
    -> std::optional<uint64_t> {
  return queryProperty<std::optional<uint64_t>>(
      QDMI_OPERATION_PROPERTY_INTERACTIONRADIUS, sites, params);
}
auto FoMaC::Device::Operation::getBlockingRadius(
    const std::vector<Site>& sites, const std::vector<double>& params) const
    -> std::optional<uint64_t> {
  return queryProperty<std::optional<uint64_t>>(
      QDMI_OPERATION_PROPERTY_BLOCKINGRADIUS, sites, params);
}
auto FoMaC::Device::Operation::getIdlingFidelity(
    const std::vector<Site>& sites, const std::vector<double>& params) const
    -> std::optional<double> {
  return queryProperty<std::optional<double>>(
      QDMI_OPERATION_PROPERTY_IDLINGFIDELITY, sites, params);
}
auto FoMaC::Device::Operation::isZoned(const std::vector<Site>& sites,
                                       const std::vector<double>& params) const
    -> std::optional<bool> {
  return queryProperty<std::optional<bool>>(QDMI_OPERATION_PROPERTY_ISZONED,
                                            sites, params);
}
auto FoMaC::Device::Operation::getSites(const std::vector<Site>& sites,
                                        const std::vector<double>& params) const
    -> std::optional<std::vector<Site>> {
  const auto& qdmiSites = queryProperty<std::optional<std::vector<QDMI_Site>>>(
      QDMI_OPERATION_PROPERTY_SITES, sites, params);
  if (!qdmiSites.has_value()) {
    return std::nullopt;
  }
  std::vector<Site> returnedSites;
  returnedSites.reserve(qdmiSites->size());
  std::ranges::transform(*qdmiSites, std::back_inserter(returnedSites),
                         [device = device_](const QDMI_Site& site) -> Site {
                           return {Token{}, device, site};
                         });
  return returnedSites;
}
auto FoMaC::Device::Operation::getMeanShuttlingSpeed(
    const std::vector<Site>& sites, const std::vector<double>& params) const
    -> std::optional<uint64_t> {
  return queryProperty<std::optional<uint64_t>>(
      QDMI_OPERATION_PROPERTY_MEANSHUTTLINGSPEED, sites, params);
}
auto FoMaC::Device::getName() const -> std::string {
  return queryProperty<std::string>(QDMI_DEVICE_PROPERTY_NAME);
}
auto FoMaC::Device::getVersion() const -> std::string {
  return queryProperty<std::string>(QDMI_DEVICE_PROPERTY_VERSION);
}
auto FoMaC::Device::getStatus() const -> QDMI_Device_Status {
  return queryProperty<QDMI_Device_Status>(QDMI_DEVICE_PROPERTY_STATUS);
}
auto FoMaC::Device::getLibraryVersion() const -> std::string {
  return queryProperty<std::string>(QDMI_DEVICE_PROPERTY_LIBRARYVERSION);
}
auto FoMaC::Device::getQubitsNum() const -> size_t {
  return queryProperty<size_t>(QDMI_DEVICE_PROPERTY_QUBITSNUM);
}
auto FoMaC::Device::getSites() const -> std::vector<Site> {
  const auto& qdmiSites =
      queryProperty<std::vector<QDMI_Site>>(QDMI_DEVICE_PROPERTY_SITES);
  std::vector<Site> sites;
  sites.reserve(qdmiSites.size());
  std::ranges::transform(qdmiSites, std::back_inserter(sites),
                         [device = device_](const QDMI_Site& site) -> Site {
                           return {Token{}, device, site};
                         });
  return sites;
}
auto FoMaC::Device::getOperations() const -> std::vector<Operation> {
  const auto& qdmiOperations = queryProperty<std::vector<QDMI_Operation>>(
      QDMI_DEVICE_PROPERTY_OPERATIONS);
  std::vector<Operation> operations;
  operations.reserve(qdmiOperations.size());
  std::ranges::transform(
      qdmiOperations, std::back_inserter(operations),
      [device = device_](const QDMI_Operation& op) -> Operation {
        return {Token{}, device, op};
      });
  return operations;
}
auto FoMaC::Device::getCouplingMap() const
    -> std::optional<std::vector<std::pair<Site, Site>>> {
  const auto& qdmiCouplingMap = queryProperty<
      std::optional<std::vector<std::pair<QDMI_Site, QDMI_Site>>>>(
      QDMI_DEVICE_PROPERTY_COUPLINGMAP);
  if (!qdmiCouplingMap.has_value()) {
    return std::nullopt;
  }
  std::vector<std::pair<Site, Site>> couplingMap;
  couplingMap.reserve(qdmiCouplingMap->size());
  std::ranges::transform(*qdmiCouplingMap, std::back_inserter(couplingMap),
                         [this](const std::pair<QDMI_Site, QDMI_Site>& pair)
                             -> std::pair<Site, Site> {
                           return {Site{Token{}, device_, pair.first},
                                   Site{Token{}, device_, pair.second}};
                         });
  return couplingMap;
}
auto FoMaC::Device::getNeedsCalibration() const -> std::optional<size_t> {
  return queryProperty<std::optional<size_t>>(
      QDMI_DEVICE_PROPERTY_NEEDSCALIBRATION);
}
auto FoMaC::Device::getLengthUnit() const -> std::optional<std::string> {
  return queryProperty<std::optional<std::string>>(
      QDMI_DEVICE_PROPERTY_LENGTHUNIT);
}
auto FoMaC::Device::getLengthScaleFactor() const -> std::optional<double> {
  return queryProperty<std::optional<double>>(
      QDMI_DEVICE_PROPERTY_LENGTHSCALEFACTOR);
}
auto FoMaC::Device::getDurationUnit() const -> std::optional<std::string> {
  return queryProperty<std::optional<std::string>>(
      QDMI_DEVICE_PROPERTY_DURATIONUNIT);
}
auto FoMaC::Device::getDurationScaleFactor() const -> std::optional<double> {
  return queryProperty<std::optional<double>>(
      QDMI_DEVICE_PROPERTY_DURATIONSCALEFACTOR);
}
auto FoMaC::Device::getMinAtomDistance() const -> std::optional<uint64_t> {
  return queryProperty<std::optional<uint64_t>>(
      QDMI_DEVICE_PROPERTY_MINATOMDISTANCE);
}

auto FoMaC::Device::submitJob(const std::string& program,
                              QDMI_Program_Format format, size_t numShots,
                              double timeout) const -> Job {
  QDMI_Job job = nullptr;
  throwIfError(QDMI_device_create_job(device_, &job), "Creating job");

  // Set program format
  throwIfError(QDMI_job_set_parameter(job, QDMI_JOB_PARAMETER_PROGRAMFORMAT,
                                      sizeof(format), &format),
               "Setting program format");

  // Set program
  throwIfError(QDMI_job_set_parameter(job, QDMI_JOB_PARAMETER_PROGRAM,
                                      program.size(), program.c_str()),
               "Setting program");

  // Set number of shots
  throwIfError(QDMI_job_set_parameter(job, QDMI_JOB_PARAMETER_SHOTSNUM,
                                      sizeof(numShots), &numShots),
               "Setting number of shots");

  // Submit the job
  throwIfError(QDMI_job_submit(job), "Submitting job");

  // Wait for the job to finish
  const auto timeoutSeconds = static_cast<size_t>(timeout);
  throwIfError(QDMI_job_wait(job, timeoutSeconds), "Waiting for job");

  return Job{job};
}

auto FoMaC::Job::check() const -> QDMI_Job_Status {
  QDMI_Job_Status status;
  throwIfError(QDMI_job_check(job_, &status), "Checking job status");
  return status;
}

auto FoMaC::Job::wait() const -> void {
  throwIfError(QDMI_job_wait(job_, 0), "Waiting for job");
}

auto FoMaC::Job::cancel() const -> void {
  throwIfError(QDMI_job_cancel(job_), "Cancelling job");
}

auto FoMaC::Job::getId() const -> std::string {
  size_t size = 0;
  throwIfError(
      QDMI_job_query_property(job_, QDMI_JOB_PROPERTY_ID, 0, nullptr, &size),
      "Querying job ID size");
  std::string id(size - 1, '\0');
  throwIfError(QDMI_job_query_property(job_, QDMI_JOB_PROPERTY_ID, size,
                                       id.data(), nullptr),
               "Querying job ID");
  return id;
}

auto FoMaC::Job::getNumShots() const -> size_t {
  size_t numShots = 0;
  throwIfError(QDMI_job_query_property(job_, QDMI_JOB_PROPERTY_SHOTSNUM,
                                       sizeof(numShots), &numShots, nullptr),
               "Querying number of shots");
  return numShots;
}

auto FoMaC::Job::getCounts() const -> std::map<std::string, size_t> {
  // Get the histogram keys
  size_t keysSize = 0;
  throwIfError(QDMI_job_get_results(job_, QDMI_JOB_RESULT_HIST_KEYS, 0, nullptr,
                                    &keysSize),
               "Querying histogram keys size");
  std::string keys(keysSize - 1, '\0');
  throwIfError(QDMI_job_get_results(job_, QDMI_JOB_RESULT_HIST_KEYS, keysSize,
                                    keys.data(), nullptr),
               "Querying histogram keys");

  // Get the histogram values
  size_t valuesSize = 0;
  throwIfError(QDMI_job_get_results(job_, QDMI_JOB_RESULT_HIST_VALUES, 0,
                                    nullptr, &valuesSize),
               "Querying histogram values size");
  std::vector<size_t> values(valuesSize / sizeof(size_t));
  throwIfError(QDMI_job_get_results(job_, QDMI_JOB_RESULT_HIST_VALUES,
                                    valuesSize, values.data(), nullptr),
               "Querying histogram values");

  // Parse the keys (comma-separated)
  std::map<std::string, size_t> counts;
  std::istringstream keysStream(keys);
  std::string key;
  size_t idx = 0;
  while (std::getline(keysStream, key, ',')) {
    if (idx < values.size()) {
      counts[key] = values[idx];
      ++idx;
    }
  }
  return counts;
}

FoMaC::FoMaC() {
  QDMI_session_alloc(&session_);
  QDMI_session_init(session_);
}
FoMaC::~FoMaC() { QDMI_session_free(session_); }
auto FoMaC::getDevices() -> std::vector<Device> {
  const auto& qdmiDevices = get().queryProperty<std::vector<QDMI_Device>>(
      QDMI_SESSION_PROPERTY_DEVICES);
  std::vector<Device> devices;
  devices.reserve(qdmiDevices.size());
  std::ranges::transform(
      qdmiDevices, std::back_inserter(devices),
      [](const QDMI_Device& dev) -> Device { return {Token{}, dev}; });
  return devices;
}
} // namespace fomac
