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

#include "qdmi/Common.hpp"

#include <algorithm>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iterator>
#include <map>
#include <optional>
#include <qdmi/client.h>
#include <regex>
#include <spdlog/spdlog.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace fomac {
auto Session::Device::Site::getIndex() const -> size_t {
  return queryProperty<size_t>(QDMI_SITE_PROPERTY_INDEX);
}
auto Session::Device::Site::getT1() const -> std::optional<uint64_t> {
  return queryProperty<std::optional<uint64_t>>(QDMI_SITE_PROPERTY_T1);
}
auto Session::Device::Site::getT2() const -> std::optional<uint64_t> {
  return queryProperty<std::optional<uint64_t>>(QDMI_SITE_PROPERTY_T2);
}
auto Session::Device::Site::getName() const -> std::optional<std::string> {
  return queryProperty<std::optional<std::string>>(QDMI_SITE_PROPERTY_NAME);
}
auto Session::Device::Site::getXCoordinate() const -> std::optional<int64_t> {
  return queryProperty<std::optional<int64_t>>(QDMI_SITE_PROPERTY_XCOORDINATE);
}
auto Session::Device::Site::getYCoordinate() const -> std::optional<int64_t> {
  return queryProperty<std::optional<int64_t>>(QDMI_SITE_PROPERTY_YCOORDINATE);
}
auto Session::Device::Site::getZCoordinate() const -> std::optional<int64_t> {
  return queryProperty<std::optional<int64_t>>(QDMI_SITE_PROPERTY_ZCOORDINATE);
}
auto Session::Device::Site::isZone() const -> bool {
  return queryProperty<std::optional<bool>>(QDMI_SITE_PROPERTY_ISZONE)
      .value_or(false);
}
auto Session::Device::Site::getXExtent() const -> std::optional<uint64_t> {
  return queryProperty<std::optional<uint64_t>>(QDMI_SITE_PROPERTY_XEXTENT);
}
auto Session::Device::Site::getYExtent() const -> std::optional<uint64_t> {
  return queryProperty<std::optional<uint64_t>>(QDMI_SITE_PROPERTY_YEXTENT);
}
auto Session::Device::Site::getZExtent() const -> std::optional<uint64_t> {
  return queryProperty<std::optional<uint64_t>>(QDMI_SITE_PROPERTY_ZEXTENT);
}
auto Session::Device::Site::getModuleIndex() const -> std::optional<uint64_t> {
  return queryProperty<std::optional<uint64_t>>(QDMI_SITE_PROPERTY_MODULEINDEX);
}
auto Session::Device::Site::getSubmoduleIndex() const
    -> std::optional<uint64_t> {
  return queryProperty<std::optional<uint64_t>>(
      QDMI_SITE_PROPERTY_SUBMODULEINDEX);
}
auto Session::Device::Operation::getName(
    const std::vector<Site>& sites, const std::vector<double>& params) const
    -> std::string {
  return queryProperty<std::string>(QDMI_OPERATION_PROPERTY_NAME, sites,
                                    params);
}
auto Session::Device::Operation::getQubitsNum(
    const std::vector<Site>& sites, const std::vector<double>& params) const
    -> std::optional<size_t> {
  return queryProperty<std::optional<size_t>>(QDMI_OPERATION_PROPERTY_QUBITSNUM,
                                              sites, params);
}
auto Session::Device::Operation::getParametersNum(
    const std::vector<Site>& sites, const std::vector<double>& params) const
    -> size_t {
  return queryProperty<size_t>(QDMI_OPERATION_PROPERTY_PARAMETERSNUM, sites,
                               params);
}
auto Session::Device::Operation::getDuration(
    const std::vector<Site>& sites, const std::vector<double>& params) const
    -> std::optional<uint64_t> {
  return queryProperty<std::optional<uint64_t>>(
      QDMI_OPERATION_PROPERTY_DURATION, sites, params);
}
auto Session::Device::Operation::getFidelity(
    const std::vector<Site>& sites, const std::vector<double>& params) const
    -> std::optional<double> {
  return queryProperty<std::optional<double>>(QDMI_OPERATION_PROPERTY_FIDELITY,
                                              sites, params);
}
auto Session::Device::Operation::getInteractionRadius(
    const std::vector<Site>& sites, const std::vector<double>& params) const
    -> std::optional<uint64_t> {
  return queryProperty<std::optional<uint64_t>>(
      QDMI_OPERATION_PROPERTY_INTERACTIONRADIUS, sites, params);
}
auto Session::Device::Operation::getBlockingRadius(
    const std::vector<Site>& sites, const std::vector<double>& params) const
    -> std::optional<uint64_t> {
  return queryProperty<std::optional<uint64_t>>(
      QDMI_OPERATION_PROPERTY_BLOCKINGRADIUS, sites, params);
}
auto Session::Device::Operation::getIdlingFidelity(
    const std::vector<Site>& sites, const std::vector<double>& params) const
    -> std::optional<double> {
  return queryProperty<std::optional<double>>(
      QDMI_OPERATION_PROPERTY_IDLINGFIDELITY, sites, params);
}
auto Session::Device::Operation::isZoned() const -> bool {
  return queryProperty<std::optional<bool>>(QDMI_OPERATION_PROPERTY_ISZONED, {},
                                            {})
      .value_or(false);
}
auto Session::Device::Operation::getSites() const
    -> std::optional<std::vector<Site>> {
  const auto& qdmiSites = queryProperty<std::optional<std::vector<QDMI_Site>>>(
      QDMI_OPERATION_PROPERTY_SITES, {}, {});
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
auto Session::Device::Operation::getSitePairs() const
    -> std::optional<std::vector<std::pair<Site, Site>>> {
  if (const auto qubitsNum = getQubitsNum({}, {});
      !qubitsNum.has_value() || *qubitsNum != 2 || isZoned()) {
    return std::nullopt; // Not a 2-qubit operation or operation is zoned
  }

  const auto sitesOpt = getSites();
  if (!sitesOpt.has_value()) {
    return std::nullopt;
  }

  const auto& sitesVec = *sitesOpt;
  if (sitesVec.empty() || sitesVec.size() % 2 != 0) {
    return std::nullopt; // Invalid: no sites or odd number of sites
  }

  std::vector<std::pair<Site, Site>> pairs;
  pairs.reserve(sitesVec.size() / 2);

  for (size_t i = 0; i < sitesVec.size(); i += 2) {
    pairs.emplace_back(sitesVec[i], sitesVec[i + 1]);
  }

  return pairs;
}
auto Session::Device::Operation::getMeanShuttlingSpeed(
    const std::vector<Site>& sites, const std::vector<double>& params) const
    -> std::optional<uint64_t> {
  return queryProperty<std::optional<uint64_t>>(
      QDMI_OPERATION_PROPERTY_MEANSHUTTLINGSPEED, sites, params);
}
auto Session::Device::getName() const -> std::string {
  return queryProperty<std::string>(QDMI_DEVICE_PROPERTY_NAME);
}
auto Session::Device::getVersion() const -> std::string {
  return queryProperty<std::string>(QDMI_DEVICE_PROPERTY_VERSION);
}
auto Session::Device::getStatus() const -> QDMI_Device_Status {
  return queryProperty<QDMI_Device_Status>(QDMI_DEVICE_PROPERTY_STATUS);
}
auto Session::Device::getLibraryVersion() const -> std::string {
  return queryProperty<std::string>(QDMI_DEVICE_PROPERTY_LIBRARYVERSION);
}
auto Session::Device::getQubitsNum() const -> size_t {
  return queryProperty<size_t>(QDMI_DEVICE_PROPERTY_QUBITSNUM);
}
auto Session::Device::getSites() const -> std::vector<Site> {
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
auto Session::Device::getRegularSites() const -> std::vector<Site> {
  auto allSites = getSites();
  const auto newEnd = std::ranges::remove_if(
      allSites, [](const Site& s) { return s.isZone(); });
  allSites.erase(newEnd.begin(), newEnd.end());
  return allSites;
}
auto Session::Device::getZones() const -> std::vector<Site> {
  const auto& allSites = getSites();
  std::vector<Site> zones;
  zones.reserve(3); // Reserve space for a typical max number of zones
  std::ranges::copy_if(allSites, std::back_inserter(zones),
                       [](const Site& s) { return s.isZone(); });
  return zones;
}
auto Session::Device::getOperations() const -> std::vector<Operation> {
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
auto Session::Device::getCouplingMap() const
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
auto Session::Device::getNeedsCalibration() const -> std::optional<size_t> {
  return queryProperty<std::optional<size_t>>(
      QDMI_DEVICE_PROPERTY_NEEDSCALIBRATION);
}
auto Session::Device::getLengthUnit() const -> std::optional<std::string> {
  return queryProperty<std::optional<std::string>>(
      QDMI_DEVICE_PROPERTY_LENGTHUNIT);
}
auto Session::Device::getLengthScaleFactor() const -> std::optional<double> {
  return queryProperty<std::optional<double>>(
      QDMI_DEVICE_PROPERTY_LENGTHSCALEFACTOR);
}
auto Session::Device::getDurationUnit() const -> std::optional<std::string> {
  return queryProperty<std::optional<std::string>>(
      QDMI_DEVICE_PROPERTY_DURATIONUNIT);
}
auto Session::Device::getDurationScaleFactor() const -> std::optional<double> {
  return queryProperty<std::optional<double>>(
      QDMI_DEVICE_PROPERTY_DURATIONSCALEFACTOR);
}
auto Session::Device::getMinAtomDistance() const -> std::optional<uint64_t> {
  return queryProperty<std::optional<uint64_t>>(
      QDMI_DEVICE_PROPERTY_MINATOMDISTANCE);
}

auto Session::Device::getSupportedProgramFormats() const
    -> std::vector<QDMI_Program_Format> {
  return queryProperty<std::vector<QDMI_Program_Format>>(
      QDMI_DEVICE_PROPERTY_SUPPORTEDPROGRAMFORMATS);
}

auto Session::Device::submitJob(const std::string& program,
                                const QDMI_Program_Format format,
                                const size_t numShots) const -> Job {
  QDMI_Job job = nullptr;
  qdmi::throwIfError(QDMI_device_create_job(device_, &job), "Creating job");
  Job jobWrapper{job}; // RAII wrapper to prevent leaks in case of exceptions

  // Set program format
  qdmi::throwIfError(QDMI_job_set_parameter(jobWrapper,
                                            QDMI_JOB_PARAMETER_PROGRAMFORMAT,
                                            sizeof(format), &format),
                     "Setting program format");

  // Set program
  qdmi::throwIfError(
      QDMI_job_set_parameter(jobWrapper, QDMI_JOB_PARAMETER_PROGRAM,
                             program.size() + 1, program.c_str()),
      "Setting program");

  // Set number of shots
  qdmi::throwIfError(QDMI_job_set_parameter(jobWrapper,
                                            QDMI_JOB_PARAMETER_SHOTSNUM,
                                            sizeof(numShots), &numShots),
                     "Setting number of shots");

  // Submit the job
  qdmi::throwIfError(QDMI_job_submit(jobWrapper), "Submitting job");
  return jobWrapper;
}

auto Session::Job::check() const -> QDMI_Job_Status {
  QDMI_Job_Status status{};
  qdmi::throwIfError(QDMI_job_check(job_, &status), "Checking job status");
  return status;
}

auto Session::Job::wait(const size_t timeout) const -> bool {
  const auto ret = QDMI_job_wait(job_, timeout);
  if (ret == QDMI_SUCCESS) {
    return true;
  }
  if (ret == QDMI_ERROR_TIMEOUT) {
    return false;
  }
  qdmi::throwIfError(ret, "Waiting for job");
  qdmi::unreachable();
}

auto Session::Job::cancel() const -> void {
  qdmi::throwIfError(QDMI_job_cancel(job_), "Cancelling job");
}

auto Session::Job::getId() const -> std::string {
  size_t size = 0;
  qdmi::throwIfError(
      QDMI_job_query_property(job_, QDMI_JOB_PROPERTY_ID, 0, nullptr, &size),
      "Querying job ID size");
  std::string id(size - 1, '\0');
  qdmi::throwIfError(QDMI_job_query_property(job_, QDMI_JOB_PROPERTY_ID, size,
                                             id.data(), nullptr),
                     "Querying job ID");
  return id;
}

auto Session::Job::getProgramFormat() const -> QDMI_Program_Format {
  QDMI_Program_Format format{};
  qdmi::throwIfError(QDMI_job_query_property(job_,
                                             QDMI_JOB_PROPERTY_PROGRAMFORMAT,
                                             sizeof(format), &format, nullptr),
                     "Querying program format");
  return format;
}

auto Session::Job::getProgram() const -> std::string {
  size_t size = 0;
  qdmi::throwIfError(QDMI_job_query_property(job_, QDMI_JOB_PROPERTY_PROGRAM, 0,
                                             nullptr, &size),
                     "Querying program size");

  std::string program(size - 1, '\0');
  qdmi::throwIfError(QDMI_job_query_property(job_, QDMI_JOB_PROPERTY_PROGRAM,
                                             size, program.data(), nullptr),
                     "Querying program");
  return program;
}

auto Session::Job::getNumShots() const -> size_t {
  size_t numShots = 0;
  qdmi::throwIfError(QDMI_job_query_property(job_, QDMI_JOB_PROPERTY_SHOTSNUM,
                                             sizeof(numShots), &numShots,
                                             nullptr),
                     "Querying number of shots");
  return numShots;
}

auto Session::Job::getShots() const -> std::vector<std::string> {
  size_t shotsSize = 0;
  qdmi::throwIfError(
      QDMI_job_get_results(job_, QDMI_JOB_RESULT_SHOTS, 0, nullptr, &shotsSize),
      "Querying shots size");

  if (shotsSize == 0) {
    return {};
  }

  std::string shots(shotsSize - 1, '\0');
  qdmi::throwIfError(QDMI_job_get_results(job_, QDMI_JOB_RESULT_SHOTS,
                                          shotsSize, shots.data(), nullptr),
                     "Querying shots");

  // Parse the shots (comma-separated)
  std::vector<std::string> shotsVec;
  const auto numShots = getNumShots();
  shotsVec.reserve(numShots);
  std::istringstream shotsStream(shots);
  std::string shot;
  while (std::getline(shotsStream, shot, ',')) {
    shotsVec.emplace_back(shot);
  }
  if (shotsVec.size() != numShots) {
    throw std::runtime_error("Number of shots mismatch");
  }

  return shotsVec;
}

auto Session::Job::getCounts() const -> std::map<std::string, size_t> {
  // Get the histogram keys
  size_t keysSize = 0;
  qdmi::throwIfError(QDMI_job_get_results(job_, QDMI_JOB_RESULT_HIST_KEYS, 0,
                                          nullptr, &keysSize),
                     "Querying histogram keys size");

  if (keysSize == 0) {
    return {}; // Empty histogram
  }

  std::string keys(keysSize - 1, '\0');
  qdmi::throwIfError(QDMI_job_get_results(job_, QDMI_JOB_RESULT_HIST_KEYS,
                                          keysSize, keys.data(), nullptr),
                     "Querying histogram keys");

  // Get the histogram values
  size_t valuesSize = 0;
  qdmi::throwIfError(QDMI_job_get_results(job_, QDMI_JOB_RESULT_HIST_VALUES, 0,
                                          nullptr, &valuesSize),
                     "Querying histogram values size");

  if (valuesSize % sizeof(size_t) != 0) {
    throw std::runtime_error(
        "Invalid histogram values size: not a multiple of size_t");
  }

  std::vector<size_t> values(valuesSize / sizeof(size_t));
  qdmi::throwIfError(QDMI_job_get_results(job_, QDMI_JOB_RESULT_HIST_VALUES,
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

  if (idx != values.size()) {
    throw std::runtime_error("Histogram key/value count mismatch");
  }

  return counts;
}

auto Session::Job::getDenseStateVector() const
    -> std::vector<std::complex<double>> {
  size_t size = 0;
  qdmi::throwIfError(QDMI_job_get_results(job_,
                                          QDMI_JOB_RESULT_STATEVECTOR_DENSE, 0,
                                          nullptr, &size),
                     "Querying dense state vector size");

  if (size % sizeof(std::complex<double>) != 0) {
    throw std::runtime_error(
        "Invalid state vector size: not a multiple of complex<double>");
  }

  std::vector<std::complex<double>> stateVector(size /
                                                sizeof(std::complex<double>));
  qdmi::throwIfError(QDMI_job_get_results(job_,
                                          QDMI_JOB_RESULT_STATEVECTOR_DENSE,
                                          size, stateVector.data(), nullptr),
                     "Querying dense state vector");
  return stateVector;
}

auto Session::Job::getDenseProbabilities() const -> std::vector<double> {
  size_t size = 0;
  qdmi::throwIfError(QDMI_job_get_results(job_,
                                          QDMI_JOB_RESULT_PROBABILITIES_DENSE,
                                          0, nullptr, &size),
                     "Querying dense probabilities size");

  if (size % sizeof(double) != 0) {
    throw std::runtime_error(
        "Invalid probabilities size: not a multiple of double");
  }

  std::vector<double> probabilities(size / sizeof(double));
  qdmi::throwIfError(QDMI_job_get_results(job_,
                                          QDMI_JOB_RESULT_PROBABILITIES_DENSE,
                                          size, probabilities.data(), nullptr),
                     "Querying dense probabilities");
  return probabilities;
}

auto Session::Job::getSparseStateVector() const
    -> std::map<std::string, std::complex<double>> {
  size_t keysSize = 0;
  qdmi::throwIfError(
      QDMI_job_get_results(job_, QDMI_JOB_RESULT_STATEVECTOR_SPARSE_KEYS, 0,
                           nullptr, &keysSize),
      "Querying sparse state vector keys size");

  if (keysSize == 0) {
    return {}; // Empty state vector
  }

  std::string keys(keysSize - 1, '\0');
  qdmi::throwIfError(
      QDMI_job_get_results(job_, QDMI_JOB_RESULT_STATEVECTOR_SPARSE_KEYS,
                           keysSize, keys.data(), nullptr),
      "Querying sparse state vector keys");

  size_t valuesSize = 0;
  qdmi::throwIfError(
      QDMI_job_get_results(job_, QDMI_JOB_RESULT_STATEVECTOR_SPARSE_VALUES, 0,
                           nullptr, &valuesSize),
      "Querying sparse state vector values size");

  if (valuesSize % sizeof(std::complex<double>) != 0) {
    throw std::runtime_error(
        "Invalid sparse state vector values size: not a multiple of "
        "complex<double>");
  }

  std::vector<std::complex<double>> values(valuesSize /
                                           sizeof(std::complex<double>));
  qdmi::throwIfError(
      QDMI_job_get_results(job_, QDMI_JOB_RESULT_STATEVECTOR_SPARSE_VALUES,
                           valuesSize, values.data(), nullptr),
      "Querying sparse state vector values");

  // Parse the keys (comma-separated)
  std::map<std::string, std::complex<double>> stateVector;
  std::istringstream keysStream(keys);
  std::string key;
  size_t idx = 0;
  while (std::getline(keysStream, key, ',')) {
    if (idx >= values.size()) {
      throw std::runtime_error("Sparse state vector key/value count mismatch");
    }
    stateVector[key] = values[idx];
    ++idx;
  }

  if (idx != values.size()) {
    throw std::runtime_error("Sparse state vector key/value count mismatch");
  }
  return stateVector;
}

auto Session::Job::getSparseProbabilities() const
    -> std::map<std::string, double> {
  size_t keysSize = 0;
  qdmi::throwIfError(
      QDMI_job_get_results(job_, QDMI_JOB_RESULT_PROBABILITIES_SPARSE_KEYS, 0,
                           nullptr, &keysSize),
      "Querying sparse probabilities keys size");

  if (keysSize == 0) {
    return {}; // Empty probabilities
  }

  std::string keys(keysSize - 1, '\0');
  qdmi::throwIfError(
      QDMI_job_get_results(job_, QDMI_JOB_RESULT_PROBABILITIES_SPARSE_KEYS,
                           keysSize, keys.data(), nullptr),
      "Querying sparse probabilities keys");

  size_t valuesSize = 0;
  qdmi::throwIfError(
      QDMI_job_get_results(job_, QDMI_JOB_RESULT_PROBABILITIES_SPARSE_VALUES, 0,
                           nullptr, &valuesSize),
      "Querying sparse probabilities values size");

  if (valuesSize % sizeof(double) != 0) {
    throw std::runtime_error(
        "Invalid sparse probabilities values size: not a multiple of double");
  }

  std::vector<double> values(valuesSize / sizeof(double));
  qdmi::throwIfError(
      QDMI_job_get_results(job_, QDMI_JOB_RESULT_PROBABILITIES_SPARSE_VALUES,
                           valuesSize, values.data(), nullptr),
      "Querying sparse probabilities values");

  // Parse the keys (comma-separated)
  std::map<std::string, double> probabilities;
  std::istringstream keysStream(keys);
  std::string key;
  size_t idx = 0;
  while (std::getline(keysStream, key, ',')) {
    if (idx >= values.size()) {
      throw std::runtime_error("Sparse probabilities key/value count mismatch");
    }
    probabilities[key] = values[idx];
    ++idx;
  }
  if (idx != values.size()) {
    throw std::runtime_error("Sparse probabilities key/value count mismatch");
  }
  return probabilities;
}

Session::Session(const SessionConfig& config) {
  const auto result = QDMI_session_alloc(&session_);
  qdmi::throwIfError(result, "Allocating QDMI session");

  // Helper to ensure session is freed if an exception is thrown during setup
  const auto cleanup = [this]() -> void {
    if (session_ != nullptr) {
      QDMI_session_free(session_);
      session_ = nullptr;
    }
  };
  // Helper to set session parameters
  const auto setParameter = [this](const std::optional<std::string>& value,
                                   QDMI_Session_Parameter param) -> void {
    if (value) {
      const auto status = static_cast<QDMI_STATUS>(QDMI_session_set_parameter(
          session_, param, value->size() + 1, value->c_str()));
      if (status == QDMI_ERROR_NOTSUPPORTED) {
        // Optional parameter not supported by session - skip it
        SPDLOG_INFO("Session parameter {} not supported (skipped)",
                    qdmi::toString(param));
        return;
      }
      if (status == QDMI_SUCCESS) {
        return;
      }
      std::ostringstream ss;
      ss << "Setting session parameter " << qdmi::toString(param) << ": "
         << qdmi::toString(status) << " (status = " << status << ")";
      qdmi::throwIfError(status, ss.str());
    }
  };

  try {
    // Validate file existence for authFile
    if (config.authFile) {
      if (!std::filesystem::exists(*config.authFile)) {
        throw std::runtime_error("Authentication file does not exist: " +
                                 *config.authFile);
      }
    }
    // Validate URL format for authUrl
    if (config.authUrl) {
      // Breakdown of the regex pattern:
      // 1. ^https?://              -> Start with http:// or https://
      // 2. (?:                     -> Start Host Group
      //      \[[a-fA-F0-9:]+\]     -> Branch A: IPv6 (Must be in brackets like
      //      [::1])
      //                            -> Note: No \b used here because ']' is a
      //                            non-word char
      //      |                     -> OR
      //      (?:                   -> Branch B: Alphanumeric Hosts (Group for
      //      \b check)
      //        (?:\d{1,3}\.){3}\d{1,3} -> IPv4 (e.g., 127.0.0.1)
      //        |                   -> OR
      //        localhost           -> Localhost
      //        |                   -> OR
      //        (?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6} ->
      //        Domain
      //      )\b                   -> End Branch B + Word Boundary (Prevents
      //      "localhostX")
      //    )                       -> End Host Group
      // 3. (?::\d+)?               -> Optional Port (e.g., :8080)
      // 4. (?:...)*$               -> Optional Path/Query params + End of
      // string
      static const std::regex URL_PATTERN(
          R"(^https?://(?:\[[a-fA-F0-9:]+\]|(?:(?:\d{1,3}\.){3}\d{1,3}|localhost|(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6})\b)(?::\d+)?(?:[-a-zA-Z0-9()@:%_\+.~#?&/=]*)$)",
          std::regex::optimize);
      if (!std::regex_match(*config.authUrl, URL_PATTERN)) {
        throw std::runtime_error("Invalid URL format: " + *config.authUrl);
      }
    }

    // Set session parameters
    setParameter(config.token, QDMI_SESSION_PARAMETER_TOKEN);
    setParameter(config.authFile, QDMI_SESSION_PARAMETER_AUTHFILE);
    setParameter(config.authUrl, QDMI_SESSION_PARAMETER_AUTHURL);
    setParameter(config.username, QDMI_SESSION_PARAMETER_USERNAME);
    setParameter(config.password, QDMI_SESSION_PARAMETER_PASSWORD);
    setParameter(config.projectId, QDMI_SESSION_PARAMETER_PROJECTID);
    setParameter(config.custom1, QDMI_SESSION_PARAMETER_CUSTOM1);
    setParameter(config.custom2, QDMI_SESSION_PARAMETER_CUSTOM2);
    setParameter(config.custom3, QDMI_SESSION_PARAMETER_CUSTOM3);
    setParameter(config.custom4, QDMI_SESSION_PARAMETER_CUSTOM4);
    setParameter(config.custom5, QDMI_SESSION_PARAMETER_CUSTOM5);

    // Initialize the session
    qdmi::throwIfError(QDMI_session_init(session_), "Initializing session");
  } catch (...) {
    cleanup();
    throw;
  }
}

Session::~Session() {
  if (session_ != nullptr) {
    QDMI_session_free(session_);
  }
}

Session::Session(Session&& other) noexcept : session_(other.session_) {
  other.session_ = nullptr;
}

Session& Session::operator=(Session&& other) noexcept {
  if (this != &other) {
    if (session_ != nullptr) {
      QDMI_session_free(session_);
    }
    session_ = other.session_;
    other.session_ = nullptr;
  }
  return *this;
}

auto Session::getDevices() -> std::vector<Device> {

  const auto& qdmiDevices =
      queryProperty<std::vector<QDMI_Device>>(QDMI_SESSION_PROPERTY_DEVICES);
  std::vector<Device> devices;
  devices.reserve(qdmiDevices.size());
  std::ranges::transform(
      qdmiDevices, std::back_inserter(devices),
      [](const QDMI_Device& dev) -> Device { return {Token{}, dev}; });
  return devices;
}
} // namespace fomac
