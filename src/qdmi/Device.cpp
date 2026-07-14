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

#include "DeviceApi.hpp"
#include "qdmi/common/Common.hpp"

#include <qdmi/constants.h>
#include <qdmi/device.h>

#include <algorithm>
#include <array>
#include <complex>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <map>
#include <memory>
#include <optional>
#include <ranges> // NOLINT(misc-include-cleaner)
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace qdmi {
namespace {
template <class Query>
[[nodiscard]] auto queryBytes(Query query, const std::string& description)
    -> std::optional<std::vector<std::byte>> {
  size_t size = 0;
  const auto sizeResult = query(0, nullptr, &size);
  if (sizeResult == QDMI_ERROR_NOTSUPPORTED) {
    return std::nullopt;
  }
  throwIfError(sizeResult, "Querying " + description + " size");
  std::vector<std::byte> bytes(size);
  if (size != 0) {
    throwIfError(query(size, bytes.data(), nullptr), "Querying " + description);
  }
  return bytes;
}

template <class T, class Query>
[[nodiscard]] auto queryValue(Query query, const std::string& description)
    -> T {
  T value{};
  throwIfError(query(sizeof(T), static_cast<void*>(&value), nullptr),
               "Querying " + description);
  return value;
}

template <class T, class Query>
[[nodiscard]] auto queryOptionalValue(Query query,
                                      const std::string& description)
    -> std::optional<T> {
  T value{};
  const auto result = query(sizeof(T), static_cast<void*>(&value), nullptr);
  if (result == QDMI_ERROR_NOTSUPPORTED) {
    return std::nullopt;
  }
  throwIfError(result, "Querying " + description);
  return value;
}

template <class T, class Query>
[[nodiscard]] auto queryVector(Query query, const std::string& description)
    -> std::vector<T> {
  size_t size = 0;
  const auto sizeResult = query(0, nullptr, &size);
  if (sizeResult == QDMI_ERROR_NOTSUPPORTED) {
    throw std::runtime_error("Querying " + description + ": Not supported.");
  }
  throwIfError(sizeResult, "Querying " + description + " size");
  if (size % sizeof(T) != 0) {
    throw std::runtime_error("Invalid byte size while querying " + description);
  }
  std::vector<T> values(size / sizeof(T));
  if (size != 0) {
    throwIfError(query(size, static_cast<void*>(values.data()), nullptr),
                 "Querying " + description);
  }
  return values;
}

template <class T, class Query>
[[nodiscard]] auto queryOptionalVector(Query query,
                                       const std::string& description)
    -> std::optional<std::vector<T>> {
  size_t size = 0;
  const auto sizeResult = query(0, nullptr, &size);
  if (sizeResult == QDMI_ERROR_NOTSUPPORTED) {
    return std::nullopt;
  }
  throwIfError(sizeResult, "Querying " + description + " size");
  if (size % sizeof(T) != 0) {
    throw std::runtime_error("Invalid byte size while querying " + description);
  }
  std::vector<T> values(size / sizeof(T));
  if (size != 0) {
    throwIfError(query(size, static_cast<void*>(values.data()), nullptr),
                 "Querying " + description);
  }
  return values;
}

template <class Query>
[[nodiscard]] auto queryString(Query query, const std::string& description)
    -> std::string {
  const auto bytes = queryBytes(std::move(query), description);
  if (!bytes || bytes->empty() || bytes->back() != std::byte{0}) {
    throw std::runtime_error("Invalid string while querying " + description);
  }
  return {reinterpret_cast<const char*>(bytes->data()), bytes->size() - 1};
}

template <class Query>
[[nodiscard]] auto queryOptionalString(Query query,
                                       const std::string& description)
    -> std::optional<std::string> {
  const auto bytes = queryBytes(std::move(query), description);
  if (!bytes) {
    return std::nullopt;
  }
  if (bytes->empty() || bytes->back() != std::byte{0}) {
    throw std::runtime_error("Invalid string while querying " + description);
  }
  return std::string(reinterpret_cast<const char*>(bytes->data()),
                     bytes->size() - 1);
}

[[nodiscard]] auto splitCommaSeparated(const std::string& values)
    -> std::vector<std::string> {
  if (values.empty()) {
    return {};
  }
  std::vector<std::string> result;
  std::istringstream stream(values);
  for (std::string value; std::getline(stream, value, ',');) {
    result.emplace_back(std::move(value));
  }
  return result;
}

[[nodiscard]] constexpr auto customOffset(const CustomProperty property)
    -> int {
  const auto offset = static_cast<int>(property) - 1;
  if (offset < 0 || offset >= 5) {
    throw std::invalid_argument("Invalid custom property selector");
  }
  return offset;
}

[[nodiscard]] constexpr auto deviceCustom(const CustomProperty property)
    -> QDMI_Device_Property {
  return static_cast<QDMI_Device_Property>(QDMI_DEVICE_PROPERTY_CUSTOM1 +
                                           customOffset(property));
}
[[nodiscard]] constexpr auto siteCustom(const CustomProperty property)
    -> QDMI_Site_Property {
  return static_cast<QDMI_Site_Property>(QDMI_SITE_PROPERTY_CUSTOM1 +
                                         customOffset(property));
}
[[nodiscard]] constexpr auto operationCustom(const CustomProperty property)
    -> QDMI_Operation_Property {
  return static_cast<QDMI_Operation_Property>(QDMI_OPERATION_PROPERTY_CUSTOM1 +
                                              customOffset(property));
}
[[nodiscard]] constexpr auto jobCustom(const CustomProperty property)
    -> QDMI_Device_Job_Property {
  return static_cast<QDMI_Device_Job_Property>(
      QDMI_DEVICE_JOB_PROPERTY_CUSTOM1 + customOffset(property));
}
[[nodiscard]] constexpr auto resultCustom(const CustomProperty property)
    -> QDMI_Job_Result {
  return static_cast<QDMI_Job_Result>(QDMI_JOB_RESULT_CUSTOM1 +
                                      customOffset(property));
}

void setCustomJobParameter(const detail::JobState& state,
                           const QDMI_Device_Job_Parameter parameter,
                           const CustomJobParameter& value) {
  const auto result = std::visit(
      [&state, parameter](const auto& typed) {
        using T = std::decay_t<decltype(typed)>;
        if constexpr (std::same_as<T, std::string>) {
          return state.device->api->setJobParameter(
              state.job, parameter, typed.size() + 1, typed.c_str());
        } else {
          return state.device->api->setJobParameter(state.job, parameter,
                                                    sizeof(T), &typed);
        }
      },
      value);
  throwIfError(result, "Setting custom parameter");
}
} // namespace

auto Device::getName() const -> std::string {
  return queryString(
      [this](const size_t size, void* value, size_t* sizeRet) {
        return state_->api->queryDevice(
            state_->session, QDMI_DEVICE_PROPERTY_NAME, size, value, sizeRet);
      },
      "device name");
}
auto Device::getVersion() const -> std::string {
  return queryString(
      [this](const size_t size, void* value, size_t* sizeRet) {
        return state_->api->queryDevice(state_->session,
                                        QDMI_DEVICE_PROPERTY_VERSION, size,
                                        value, sizeRet);
      },
      "device version");
}
auto Device::getStatus() const -> DeviceStatus {
  const auto status = queryValue<QDMI_Device_Status>(
      [this](const size_t size, void* value, size_t* sizeRet) {
        return state_->api->queryDevice(
            state_->session, QDMI_DEVICE_PROPERTY_STATUS, size, value, sizeRet);
      },
      "device status");
  return static_cast<DeviceStatus>(status);
}
auto Device::getLibraryVersion() const -> std::string {
  return queryString(
      [this](const size_t size, void* value, size_t* sizeRet) {
        return state_->api->queryDevice(state_->session,
                                        QDMI_DEVICE_PROPERTY_LIBRARYVERSION,
                                        size, value, sizeRet);
      },
      "device library version");
}
auto Device::getQubitsNum() const -> size_t {
  return queryValue<size_t>(
      [this](const size_t size, void* value, size_t* sizeRet) {
        return state_->api->queryDevice(state_->session,
                                        QDMI_DEVICE_PROPERTY_QUBITSNUM, size,
                                        value, sizeRet);
      },
      "device qubit count");
}
auto Device::getSites() const -> std::vector<Site> {
  const auto handles = queryVector<QDMI_Site>(
      [this](const size_t size, void* value, size_t* sizeRet) {
        return state_->api->queryDevice(
            state_->session, QDMI_DEVICE_PROPERTY_SITES, size, value, sizeRet);
      },
      "device sites");
  std::vector<Site> sites;
  sites.reserve(handles.size());
  std::ranges::transform(handles, std::back_inserter(sites),
                         [this](auto* handle) { return Site(state_, handle); });
  return sites;
}
auto Device::getRegularSites() const -> std::vector<Site> {
  auto sites = getSites();
  std::erase_if(sites, [](const Site& site) { return site.isZone(); });
  return sites;
}
auto Device::getZones() const -> std::vector<Site> {
  auto sites = getSites();
  std::erase_if(sites, [](const Site& site) { return !site.isZone(); });
  return sites;
}
auto Device::getOperations() const -> std::vector<Operation> {
  const auto handles = queryVector<QDMI_Operation>(
      [this](const size_t size, void* value, size_t* sizeRet) {
        return state_->api->queryDevice(state_->session,
                                        QDMI_DEVICE_PROPERTY_OPERATIONS, size,
                                        value, sizeRet);
      },
      "device operations");
  std::vector<Operation> operations;
  operations.reserve(handles.size());
  std::ranges::transform(
      handles, std::back_inserter(operations),
      [this](auto* handle) { return Operation(state_, handle); });
  return operations;
}
auto Device::getCouplingMap() const
    -> std::optional<std::vector<std::pair<Site, Site>>> {
  struct Pair {
    QDMI_Site first;
    QDMI_Site second;
  };
  const auto pairs = queryOptionalVector<Pair>(
      [this](const size_t size, void* value, size_t* sizeRet) {
        return state_->api->queryDevice(state_->session,
                                        QDMI_DEVICE_PROPERTY_COUPLINGMAP, size,
                                        value, sizeRet);
      },
      "device coupling map");
  if (!pairs) {
    return std::nullopt;
  }
  std::vector<std::pair<Site, Site>> result;
  result.reserve(pairs->size());
  std::ranges::transform(
      *pairs, std::back_inserter(result), [this](const Pair& pair) {
        return std::pair{Site(state_, pair.first), Site(state_, pair.second)};
      });
  return result;
}

#define DEVICE_OPTIONAL_VALUE(method, type, property, description)             \
  auto Device::method() const -> std::optional<type> {                         \
    return queryOptionalValue<type>(                                           \
        [this](const size_t size, void* value, size_t* sizeRet) {              \
          return state_->api->queryDevice(state_->session, property, size,     \
                                          value, sizeRet);                     \
        },                                                                     \
        description);                                                          \
  }
DEVICE_OPTIONAL_VALUE(getNeedsCalibration, size_t,
                      QDMI_DEVICE_PROPERTY_NEEDSCALIBRATION,
                      "device calibration requirement")
DEVICE_OPTIONAL_VALUE(getLengthScaleFactor, double,
                      QDMI_DEVICE_PROPERTY_LENGTHSCALEFACTOR,
                      "device length scale")
DEVICE_OPTIONAL_VALUE(getDurationScaleFactor, double,
                      QDMI_DEVICE_PROPERTY_DURATIONSCALEFACTOR,
                      "device duration scale")
DEVICE_OPTIONAL_VALUE(getMinAtomDistance, uint64_t,
                      QDMI_DEVICE_PROPERTY_MINATOMDISTANCE,
                      "device minimum atom distance")
#undef DEVICE_OPTIONAL_VALUE

auto Device::getLengthUnit() const -> std::optional<std::string> {
  return queryOptionalString(
      [this](const size_t size, void* value, size_t* sizeRet) {
        return state_->api->queryDevice(state_->session,
                                        QDMI_DEVICE_PROPERTY_LENGTHUNIT, size,
                                        value, sizeRet);
      },
      "device length unit");
}
auto Device::getDurationUnit() const -> std::optional<std::string> {
  return queryOptionalString(
      [this](const size_t size, void* value, size_t* sizeRet) {
        return state_->api->queryDevice(state_->session,
                                        QDMI_DEVICE_PROPERTY_DURATIONUNIT, size,
                                        value, sizeRet);
      },
      "device duration unit");
}
auto Device::getSupportedProgramFormats() const -> std::vector<ProgramFormat> {
  const auto formats = queryVector<QDMI_Program_Format>(
      [this](const size_t size, void* value, size_t* sizeRet) {
        return state_->api->queryDevice(
            state_->session, QDMI_DEVICE_PROPERTY_SUPPORTEDPROGRAMFORMATS, size,
            value, sizeRet);
      },
      "supported program formats");
  std::vector<ProgramFormat> result;
  result.reserve(formats.size());
  std::ranges::transform(formats, std::back_inserter(result), [](const auto f) {
    return static_cast<ProgramFormat>(f);
  });
  return result;
}
auto Device::getChildDevices() const -> std::vector<Device> {
  std::vector<Device> result;
  result.reserve(state_->children.size());
  std::ranges::transform(state_->children, std::back_inserter(result),
                         [](const auto& child) { return Device(child); });
  return result;
}
auto Device::queryCustomPropertyBytes(const CustomProperty property) const
    -> std::optional<std::vector<std::byte>> {
  return queryBytes(
      [this, property](const size_t size, void* value, size_t* sizeRet) {
        return state_->api->queryDevice(state_->session, deviceCustom(property),
                                        size, value, sizeRet);
      },
      "custom device property");
}
auto Device::submitJob(const std::string& program, const ProgramFormat format,
                       const size_t numShots,
                       const std::optional<CustomJobParameter>& custom1,
                       const std::optional<CustomJobParameter>& custom2,
                       const std::optional<CustomJobParameter>& custom3,
                       const std::optional<CustomJobParameter>& custom4,
                       const std::optional<CustomJobParameter>& custom5) const
    -> Job {
  auto jobState = std::make_shared<detail::JobState>(state_);
  const auto qdmiFormat = static_cast<QDMI_Program_Format>(format);
  throwIfError(jobState->device->api->setJobParameter(
                   jobState->job, QDMI_DEVICE_JOB_PARAMETER_PROGRAMFORMAT,
                   sizeof(qdmiFormat), &qdmiFormat),
               "Setting program format");
  throwIfError(jobState->device->api->setJobParameter(
                   jobState->job, QDMI_DEVICE_JOB_PARAMETER_PROGRAM,
                   program.size() + 1, program.c_str()),
               "Setting program");
  throwIfError(jobState->device->api->setJobParameter(
                   jobState->job, QDMI_DEVICE_JOB_PARAMETER_SHOTSNUM,
                   sizeof(numShots), &numShots),
               "Setting number of shots");
  const std::array customValues{&custom1, &custom2, &custom3, &custom4,
                                &custom5};
  for (size_t i = 0; i < customValues.size(); ++i) {
    if (*customValues[i]) {
      setCustomJobParameter(
          *jobState,
          static_cast<QDMI_Device_Job_Parameter>(
              QDMI_DEVICE_JOB_PARAMETER_CUSTOM1 + static_cast<int>(i)),
          **customValues[i]);
    }
  }
  jobState->device->api->submitJob(jobState->job);
  return Job(std::move(jobState));
}

auto Job::check() const -> JobStatus {
  return static_cast<JobStatus>(state_->device->api->checkJob(state_->job));
}
auto Job::wait(const size_t timeout) const -> bool {
  return state_->device->api->waitJob(state_->job, timeout);
}
void Job::cancel() const { state_->device->api->cancelJob(state_->job); }
auto Job::getId() const -> std::string {
  return queryString(
      [this](const size_t size, void* value, size_t* sizeRet) {
        return state_->device->api->queryJobProperty(
            state_->job, QDMI_DEVICE_JOB_PROPERTY_ID, size, value, sizeRet);
      },
      "job ID");
}
auto Job::getProgramFormat() const -> ProgramFormat {
  const auto format = queryValue<QDMI_Program_Format>(
      [this](const size_t size, void* value, size_t* sizeRet) {
        return state_->device->api->queryJobProperty(
            state_->job, QDMI_DEVICE_JOB_PROPERTY_PROGRAMFORMAT, size, value,
            sizeRet);
      },
      "job program format");
  return static_cast<ProgramFormat>(format);
}
auto Job::getProgram() const -> std::string {
  return queryString(
      [this](const size_t size, void* value, size_t* sizeRet) {
        return state_->device->api->queryJobProperty(
            state_->job, QDMI_DEVICE_JOB_PROPERTY_PROGRAM, size, value,
            sizeRet);
      },
      "job program");
}
auto Job::getNumShots() const -> size_t {
  return queryValue<size_t>(
      [this](const size_t size, void* value, size_t* sizeRet) {
        return state_->device->api->queryJobProperty(
            state_->job, QDMI_DEVICE_JOB_PROPERTY_SHOTSNUM, size, value,
            sizeRet);
      },
      "job shot count");
}
auto Job::queryCustomPropertyBytes(const CustomProperty property) const
    -> std::optional<std::vector<std::byte>> {
  return queryBytes(
      [this, property](const size_t size, void* value, size_t* sizeRet) {
        return state_->device->api->queryJobProperty(
            state_->job, jobCustom(property), size, value, sizeRet);
      },
      "custom job property");
}
auto Job::getCustomResultBytes(const CustomProperty property) const
    -> std::optional<std::vector<std::byte>> {
  return queryBytes(
      [this, property](const size_t size, void* value, size_t* sizeRet) {
        return state_->device->api->getJobResult(
            state_->job, resultCustom(property), size, value, sizeRet);
      },
      "custom job result");
}
auto Job::getShots() const -> std::vector<std::string> {
  const auto bytes = queryBytes(
      [this](const size_t size, void* value, size_t* sizeRet) {
        return state_->device->api->getJobResult(
            state_->job, QDMI_JOB_RESULT_SHOTS, size, value, sizeRet);
      },
      "job shots");
  if (!bytes) {
    throw std::runtime_error("Querying job shots: Not supported.");
  }
  if (bytes->empty()) {
    return {};
  }
  if (bytes->back() != std::byte{0}) {
    throw std::runtime_error("Invalid string while querying job shots");
  }
  return splitCommaSeparated(std::string(
      reinterpret_cast<const char*>(bytes->data()), bytes->size() - 1));
}
auto Job::getCounts() const -> std::map<std::string, size_t> {
  const auto keys = splitCommaSeparated(queryString(
      [this](const size_t size, void* value, size_t* sizeRet) {
        return state_->device->api->getJobResult(
            state_->job, QDMI_JOB_RESULT_HIST_KEYS, size, value, sizeRet);
      },
      "histogram keys"));
  const auto values = queryVector<size_t>(
      [this](const size_t size, void* value, size_t* sizeRet) {
        return state_->device->api->getJobResult(
            state_->job, QDMI_JOB_RESULT_HIST_VALUES, size, value, sizeRet);
      },
      "histogram values");
  if (keys.size() != values.size()) {
    throw std::runtime_error("Histogram key/value lengths do not match");
  }
  std::map<std::string, size_t> result;
  for (size_t i = 0; i < keys.size(); ++i) {
    result.emplace(keys[i], values[i]);
  }
  return result;
}
auto Job::getDenseStateVector() const -> std::vector<std::complex<double>> {
  return queryVector<std::complex<double>>(
      [this](const size_t size, void* value, size_t* sizeRet) {
        return state_->device->api->getJobResult(
            state_->job, QDMI_JOB_RESULT_STATEVECTOR_DENSE, size, value,
            sizeRet);
      },
      "dense state vector");
}
auto Job::getDenseProbabilities() const -> std::vector<double> {
  return queryVector<double>(
      [this](const size_t size, void* value, size_t* sizeRet) {
        return state_->device->api->getJobResult(
            state_->job, QDMI_JOB_RESULT_PROBABILITIES_DENSE, size, value,
            sizeRet);
      },
      "dense probabilities");
}
auto Job::getSparseStateVector() const
    -> std::map<std::string, std::complex<double>> {
  const auto keys = splitCommaSeparated(queryString(
      [this](const size_t size, void* value, size_t* sizeRet) {
        return state_->device->api->getJobResult(
            state_->job, QDMI_JOB_RESULT_STATEVECTOR_SPARSE_KEYS, size, value,
            sizeRet);
      },
      "sparse state-vector keys"));
  const auto values = queryVector<std::complex<double>>(
      [this](const size_t size, void* value, size_t* sizeRet) {
        return state_->device->api->getJobResult(
            state_->job, QDMI_JOB_RESULT_STATEVECTOR_SPARSE_VALUES, size, value,
            sizeRet);
      },
      "sparse state-vector values");
  if (keys.size() != values.size()) {
    throw std::runtime_error("Sparse state-vector lengths do not match");
  }
  std::map<std::string, std::complex<double>> result;
  for (size_t i = 0; i < keys.size(); ++i) {
    result.emplace(keys[i], values[i]);
  }
  return result;
}
auto Job::getSparseProbabilities() const -> std::map<std::string, double> {
  const auto keys = splitCommaSeparated(queryString(
      [this](const size_t size, void* value, size_t* sizeRet) {
        return state_->device->api->getJobResult(
            state_->job, QDMI_JOB_RESULT_PROBABILITIES_SPARSE_KEYS, size, value,
            sizeRet);
      },
      "sparse probability keys"));
  const auto values = queryVector<double>(
      [this](const size_t size, void* value, size_t* sizeRet) {
        return state_->device->api->getJobResult(
            state_->job, QDMI_JOB_RESULT_PROBABILITIES_SPARSE_VALUES, size,
            value, sizeRet);
      },
      "sparse probability values");
  if (keys.size() != values.size()) {
    throw std::runtime_error("Sparse probability lengths do not match");
  }
  std::map<std::string, double> result;
  for (size_t i = 0; i < keys.size(); ++i) {
    result.emplace(keys[i], values[i]);
  }
  return result;
}

#define SITE_OPTIONAL_VALUE(method, type, property, description)               \
  auto Site::method() const -> std::optional<type> {                           \
    return queryOptionalValue<type>(                                           \
        [this](const size_t size, void* value, size_t* sizeRet) {              \
          return state_->api->querySite(state_->session,                       \
                                        static_cast<QDMI_Site>(handle_),       \
                                        property, size, value, sizeRet);       \
        },                                                                     \
        description);                                                          \
  }
auto Site::getIndex() const -> size_t {
  return queryValue<size_t>(
      [this](const size_t size, void* value, size_t* sizeRet) {
        return state_->api->querySite(
            state_->session, static_cast<QDMI_Site>(handle_),
            QDMI_SITE_PROPERTY_INDEX, size, value, sizeRet);
      },
      "site index");
}
SITE_OPTIONAL_VALUE(getT1, uint64_t, QDMI_SITE_PROPERTY_T1, "site T1")
SITE_OPTIONAL_VALUE(getT2, uint64_t, QDMI_SITE_PROPERTY_T2, "site T2")
SITE_OPTIONAL_VALUE(getXCoordinate, int64_t, QDMI_SITE_PROPERTY_XCOORDINATE,
                    "site x coordinate")
SITE_OPTIONAL_VALUE(getYCoordinate, int64_t, QDMI_SITE_PROPERTY_YCOORDINATE,
                    "site y coordinate")
SITE_OPTIONAL_VALUE(getZCoordinate, int64_t, QDMI_SITE_PROPERTY_ZCOORDINATE,
                    "site z coordinate")
SITE_OPTIONAL_VALUE(getXExtent, uint64_t, QDMI_SITE_PROPERTY_XEXTENT,
                    "site x extent")
SITE_OPTIONAL_VALUE(getYExtent, uint64_t, QDMI_SITE_PROPERTY_YEXTENT,
                    "site y extent")
SITE_OPTIONAL_VALUE(getZExtent, uint64_t, QDMI_SITE_PROPERTY_ZEXTENT,
                    "site z extent")
SITE_OPTIONAL_VALUE(getModuleIndex, uint64_t, QDMI_SITE_PROPERTY_MODULEINDEX,
                    "site module index")
SITE_OPTIONAL_VALUE(getSubmoduleIndex, uint64_t,
                    QDMI_SITE_PROPERTY_SUBMODULEINDEX, "site submodule index")
#undef SITE_OPTIONAL_VALUE
auto Site::getName() const -> std::optional<std::string> {
  return queryOptionalString(
      [this](const size_t size, void* value, size_t* sizeRet) {
        return state_->api->querySite(
            state_->session, static_cast<QDMI_Site>(handle_),
            QDMI_SITE_PROPERTY_NAME, size, value, sizeRet);
      },
      "site name");
}
auto Site::isZone() const -> bool {
  return queryOptionalValue<bool>(
             [this](const size_t size, void* value, size_t* sizeRet) {
               return state_->api->querySite(
                   state_->session, static_cast<QDMI_Site>(handle_),
                   QDMI_SITE_PROPERTY_ISZONE, size, value, sizeRet);
             },
             "site zone flag")
      .value_or(false);
}
auto Site::queryCustomPropertyBytes(const CustomProperty property) const
    -> std::optional<std::vector<std::byte>> {
  return queryBytes(
      [this, property](const size_t size, void* value, size_t* sizeRet) {
        return state_->api->querySite(
            state_->session, static_cast<QDMI_Site>(handle_),
            siteCustom(property), size, value, sizeRet);
      },
      "custom site property");
}

auto Operation::siteHandles(const std::vector<Site>& sites)
    -> std::vector<void*> {
  std::vector<void*> handles;
  handles.reserve(sites.size());
  std::ranges::transform(
      sites, std::back_inserter(handles),
      [](const Site& site) { return static_cast<QDMI_Site>(site.handle_); });
  return handles;
}

#define OPERATION_OPTIONAL_VALUE(method, type, property, description)          \
  auto Operation::method(const std::vector<Site>& sites,                       \
                         const std::vector<double>& params) const              \
      -> std::optional<type> {                                                 \
    const auto opaqueHandles = siteHandles(sites);                             \
    const auto* handles =                                                      \
        reinterpret_cast<const QDMI_Site*>(opaqueHandles.data());              \
    return queryOptionalValue<type>(                                           \
        [this, &opaqueHandles, handles,                                        \
         &params](const size_t size, void* value, size_t* sizeRet) {           \
          return state_->api->queryOperation(                                  \
              state_->session, static_cast<QDMI_Operation>(handle_),           \
              opaqueHandles.size(), handles, params.size(), params.data(),     \
              property, size, value, sizeRet);                                 \
        },                                                                     \
        description);                                                          \
  }
auto Operation::getName(const std::vector<Site>& sites,
                        const std::vector<double>& params) const
    -> std::string {
  const auto opaqueHandles = siteHandles(sites);
  const auto* handles =
      reinterpret_cast<const QDMI_Site*>(opaqueHandles.data());
  return queryString(
      [this, &opaqueHandles, handles, &params](const size_t size, void* value,
                                               size_t* sizeRet) {
        return state_->api->queryOperation(
            state_->session, static_cast<QDMI_Operation>(handle_),
            opaqueHandles.size(), handles, params.size(), params.data(),
            QDMI_OPERATION_PROPERTY_NAME, size, value, sizeRet);
      },
      "operation name");
}
OPERATION_OPTIONAL_VALUE(getQubitsNum, size_t,
                         QDMI_OPERATION_PROPERTY_QUBITSNUM,
                         "operation qubit count")
OPERATION_OPTIONAL_VALUE(getDuration, uint64_t,
                         QDMI_OPERATION_PROPERTY_DURATION, "operation duration")
OPERATION_OPTIONAL_VALUE(getFidelity, double, QDMI_OPERATION_PROPERTY_FIDELITY,
                         "operation fidelity")
OPERATION_OPTIONAL_VALUE(getInteractionRadius, uint64_t,
                         QDMI_OPERATION_PROPERTY_INTERACTIONRADIUS,
                         "operation interaction radius")
OPERATION_OPTIONAL_VALUE(getBlockingRadius, uint64_t,
                         QDMI_OPERATION_PROPERTY_BLOCKINGRADIUS,
                         "operation blocking radius")
OPERATION_OPTIONAL_VALUE(getIdlingFidelity, double,
                         QDMI_OPERATION_PROPERTY_IDLINGFIDELITY,
                         "operation idling fidelity")
OPERATION_OPTIONAL_VALUE(getMeanShuttlingSpeed, uint64_t,
                         QDMI_OPERATION_PROPERTY_MEANSHUTTLINGSPEED,
                         "operation mean shuttling speed")
#undef OPERATION_OPTIONAL_VALUE
auto Operation::getParametersNum(const std::vector<Site>& sites,
                                 const std::vector<double>& params) const
    -> size_t {
  const auto opaqueHandles = siteHandles(sites);
  const auto* handles =
      reinterpret_cast<const QDMI_Site*>(opaqueHandles.data());
  return queryValue<size_t>(
      [this, &opaqueHandles, handles, &params](const size_t size, void* value,
                                               size_t* sizeRet) {
        return state_->api->queryOperation(
            state_->session, static_cast<QDMI_Operation>(handle_),
            opaqueHandles.size(), handles, params.size(), params.data(),
            QDMI_OPERATION_PROPERTY_PARAMETERSNUM, size, value, sizeRet);
      },
      "operation parameter count");
}
auto Operation::isZoned() const -> bool {
  const std::vector<QDMI_Site> sites;
  const std::vector<double> params;
  return queryOptionalValue<bool>(
             [this, &sites, &params](const size_t size, void* value,
                                     size_t* sizeRet) {
               return state_->api->queryOperation(
                   state_->session, static_cast<QDMI_Operation>(handle_), 0,
                   sites.data(), 0, params.data(),
                   QDMI_OPERATION_PROPERTY_ISZONED, size, value, sizeRet);
             },
             "operation zone flag")
      .value_or(false);
}
auto Operation::getSites() const -> std::optional<std::vector<Site>> {
  const std::vector<QDMI_Site> sites;
  const std::vector<double> params;
  const auto handles = queryOptionalVector<QDMI_Site>(
      [this, &sites, &params](const size_t size, void* value, size_t* sizeRet) {
        return state_->api->queryOperation(
            state_->session, static_cast<QDMI_Operation>(handle_), 0,
            sites.data(), 0, params.data(), QDMI_OPERATION_PROPERTY_SITES, size,
            value, sizeRet);
      },
      "operation sites");
  if (!handles) {
    return std::nullopt;
  }
  std::vector<Site> result;
  result.reserve(handles->size());
  std::ranges::transform(*handles, std::back_inserter(result),
                         [this](auto* site) { return Site(state_, site); });
  return result;
}
auto Operation::getSitePairs() const
    -> std::optional<std::vector<std::pair<Site, Site>>> {
  if (isZoned() || getQubitsNum().value_or(0) != 2) {
    return std::nullopt;
  }
  const auto sites = getSites();
  if (!sites || sites->empty() || sites->size() % 2 != 0) {
    return std::nullopt;
  }
  std::vector<std::pair<Site, Site>> pairs;
  pairs.reserve(sites->size() / 2);
  for (size_t i = 0; i < sites->size(); i += 2) {
    pairs.emplace_back((*sites)[i], (*sites)[i + 1]);
  }
  return pairs;
}
auto Operation::queryCustomPropertyBytes(
    const CustomProperty property, const std::vector<Site>& sites,
    const std::vector<double>& params) const
    -> std::optional<std::vector<std::byte>> {
  const auto opaqueHandles = siteHandles(sites);
  const auto* handles =
      reinterpret_cast<const QDMI_Site*>(opaqueHandles.data());
  return queryBytes(
      [this, property, &opaqueHandles, handles,
       &params](const size_t size, void* value, size_t* sizeRet) {
        return state_->api->queryOperation(
            state_->session, static_cast<QDMI_Operation>(handle_),
            opaqueHandles.size(), handles, params.size(), params.data(),
            operationCustom(property), size, value, sizeRet);
      },
      "custom operation property");
}
} // namespace qdmi
