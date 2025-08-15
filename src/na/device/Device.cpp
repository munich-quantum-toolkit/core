/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

/** @file
 * @brief The MQT QDMI device implementation for neutral atom devices.
 */

#include "na/device/Device.hpp"

#include "mqt_na_qdmi/device.h"
#include "na/device/DeviceMemberInitializers.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <unordered_set>
#include <utility>
#include <vector>

// NOLINTBEGIN(bugprone-macro-parentheses)
#define ADD_SINGLE_VALUE_PROPERTY(prop_name, prop_type, prop_value, prop,      \
                                  size, value, size_ret)                       \
  {                                                                            \
    if ((prop) == (prop_name)) {                                               \
      if ((value) != nullptr) {                                                \
        if ((size) < sizeof(prop_type)) {                                      \
          return QDMI_ERROR_INVALIDARGUMENT;                                   \
        }                                                                      \
        *static_cast<prop_type*>(value) = prop_value;                          \
      }                                                                        \
      if ((size_ret) != nullptr) {                                             \
        *size_ret = sizeof(prop_type);                                         \
      }                                                                        \
      return QDMI_SUCCESS;                                                     \
    }                                                                          \
  }

#ifdef _WIN32
#define STRNCPY(dest, src, size)                                               \
  strncpy_s(static_cast<char*>(dest), size, src, size);
#else
#define STRNCPY(dest, src, size) strncpy(static_cast<char*>(dest), src, size);
#endif

#define ADD_STRING_PROPERTY(prop_name, prop_value, prop, size, value,          \
                            size_ret)                                          \
  {                                                                            \
    if ((prop) == (prop_name)) {                                               \
      if ((value) != nullptr) {                                                \
        if ((size) < strlen(prop_value) + 1) {                                 \
          return QDMI_ERROR_INVALIDARGUMENT;                                   \
        }                                                                      \
        STRNCPY(value, prop_value, size);                                      \
        /* NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic) */  \
        static_cast<char*>(value)[size - 1] = '\0';                            \
      }                                                                        \
      if ((size_ret) != nullptr) {                                             \
        *size_ret = strlen(prop_value) + 1;                                    \
      }                                                                        \
      return QDMI_SUCCESS;                                                     \
    }                                                                          \
  }

#define ADD_LIST_PROPERTY(prop_name, prop_type, prop_values, prop, size,       \
                          value, size_ret)                                     \
  {                                                                            \
    if ((prop) == (prop_name)) {                                               \
      if ((value) != nullptr) {                                                \
        if ((size) < (prop_values).size() * sizeof(prop_type)) {               \
          return QDMI_ERROR_INVALIDARGUMENT;                                   \
        }                                                                      \
        memcpy(static_cast<void*>(value),                                      \
               static_cast<const void*>((prop_values).data()),                 \
               (prop_values).size() * sizeof(prop_type));                      \
      }                                                                        \
      if ((size_ret) != nullptr) {                                             \
        *size_ret = (prop_values).size() * sizeof(prop_type);                  \
      }                                                                        \
      return QDMI_SUCCESS;                                                     \
    }                                                                          \
  }
// NOLINTEND(bugprone-macro-parentheses)

namespace qdmi {
Device::Device() {
  // NOLINTBEGIN(cppcoreguidelines-prefer-member-initializer)
  INITIALIZE_NAME(name_);
  INITIALIZE_QUBITSNUM(qubitsNum_);
  // NOLINTEND(cppcoreguidelines-prefer-member-initializer)
  INITIALIZE_LENGTHUNIT(lengthUnit_);
  INITIALIZE_DURATIONUNIT(durationUnit_);
  // NOLINTNEXTLINE(misc-const-correctness)
  INITIALIZE_SITES(sites_);
  INITIALIZE_OPERATIONS(operations_);
}
auto Device::sessionAlloc(MQT_NA_QDMI_Device_Session* session) -> int {
  if (session == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  auto uniqueSession = std::make_unique<MQT_NA_QDMI_Device_Session_impl_d>();
  const auto& it =
      sessions_.emplace(uniqueSession.get(), std::move(uniqueSession)).first;
  // get the key, i.e., the raw pointer to the session from the map iterator
  *session = it->first;
  return QDMI_SUCCESS;
}
auto Device::sessionFree(MQT_NA_QDMI_Device_Session session) -> void {
  if (session != nullptr) {
    if (const auto& it = sessions_.find(session); it != sessions_.end()) {
      sessions_.erase(it);
    }
  }
}
auto Device::queryProperty(const QDMI_Device_Property prop, const size_t size,
                           void* value, size_t* sizeRet) -> int {
  if ((value != nullptr && size == 0) || prop >= QDMI_DEVICE_PROPERTY_MAX) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  ADD_STRING_PROPERTY(QDMI_DEVICE_PROPERTY_NAME, name_.c_str(), prop, size,
                      value, sizeRet)
  // NOLINTNEXTLINE(misc-include-cleaner)
  ADD_STRING_PROPERTY(QDMI_DEVICE_PROPERTY_VERSION, MQT_CORE_VERSION, prop,
                      size, value, sizeRet)
  // NOLINTNEXTLINE(misc-include-cleaner)
  ADD_STRING_PROPERTY(QDMI_DEVICE_PROPERTY_LIBRARYVERSION, QDMI_VERSION, prop,
                      size, value, sizeRet)
  ADD_SINGLE_VALUE_PROPERTY(QDMI_DEVICE_PROPERTY_STATUS, QDMI_Device_Status,
                            QDMI_DEVICE_STATUS_IDLE, prop, size, value, sizeRet)
  ADD_SINGLE_VALUE_PROPERTY(QDMI_DEVICE_PROPERTY_QUBITSNUM, size_t, qubitsNum_,
                            prop, size, value, sizeRet)
  // This device never needs calibration
  ADD_SINGLE_VALUE_PROPERTY(QDMI_DEVICE_PROPERTY_NEEDSCALIBRATION, size_t, 0,
                            prop, size, value, sizeRet)
  // This device does not support pulse-level control
  ADD_SINGLE_VALUE_PROPERTY(
      QDMI_DEVICE_PROPERTY_PULSESUPPORT, QDMI_Device_Pulse_Support_Level,
      QDMI_DEVICE_PULSE_SUPPORT_LEVEL_NONE, prop, size, value, sizeRet)
  ADD_STRING_PROPERTY(QDMI_DEVICE_PROPERTY_LENGTHUNIT, lengthUnit_.unit.c_str(),
                      prop, size, value, sizeRet)
  ADD_SINGLE_VALUE_PROPERTY(QDMI_DEVICE_PROPERTY_LENGTHSCALEFACTOR, double,
                            lengthUnit_.scaleFactor, prop, size, value, sizeRet)
  ADD_STRING_PROPERTY(QDMI_DEVICE_PROPERTY_DURATIONUNIT,
                      durationUnit_.unit.c_str(), prop, size, value, sizeRet)
  ADD_SINGLE_VALUE_PROPERTY(QDMI_DEVICE_PROPERTY_DURATIONSCALEFACTOR, double,
                            durationUnit_.scaleFactor, prop, size, value,
                            sizeRet)
  ADD_LIST_PROPERTY(QDMI_DEVICE_PROPERTY_SITES, MQT_NA_QDMI_Site, sites_, prop,
                    size, value, sizeRet)
  ADD_LIST_PROPERTY(QDMI_DEVICE_PROPERTY_OPERATIONS, MQT_NA_QDMI_Operation,
                    operations_, prop, size, value, sizeRet)
  return QDMI_ERROR_NOTSUPPORTED;
}
} // namespace qdmi

auto MQT_NA_QDMI_Device_Session_impl_d::init() -> int {
  if (status_ != Status::ALLOCATED) {
    return QDMI_ERROR_BADSTATE;
  }
  status_ = Status::INITIALIZED;
  return QDMI_SUCCESS;
}
auto MQT_NA_QDMI_Device_Session_impl_d::setParameter(
    QDMI_Device_Session_Parameter param, const size_t size,
    const void* value) const -> int {
  if ((value != nullptr && size == 0) ||
      param >= QDMI_DEVICE_SESSION_PARAMETER_MAX) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  if (status_ != Status::ALLOCATED) {
    return QDMI_ERROR_BADSTATE;
  }
  return QDMI_ERROR_NOTSUPPORTED;
}
auto MQT_NA_QDMI_Device_Session_impl_d::createDeviceJob(
    // NOLINTNEXTLINE(readability-non-const-parameter)
    MQT_NA_QDMI_Device_Job* job) -> int {
  if (job == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  if (status_ == Status::ALLOCATED) {
    return QDMI_ERROR_BADSTATE;
  }
  auto uniqueJob = std::make_unique<MQT_NA_QDMI_Device_Job_impl_d>(this);
  *job = jobs_.emplace(uniqueJob.get(), std::move(uniqueJob)).first->first;
  return QDMI_SUCCESS;
}
auto MQT_NA_QDMI_Device_Session_impl_d::freeDeviceJob(
    MQT_NA_QDMI_Device_Job job) -> void {
  if (job != nullptr) {
    jobs_.erase(job);
  }
}
auto MQT_NA_QDMI_Device_Session_impl_d::queryDeviceProperty(
    const QDMI_Device_Property prop, const size_t size, void* value,
    size_t* sizeRet) const -> int {
  if (status_ != Status::INITIALIZED) {
    return QDMI_ERROR_BADSTATE;
  }
  return qdmi::Device::get().queryProperty(prop, size, value, sizeRet);
}
auto MQT_NA_QDMI_Device_Session_impl_d::querySiteProperty(
    MQT_NA_QDMI_Site site, const QDMI_Site_Property prop, const size_t size,
    void* value, size_t* sizeRet) const -> int {
  if (site == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  if (status_ != Status::INITIALIZED) {
    return QDMI_ERROR_BADSTATE;
  }
  return site->queryProperty(prop, size, value, sizeRet);
}
auto MQT_NA_QDMI_Device_Session_impl_d::queryOperationProperty(
    MQT_NA_QDMI_Operation operation, const size_t numSites,
    const MQT_NA_QDMI_Site* sites, const size_t numParams, const double* params,
    const QDMI_Operation_Property prop, const size_t size, void* value,
    size_t* sizeRet) const -> int {
  if (operation == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  if (status_ != Status::INITIALIZED) {
    return QDMI_ERROR_BADSTATE;
  }
  return operation->queryProperty(numSites, sites, numParams, params, prop,
                                  size, value, sizeRet);
}
auto MQT_NA_QDMI_Device_Job_impl_d::free() -> void {
  session_->freeDeviceJob(this);
}
auto MQT_NA_QDMI_Device_Job_impl_d::setParameter(
    const QDMI_Device_Job_Parameter param, const size_t size, const void* value)
    -> int {
  if ((value != nullptr && size == 0) ||
      param >= QDMI_DEVICE_JOB_PARAMETER_MAX) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  return QDMI_ERROR_NOTSUPPORTED;
}
auto MQT_NA_QDMI_Device_Job_impl_d::queryProperty(
    // NOLINTNEXTLINE(readability-non-const-parameter)
    const QDMI_Device_Job_Property prop, const size_t size, void* value,
    [[maybe_unused]] size_t* sizeRet) -> int {
  if ((value != nullptr && size == 0) || prop >= QDMI_DEVICE_JOB_PROPERTY_MAX) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  return QDMI_ERROR_NOTSUPPORTED;
}
auto MQT_NA_QDMI_Device_Job_impl_d::submit() -> int {
  return QDMI_ERROR_NOTSUPPORTED;
}
auto MQT_NA_QDMI_Device_Job_impl_d::cancel() -> int {
  return QDMI_ERROR_NOTSUPPORTED;
}
// NOLINTNEXTLINE(readability-non-const-parameter)
auto MQT_NA_QDMI_Device_Job_impl_d::check(QDMI_Job_Status* status) -> int {
  if (status == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  return QDMI_ERROR_NOTSUPPORTED;
}
auto MQT_NA_QDMI_Device_Job_impl_d::wait([[maybe_unused]] const size_t timeout)
    -> int {
  return QDMI_ERROR_NOTSUPPORTED;
}
auto MQT_NA_QDMI_Device_Job_impl_d::getResults(
    QDMI_Job_Result result,
    // NOLINTNEXTLINE(readability-non-const-parameter)
    const size_t size, void* data, [[maybe_unused]] size_t* sizeRet) -> int {
  if ((data != nullptr && size == 0) || result >= QDMI_JOB_RESULT_MAX) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  return QDMI_ERROR_NOTSUPPORTED;
}
MQT_NA_QDMI_Site_impl_d::MQT_NA_QDMI_Site_impl_d(const uint64_t id,
                                                 const uint64_t module,
                                                 const uint64_t subModule,
                                                 const int64_t x,
                                                 const int64_t y)
    : id_(id), moduleId_(module), subModuleId_(subModule), x_(x), y_(y) {
  INITIALIZE_DECOHERENCETIMES(decoherenceTimes_);
}
MQT_NA_QDMI_Site_impl_d::MQT_NA_QDMI_Site_impl_d(const uint64_t id,
                                                 const int64_t x,
                                                 const int64_t y,
                                                 const uint64_t width,
                                                 const uint64_t height)
    : id_(id), x_(x), y_(y), xExtent_(width), yExtent_(height), isZone(true) {
  INITIALIZE_DECOHERENCETIMES(decoherenceTimes_);
}
auto MQT_NA_QDMI_Site_impl_d::makeUniqueSite(const uint64_t id,
                                             const uint64_t moduleId,
                                             const uint64_t subModuleId,
                                             const int64_t x, const int64_t y)
    -> std::unique_ptr<MQT_NA_QDMI_Site_impl_d> {
  const MQT_NA_QDMI_Site_impl_d site(id, moduleId, subModuleId, x, y);
  return std::make_unique<MQT_NA_QDMI_Site_impl_d>(site);
}
auto MQT_NA_QDMI_Site_impl_d::makeUniqueZone(const uint64_t id, const int64_t x,
                                             const int64_t y,
                                             const uint64_t width,
                                             const uint64_t height)
    -> std::unique_ptr<MQT_NA_QDMI_Site_impl_d> {
  const MQT_NA_QDMI_Site_impl_d site(id, x, y, width, height);
  return std::make_unique<MQT_NA_QDMI_Site_impl_d>(site);
}
auto MQT_NA_QDMI_Site_impl_d::queryProperty(const QDMI_Site_Property prop,
                                            const size_t size, void* value,
                                            size_t* sizeRet) const -> int {
  if ((value != nullptr && size == 0) || prop >= QDMI_SITE_PROPERTY_MAX) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  ADD_SINGLE_VALUE_PROPERTY(QDMI_SITE_PROPERTY_INDEX, uint64_t, id_, prop, size,
                            value, sizeRet)
  ADD_SINGLE_VALUE_PROPERTY(QDMI_SITE_PROPERTY_XCOORDINATE, int64_t, x_, prop,
                            size, value, sizeRet)
  ADD_SINGLE_VALUE_PROPERTY(QDMI_SITE_PROPERTY_YCOORDINATE, int64_t, y_, prop,
                            size, value, sizeRet)
  ADD_SINGLE_VALUE_PROPERTY(QDMI_SITE_PROPERTY_T1, uint64_t,
                            decoherenceTimes_.t1_, prop, size, value, sizeRet)
  ADD_SINGLE_VALUE_PROPERTY(QDMI_SITE_PROPERTY_T2, uint64_t,
                            decoherenceTimes_.t2_, prop, size, value, sizeRet)
  ADD_SINGLE_VALUE_PROPERTY(QDMI_SITE_PROPERTY_ISZONE, bool, isZone, prop, size,
                            value, sizeRet)
  if (isZone) {
    ADD_SINGLE_VALUE_PROPERTY(QDMI_SITE_PROPERTY_XEXTENT, uint64_t, xExtent_,
                              prop, size, value, sizeRet)
    ADD_SINGLE_VALUE_PROPERTY(QDMI_SITE_PROPERTY_YEXTENT, uint64_t, yExtent_,
                              prop, size, value, sizeRet)
  } else {
    ADD_SINGLE_VALUE_PROPERTY(QDMI_SITE_PROPERTY_MODULEINDEX, uint64_t,
                              moduleId_, prop, size, value, sizeRet)
    ADD_SINGLE_VALUE_PROPERTY(QDMI_SITE_PROPERTY_SUBMODULEINDEX, uint64_t,
                              subModuleId_, prop, size, value, sizeRet)
  }
  return QDMI_ERROR_NOTSUPPORTED;
}
MQT_NA_QDMI_Operation_impl_d::MQT_NA_QDMI_Operation_impl_d(
    std::string name, const size_t numParameters, const size_t numQubits,
    const uint64_t duration, const double fidelity, MQT_NA_QDMI_Site zone)
    : name_(std::move(name)), numParameters_(numParameters),
      numQubits_(numQubits), duration_(duration), fidelity_(fidelity),
      isZoned_(true) {
  supportedSites_.emplace_back(zone);
}
MQT_NA_QDMI_Operation_impl_d::MQT_NA_QDMI_Operation_impl_d(
    std::string name, const size_t numParameters, const size_t numQubits,
    const uint64_t duration, const double fidelity,
    const uint64_t interactionRadius, uint64_t blockingRadius,
    MQT_NA_QDMI_Site zone)
    : name_(std::move(name)), numParameters_(numParameters),
      numQubits_(numQubits), duration_(duration), fidelity_(fidelity),
      interactionRadius_(interactionRadius), blockingRadius_(blockingRadius),
      isZoned_(true) {
  supportedSites_.emplace_back(zone);
}
MQT_NA_QDMI_Operation_impl_d::MQT_NA_QDMI_Operation_impl_d(
    std::string name, const size_t numParameters, const size_t numQubits,
    const uint64_t duration, const double fidelity,
    const std::vector<MQT_NA_QDMI_Site>& sites)
    : name_(std::move(name)), numParameters_(numParameters),
      numQubits_(numQubits), duration_(duration), fidelity_(fidelity),
      supportedSites_(sites) {}
MQT_NA_QDMI_Operation_impl_d::MQT_NA_QDMI_Operation_impl_d(
    std::string name, const size_t numParameters, const size_t numQubits,
    const uint64_t duration, const double fidelity,
    const uint64_t interactionRadius, uint64_t blockingRadius,
    const std::vector<MQT_NA_QDMI_Site>& sites)
    : name_(std::move(name)), numParameters_(numParameters),
      numQubits_(numQubits), duration_(duration), fidelity_(fidelity),
      interactionRadius_(interactionRadius), blockingRadius_(blockingRadius),
      supportedSites_(sites) {}
MQT_NA_QDMI_Operation_impl_d::MQT_NA_QDMI_Operation_impl_d(
    std::string name, size_t numParameters, uint64_t duration, double fidelity,
    MQT_NA_QDMI_Site zone)
    : name_(std::move(name)), numParameters_(numParameters),
      duration_(duration), fidelity_(fidelity), isZoned_(true) {
  supportedSites_.emplace_back(zone);
}
MQT_NA_QDMI_Operation_impl_d::MQT_NA_QDMI_Operation_impl_d(
    std::string name, size_t numParameters, MQT_NA_QDMI_Site zone,
    uint64_t meanShuttlingSpeed)
    : name_(std::move(name)), numParameters_(numParameters), numQubits_(0),
      meanShuttlingSpeed_(meanShuttlingSpeed), isZoned_(true) {
  supportedSites_.emplace_back(zone);
}
auto MQT_NA_QDMI_Operation_impl_d::makeUniqueGlobalSingleQubit(
    const std::string& name, size_t numParameters, uint64_t duration,
    double fidelity, MQT_NA_QDMI_Site zone)
    -> std::unique_ptr<MQT_NA_QDMI_Operation_impl_d> {
  MQT_NA_QDMI_Operation_impl_d op(name, numParameters, 1, duration, fidelity,
                                  zone);
  return std::make_unique<MQT_NA_QDMI_Operation_impl_d>(std::move(op));
}
auto MQT_NA_QDMI_Operation_impl_d::makeUniqueGlobalMultiQubit(
    const std::string& name, size_t numParameters, size_t numQubits,
    uint64_t duration, double fidelity, uint64_t interactionRadius,
    uint64_t blockingRadius, MQT_NA_QDMI_Site zone)
    -> std::unique_ptr<MQT_NA_QDMI_Operation_impl_d> {
  MQT_NA_QDMI_Operation_impl_d op(name, numParameters, numQubits, duration,
                                  fidelity, interactionRadius, blockingRadius,
                                  zone);
  return std::make_unique<MQT_NA_QDMI_Operation_impl_d>(std::move(op));
}
auto MQT_NA_QDMI_Operation_impl_d::makeUniqueLocalSingleQubit(
    const std::string& name, size_t numParameters, uint64_t duration,
    double fidelity, const std::vector<MQT_NA_QDMI_Site>& sites)
    -> std::unique_ptr<MQT_NA_QDMI_Operation_impl_d> {
  MQT_NA_QDMI_Operation_impl_d op(name, numParameters, 1, duration, fidelity,
                                  sites);
  return std::make_unique<MQT_NA_QDMI_Operation_impl_d>(std::move(op));
}
auto MQT_NA_QDMI_Operation_impl_d::makeUniqueLocalTwoQubit(
    const std::string& name, size_t numParameters, size_t numQubits,
    uint64_t duration, double fidelity, uint64_t interactionRadius,
    uint64_t blockingRadius,
    const std::vector<std::pair<MQT_NA_QDMI_Site, MQT_NA_QDMI_Site>>& sites)
    -> std::unique_ptr<MQT_NA_QDMI_Operation_impl_d> {
  // TODO: can that be solved by a cast from one vector to the other
  std::vector<MQT_NA_QDMI_Site> sitesFlattened;
  std::ranges::for_each(sites, [&sitesFlattened](const auto& sitePair) {
    sitesFlattened.emplace_back(sitePair.first);
    sitesFlattened.emplace_back(sitePair.second);
  });
  MQT_NA_QDMI_Operation_impl_d op(name, numParameters, numQubits, duration,
                                  fidelity, interactionRadius, blockingRadius,
                                  sitesFlattened);
  return std::make_unique<MQT_NA_QDMI_Operation_impl_d>(std::move(op));
}
auto MQT_NA_QDMI_Operation_impl_d::makeUniqueShuttlingLoad(
    const std::string& name, size_t numParameters, uint64_t duration,
    double fidelity, MQT_NA_QDMI_Site zone)
    -> std::unique_ptr<MQT_NA_QDMI_Operation_impl_d> {
  MQT_NA_QDMI_Operation_impl_d op(name, numParameters, duration, fidelity,
                                  zone);
  return std::make_unique<MQT_NA_QDMI_Operation_impl_d>(std::move(op));
}
auto MQT_NA_QDMI_Operation_impl_d::makeUniqueShuttlingMove(
    const std::string& name, size_t numParameters, MQT_NA_QDMI_Site zone,
    const uint64_t meanShuttlingSpeed)
    -> std::unique_ptr<MQT_NA_QDMI_Operation_impl_d> {
  MQT_NA_QDMI_Operation_impl_d op(name, numParameters, zone,
                                  meanShuttlingSpeed);
  return std::make_unique<MQT_NA_QDMI_Operation_impl_d>(std::move(op));
}
auto MQT_NA_QDMI_Operation_impl_d::makeUniqueShuttlingStore(
    const std::string& name, size_t numParameters, uint64_t duration,
    double fidelity, MQT_NA_QDMI_Site zone)
    -> std::unique_ptr<MQT_NA_QDMI_Operation_impl_d> {
  MQT_NA_QDMI_Operation_impl_d op(name, numParameters, duration, fidelity,
                                  zone);
  return std::make_unique<MQT_NA_QDMI_Operation_impl_d>(std::move(op));
}
auto MQT_NA_QDMI_Operation_impl_d::queryProperty(
    const size_t numSites, const MQT_NA_QDMI_Site* sites,
    const size_t numParams, const double* params,
    const QDMI_Operation_Property prop, const size_t size, void* value,
    size_t* sizeRet) const -> int {
  if ((sites != nullptr && numSites == 0) ||
      (params != nullptr && numParams == 0) ||
      (value != nullptr && size == 0) || prop >= QDMI_OPERATION_PROPERTY_MAX ||
      (isZoned_ && numSites > 1) ||
      (!isZoned_ && numSites > 0 && numQubits_ != numSites)) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  if (sites != nullptr) {
    // todo: Can this be improves, esp. the linear running time to
    // constant/logartihmic
    for (const MQT_NA_QDMI_Site* site = sites;
         // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
         std::cmp_less(site - sites, numSites); ++site) {
      if (std::ranges::find(supportedSites_, *site) == supportedSites_.cend()) {
        return QDMI_ERROR_NOTSUPPORTED;
      }
    }
    // Check for 2-qubit gates whether sites are within interaction radius
    if (numSites == 2) {
      const auto* sitePair = reinterpret_cast<
          const std::pair<MQT_NA_QDMI_Site, MQT_NA_QDMI_Site>*>(sites);
      const auto xDistance = sitePair->second->x_ - sitePair->first->x_;
      const auto yDistance = sitePair->second->y_ - sitePair->first->y_;
      const auto distance =
          std::sqrt(std::pow(xDistance, 2) + std::pow(yDistance, 2));
      if (distance > interactionRadius_) {
        return QDMI_ERROR_NOTSUPPORTED;
      }
    }
  }
  ADD_STRING_PROPERTY(QDMI_OPERATION_PROPERTY_NAME, name_.c_str(), prop, size,
                      value, sizeRet)
  ADD_SINGLE_VALUE_PROPERTY(QDMI_OPERATION_PROPERTY_PARAMETERSNUM, size_t,
                            numParameters_, prop, size, value, sizeRet)
  const std::vector sitesVec(supportedSites_.cbegin(), supportedSites_.cend());
  ADD_LIST_PROPERTY(QDMI_OPERATION_PROPERTY_SITES, MQT_NA_QDMI_Site, sitesVec,
                    prop, size, value, sizeRet)
  if (interactionRadius_) {
    ADD_SINGLE_VALUE_PROPERTY(QDMI_OPERATION_PROPERTY_INTERACTIONRADIUS,
                              uint64_t, *interactionRadius_, prop, size, value,
                              sizeRet)
  }
  if (blockingRadius_) {
    ADD_SINGLE_VALUE_PROPERTY(QDMI_OPERATION_PROPERTY_BLOCKINGRADIUS, uint64_t,
                              *blockingRadius_, prop, size, value, sizeRet)
  }
  if (meanShuttlingSpeed_) {
    ADD_SINGLE_VALUE_PROPERTY(QDMI_OPERATION_PROPERTY_MEANSHUTTLINGSPEED,
                              uint64_t, *meanShuttlingSpeed_, prop, size, value,
                              sizeRet)
  }
  if (duration_) {
    ADD_SINGLE_VALUE_PROPERTY(QDMI_OPERATION_PROPERTY_DURATION, uint64_t,
                              *duration_, prop, size, value, sizeRet)
  }
  if (fidelity_) {
    ADD_SINGLE_VALUE_PROPERTY(QDMI_OPERATION_PROPERTY_FIDELITY, double,
                              *fidelity_, prop, size, value, sizeRet)
  }
  if (numQubits_) {
    ADD_SINGLE_VALUE_PROPERTY(QDMI_OPERATION_PROPERTY_QUBITSNUM, size_t,
                              *numQubits_, prop, size, value, sizeRet)
  }
  ADD_SINGLE_VALUE_PROPERTY(QDMI_OPERATION_PROPERTY_ISZONED, bool, isZoned_,
                            prop, size, value, sizeRet)
  return QDMI_ERROR_NOTSUPPORTED;
}

int MQT_NA_QDMI_device_initialize() {
  std::ignore = qdmi::Device::get(); // Ensure the singleton is created
  return QDMI_SUCCESS;
}

int MQT_NA_QDMI_device_finalize() { return QDMI_SUCCESS; }

int MQT_NA_QDMI_device_session_alloc(MQT_NA_QDMI_Device_Session* session) {
  return qdmi::Device::get().sessionAlloc(session);
}

int MQT_NA_QDMI_device_session_init(MQT_NA_QDMI_Device_Session session) {
  if (session == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  return session->init();
}

void MQT_NA_QDMI_device_session_free(MQT_NA_QDMI_Device_Session session) {
  qdmi::Device::get().sessionFree(session);
}

int MQT_NA_QDMI_device_session_set_parameter(
    MQT_NA_QDMI_Device_Session session, QDMI_Device_Session_Parameter param,
    const size_t size, const void* value) {
  if (session == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  return session->setParameter(param, size, value);
}

int MQT_NA_QDMI_device_session_create_device_job(
    MQT_NA_QDMI_Device_Session session, MQT_NA_QDMI_Device_Job* job) {
  if (session == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  return session->createDeviceJob(job);
}

void MQT_NA_QDMI_device_job_free(MQT_NA_QDMI_Device_Job job) { job->free(); }

int MQT_NA_QDMI_device_job_set_parameter(MQT_NA_QDMI_Device_Job job,
                                         const QDMI_Device_Job_Parameter param,
                                         const size_t size, const void* value) {
  if (job == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  // The called function is only static because jobs are not supported. However,
  // we keep the function call for a complete implementation of QDMI and silence
  // the respective warning.
  //===--------------------------------------------------------------------===//
  // NOLINTNEXTLINE(readability-static-accessed-through-instance)
  return job->setParameter(param, size, value);
}

int MQT_NA_QDMI_device_job_query_property(MQT_NA_QDMI_Device_Job job,
                                          const QDMI_Device_Job_Property prop,
                                          const size_t size, void* value,
                                          size_t* sizeRet) {
  if (job == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  // The called function is only static because jobs are not supported. We keep
  // the function call, however, as it is needed to free the job when the jobs
  // are supported.
  //===--------------------------------------------------------------------===//
  // NOLINTNEXTLINE(readability-static-accessed-through-instance)
  return job->queryProperty(prop, size, value, sizeRet);
}

int MQT_NA_QDMI_device_job_submit(MQT_NA_QDMI_Device_Job job) {
  if (job == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  // The called function is only static because jobs are not supported. We keep
  // the function call, however, as it is needed to free the job when the jobs
  // are supported.
  //===--------------------------------------------------------------------===//
  // NOLINTNEXTLINE(readability-static-accessed-through-instance)
  return job->submit();
}

int MQT_NA_QDMI_device_job_cancel(MQT_NA_QDMI_Device_Job job) {
  if (job == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  // The called function is only static because jobs are not supported. We keep
  // the function call, however, as it is needed to free the job when the jobs
  // are supported.
  //===--------------------------------------------------------------------===//
  // NOLINTNEXTLINE(readability-static-accessed-through-instance)
  return job->cancel();
}

int MQT_NA_QDMI_device_job_check(MQT_NA_QDMI_Device_Job job,
                                 QDMI_Job_Status* status) {
  if (job == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  // The called function is only static because jobs are not supported. We keep
  // the function call, however, as it is needed to free the job when the jobs
  // are supported.
  //===--------------------------------------------------------------------===//
  // NOLINTNEXTLINE(readability-static-accessed-through-instance)
  return job->check(status);
}

int MQT_NA_QDMI_device_job_wait(MQT_NA_QDMI_Device_Job job,
                                const size_t timeout) {
  if (job == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  // The called function is only static because jobs are not supported. We keep
  // the function call, however, as it is needed to free the job when the jobs
  // are supported.
  //===--------------------------------------------------------------------===//
  // NOLINTNEXTLINE(readability-static-accessed-through-instance)
  return job->wait(timeout);
}

int MQT_NA_QDMI_device_job_get_results(MQT_NA_QDMI_Device_Job job,
                                       QDMI_Job_Result result,
                                       const size_t size, void* data,
                                       size_t* sizeRet) {
  if (job == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  // The called function is only static because jobs are not supported. We keep
  // the function call, however, as it is needed to free the job when the jobs
  // are supported.
  //===--------------------------------------------------------------------===//
  // NOLINTNEXTLINE(readability-static-accessed-through-instance)
  return job->getResults(result, size, data, sizeRet);
}

int MQT_NA_QDMI_device_session_query_device_property(
    MQT_NA_QDMI_Device_Session session, const QDMI_Device_Property prop,
    const size_t size, void* value, size_t* sizeRet) {
  if (session == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  return session->queryDeviceProperty(prop, size, value, sizeRet);
}

int MQT_NA_QDMI_device_session_query_site_property(
    MQT_NA_QDMI_Device_Session session, MQT_NA_QDMI_Site site,
    const QDMI_Site_Property prop, const size_t size, void* value,
    size_t* sizeRet) {
  if (session == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  return session->querySiteProperty(site, prop, size, value, sizeRet);
}

int MQT_NA_QDMI_device_session_query_operation_property(
    MQT_NA_QDMI_Device_Session session, MQT_NA_QDMI_Operation operation,
    const size_t numSites, const MQT_NA_QDMI_Site* sites,
    const size_t numParams, const double* params,
    const QDMI_Operation_Property prop, const size_t size, void* value,
    size_t* sizeRet) {
  if (session == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  return session->queryOperationProperty(operation, numSites, sites, numParams,
                                         params, prop, size, value, sizeRet);
}
