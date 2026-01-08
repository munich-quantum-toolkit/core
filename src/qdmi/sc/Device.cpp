/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

/** @file
 * @brief The MQT QDMI device implementation for superconducting devices.
 */

#include "qdmi/sc/Device.hpp"

#include "mqt_sc_qdmi/device.h"
#include "qdmi/Common.hpp"
#include "qdmi/sc/DeviceMemberInitializers.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <span>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace qdmi::sc {

Device::Device() {
  // NOLINTBEGIN(cppcoreguidelines-prefer-member-initializer)
  INITIALIZE_NAME(name_);
  INITIALIZE_QUBITSNUM(qubitsNum_);
  // NOLINTEND(cppcoreguidelines-prefer-member-initializer)
  // NOLINTNEXTLINE(misc-const-correctness)
  INITIALIZE_SITES(sites_);
  INITIALIZE_COUPLINGMAP(couplingMap_);
  INITIALIZE_OPERATIONS(operations_);
}
Device::~Device() {
  // Explicitly clear sessions before destruction to avoid spurious segfaults
  sessions_.clear();
}
auto Device::get() -> Device& {
  // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
  static auto* instance = new Device();
  // The instance is intentionally leaked to avoid static deinitialization
  // issues (cf. static (de)initialization order fiasco)
  return *instance;
}
auto Device::sessionAlloc(MQT_SC_QDMI_Device_Session* session) -> int {
  if (session == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  auto uniqueSession = std::make_unique<MQT_SC_QDMI_Device_Session_impl_d>();
  const auto& it =
      sessions_.emplace(uniqueSession.get(), std::move(uniqueSession)).first;
  // get the key, i.e., the raw pointer to the session from the map iterator
  *session = it->first;
  return QDMI_SUCCESS;
}
auto Device::sessionFree(MQT_SC_QDMI_Device_Session session) -> void {
  if (session != nullptr) {
    if (const auto& it = sessions_.find(session); it != sessions_.end()) {
      sessions_.erase(it);
    }
  }
}
auto Device::queryProperty(const QDMI_Device_Property prop, const size_t size,
                           void* value, size_t* sizeRet) const -> int {
  if ((value != nullptr && size == 0) ||
      IS_INVALID_ARGUMENT(prop, QDMI_DEVICE_PROPERTY)) {
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
  ADD_LIST_PROPERTY(QDMI_DEVICE_PROPERTY_SITES, MQT_SC_QDMI_Site, sites_, prop,
                    size, value, sizeRet)
  ADD_LIST_PROPERTY(QDMI_DEVICE_PROPERTY_COUPLINGMAP, MQT_SC_QDMI_Site,
                    couplingMap_, prop, size, value, sizeRet)
  ADD_LIST_PROPERTY(QDMI_DEVICE_PROPERTY_OPERATIONS, MQT_SC_QDMI_Operation,
                    operations_, prop, size, value, sizeRet)
  if (prop == (QDMI_DEVICE_PROPERTY_SUPPORTEDPROGRAMFORMATS)) {
    if (value != nullptr && size > 0) {
      return QDMI_ERROR_INVALIDARGUMENT;
    }
    if (sizeRet != nullptr) {
      *sizeRet = 0;
    }
    return QDMI_SUCCESS;
  }
  return QDMI_ERROR_NOTSUPPORTED;
}
} // namespace qdmi::sc

auto MQT_SC_QDMI_Device_Session_impl_d::init() -> int {
  if (status_ != Status::ALLOCATED) {
    return QDMI_ERROR_BADSTATE;
  }
  status_ = Status::INITIALIZED;
  return QDMI_SUCCESS;
}
auto MQT_SC_QDMI_Device_Session_impl_d::setParameter(
    QDMI_Device_Session_Parameter param, const size_t size,
    const void* value) const -> int {
  if ((value != nullptr && size == 0) ||
      IS_INVALID_ARGUMENT(param, QDMI_DEVICE_SESSION_PARAMETER)) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  if (status_ != Status::ALLOCATED) {
    return QDMI_ERROR_BADSTATE;
  }
  return QDMI_ERROR_NOTSUPPORTED;
}
auto MQT_SC_QDMI_Device_Session_impl_d::createDeviceJob(
    // NOLINTNEXTLINE(readability-non-const-parameter)
    MQT_SC_QDMI_Device_Job* job) -> int {
  if (job == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  if (status_ != Status::INITIALIZED) {
    return QDMI_ERROR_BADSTATE;
  }
  auto uniqueJob = std::make_unique<MQT_SC_QDMI_Device_Job_impl_d>(this);
  *job = jobs_.emplace(uniqueJob.get(), std::move(uniqueJob)).first->first;
  return QDMI_SUCCESS;
}
auto MQT_SC_QDMI_Device_Session_impl_d::freeDeviceJob(
    MQT_SC_QDMI_Device_Job job) -> void {
  if (job != nullptr) {
    jobs_.erase(job);
  }
}
auto MQT_SC_QDMI_Device_Session_impl_d::queryDeviceProperty(
    const QDMI_Device_Property prop, const size_t size, void* value,
    size_t* sizeRet) const -> int {
  if (status_ != Status::INITIALIZED) {
    return QDMI_ERROR_BADSTATE;
  }
  return qdmi::sc::Device::get().queryProperty(prop, size, value, sizeRet);
}
auto MQT_SC_QDMI_Device_Session_impl_d::querySiteProperty(
    MQT_SC_QDMI_Site site, const QDMI_Site_Property prop, const size_t size,
    void* value, size_t* sizeRet) const -> int {
  if (site == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  if (status_ != Status::INITIALIZED) {
    return QDMI_ERROR_BADSTATE;
  }
  return site->queryProperty(prop, size, value, sizeRet);
}
auto MQT_SC_QDMI_Device_Session_impl_d::queryOperationProperty(
    MQT_SC_QDMI_Operation operation, const size_t numSites,
    const MQT_SC_QDMI_Site* sites, const size_t numParams, const double* params,
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
auto MQT_SC_QDMI_Device_Job_impl_d::free() -> void {
  session_->freeDeviceJob(this);
}
// NOLINTNEXTLINE(readability-convert-member-functions-to-static)
auto MQT_SC_QDMI_Device_Job_impl_d::setParameter(
    const QDMI_Device_Job_Parameter param, const size_t size, const void* value)
    -> int {
  if ((value != nullptr && size == 0) ||
      IS_INVALID_ARGUMENT(param, QDMI_DEVICE_JOB_PARAMETER)) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  return QDMI_ERROR_NOTSUPPORTED;
}
// NOLINTNEXTLINE(readability-convert-member-functions-to-static)
auto MQT_SC_QDMI_Device_Job_impl_d::queryProperty(
    // NOLINTNEXTLINE(readability-non-const-parameter)
    const QDMI_Device_Job_Property prop, const size_t size, void* value,
    [[maybe_unused]] size_t* sizeRet) -> int {
  if ((value != nullptr && size == 0) ||
      IS_INVALID_ARGUMENT(prop, QDMI_DEVICE_JOB_PROPERTY)) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  return QDMI_ERROR_NOTSUPPORTED;
}
// NOLINTNEXTLINE(readability-convert-member-functions-to-static)
auto MQT_SC_QDMI_Device_Job_impl_d::submit() -> int {
  return QDMI_ERROR_NOTSUPPORTED;
}
// NOLINTNEXTLINE(readability-convert-member-functions-to-static)
auto MQT_SC_QDMI_Device_Job_impl_d::cancel() -> int {
  return QDMI_ERROR_NOTSUPPORTED;
}
// NOLINTNEXTLINE(readability-convert-member-functions-to-static,readability-non-const-parameter)
auto MQT_SC_QDMI_Device_Job_impl_d::check(QDMI_Job_Status* status) -> int {
  if (status == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  return QDMI_ERROR_NOTSUPPORTED;
}
// NOLINTNEXTLINE(readability-convert-member-functions-to-static)
auto MQT_SC_QDMI_Device_Job_impl_d::wait([[maybe_unused]] const size_t timeout)
    -> int {
  return QDMI_ERROR_NOTSUPPORTED;
}
// NOLINTNEXTLINE(readability-convert-member-functions-to-static)
auto MQT_SC_QDMI_Device_Job_impl_d::getResults(
    QDMI_Job_Result result,
    // NOLINTNEXTLINE(readability-non-const-parameter)
    const size_t size, void* data, [[maybe_unused]] size_t* sizeRet) -> int {
  if ((data != nullptr && size == 0) ||
      IS_INVALID_ARGUMENT(result, QDMI_JOB_RESULT)) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  return QDMI_ERROR_NOTSUPPORTED;
}
auto MQT_SC_QDMI_Site_impl_d::makeUniqueSite(const uint64_t id)
    -> std::unique_ptr<MQT_SC_QDMI_Site_impl_d> {
  const MQT_SC_QDMI_Site_impl_d site(id);
  return std::make_unique<MQT_SC_QDMI_Site_impl_d>(site);
}
auto MQT_SC_QDMI_Site_impl_d::queryProperty(const QDMI_Site_Property prop,
                                            const size_t size, void* value,
                                            size_t* sizeRet) const -> int {
  if ((value != nullptr && size == 0) ||
      IS_INVALID_ARGUMENT(prop, QDMI_SITE_PROPERTY)) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  ADD_SINGLE_VALUE_PROPERTY(QDMI_SITE_PROPERTY_INDEX, uint64_t, id_, prop, size,
                            value, sizeRet)
  return QDMI_ERROR_NOTSUPPORTED;
}
auto MQT_SC_QDMI_Operation_impl_d::sortSites() -> void {
  std::visit(
      [](auto& sites) {
        using T = std::decay_t<decltype(sites)>;
        if constexpr (std::is_same_v<T, std::vector<MQT_SC_QDMI_Site>>) {
          // Single-qubit: sort flat list by pointer address
          std::ranges::sort(sites, std::less<MQT_SC_QDMI_Site>{});
        } else if constexpr (std::is_same_v<
                                 T, std::vector<std::pair<MQT_SC_QDMI_Site,
                                                          MQT_SC_QDMI_Site>>>) {
          // Two-qubit: normalize each pair (first < second)
          // Use std::less for proper total order (pointer comparison with
          // operator> invokes undefined behavior)
          std::ranges::for_each(sites, [](auto& p) {
            if (std::less<MQT_SC_QDMI_Site>{}(p.second, p.first)) {
              std::swap(p.first, p.second);
            }
          });
          std::ranges::sort(sites);
        }
        // more cases go here if needed in the future
      },
      supportedSites_);
}
MQT_SC_QDMI_Operation_impl_d::MQT_SC_QDMI_Operation_impl_d(
    std::string name, const size_t numParameters,
    const std::vector<MQT_SC_QDMI_Site>& sites)
    : name_(std::move(name)), numParameters_(numParameters), numQubits_(1),
      supportedSites_(sites) {
  sortSites();
}
MQT_SC_QDMI_Operation_impl_d::MQT_SC_QDMI_Operation_impl_d(
    std::string name, const size_t numParameters,
    const std::vector<std::pair<MQT_SC_QDMI_Site, MQT_SC_QDMI_Site>>& sites)
    : name_(std::move(name)), numParameters_(numParameters), numQubits_(2),
      supportedSites_(sites) {
  sortSites();
}
auto MQT_SC_QDMI_Operation_impl_d::makeUniqueSingleQubit(
    std::string name, const size_t numParameters,
    const std::vector<MQT_SC_QDMI_Site>& sites)
    -> std::unique_ptr<MQT_SC_QDMI_Operation_impl_d> {
  const MQT_SC_QDMI_Operation_impl_d op(std::move(name), numParameters, sites);
  return std::make_unique<MQT_SC_QDMI_Operation_impl_d>(op);
}
auto MQT_SC_QDMI_Operation_impl_d::makeUniqueTwoQubit(
    std::string name, const size_t numParameters,
    const std::vector<std::pair<MQT_SC_QDMI_Site, MQT_SC_QDMI_Site>>& sites)
    -> std::unique_ptr<MQT_SC_QDMI_Operation_impl_d> {
  const MQT_SC_QDMI_Operation_impl_d op(std::move(name), numParameters, sites);
  return std::make_unique<MQT_SC_QDMI_Operation_impl_d>(op);
}
auto MQT_SC_QDMI_Operation_impl_d::queryProperty(
    const size_t numSites, const MQT_SC_QDMI_Site* sites,
    const size_t numParams, const double* params,
    const QDMI_Operation_Property prop, const size_t size, void* value,
    size_t* sizeRet) const -> int {
  if ((sites != nullptr && numSites == 0) ||
      (params != nullptr && numParams == 0) ||
      (value != nullptr && size == 0) ||
      IS_INVALID_ARGUMENT(prop, QDMI_OPERATION_PROPERTY)) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  if (sites != nullptr) {
    // If numQubits_ == 1 or isZoned_ == true
    if (numSites == 1) {
      // If the (single) site is not supported, return with an error
      const bool found = std::visit(
          [sites]<typename S>(const S& storedSites) -> bool {
            using T = std::decay_t<S>;
            if constexpr (std::is_same_v<T, std::vector<MQT_SC_QDMI_Site>>) {
              return std::ranges::binary_search(storedSites, *sites,
                                                std::less<MQT_SC_QDMI_Site>{});
            }
            return false; // Wrong variant type
          },
          supportedSites_);
      if (!found) {
        return QDMI_ERROR_NOTSUPPORTED;
      }
    } else if (numSites == 2) {
      // NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic)
      const std::pair needle = std::less<MQT_SC_QDMI_Site>{}(sites[0], sites[1])
                                   ? std::make_pair(sites[0], sites[1])
                                   : std::make_pair(sites[1], sites[0]);
      // NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic)
      // if the pair of sites is not supported, return with an error
      const bool found = std::visit(
          [&needle]<typename S>(const S& storedSites) -> bool {
            using T = std::decay_t<S>;
            if constexpr (std::is_same_v<
                              T, std::vector<std::pair<MQT_SC_QDMI_Site,
                                                       MQT_SC_QDMI_Site>>>) {
              return std::ranges::binary_search(storedSites, needle);
            }
            return false; // Wrong variant type
          },
          supportedSites_);
      if (!found) {
        return QDMI_ERROR_NOTSUPPORTED;
      }
    } // this device does not support operations with more than two qubits
  }
  ADD_STRING_PROPERTY(QDMI_OPERATION_PROPERTY_NAME, name_.c_str(), prop, size,
                      value, sizeRet)
  ADD_SINGLE_VALUE_PROPERTY(QDMI_OPERATION_PROPERTY_PARAMETERSNUM, size_t,
                            numParameters_, prop, size, value, sizeRet)
  ADD_SINGLE_VALUE_PROPERTY(QDMI_OPERATION_PROPERTY_QUBITSNUM, size_t,
                            numQubits_, prop, size, value, sizeRet)
  if (prop == QDMI_OPERATION_PROPERTY_SITES) {
    return std::visit(
        [&]<typename S>(const S& storedSites) -> int {
          using T = std::decay_t<S>;
          if constexpr (std::is_same_v<T, std::vector<MQT_SC_QDMI_Site>>) {
            // Single-qubit: return flat array
            ADD_LIST_PROPERTY(QDMI_OPERATION_PROPERTY_SITES, MQT_SC_QDMI_Site,
                              storedSites, prop, size, value, sizeRet)
          } else if constexpr (std::is_same_v<
                                   T,
                                   std::vector<std::pair<MQT_SC_QDMI_Site,
                                                         MQT_SC_QDMI_Site>>>) {
            // Ensure std::pair has standard layout and expected size
            static_assert(std::is_standard_layout_v<
                          std::pair<MQT_SC_QDMI_Site, MQT_SC_QDMI_Site>>);
            static_assert(
                sizeof(std::pair<MQT_SC_QDMI_Site, MQT_SC_QDMI_Site>) ==
                2 * sizeof(MQT_SC_QDMI_Site));
            // Two-qubit: reinterpret as flat array of sites using std::span
            // std::pair has standard layout, so the memory layout of
            // vector<pair<Site, Site>> is equivalent to Site[2*N]
            const auto flatView = std::span<const MQT_SC_QDMI_Site>(
                // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
                reinterpret_cast<const MQT_SC_QDMI_Site*>(storedSites.data()),
                storedSites.size() * 2);
            ADD_LIST_PROPERTY(QDMI_OPERATION_PROPERTY_SITES, MQT_SC_QDMI_Site,
                              flatView, prop, size, value, sizeRet)
          }
          // more cases go here if needed in the future
          return QDMI_ERROR_NOTSUPPORTED;
        },
        supportedSites_);
  }
  return QDMI_ERROR_NOTSUPPORTED;
}

int MQT_SC_QDMI_device_initialize() {
  // ensure the singleton is initialized
  std::ignore = qdmi::sc::Device::get();
  return QDMI_SUCCESS;
}

int MQT_SC_QDMI_device_finalize() { return QDMI_SUCCESS; }

int MQT_SC_QDMI_device_session_alloc(MQT_SC_QDMI_Device_Session* session) {
  return qdmi::sc::Device::get().sessionAlloc(session);
}

int MQT_SC_QDMI_device_session_init(MQT_SC_QDMI_Device_Session session) {
  if (session == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  return session->init();
}

void MQT_SC_QDMI_device_session_free(MQT_SC_QDMI_Device_Session session) {
  qdmi::sc::Device::get().sessionFree(session);
}

int MQT_SC_QDMI_device_session_set_parameter(
    MQT_SC_QDMI_Device_Session session, QDMI_Device_Session_Parameter param,
    const size_t size, const void* value) {
  if (session == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  return session->setParameter(param, size, value);
}

int MQT_SC_QDMI_device_session_create_device_job(
    MQT_SC_QDMI_Device_Session session, MQT_SC_QDMI_Device_Job* job) {
  if (session == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  return session->createDeviceJob(job);
}

void MQT_SC_QDMI_device_job_free(MQT_SC_QDMI_Device_Job job) { job->free(); }

int MQT_SC_QDMI_device_job_set_parameter(MQT_SC_QDMI_Device_Job job,
                                         const QDMI_Device_Job_Parameter param,
                                         const size_t size, const void* value) {
  if (job == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  return job->setParameter(param, size, value);
}

int MQT_SC_QDMI_device_job_query_property(MQT_SC_QDMI_Device_Job job,
                                          const QDMI_Device_Job_Property prop,
                                          const size_t size, void* value,
                                          size_t* sizeRet) {
  if (job == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  return job->queryProperty(prop, size, value, sizeRet);
}

int MQT_SC_QDMI_device_job_submit(MQT_SC_QDMI_Device_Job job) {
  if (job == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  return job->submit();
}

int MQT_SC_QDMI_device_job_cancel(MQT_SC_QDMI_Device_Job job) {
  if (job == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  return job->cancel();
}

int MQT_SC_QDMI_device_job_check(MQT_SC_QDMI_Device_Job job,
                                 QDMI_Job_Status* status) {
  if (job == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  return job->check(status);
}

int MQT_SC_QDMI_device_job_wait(MQT_SC_QDMI_Device_Job job,
                                const size_t timeout) {
  if (job == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  return job->wait(timeout);
}

int MQT_SC_QDMI_device_job_get_results(MQT_SC_QDMI_Device_Job job,
                                       QDMI_Job_Result result,
                                       const size_t size, void* data,
                                       size_t* sizeRet) {
  if (job == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  return job->getResults(result, size, data, sizeRet);
}

int MQT_SC_QDMI_device_session_query_device_property(
    MQT_SC_QDMI_Device_Session session, const QDMI_Device_Property prop,
    const size_t size, void* value, size_t* sizeRet) {
  if (session == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  return session->queryDeviceProperty(prop, size, value, sizeRet);
}

int MQT_SC_QDMI_device_session_query_site_property(
    MQT_SC_QDMI_Device_Session session, MQT_SC_QDMI_Site site,
    const QDMI_Site_Property prop, const size_t size, void* value,
    size_t* sizeRet) {
  if (session == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  return session->querySiteProperty(site, prop, size, value, sizeRet);
}

int MQT_SC_QDMI_device_session_query_operation_property(
    MQT_SC_QDMI_Device_Session session, MQT_SC_QDMI_Operation operation,
    const size_t numSites, const MQT_SC_QDMI_Site* sites,
    const size_t numParams, const double* params,
    const QDMI_Operation_Property prop, const size_t size, void* value,
    size_t* sizeRet) {
  if (session == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  return session->queryOperationProperty(operation, numSites, sites, numParams,
                                         params, prop, size, value, sizeRet);
}
