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

#include "Initializer.hpp"
#include "mqt_na_qdmi/device.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <utility>

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

#define ADD_STRING_PROPERTY(prop_name, prop_value, prop, size, value,          \
                            size_ret)                                          \
  {                                                                            \
    if ((prop) == (prop_name)) {                                               \
      if ((value) != nullptr) {                                                \
        if ((size) < strlen(prop_value) + 1) {                                 \
          return QDMI_ERROR_INVALIDARGUMENT;                                   \
        }                                                                      \
        strncpy(static_cast<char*>(value), prop_value, size);                  \
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
  // NOLINTNEXTLINE(cppcoreguidelines-prefer-member-initializer)
  INITIALIZE_NAME(name);
  // NOLINTNEXTLINE(cppcoreguidelines-prefer-member-initializer)
  INITIALIZE_QUBITSNUM(qubitsNum);
  INITIALIZE_OPERATIONS(operations);
  INITIALIZE_SITES(sites);
}
Device::~Device() { sessions.clear(); }
auto Device::sessionAlloc(MQT_NA_QDMI_Device_Session* session) -> int {
  if (session == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  auto uniqueSession = std::make_unique<MQT_NA_QDMI_Device_Session_impl_d>();
  *session = sessions.emplace(uniqueSession.get(), std::move(uniqueSession))
                 .first->first;
  return QDMI_SUCCESS;
}
auto Device::sessionFree(MQT_NA_QDMI_Device_Session session) -> void {
  if (session != nullptr) {
    if (const auto& it = sessions.find(session); it != sessions.end()) {
      sessions.erase(it);
    }
  }
}
auto Device::queryProperty(const QDMI_Device_Property prop, const size_t size,
                           void* value, size_t* sizeRet) -> int {
  if ((value != nullptr && size == 0) || prop >= QDMI_DEVICE_PROPERTY_MAX) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  ADD_STRING_PROPERTY(QDMI_DEVICE_PROPERTY_NAME, name.c_str(), prop, size,
                      value, sizeRet)
  // NOLINTNEXTLINE(misc-include-cleaner)
  ADD_STRING_PROPERTY(QDMI_DEVICE_PROPERTY_VERSION, MQT_CORE_VERSION, prop,
                      size, value, sizeRet)
  // NOLINTNEXTLINE(misc-include-cleaner)
  ADD_STRING_PROPERTY(QDMI_DEVICE_PROPERTY_LIBRARYVERSION, QDMI_VERSION, prop,
                      size, value, sizeRet)
  ADD_SINGLE_VALUE_PROPERTY(QDMI_DEVICE_PROPERTY_STATUS, QDMI_Device_Status,
                            QDMI_DEVICE_STATUS_IDLE, prop, size, value, sizeRet)
  ADD_SINGLE_VALUE_PROPERTY(QDMI_DEVICE_PROPERTY_QUBITSNUM, size_t, qubitsNum,
                            prop, size, value, sizeRet)
  // This device never needs calibration
  ADD_SINGLE_VALUE_PROPERTY(QDMI_DEVICE_PROPERTY_NEEDSCALIBRATION, size_t, 0,
                            prop, size, value, sizeRet)
  ADD_LIST_PROPERTY(QDMI_DEVICE_PROPERTY_SITES, MQT_NA_QDMI_Site, sites, prop,
                    size, value, sizeRet)
  ADD_LIST_PROPERTY(QDMI_DEVICE_PROPERTY_OPERATIONS, MQT_NA_QDMI_Operation,
                    operations, prop, size, value, sizeRet)
  return QDMI_ERROR_NOTSUPPORTED;
}
} // namespace qdmi

auto MQT_NA_QDMI_Device_Session_impl_d::init() -> int {
  if (status != Status::ALLOCATED) {
    return QDMI_ERROR_BADSTATE;
  }
  status = Status::INITIALIZED;
  return QDMI_SUCCESS;
}
auto MQT_NA_QDMI_Device_Session_impl_d::setParameter(
    QDMI_Device_Session_Parameter param, const size_t size,
    const void* value) const -> int {
  if ((value != nullptr && size == 0) ||
      param >= QDMI_DEVICE_SESSION_PARAMETER_MAX) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  if (status != Status::ALLOCATED) {
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
  if (status == Status::ALLOCATED) {
    return QDMI_ERROR_BADSTATE;
  }
  auto uniqueJob = std::make_unique<MQT_NA_QDMI_Device_Job_impl_d>(this);
  *job = jobs.emplace(uniqueJob.get(), std::move(uniqueJob)).first->first;
  return QDMI_SUCCESS;
}
auto MQT_NA_QDMI_Device_Session_impl_d::freeDeviceJob(
    [[maybe_unused]] MQT_NA_QDMI_Device_Job job) -> void {
  if (job != nullptr) {
    if (const auto& it = jobs.find(job); it != jobs.end()) {
      jobs.erase(it);
    }
  }
}
auto MQT_NA_QDMI_Device_Session_impl_d::queryDeviceProperty(
    const QDMI_Device_Property prop, const size_t size, void* value,
    size_t* sizeRet) const -> int {
  if (status != Status::INITIALIZED) {
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
  if (status != Status::INITIALIZED) {
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
  if (status != Status::INITIALIZED) {
    return QDMI_ERROR_BADSTATE;
  }
  return operation->queryProperty(numSites, sites, numParams, params, prop,
                                  size, value, sizeRet);
}
auto MQT_NA_QDMI_Device_Job_impl_d::free() -> void {
  // The called function is only static because jobs are not supported. We keep
  // the function call, however, as it is needed to free the job when the jobs
  // are supported.
  //===--------------------------------------------------------------------===//
  // NOLINTNEXTLINE(readability-static-accessed-through-instance)
  session->freeDeviceJob(this);
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
MQT_NA_QDMI_Site_impl_d::MQT_NA_QDMI_Site_impl_d(const size_t id,
                                                 const size_t module,
                                                 const size_t subModule,
                                                 const int64_t x,
                                                 const int64_t y)
    : id(id), module(module), subModule(subModule), x(x), y(y) {
  INITIALIZE_T1(decoherenceTimes.t1);
  INITIALIZE_T2(decoherenceTimes.t2);
}
auto MQT_NA_QDMI_Site_impl_d::queryProperty(const QDMI_Site_Property prop,
                                            const size_t size, void* value,
                                            size_t* sizeRet) const -> int {
  if ((value != nullptr && size == 0) ||
      (prop != QDMI_SITE_PROPERTY_CUSTOM1 &&
       prop != QDMI_SITE_PROPERTY_CUSTOM2 &&
       prop != QDMI_SITE_PROPERTY_CUSTOM3 &&
       prop != QDMI_SITE_PROPERTY_CUSTOM4 && prop >= QDMI_SITE_PROPERTY_MAX)) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  ADD_SINGLE_VALUE_PROPERTY(QDMI_SITE_PROPERTY_INDEX, size_t, id, prop, size,
                            value, sizeRet)
  ADD_SINGLE_VALUE_PROPERTY(QDMI_SITE_PROPERTY_CUSTOM1, size_t, module, prop,
                            size, value, sizeRet)
  ADD_SINGLE_VALUE_PROPERTY(QDMI_SITE_PROPERTY_CUSTOM2, size_t, subModule, prop,
                            size, value, sizeRet)
  ADD_SINGLE_VALUE_PROPERTY(QDMI_SITE_PROPERTY_CUSTOM3, int64_t, x, prop, size,
                            value, sizeRet)
  ADD_SINGLE_VALUE_PROPERTY(QDMI_SITE_PROPERTY_CUSTOM4, int64_t, y, prop, size,
                            value, sizeRet)
  ADD_SINGLE_VALUE_PROPERTY(QDMI_SITE_PROPERTY_T1, double, decoherenceTimes.t1,
                            prop, size, value, sizeRet)
  ADD_SINGLE_VALUE_PROPERTY(QDMI_SITE_PROPERTY_T2, double, decoherenceTimes.t2,
                            prop, size, value, sizeRet)
  return QDMI_ERROR_NOTSUPPORTED;
}
auto MQT_NA_QDMI_Operation_impl_d::isShuttling(const Type type) -> bool {
  switch (type) {
  case Type::ShuttlingLoad:
  case Type::ShuttlingMove:
  case Type::ShuttlingStore:
    return true;
  default:
    return false;
  }
}
auto MQT_NA_QDMI_Operation_impl_d::queryProperty(
    const size_t numSites, const MQT_NA_QDMI_Site* sites,
    const size_t numParams, const double* params,
    const QDMI_Operation_Property prop, const size_t size, void* value,
    size_t* sizeRet) const -> int {
  if ((sites != nullptr && numSites == 0) ||
      (params != nullptr && numParams == 0) ||
      (value != nullptr && size == 0) || prop >= QDMI_OPERATION_PROPERTY_MAX) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  ADD_STRING_PROPERTY(QDMI_OPERATION_PROPERTY_NAME, name.c_str(), prop, size,
                      value, sizeRet)
  ADD_SINGLE_VALUE_PROPERTY(QDMI_OPERATION_PROPERTY_PARAMETERSNUM, size_t,
                            numParameters, prop, size, value, sizeRet)
  if (type != Type::ShuttlingMove) {
    ADD_SINGLE_VALUE_PROPERTY(QDMI_OPERATION_PROPERTY_DURATION, double,
                              duration, prop, size, value, sizeRet)
    ADD_SINGLE_VALUE_PROPERTY(QDMI_OPERATION_PROPERTY_FIDELITY, double,
                              fidelity, prop, size, value, sizeRet)
  }
  if (!isShuttling(type)) {
    ADD_SINGLE_VALUE_PROPERTY(QDMI_OPERATION_PROPERTY_QUBITSNUM, size_t,
                              numQubits, prop, size, value, sizeRet)
  }
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
  // The called function is only static because jobs are not supported. We keep
  // the function call, however, as it is needed to free the job when the jobs
  // are supported.
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
