/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "qdmi/Driver.hpp"

#include "mqt_na_qdmi/device.h"

#include <cassert>
#include <cstddef>
#include <cstring>
#include <dlfcn.h>
#include <exception>
#include <memory>
#include <qdmi/client.h>
#include <qdmi/device.h>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace qdmi {
// Macro to load a static symbol from a statically linked library.
// @param prefix is the prefix used for the function names in the library.
// @param symbol is the name of the symbol to load.
#define LOAD_STATIC_SYMBOL(prefix, symbol)                                     \
  {                                                                            \
    (symbol) = reinterpret_cast<decltype(symbol)>(prefix##_QDMI_##symbol);     \
  }
#define DEFINE_STATIC_LIBRARY(prefix)                                          \
  prefix##DeviceLibrary::prefix##DeviceLibrary() {                             \
    /* NOLINTBEGIN(cppcoreguidelines-pro-type-reinterpret-cast) */             \
    /* load the function symbols from the static library */                    \
    LOAD_STATIC_SYMBOL(prefix, device_initialize)                              \
    LOAD_STATIC_SYMBOL(prefix, device_finalize)                                \
    /* device session interface */                                             \
    LOAD_STATIC_SYMBOL(prefix, device_session_alloc)                           \
    LOAD_STATIC_SYMBOL(prefix, device_session_init)                            \
    LOAD_STATIC_SYMBOL(prefix, device_session_free)                            \
    LOAD_STATIC_SYMBOL(prefix, device_session_set_parameter)                   \
    /* device job interface */                                                 \
    LOAD_STATIC_SYMBOL(prefix, device_session_create_device_job)               \
    LOAD_STATIC_SYMBOL(prefix, device_job_free)                                \
    LOAD_STATIC_SYMBOL(prefix, device_job_set_parameter)                       \
    LOAD_STATIC_SYMBOL(prefix, device_job_query_property)                      \
    LOAD_STATIC_SYMBOL(prefix, device_job_submit)                              \
    LOAD_STATIC_SYMBOL(prefix, device_job_cancel)                              \
    LOAD_STATIC_SYMBOL(prefix, device_job_check)                               \
    LOAD_STATIC_SYMBOL(prefix, device_job_wait)                                \
    LOAD_STATIC_SYMBOL(prefix, device_job_get_results)                         \
    /* device query interface */                                               \
    LOAD_STATIC_SYMBOL(prefix, device_session_query_device_property)           \
    LOAD_STATIC_SYMBOL(prefix, device_session_query_site_property)             \
    LOAD_STATIC_SYMBOL(prefix, device_session_query_operation_property)        \
    /* NOLINTEND(cppcoreguidelines-pro-type-reinterpret-cast) */               \
    /* initialize the device */                                                \
    device_initialize();                                                       \
  }                                                                            \
                                                                               \
  prefix##DeviceLibrary::~prefix##DeviceLibrary() {                            \
    /* Check if QDMI_device_finalize is not NULL before calling it. */         \
    if (device_finalize != nullptr) {                                          \
      device_finalize();                                                       \
    }                                                                          \
  }
DEFINE_STATIC_LIBRARY(MQT_NA)
DynamicDeviceLibrary::DynamicDeviceLibrary(const std::string& libName,
                                           const std::string& prefix)
    : libHandle_(dlopen(libName.c_str(), RTLD_NOW | RTLD_LOCAL)) {
  if (libHandle_ == nullptr) {
    throw std::runtime_error("Couldn't open the device library: " + libName);
  }
  if (openLibHandles().contains(libHandle_)) {
    // dlopen uses reference counting, so we need to decrement the reference
    // count that was increased by dlopen.
    dlclose(libHandle_);
    throw std::runtime_error("Device library already loaded: " + libName);
  }
  openLibHandles().emplace(libHandle_);
  // Macro for loading a symbol from the dynamic library.
  // @param symbol is the name of the symbol to load.
#define LOAD_DYNAMIC_SYMBOL(symbol)                                            \
  {                                                                            \
    const std::string symbolName = std::string(prefix) + "_QDMI_" + #symbol;   \
    (symbol) = reinterpret_cast<decltype(symbol)>(                             \
        dlsym(libHandle, symbolName.c_str()));                                 \
    if ((symbol) == nullptr) {                                                 \
      throw std::runtime_error("Failed to load symbol: " + symbolName);        \
    }                                                                          \
  }
  try {
    // NOLINTBEGIN(cppcoreguidelines-pro-type-reinterpret-cast)
    // load the function symbols from the dynamic library
    LOAD_DYNAMIC_SYMBOL(device_initialize)
    LOAD_DYNAMIC_SYMBOL(device_finalize)
    // device session interface
    LOAD_DYNAMIC_SYMBOL(device_session_alloc)
    LOAD_DYNAMIC_SYMBOL(device_session_init)
    LOAD_DYNAMIC_SYMBOL(device_session_free)
    LOAD_DYNAMIC_SYMBOL(device_session_set_parameter)
    // device job interface
    LOAD_DYNAMIC_SYMBOL(device_session_create_device_job)
    LOAD_DYNAMIC_SYMBOL(device_job_free)
    LOAD_DYNAMIC_SYMBOL(device_job_set_parameter)
    LOAD_DYNAMIC_SYMBOL(device_job_query_property)
    LOAD_DYNAMIC_SYMBOL(device_job_submit)
    LOAD_DYNAMIC_SYMBOL(device_job_cancel)
    LOAD_DYNAMIC_SYMBOL(device_job_check)
    LOAD_DYNAMIC_SYMBOL(device_job_wait)
    LOAD_DYNAMIC_SYMBOL(device_job_get_results)
    // device query interface
    LOAD_DYNAMIC_SYMBOL(device_session_query_device_property)
    LOAD_DYNAMIC_SYMBOL(device_session_query_site_property)
    LOAD_DYNAMIC_SYMBOL(device_session_query_operation_property)
    // NOLINTEND(cppcoreguidelines-pro-type-reinterpret-cast)
  } catch (const std::exception&) {
    dlclose(libHandle_);
    throw;
  }
  // initialize the device
  device_initialize();
}

DynamicDeviceLibrary::~DynamicDeviceLibrary() {
  // Check if QDMI_device_finalize is not NULL before calling it.
  if (device_finalize != nullptr) {
    device_finalize();
  }
  // close the dynamic library
  if (libHandle_ != nullptr) {
    dlclose(libHandle_);
  }
}
} // namespace qdmi

QDMI_Device_impl_d::QDMI_Device_impl_d(
    std::unique_ptr<qdmi::DeviceLibrary>&& lib)
    : library_(std::move(lib)) {
  if (library_->device_session_alloc(&deviceSession_) != QDMI_SUCCESS) {
    throw std::runtime_error("Failed to allocate device session");
  }
  if (library_->device_session_init(deviceSession_) != QDMI_SUCCESS) {
    library_->device_session_free(deviceSession_);
    throw std::runtime_error("Failed to initialize device session");
  }
}

auto QDMI_Device_impl_d::createJob(QDMI_Job* job) -> int {
  if (job == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  QDMI_Device_Job deviceJob = nullptr;
  auto result =
      library_->device_session_create_device_job(deviceSession_, &deviceJob);
  if (result != QDMI_SUCCESS) {
    return result;
  }
  auto uniqueJob =
      std::make_unique<QDMI_Job_impl_d>(library_.get(), deviceJob, this);
  *job = jobs_.emplace(uniqueJob.get(), std::move(uniqueJob)).first->first;
  return QDMI_SUCCESS;
}

auto QDMI_Device_impl_d::freeJob(QDMI_Job job) -> void {
  if (job != nullptr) {
    jobs_.erase(job);
  }
}

auto QDMI_Device_impl_d::queryDeviceProperty(QDMI_Device_Property prop,
                                             const size_t size, void* value,
                                             size_t* sizeRet) const -> int {
  return library_->device_session_query_device_property(deviceSession_, prop,
                                                        size, value, sizeRet);
}

auto QDMI_Device_impl_d::querySiteProperty(QDMI_Site site,
                                           QDMI_Site_Property prop,
                                           const size_t size, void* value,
                                           size_t* sizeRet) const -> int {
  return library_->device_session_query_site_property(
      deviceSession_, site, prop, size, value, sizeRet);
}

auto QDMI_Device_impl_d::queryOperationProperty(
    QDMI_Operation operation, const size_t numSites, const QDMI_Site* sites,
    const size_t numParams, const double* params, QDMI_Operation_Property prop,
    const size_t size, void* value, size_t* sizeRet) const -> int {
  return library_->device_session_query_operation_property(
      deviceSession_, operation, numSites, sites, numParams, params, prop, size,
      value, sizeRet);
}

QDMI_Job_impl_d::~QDMI_Job_impl_d() { library_->device_job_free(deviceJob_); }
auto QDMI_Job_impl_d::setParameter(QDMI_Job_Parameter param, const size_t size,
                                   const void* value) const -> int {
  switch (param) {
  case QDMI_JOB_PARAMETER_PROGRAM:
    return library_->device_job_set_parameter(
        deviceJob_, QDMI_DEVICE_JOB_PARAMETER_PROGRAM, size, value);
  case QDMI_JOB_PARAMETER_PROGRAMFORMAT:
    return library_->device_job_set_parameter(
        deviceJob_, QDMI_DEVICE_JOB_PARAMETER_PROGRAMFORMAT, size, value);
  case QDMI_JOB_PARAMETER_SHOTSNUM:
    return library_->device_job_set_parameter(
        deviceJob_, QDMI_DEVICE_JOB_PARAMETER_SHOTSNUM, size, value);
  default:
    return QDMI_ERROR_NOTSUPPORTED;
  }
}

auto QDMI_Job_impl_d::queryProperty(QDMI_Job_Property prop, const size_t size,
                                    void* value, size_t* sizeRet) const -> int {
  switch (prop) {
  case QDMI_JOB_PROPERTY_ID:
    return library_->device_job_query_property(
        deviceJob_, QDMI_DEVICE_JOB_PROPERTY_ID, size, value, sizeRet);
  case QDMI_JOB_PROPERTY_PROGRAM:
    return library_->device_job_query_property(
        deviceJob_, QDMI_DEVICE_JOB_PROPERTY_PROGRAM, size, value, sizeRet);
  case QDMI_JOB_PROPERTY_PROGRAMFORMAT:
    return library_->device_job_query_property(
        deviceJob_, QDMI_DEVICE_JOB_PROPERTY_PROGRAMFORMAT, size, value,
        sizeRet);
  case QDMI_JOB_PROPERTY_SHOTSNUM:
    return library_->device_job_query_property(
        deviceJob_, QDMI_DEVICE_JOB_PROPERTY_SHOTSNUM, size, value, sizeRet);
  default:
    return QDMI_ERROR_NOTSUPPORTED;
  }
}

auto QDMI_Job_impl_d::submit() const -> int {
  return library_->device_job_submit(deviceJob_);
}

auto QDMI_Job_impl_d::cancel() const -> int {
  return library_->device_job_cancel(deviceJob_);
}

auto QDMI_Job_impl_d::check(QDMI_Job_Status* status) const -> int {
  return library_->device_job_check(deviceJob_, status);
}

auto QDMI_Job_impl_d::wait(size_t timeout) const -> int {
  return library_->device_job_wait(deviceJob_, timeout);
}

auto QDMI_Job_impl_d::getResults(QDMI_Job_Result result, const size_t size,
                                 void* data, size_t* sizeRet) const -> int {
  return library_->device_job_get_results(deviceJob_, result, size, data,
                                          sizeRet);
}

auto QDMI_Job_impl_d::free() -> void { device_->freeJob(this); }

auto QDMI_Session_impl_d::init() -> int {
  if (status_ != qdmi::SessionStatus::ALLOCATED) {
    return QDMI_ERROR_BADSTATE;
  }
  status_ = qdmi::SessionStatus::INITIALIZED;
  return QDMI_SUCCESS;
}

auto QDMI_Session_impl_d::setParameter(QDMI_Session_Parameter param,
                                       const size_t size,
                                       const void* value) const -> int {
  if ((value != nullptr && size == 0) || param >= QDMI_SESSION_PARAMETER_MAX) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  if (status_ != qdmi::SessionStatus::ALLOCATED) {
    return QDMI_ERROR_BADSTATE;
  }
  return QDMI_ERROR_NOTSUPPORTED;
}

auto QDMI_Session_impl_d::querySessionProperty(QDMI_Session_Property prop,
                                               size_t size, void* value,
                                               size_t* sizeRet) const -> int {
  if ((value != nullptr && size == 0) || prop >= QDMI_SESSION_PROPERTY_MAX) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  if (status_ != qdmi::SessionStatus::INITIALIZED) {
    return QDMI_ERROR_BADSTATE;
  }
  if (prop == QDMI_SESSION_PROPERTY_DEVICES) {
    if (value != nullptr) {
      if (size < devices_->size() * sizeof(QDMI_Device)) {
        return QDMI_ERROR_INVALIDARGUMENT;
      }
      memcpy(value, static_cast<const void*>(devices_->data()),
             devices_->size() * sizeof(QDMI_Device));
    }
    if (sizeRet != nullptr) {
      *sizeRet = devices_->size() * sizeof(QDMI_Device);
    }
    return QDMI_SUCCESS;
  }
  return QDMI_ERROR_NOTSUPPORTED;
}

namespace qdmi {
Driver::Driver() {
  devices_.emplace_back(std::make_unique<QDMI_Device_impl_d>(
      std::make_unique<MQT_NADeviceLibrary>()));
}

Driver::~Driver() {
  sessions_.clear();
  devices_.clear();
}

auto Driver::addDynamicDeviceLibrary(const std::string& libName,
                                     const std::string& prefix) -> void {
  devices_.emplace_back(std::make_unique<QDMI_Device_impl_d>(
      std::make_unique<DynamicDeviceLibrary>(libName, prefix)));
}

auto Driver::sessionAlloc(QDMI_Session* session) -> int {
  if (session == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  auto uniqueSession = std::make_unique<QDMI_Session_impl_d>(devices_);
  *session = sessions_.emplace(uniqueSession.get(), std::move(uniqueSession))
                 .first->first;
  return QDMI_SUCCESS;
}

auto Driver::sessionFree(QDMI_Session session) -> void {
  if (session != nullptr) {
    if (const auto& it = sessions_.find(session); it != sessions_.end()) {
      sessions_.erase(it);
    }
  }
}
} // namespace qdmi

int QDMI_session_alloc(QDMI_Session* session) {
  return qdmi::Driver::get().sessionAlloc(session);
}

int QDMI_session_init(QDMI_Session session) {
  if (session == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  return session->init();
}

void QDMI_session_free(QDMI_Session session) {
  qdmi::Driver::get().sessionFree(session);
}

int QDMI_session_set_parameter(QDMI_Session session,
                               QDMI_Session_Parameter param, const size_t size,
                               const void* value) {
  if (session == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  return session->setParameter(param, size, value);
}

int QDMI_session_query_session_property(QDMI_Session session,
                                        QDMI_Session_Property prop, size_t size,
                                        void* value, size_t* sizeRet) {
  if (session == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  return session->querySessionProperty(prop, size, value, sizeRet);
}

int QDMI_device_create_job(QDMI_Device dev, QDMI_Job* job) {
  if (dev == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  return dev->createJob(job);
}

void QDMI_job_free(QDMI_Job job) {
  if (job != nullptr) {
    job->free();
  }
}

int QDMI_job_set_parameter(QDMI_Job job, QDMI_Job_Parameter param,
                           const size_t size, const void* value) {
  if (job == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  return job->setParameter(param, size, value);
}

int QDMI_job_query_property(QDMI_Job job, QDMI_Job_Property prop,
                            const size_t size, void* value, size_t* sizeRet) {
  if (job == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  return job->queryProperty(prop, size, value, sizeRet);
}

int QDMI_job_submit(QDMI_Job job) {
  if (job == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  return job->submit();
}

int QDMI_job_cancel(QDMI_Job job) {
  if (job == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  return job->cancel();
}

int QDMI_job_check(QDMI_Job job, QDMI_Job_Status* status) {
  if (job == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  return job->check(status);
}

int QDMI_job_wait(QDMI_Job job, size_t timeout) {
  if (job == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  return job->wait(timeout);
}

int QDMI_job_get_results(QDMI_Job job, QDMI_Job_Result result,
                         const size_t size, void* data, size_t* sizeRet) {
  if (job == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  return job->getResults(result, size, data, sizeRet);
}

int QDMI_device_query_device_property(QDMI_Device device,
                                      QDMI_Device_Property prop,
                                      const size_t size, void* value,
                                      size_t* sizeRet) {
  if (device == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  return device->queryDeviceProperty(prop, size, value, sizeRet);
}

int QDMI_device_query_site_property(QDMI_Device device, QDMI_Site site,
                                    QDMI_Site_Property prop, const size_t size,
                                    void* value, size_t* sizeRet) {
  if (device == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  return device->querySiteProperty(site, prop, size, value, sizeRet);
}

int QDMI_device_query_operation_property(
    QDMI_Device device, QDMI_Operation operation, const size_t numSites,
    const QDMI_Site* sites, const size_t numParams, const double* params,
    QDMI_Operation_Property prop, const size_t size, void* value,
    size_t* sizeRet) {
  if (device == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  return device->queryOperationProperty(operation, numSites, sites, numParams,
                                        params, prop, size, value, sizeRet);
}
