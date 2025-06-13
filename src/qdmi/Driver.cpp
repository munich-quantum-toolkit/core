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
 * @brief An example driver implementation in C++.
 */

#include "qdmi/Driver.hpp"

#include "qdmi/client.h"
#include "qdmi/device.h"
#include "spdlog/sinks/basic_file_sink-inl.h"

#include <cstdlib>
#include <dlfcn.h>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace {
enum class SessionStatus : uint8_t {
  ALLOCATED,  ///< The session has been allocated but not initialized
  INITIALIZED ///< The session has been initialized and is ready for use
};
} // namespace

/**
 * @brief Definition of the QDMI Library.
 */
struct DeviceLibrary {
  void* lib_handle = nullptr;

  // NOLINTBEGIN(readability-identifier-naming)
  /// Function pointer to @ref QDMI_device_initialize.
  decltype(QDMI_device_initialize)* device_initialize{};
  /// Function pointer to @ref QDMI_device_finalize.
  decltype(QDMI_device_finalize)* device_finalize{};
  /// Function pointer to @ref QDMI_device_session_alloc.
  decltype(QDMI_device_session_alloc)* device_session_alloc{};
  /// Function pointer to @ref QDMI_device_session_init.
  decltype(QDMI_device_session_init)* device_session_init{};
  /// Function pointer to @ref QDMI_device_session_free.
  decltype(QDMI_device_session_free)* device_session_free{};
  /// Function pointer to @ref QDMI_device_session_set_parameter.
  decltype(QDMI_device_session_set_parameter)* device_session_set_parameter{};
  /// Function pointer to @ref QDMI_device_session_create_device_job.
  decltype(QDMI_device_session_create_device_job)*
      device_session_create_device_job{};
  /// Function pointer to @ref QDMI_device_job_free.
  decltype(QDMI_device_job_free)* device_job_free{};
  /// Function pointer to @ref QDMI_device_job_set_parameter.
  decltype(QDMI_device_job_set_parameter)* device_job_set_parameter{};
  /// Function pointer to @ref QDMI_device_job_submit.
  decltype(QDMI_device_job_submit)* device_job_submit{};
  /// Function pointer to @ref QDMI_device_job_cancel.
  decltype(QDMI_device_job_cancel)* device_job_cancel{};
  /// Function pointer to @ref QDMI_device_job_check.
  decltype(QDMI_device_job_check)* device_job_check{};
  /// Function pointer to @ref QDMI_device_job_wait.
  decltype(QDMI_device_job_wait)* device_job_wait{};
  /// Function pointer to @ref QDMI_device_job_get_results.
  decltype(QDMI_device_job_get_results)* device_job_get_results{};
  /// Function pointer to @ref QDMI_device_session_query_device_property.
  decltype(QDMI_device_session_query_device_property)*
      device_session_query_device_property{};
  /// Function pointer to @ref QDMI_device_session_query_site_property.
  decltype(QDMI_device_session_query_site_property)*
      device_session_query_site_property{};
  /// Function pointer to @ref QDMI_device_session_query_operation_property.
  decltype(QDMI_device_session_query_operation_property)*
      device_session_query_operation_property{};
  // NOLINTEND(readability-identifier-naming)

  // default constructor
  DeviceLibrary() = default;

  // delete copy constructor, copy assignment, move constructor, move assignment
  // to allow only one instance and proper destruction of the dynamic library.
  DeviceLibrary(const DeviceLibrary&) = delete;
  DeviceLibrary& operator=(const DeviceLibrary&) = delete;
  DeviceLibrary(DeviceLibrary&&) = delete;
  DeviceLibrary& operator=(DeviceLibrary&&) = delete;

  // destructor
  ~DeviceLibrary() {
    // Check if QDMI_device_finalize is not NULL before calling it.
    if (device_finalize != nullptr) {
      device_finalize();
    }
    // close the dynamic library
    if (lib_handle != nullptr) {
      dlclose(lib_handle);
    }
  }
};

/**
 * @brief Definition of the QDMI Device.
 */
struct QDMI_Device_impl_d {
  const DeviceLibrary* library = nullptr;
  QDMI_Device_Session session = nullptr;

  // destructor
  ~QDMI_Device_impl_d() {
    if (library != nullptr && session != nullptr) {
      library->device_session_free(session);
    }
  }
};

/**
 * @brief Definition of the QDMI Session.
 */
struct QDMI_Session_impl_d {
  SessionStatus status = SessionStatus::ALLOCATED;
};

/**
 * @brief Definition of the QDMI Job.
 */
struct QDMI_Job_impl_d {
  QDMI_Device_Job deviceJob = nullptr;
  QDMI_Device device = nullptr;
};

namespace {

[[nodiscard]] auto deviceLibraries()
    -> std::unordered_map<void*, std::unique_ptr<DeviceLibrary>>& {
  static std::unordered_map<void*, std::unique_ptr<DeviceLibrary>> libraries;
  return libraries;
}

#define LOAD_SYMBOL(device, prefix, symbol)                                    \
  {                                                                            \
    const std::string symbol_name = std::string(prefix) + "_QDMI_" + #symbol;  \
    (device).symbol = reinterpret_cast<decltype((device).symbol)>(             \
        dlsym((device).lib_handle, symbol_name.c_str()));                      \
    if ((device).symbol == nullptr) {                                          \
      throw std::runtime_error("Failed to load symbol: " + symbol_name);       \
    }                                                                          \
  }

void addDeviceLibrary(const std::string& libName, const std::string& prefix) {
  auto* lib_handle = dlopen(libName.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (lib_handle == nullptr) {
    throw std::runtime_error("Couldn't open the device library: " + libName);
  }
  if (const auto it = deviceLibraries().find(lib_handle);
      it != deviceLibraries().end()) {
    // dlopen uses reference counting, so we need to decrement the reference
    // count that was increased by dlopen.
    dlclose(lib_handle);
    return;
  }
  auto it = deviceLibraries()
                .emplace(lib_handle, std::make_unique<DeviceLibrary>())
                .first;
  auto& library = *it->second;
  library.lib_handle = lib_handle;

  try {
    // NOLINTBEGIN(cppcoreguidelines-pro-type-reinterpret-cast)

    // load the function symbols from the dynamic library
    LOAD_SYMBOL(library, prefix, device_initialize)
    LOAD_SYMBOL(library, prefix, device_finalize)
    // device session interface
    LOAD_SYMBOL(library, prefix, device_session_alloc)
    LOAD_SYMBOL(library, prefix, device_session_init)
    LOAD_SYMBOL(library, prefix, device_session_free)
    LOAD_SYMBOL(library, prefix, device_session_set_parameter)
    // device job interface
    LOAD_SYMBOL(library, prefix, device_session_create_device_job)
    LOAD_SYMBOL(library, prefix, device_job_free)
    LOAD_SYMBOL(library, prefix, device_job_set_parameter)
    LOAD_SYMBOL(library, prefix, device_job_submit)
    LOAD_SYMBOL(library, prefix, device_job_cancel)
    LOAD_SYMBOL(library, prefix, device_job_check)
    LOAD_SYMBOL(library, prefix, device_job_wait)
    LOAD_SYMBOL(library, prefix, device_job_get_results)
    // device query interface
    LOAD_SYMBOL(library, prefix, device_session_query_device_property)
    LOAD_SYMBOL(library, prefix, device_session_query_site_property)
    LOAD_SYMBOL(library, prefix, device_session_query_operation_property)

    // NOLINTEND(cppcoreguidelines-pro-type-reinterpret-cast)
  } catch (const std::exception&) {
    dlclose(lib_handle);
    throw;
  }
  // initialize the device
  library.device_initialize();
}

[[nodiscard]] auto devices()
    -> std::vector<std::unique_ptr<QDMI_Device_impl_d>>& {
  static std::vector<std::unique_ptr<QDMI_Device_impl_d>> devices;
  return devices;
}

auto addDevice(const DeviceLibrary& library) -> void {
  auto& device =
      *devices().emplace_back(std::make_unique<QDMI_Device_impl_d>());
  device.library = &library;
  device.library->device_session_alloc(&device.session);
  device.library->device_session_init(device.session);
}

[[nodiscard]] auto sessions()
    -> std::vector<std::unique_ptr<QDMI_Session_impl_d>>& {
  static std::vector<std::unique_ptr<QDMI_Session_impl_d>> sessions;
  return sessions;
}

} // namespace

namespace na {
auto initialize(const std::vector<Library>& additionalLibraries) -> void {
  std::vector<Library> knownLibraries;
  std::stringstream libs(KNOWN_LIBRARIES);
  std::stringstream prefixes(KNOWN_LIBRARY_PREFIXES);
  std::string lib;
  std::string prefix;
  while (std::getline(libs, lib, ' ') && std::getline(prefixes, prefix, ' ')) {
    knownLibraries.emplace_back(Library{prefix, lib});
  }
  for (const auto& [prefix, path] : knownLibraries) {
    addDeviceLibrary(path, prefix);
  }
  for (const auto& [prefix, path] : additionalLibraries) {
    addDeviceLibrary(path, prefix);
  }
  for (const auto& deviceLib : deviceLibraries()) {
    addDevice(*deviceLib.second);
  }
}

auto finalize() -> void {}
} // namespace na

int QDMI_session_alloc(QDMI_Session* session) {
  if (session == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
  *session = new QDMI_Session_impl_d();
  return QDMI_SUCCESS;
}

int QDMI_session_init(QDMI_Session session) {
  if (session == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  if (session->status != SessionStatus::ALLOCATED) {
    return QDMI_ERROR_BADSTATE;
  }
  session->status = SessionStatus::INITIALIZED;
  return QDMI_SUCCESS;
}

void QDMI_session_free(QDMI_Session session) {
  // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
  delete session;
}

int QDMI_session_set_parameter(QDMI_Session session,
                               QDMI_Session_Parameter param, const size_t size,
                               const void* value) {
  if (session == nullptr || (value != nullptr && size == 0) ||
      param >= QDMI_SESSION_PARAMETER_MAX) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  if (session->status != SessionStatus::ALLOCATED) {
    return QDMI_ERROR_BADSTATE;
  }
  return QDMI_ERROR_NOTSUPPORTED;
}

int QDMI_session_query_session_property(QDMI_Session session,
                                        QDMI_Session_Property prop, size_t size,
                                        void* value, size_t* sizeRet) {
  if (session == nullptr || (value != nullptr && size == 0) ||
      prop >= QDMI_SESSION_PROPERTY_MAX) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  if (session->status != SessionStatus::INITIALIZED) {
    return QDMI_ERROR_BADSTATE;
  }
  if (prop == (QDMI_SESSION_PROPERTY_DEVICES)) {
    if (value != nullptr) {
      if (size < devices().size() * sizeof(QDMI_Device)) {
        return QDMI_ERROR_INVALIDARGUMENT;
      }
      memcpy(value, static_cast<const void*>(devices().data()),
             devices().size() * sizeof(QDMI_Device));
    }
    if (sizeRet != nullptr) {
      *sizeRet = devices().size() * sizeof(QDMI_Device);
    }
    return QDMI_SUCCESS;
  }
  return QDMI_ERROR_NOTSUPPORTED;
}

int QDMI_device_create_job(QDMI_Device dev, QDMI_Job* job) {
  if (dev == nullptr || job == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
  *job = new QDMI_Job_impl_d();
  (*job)->device = dev;
  return dev->library->device_session_create_device_job(dev->session,
                                                        &(*job)->deviceJob);
}

void QDMI_job_free(QDMI_Job job) {
  if (job != nullptr) {
    job->device->library->device_job_free(job->deviceJob);
    // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
    delete job;
  }
}

int QDMI_job_set_parameter(QDMI_Job job, QDMI_Job_Parameter param,
                           const size_t size, const void* value) {
  if (job == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  return job->device->library->device_job_set_parameter(
      job->deviceJob, static_cast<QDMI_Device_Job_Parameter>(param), size,
      value);
}

int QDMI_job_submit(QDMI_Job job) {
  if (job == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  return job->device->library->device_job_submit(job->deviceJob);
}

int QDMI_job_cancel(QDMI_Job job) {
  if (job == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  return job->device->library->device_job_cancel(job->deviceJob);
}

int QDMI_job_check(QDMI_Job job, QDMI_Job_Status* status) {
  if (job == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  return job->device->library->device_job_check(job->deviceJob, status);
}

int QDMI_job_wait(QDMI_Job job) {
  if (job == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  return job->device->library->device_job_wait(job->deviceJob);
}

int QDMI_job_get_results(QDMI_Job job, QDMI_Job_Result result,
                         const size_t size, void* data, size_t* sizeRet) {
  if (job == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  return job->device->library->device_job_get_results(job->deviceJob, result,
                                                      size, data, sizeRet);
}

int QDMI_device_query_device_property(QDMI_Device device,
                                      QDMI_Device_Property prop,
                                      const size_t size, void* value,
                                      size_t* sizeRet) {
  if (device == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  return device->library->device_session_query_device_property(
      device->session, prop, size, value, sizeRet);
}

int QDMI_device_query_site_property(QDMI_Device device, QDMI_Site site,
                                    QDMI_Site_Property prop, const size_t size,
                                    void* value, size_t* sizeRet) {
  if (device == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  return device->library->device_session_query_site_property(
      device->session, site, prop, size, value, sizeRet);
}

int QDMI_device_query_operation_property(
    QDMI_Device device, QDMI_Operation operation, const size_t numSites,
    const QDMI_Site* sites, const size_t numParams, const double* params,
    QDMI_Operation_Property prop, const size_t size, void* value,
    size_t* sizeRet) {
  if (device == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  return device->library->device_session_query_operation_property(
      device->session, operation, numSites, sites, numParams, params, prop,
      size, value, sizeRet);
}
