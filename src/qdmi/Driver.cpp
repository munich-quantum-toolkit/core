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
 * An example driver implementation in C++.
 */

#define DEVICE_LIST_UPPERCASE (MQT_NA)
#define DEVICE_LIST_LOWERCASE (mqt_na)

#include "qdmi/Driver.hpp"

#include "qdmi/Macros.hpp"
#include "qdmi/client.h"
#include "qdmi/device.h"

#include <cstddef>
#include <dlfcn.h>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

// Include the device headers for the devices in the list. The device headers
// are expected to be available under `<prefix>_qdmi/device.h`.
// NOLINTBEGIN(readability-duplicate-include)
#include INCLUDE(NTH_MAX(0, DEVICE_LIST_LOWERCASE))
#include INCLUDE(NTH_MAX(1, DEVICE_LIST_LOWERCASE))
#include INCLUDE(NTH_MAX(2, DEVICE_LIST_LOWERCASE))
#include INCLUDE(NTH_MAX(3, DEVICE_LIST_LOWERCASE))
#include INCLUDE(NTH_MAX(4, DEVICE_LIST_LOWERCASE))
#include INCLUDE(NTH_MAX(5, DEVICE_LIST_LOWERCASE))
#include INCLUDE(NTH_MAX(6, DEVICE_LIST_LOWERCASE))
#include INCLUDE(NTH_MAX(7, DEVICE_LIST_LOWERCASE))
#include INCLUDE(NTH_MAX(8, DEVICE_LIST_LOWERCASE))
#include INCLUDE(NTH_MAX(9, DEVICE_LIST_LOWERCASE))
// NOLINTEND(readability-duplicate-include)

namespace qc {
// Anonymous namespace to enforce internal linkage
namespace {
/**
 * @brief The status of a session.
 * @details This enum defines the possible states of a session in the QDMI
 * library. A session can be either allocated or initialized.
 */
enum class SessionStatus : uint8_t {
  ALLOCATED,  ///< The session has been allocated but not initialized
  INITIALIZED ///< The session has been initialized and is ready for use
};

/**
 * Definition of the QDMI Library.
 */
struct DeviceLibrary {
  void* libHandle = nullptr; ///< Handle to the dynamic library

  // we keep the naming scheme of QDMI, i.e., snail_case for function names,
  // here to ease the `LOAD_SYMBOL` macro later on.
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

  /// Construct a new DeviceLibrary instance.
  DeviceLibrary() = default;

  // delete copy constructor, copy assignment, move constructor, move assignment
  // to allow only one instance and proper destruction of the dynamic library.
  DeviceLibrary(const DeviceLibrary&) = delete;
  DeviceLibrary& operator=(const DeviceLibrary&) = delete;
  DeviceLibrary(DeviceLibrary&&) = delete;
  DeviceLibrary& operator=(DeviceLibrary&&) = delete;

  ~DeviceLibrary() {
    // Check if QDMI_device_finalize is not NULL before calling it.
    if (device_finalize != nullptr) {
      device_finalize();
    }
    // close the dynamic library
    if (libHandle != nullptr) {
      dlclose(libHandle);
    }
  }
};
} // namespace
} // namespace qc

/**
 * Definition of the QDMI Device.
 */
struct QDMI_Device_impl_d {
  /// The device library that provides the device interface functions.
  const qc::DeviceLibrary* library = nullptr;
  /// The device session handle.
  QDMI_Device_Session session = nullptr;

  ~QDMI_Device_impl_d() {
    if (library != nullptr && session != nullptr) {
      library->device_session_free(session);
    }
  }
};

/**
 * Definition of the QDMI Session.
 */
struct QDMI_Session_impl_d {
  /// The status of the session.
  qc::SessionStatus status = qc::SessionStatus::ALLOCATED;
};

/**
 * Definition of the QDMI Job.
 */
struct QDMI_Job_impl_d {
  QDMI_Device_Job deviceJob = nullptr; ///< The device job handle.
  QDMI_Device device = nullptr; ///< The device handle associated with the job.
};

namespace qc {
// Anonymous namespace to enforce internal linkage
namespace {
/**
 * Returns a reference to the array of static device libraries.
 * @returns a reference to an array of unique pointers to DeviceLibrary objects.
 */
[[nodiscard]] auto staticDeviceLibraries()
    -> std::array<std::unique_ptr<DeviceLibrary>,
                  SIZE(DEVICE_LIST_UPPERCASE)>& {
  static std::array<std::unique_ptr<DeviceLibrary>, SIZE(DEVICE_LIST_UPPERCASE)>
      libraries;
  return libraries;
}

// Map the device interface functions to functions with generic (un-prefixed)
// argument types. Those functions are later used in the DeviceLibrary to ease
// the implementation of the driver. For distinction between the device
// interface functions and those defined here, we drop `QDMI_` in the middle of
// the function name and use `prefix` as a prefix for the function name.
#define ADD_DEVICE(prefix)                                                     \
  int prefix##_device_initialize(void) {                                       \
    return prefix##_QDMI_device_initialize();                                  \
  }                                                                            \
  int prefix##_device_finalize(void) {                                         \
    return prefix##_QDMI_device_finalize();                                    \
  }                                                                            \
  int prefix##_device_session_alloc(QDMI_Device_Session* session) {            \
    return prefix##_QDMI_device_session_alloc(                                 \
        PREFIX_PTR_CAST(prefix, Device_Session, session));                     \
  }                                                                            \
  int prefix##_device_session_set_parameter(                                   \
      QDMI_Device_Session session, QDMI_Device_Session_Parameter param,        \
      size_t size, const void* value) {                                        \
    return prefix##_QDMI_device_session_set_parameter(                         \
        PREFIX_CAST(prefix, Device_Session, session), param, size, value);     \
  }                                                                            \
  int prefix##_device_session_init(QDMI_Device_Session session) {              \
    return prefix##_QDMI_device_session_init(                                  \
        PREFIX_CAST(prefix, Device_Session, session));                         \
  }                                                                            \
  void prefix##_device_session_free(QDMI_Device_Session session) {             \
    return prefix##_QDMI_device_session_free(                                  \
        PREFIX_CAST(prefix, Device_Session, session));                         \
  }                                                                            \
  int prefix##_device_session_query_device_property(                           \
      QDMI_Device_Session session, QDMI_Device_Property prop, size_t size,     \
      void* value, size_t* size_ret) {                                         \
    return prefix##_QDMI_device_session_query_device_property(                 \
        PREFIX_CAST(prefix, Device_Session, session), prop, size, value,       \
        size_ret);                                                             \
  }                                                                            \
  int prefix##_device_session_query_site_property(                             \
      QDMI_Device_Session session, QDMI_Site site, QDMI_Site_Property prop,    \
      size_t size, void* value, size_t* size_ret) {                            \
    return prefix##_QDMI_device_session_query_site_property(                   \
        PREFIX_CAST(prefix, Device_Session, session),                          \
        PREFIX_CAST(prefix, Site, site), prop, size, value, size_ret);         \
  }                                                                            \
  int prefix##_device_session_query_operation_property(                        \
      QDMI_Device_Session session, QDMI_Operation operation, size_t num_sites, \
      const QDMI_Site* sites, size_t num_params, const double* params,         \
      QDMI_Operation_Property prop, size_t size, void* value,                  \
      size_t* size_ret) {                                                      \
    return prefix##_QDMI_device_session_query_operation_property(              \
        PREFIX_CAST(prefix, Device_Session, session),                          \
        PREFIX_CAST(prefix, Operation, operation), num_sites,                  \
        PREFIX_CONST_PTR_CAST(prefix, Site, sites), num_params, params, prop,  \
        size, value, size_ret);                                                \
  }                                                                            \
  int prefix##_device_session_create_device_job(QDMI_Device_Session session,   \
                                                QDMI_Device_Job* job) {        \
    return prefix##_QDMI_device_session_create_device_job(                     \
        PREFIX_CAST(prefix, Device_Session, session),                          \
        PREFIX_PTR_CAST(prefix, Device_Job, job));                             \
  }                                                                            \
  int prefix##_device_job_set_parameter(QDMI_Device_Job job,                   \
                                        QDMI_Device_Job_Parameter param,       \
                                        size_t size, const void* value) {      \
    return prefix##_QDMI_device_job_set_parameter(                             \
        PREFIX_CAST(prefix, Device_Job, job), param, size, value);             \
  }                                                                            \
  int prefix##_device_job_submit(QDMI_Device_Job job) {                        \
    return prefix##_QDMI_device_job_submit(                                    \
        PREFIX_CAST(prefix, Device_Job, job));                                 \
  }                                                                            \
  int prefix##_device_job_cancel(QDMI_Device_Job job) {                        \
    return prefix##_QDMI_device_job_cancel(                                    \
        PREFIX_CAST(prefix, Device_Job, job));                                 \
  }                                                                            \
  int prefix##_device_job_check(QDMI_Device_Job job,                           \
                                QDMI_Job_Status* status) {                     \
    return prefix##_QDMI_device_job_check(                                     \
        PREFIX_CAST(prefix, Device_Job, job), status);                         \
  }                                                                            \
  int prefix##_device_job_wait(QDMI_Device_Job job) {                          \
    return prefix##_QDMI_device_job_wait(                                      \
        PREFIX_CAST(prefix, Device_Job, job));                                 \
  }                                                                            \
  int prefix##_device_job_get_results(QDMI_Device_Job job,                     \
                                      QDMI_Job_Result result, size_t size,     \
                                      void* data, size_t* size_ret) {          \
    return prefix##_QDMI_device_job_get_results(                               \
        PREFIX_CAST(prefix, Device_Job, job), result, size, data, size_ret);   \
  }                                                                            \
  void prefix##_device_job_free(QDMI_Device_Job job) {                         \
    return prefix##_QDMI_device_job_free(                                      \
        PREFIX_CAST(prefix, Device_Job, job));                                 \
  }

// Call `ADD_DEVICE` for each device in the list of devices. See `ADD_DEVICE`
// for more information.
ITERATE(ADD_DEVICE, DEVICE_LIST_UPPERCASE)

// Add a static device library to the list of static device libraries. The `id`
// is the index in the static device libraries array, and `prefix` is the prefix
// used for the device interface functions.
#define ADD_STATIC_DEVICE_LIBRARY(id, prefix)                                  \
  auto& library =                                                              \
      *(staticDeviceLibraries()[id] = std::make_unique<DeviceLibrary>());      \
  /* static library has no handle */                                           \
  library.libHandle = nullptr;                                                 \
  /* load the function symbols from the dynamic library */                     \
  library.device_initialize = prefix##_##device_initialize;                    \
  library.device_finalize = prefix##_##device_finalize;                        \
  /* device session interface */                                               \
  library.device_session_alloc = prefix##_##device_session_alloc;              \
  library.device_session_init = prefix##_##device_session_init;                \
  library.device_session_free = prefix##_##device_session_free;                \
  library.device_session_set_parameter =                                       \
      prefix##_##device_session_set_parameter;                                 \
  /* device job interface */                                                   \
  library.device_session_create_device_job =                                   \
      prefix##_##device_session_create_device_job;                             \
  library.device_job_free = prefix##_##device_job_free;                        \
  library.device_job_set_parameter = prefix##_##device_job_set_parameter;      \
  library.device_job_submit = prefix##_##device_job_submit;                    \
  library.device_job_cancel = prefix##_##device_job_cancel;                    \
  library.device_job_check = prefix##_##device_job_check;                      \
  library.device_job_wait = prefix##_##device_job_wait;                        \
  library.device_job_get_results = prefix##_##device_job_get_results;          \
  /* device query interface */                                                 \
  library.device_session_query_device_property =                               \
      prefix##_##device_session_query_device_property;                         \
  library.device_session_query_site_property =                                 \
      prefix##_##device_session_query_site_property;                           \
  library.device_session_query_operation_property =                            \
      prefix##_##device_session_query_operation_property;                      \
  /* initialize the device */                                                  \
  library.device_initialize();

/**
 * Returns a reference to the map of library handles returned by `dlopen`
 * to the dynamic device libraries.
 * @returns a reference to a map library handle to unique pointer of
 * DeviceLibrary objects.
 */
[[nodiscard]] auto dynamicDeviceLibraries()
    -> std::unordered_map<void*, std::unique_ptr<DeviceLibrary>>& {
  static std::unordered_map<void*, std::unique_ptr<DeviceLibrary>> libraries;
  return libraries;
}

// Load a symbol from the dynamic library. The `library` is the DeviceLibrary
// object, `prefix` is the prefix used for the device interface functions, and
// `symbol` is the name of the symbol to load.
#define LOAD_SYMBOL(library, prefix, symbol)                                   \
  {                                                                            \
    const std::string symbolName = std::string(prefix) + "_QDMI_" + #symbol;   \
    (library).symbol = reinterpret_cast<decltype((library).symbol)>(           \
        dlsym((library).libHandle, symbolName.c_str()));                       \
    if ((library).symbol == nullptr) {                                         \
      throw std::runtime_error("Failed to load symbol: " + symbolName);        \
    }                                                                          \
  }

/**
 * Adds a dynamic device library to the list of dynamic device libraries.
 * @param libName is the path to the dynamic library to load.
 * @param prefix is the prefix used for the device interface functions.
 */
void addDynamicDeviceLibrary(const std::string& libName,
                             const std::string& prefix) {
  auto* libHandle = dlopen(libName.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (libHandle == nullptr) {
    throw std::runtime_error("Couldn't open the device library: " + libName);
  }
  if (const auto it = dynamicDeviceLibraries().find(libHandle);
      it != dynamicDeviceLibraries().end()) {
    // dlopen uses reference counting, so we need to decrement the reference
    // count that was increased by dlopen.
    dlclose(libHandle);
    return;
  }
  auto it = dynamicDeviceLibraries()
                .emplace(libHandle, std::make_unique<DeviceLibrary>())
                .first;
  auto& library = *it->second;
  library.libHandle = libHandle;

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
    dlclose(libHandle);
    throw;
  }
  // initialize the device
  library.device_initialize();
}

/**
 * Returns a reference to the vector of devices.
 * @returns a reference to a vector of unique pointers to QDMI_Device_impl_d
 * objects.
 */
[[nodiscard]] auto devices()
    -> std::vector<std::unique_ptr<QDMI_Device_impl_d>>& {
  static std::vector<std::unique_ptr<QDMI_Device_impl_d>> devices;
  return devices;
}

/**
 * Adds a device to the list of devices. The device is initialized with the
 * given library.
 * @param library is the DeviceLibrary object that provides the device interface
 * functions.
 */
auto addDevice(const DeviceLibrary& library) -> void {
  auto& device =
      *devices().emplace_back(std::make_unique<QDMI_Device_impl_d>());
  device.library = &library;
  device.library->device_session_alloc(&device.session);
  device.library->device_session_init(device.session);
}

/**
 * Returns a reference to the map of sessions. The map is used to store the
 * sessions allocated by the QDMI library.
 * @returns a reference to a map of QDMI_Session to their corresponding unique
 * pointers of QDMI_Session_impl_d objects.
 */
[[nodiscard]] auto sessions()
    -> std::unordered_map<QDMI_Session, std::unique_ptr<QDMI_Session_impl_d>>& {
  static std::unordered_map<QDMI_Session, std::unique_ptr<QDMI_Session_impl_d>>
      sessions;
  return sessions;
}
} // namespace

auto initialize(const std::vector<Library>& additionalLibraries) -> void {
  // Initialize known static device libraries
  ITERATE_I(ADD_STATIC_DEVICE_LIBRARY, DEVICE_LIST_UPPERCASE)
  // Load additional dynamic device libraries
  for (const auto& [prefix, path] : additionalLibraries) {
    addDynamicDeviceLibrary(path, prefix);
  }
  // Add all static and dynamic device libraries to the device list
  for (const auto& lib : staticDeviceLibraries()) {
    addDevice(*lib);
  }
  for (const auto& [handle, lib] : dynamicDeviceLibraries()) {
    addDevice(*lib);
  }
}

auto finalize() -> void {
  // Free all existing sessions
  while (!sessions().empty()) {
    QDMI_session_free(sessions().begin()->first);
  }
  // By clearing the devices, we ensure that all device sessions are freed and
  // the dynamic libraries are closed, see the destructor of `DeviceLibrary`
  // and `QDMI_Device_impl_d` for details.
  devices().clear();
  dynamicDeviceLibraries().clear();
  // Clear the static device libraries in the same way just that there is no
  // `clear` method for the array.
  for (auto& lib : staticDeviceLibraries()) {
    lib.reset();
  }
}
} // namespace qc

int QDMI_session_alloc(QDMI_Session* session) {
  if (session == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  auto uniqueSession = std::make_unique<QDMI_Session_impl_d>();
  *session = qc::sessions()
                 .emplace(uniqueSession.get(), std::move(uniqueSession))
                 .first->first;
  return QDMI_SUCCESS;
}

int QDMI_session_init(QDMI_Session session) {
  if (session == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  if (session->status != qc::SessionStatus::ALLOCATED) {
    return QDMI_ERROR_BADSTATE;
  }
  session->status = qc::SessionStatus::INITIALIZED;
  return QDMI_SUCCESS;
}

void QDMI_session_free(QDMI_Session session) { qc::sessions().erase(session); }

int QDMI_session_set_parameter(QDMI_Session session,
                               QDMI_Session_Parameter param, const size_t size,
                               const void* value) {
  if (session == nullptr || (value != nullptr && size == 0) ||
      param >= QDMI_SESSION_PARAMETER_MAX) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  if (session->status != qc::SessionStatus::ALLOCATED) {
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
  if (session->status != qc::SessionStatus::INITIALIZED) {
    return QDMI_ERROR_BADSTATE;
  }
  if (prop == (QDMI_SESSION_PROPERTY_DEVICES)) {
    if (value != nullptr) {
      if (size < qc::devices().size() * sizeof(QDMI_Device)) {
        return QDMI_ERROR_INVALIDARGUMENT;
      }
      memcpy(value, static_cast<const void*>(qc::devices().data()),
             qc::devices().size() * sizeof(QDMI_Device));
    }
    if (sizeRet != nullptr) {
      *sizeRet = qc::devices().size() * sizeof(QDMI_Device);
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
