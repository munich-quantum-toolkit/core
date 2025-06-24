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
 * @brief The MQT QDMI driver implementation.
 * @details This driver loads all statically known and linked QDMI device
 * libraries. Those are introduced to the driver via the macros
 * `DEVICE_LIST_UPPERCASE` and `DEVICE_LIST_LOWERCASE`. Additional devices
 * can be added dynamically by providing the respective information to the
 * @ref qdmi::Driver::initialize function that initializes the driver.
 */

#pragma once

#include "mqt_na_qdmi/device.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <qdmi/client.h>
#include <qdmi/device.h>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace qdmi {
/**
 * Definition of the QDMI Library.
 */
struct DeviceLibrary {
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
  /// Function pointer to @ref QDMI_device_job_query_property.
  decltype(QDMI_device_job_query_property)* device_job_query_property{};
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

  DeviceLibrary() = default;
  // delete copy constructor and copy assignment operator
  DeviceLibrary(const DeviceLibrary&) = delete;
  DeviceLibrary& operator=(const DeviceLibrary&) = delete;
  // define move constructor and move assignment operator
  DeviceLibrary(DeviceLibrary&&) = default;
  DeviceLibrary& operator=(DeviceLibrary&&) = default;
  virtual ~DeviceLibrary() = default;
};
class DynamicDeviceLibrary : public DeviceLibrary {
  /// Handle to the dynamic library.
  void* libHandle;

  static auto openLibHandles() -> std::unordered_set<void*>& {
    static std::unordered_set<void*> libHandles;
    return libHandles;
  }

public:
  DynamicDeviceLibrary(const std::string& libName, const std::string& prefix);

  ~DynamicDeviceLibrary() override;
};
#define LOAD_STATIC_SYMBOL(prefix, symbol)                                     \
  {                                                                            \
    (symbol) = reinterpret_cast<decltype(symbol)>(prefix##_QDMI_##symbol);     \
  }
#define ADD_STATIC_LIBRARY(prefix)                                             \
  class prefix##DeviceLibrary : public DeviceLibrary {                         \
  public:                                                                      \
    prefix##DeviceLibrary() {                                                  \
      /* NOLINTBEGIN(cppcoreguidelines-pro-type-reinterpret-cast) */           \
      /* load the function symbols from the dynamic library */                 \
      LOAD_STATIC_SYMBOL(prefix, device_initialize)                            \
      LOAD_STATIC_SYMBOL(prefix, device_finalize)                              \
      /* device session interface */                                           \
      LOAD_STATIC_SYMBOL(prefix, device_session_alloc)                         \
      LOAD_STATIC_SYMBOL(prefix, device_session_init)                          \
      LOAD_STATIC_SYMBOL(prefix, device_session_free)                          \
      LOAD_STATIC_SYMBOL(prefix, device_session_set_parameter)                 \
      /* device job interface */                                               \
      LOAD_STATIC_SYMBOL(prefix, device_session_create_device_job)             \
      LOAD_STATIC_SYMBOL(prefix, device_job_free)                              \
      LOAD_STATIC_SYMBOL(prefix, device_job_set_parameter)                     \
      LOAD_STATIC_SYMBOL(prefix, device_job_query_property)                    \
      LOAD_STATIC_SYMBOL(prefix, device_job_submit)                            \
      LOAD_STATIC_SYMBOL(prefix, device_job_cancel)                            \
      LOAD_STATIC_SYMBOL(prefix, device_job_check)                             \
      LOAD_STATIC_SYMBOL(prefix, device_job_wait)                              \
      LOAD_STATIC_SYMBOL(prefix, device_job_get_results)                       \
      /* device query interface */                                             \
      LOAD_STATIC_SYMBOL(prefix, device_session_query_device_property)         \
      LOAD_STATIC_SYMBOL(prefix, device_session_query_site_property)           \
      LOAD_STATIC_SYMBOL(prefix, device_session_query_operation_property)      \
      /* NOLINTEND(cppcoreguidelines-pro-type-reinterpret-cast) */             \
      /* initialize the device */                                              \
      device_initialize();                                                     \
    }                                                                          \
                                                                               \
    ~prefix##DeviceLibrary() override {                                        \
      /* Check if QDMI_device_finalize is not NULL before calling it. */       \
      if (device_finalize != nullptr) {                                        \
        device_finalize();                                                     \
      }                                                                        \
    }                                                                          \
  };
ADD_STATIC_LIBRARY(MQT_NA)

/**
 * @brief The status of a session.
 * @details This enum defines the possible states of a session in the QDMI
 * library. A session can be either allocated or initialized.
 */
enum class SessionStatus : uint8_t {
  ALLOCATED,  ///< The session has been allocated but not initialized
  INITIALIZED ///< The session has been initialized and is ready for use
};
} // namespace qdmi

/**
 * Definition of the QDMI Device.
 */
struct QDMI_Device_impl_d {
private:
  /// The device library that provides the device interface functions.
  std::unique_ptr<qdmi::DeviceLibrary> library;
  /// The device session handle.
  QDMI_Device_Session deviceSession = nullptr;
  /**
   * @brief Map of jobs to their corresponding unique pointers of
   * QDMI_Job_impl_d objects.
   */
  std::unordered_map<QDMI_Job, std::unique_ptr<QDMI_Job_impl_d>> jobs;

public:
  explicit QDMI_Device_impl_d(std::unique_ptr<qdmi::DeviceLibrary>&& lib);

  ~QDMI_Device_impl_d() {
    jobs.clear();
    if (deviceSession != nullptr) {
      library->device_session_free(deviceSession);
    }
  }

  /// @copydoc QDMI_device_create_job
  auto createJob(QDMI_Job* job) -> int;

  /// @copydoc QDMI_job_free
  auto jobFree(QDMI_Job job) -> void;

  /// @copydoc QDMI_device_query_device_property
  auto queryDeviceProperty(QDMI_Device_Property prop, size_t size, void* value,
                           size_t* sizeRet) const -> int;

  /// @copydoc QDMI_device_query_site_property
  auto querySiteProperty(QDMI_Site site, QDMI_Site_Property prop, size_t size,
                         void* value, size_t* sizeRet) const -> int;

  /// @copydoc QDMI_device_query_operation_property
  auto queryOperationProperty(QDMI_Operation operation, size_t numSites,
                              const QDMI_Site* sites, size_t numParams,
                              const double* params,
                              QDMI_Operation_Property prop, size_t size,
                              void* value, size_t* sizeRet) const -> int;
};

/**
 * Definition of the QDMI Job.
 */
struct QDMI_Job_impl_d {
private:
  QDMI_Device_Job deviceJob = nullptr; ///< The device job handle.
  QDMI_Device device = nullptr;        ///< The device associated with the job.
  const qdmi::DeviceLibrary* library = nullptr;

public:
  explicit QDMI_Job_impl_d(const qdmi::DeviceLibrary* library,
                           QDMI_Device_Job deviceJob)
      : deviceJob(deviceJob), library(library) {}

  ~QDMI_Job_impl_d();

  /// @brief Get the device associated with this job.
  [[nodiscard]] auto getDevice() const -> QDMI_Device;

  /// @copydoc QDMI_job_set_parameter
  auto setParameter(QDMI_Job_Parameter param, size_t size,
                    const void* value) const -> int;

  /// @copydoc QDMI_job_query_property
  auto queryProperty(QDMI_Job_Property prop, size_t size, void* value,
                     size_t* sizeRet) const -> int;

  /// @copydoc QDMI_job_submit
  [[nodiscard]] auto submit() const -> int;

  /// @copydoc QDMI_job_cancel
  [[nodiscard]] auto cancel() const -> int;

  /// @copydoc QDMI_job_check
  auto check(QDMI_Job_Status* status) const -> int;

  /// @copydoc QDMI_job_wait
  [[nodiscard]] auto wait(size_t timeout) const -> int;

  /// @copydoc QDMI_job_get_results
  auto getResults(QDMI_Job_Result result, size_t size, void* data,
                  size_t* sizeRet) const -> int;
};

/**
 * Definition of the QDMI Session.
 */
struct QDMI_Session_impl_d {
private:
  /// The status of the session.
  qdmi::SessionStatus status = qdmi::SessionStatus::ALLOCATED;
  /// A pointer to the list of all devices.
  const std::vector<std::unique_ptr<QDMI_Device_impl_d>>* devices;

public:
  /// @brief Constructor for the QDMI session.
  explicit QDMI_Session_impl_d(
      const std::vector<std::unique_ptr<QDMI_Device_impl_d>>& devices)
      : devices(&devices) {}

  /// @copydoc QDMI_session_init
  auto init() -> int;

  /// @copydoc QDMI_session_set_parameter
  auto setParameter(QDMI_Session_Parameter param, size_t size,
                    const void* value) const -> int;

  /// @copydoc QDMI_session_query_session_property
  auto querySessionProperty(QDMI_Session_Property prop, size_t size,
                            void* value, size_t* sizeRet) const -> int;
};

namespace qdmi {
/** @brief The QDMI driver class.
 * @details This class is a singleton that manages the QDMI libraries and
 * sessions. It is responsible for loading the libraries, allocating sessions,
 * and providing access to the devices.
 */
class Driver final {
  /// @brief Private constructor to enforce the singleton pattern.
  Driver();

  /**
   * @brief Map of sessions to their corresponding unique pointers of
   * QDMI_Session_impl_d objects.
   */
  std::unordered_map<QDMI_Session, std::unique_ptr<QDMI_Session_impl_d>>
      sessions;

  /**
   * @brief Map of devices to their corresponding unique pointers of
   * QDMI_Device_impl_d objects.
   */
  std::vector<std::unique_ptr<QDMI_Device_impl_d>> devices;

public:
  // Delete copy constructors and assignment operators to prevent copying the
  // singleton instance.
  Driver(const Driver&) = delete;
  Driver& operator=(const Driver&) = delete;

  /// @brief Get the singleton instance
  static auto get() -> Driver& {
    static Driver instance;
    return instance;
  }

  ~Driver();

  /**
   * Adds a dynamic device library to the list of dynamic device libraries.
   * @param libName is the path to the dynamic library to load.
   * @param prefix is the prefix used for the device interface functions.
   */
  auto addDynamicDeviceLibrary(const std::string& libName,
                               const std::string& prefix) -> void;

  /// @copydoc QDMI_session_alloc
  auto sessionAlloc(QDMI_Session* session) -> int;

  /// @copydoc QDMI_session_free
  auto sessionFree(QDMI_Session session) -> void;
};
} // namespace qdmi
