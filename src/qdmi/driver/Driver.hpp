/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

/// @file Driver.hpp
/// QDMI driver implementation interfaces.

#pragma once

#include "qdmi/common/Common.hpp"

#include "qdmi/client.h"
#include "qdmi/device.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/ilist.h"
#include "llvm/ADT/ilist_node.h"
#include "llvm/Support/Error.h"

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace qdmi {

class Session;

/// Inline JSON used to configure one QDMI device session.
struct InlineDeviceConfiguration {
  std::string json;
};

/// JSON file used to configure one QDMI device session.
struct FileDeviceConfiguration {
  std::filesystem::path path;
};

/// One replaceable source for a runtime device description.
using DeviceConfigurationSource =
    std::variant<InlineDeviceConfiguration, FileDeviceConfiguration>;

/// Configuration for device session parameters.
///
/// This struct holds optional parameters that can be set on a device
/// session before initialization. All parameters are optional.
struct DeviceSessionConfig {
  /// Base URL for API endpoint
  std::optional<std::string> baseUrl;
  /// Authentication token
  std::optional<std::string> token;
  /// Path to file containing authentication information
  std::optional<std::filesystem::path> authFile;
  /// URL to authentication server
  std::optional<std::string> authUrl;
  /// Username for authentication
  std::optional<std::string> username;
  /// Password for authentication
  std::optional<std::string> password;
  /// Typed runtime device-description source.
  std::optional<DeviceConfigurationSource> deviceConfiguration;
  /// Custom configuration parameter 1
  std::optional<CustomSessionParameter> custom1;
  /// Custom configuration parameter 2
  std::optional<CustomSessionParameter> custom2;
  /// Custom configuration parameter 3
  std::optional<CustomSessionParameter> custom3;
  /// Custom configuration parameter 4
  std::optional<CustomSessionParameter> custom4;
  /// Custom configuration parameter 5
  std::optional<CustomSessionParameter> custom5;
};

/// Stable registration for a QDMI device.
///
/// Registration records this metadata without loading the native
/// library. Use @ref Driver::open to create the corresponding device session.
struct DeviceDefinition {
  /// Stable identifier used to open this device.
  std::string id;
  /// Path to the native QDMI device library.
  std::filesystem::path library;
  /// Prefix used for the QDMI device interface functions.
  std::string prefix;
  /// Parameters applied before the device session is initialized.
  DeviceSessionConfig session;
};

/// Function table for one QDMI device implementation.
struct DeviceAPI {
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
  /// Function pointer to @ref QDMI_device_session_retrieve_device_job_by_id.
  decltype(QDMI_device_session_retrieve_device_job_by_id)*
      device_session_retrieve_device_job_by_id{};
  /// Function pointer to @ref QDMI_device_job_free.
  decltype(QDMI_device_job_free)* device_job_free{};
  /// Function pointer to @ref QDMI_device_job_set_parameter.
  decltype(QDMI_device_job_set_parameter)* device_job_set_parameter{};
  decltype(QDMI_device_job_set_programs)* device_job_set_programs{};
  decltype(QDMI_device_job_get_program)* device_job_get_program{};
  decltype(QDMI_device_job_get_program_status)* device_job_get_program_status{};
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

  // Default constructor
  DeviceAPI() = default;
  // delete copy constructor and copy assignment operator
  DeviceAPI(const DeviceAPI&) = delete;
  DeviceAPI& operator=(const DeviceAPI&) = delete;
  // define move constructor and move assignment operator
  DeviceAPI(DeviceAPI&&) = default;
  DeviceAPI& operator=(DeviceAPI&&) = default;
  // destructor should be virtual to allow for polymorphic deletion
  virtual ~DeviceAPI() = default;
};

/// Owns a dynamic device library and its QDMI function table.
class LoadedDeviceAPI final : public DeviceAPI {
  void* libHandle_;
  bool initialized_ = false;
  explicit LoadedDeviceAPI(void* handle) : libHandle_(handle) {}
  llvm::Error initialize(const std::string& prefix);
  friend llvm::Expected<LoadedDeviceAPI*>
  loadDeviceAPI(const std::string& libName, const std::string& prefix);

public:
  /// Finalize a successfully initialized device and close the library.
  ~LoadedDeviceAPI() override;
};

/// The status of a session.
///
/// This enum defines the possible states of a session in the QDMI
/// library. A session can be either allocated or initialized.
enum class SessionStatus : uint8_t {
  ALLOCATED,   ///< The session has been allocated but not initialized
  INITIALIZED, ///< The session has been initialized and is ready for use
};
} // namespace qdmi

/// Definition of the QDMI Device.
struct QDMI_Device_impl_d {
private:
  /// Stable ID assigned by the driver.
  std::string id_;

  /// Function table owned by the process-lifetime library cache or caller.
  const qdmi::DeviceAPI* api_;
  /// The device session handle.
  QDMI_Device_Session deviceSession_ = nullptr;
  /// Client-facing wrappers for direct child devices.
  std::vector<std::unique_ptr<QDMI_Device_impl_d>> childDevices_;
  /// Guards the owned job wrappers.
  std::mutex jobsMutex_;
  /// Intrusive ownership avoids allocating a second node for each job.
  llvm::iplist<QDMI_Job_impl_d> jobs_;

  int adoptJob(int status, QDMI_Device_Job deviceJob,
               std::unique_ptr<QDMI_Job_impl_d> wrapper, QDMI_Job* job);

  explicit QDMI_Device_impl_d(const qdmi::DeviceAPI& library)
      : api_(&library) {}
  llvm::Error initialize(const qdmi::DeviceSessionConfig& config,
                         QDMI_Child_Device childDevice, bool strict);

public:
  /// Open a device session and its children.
  ///
  /// Failure releases partial sessions.
  [[nodiscard]] static llvm::Expected<std::unique_ptr<QDMI_Device_impl_d>>
  create(const qdmi::DeviceAPI& library,
         const qdmi::DeviceSessionConfig& config = {},
         QDMI_Child_Device childDevice = nullptr, std::string id = {},
         bool strict = false);

  /// Destructor for the QDMI device.
  ///
  /// This destructor frees the device session and frees its jobs.
  ~QDMI_Device_impl_d();

  /// Returns The device interface function table.
  [[nodiscard]] const qdmi::DeviceAPI& api() const { return *api_; }

  /// Creates a job for the device.
  ///
  /// @see QDMI_device_create_job
  int createJob(QDMI_Job* job);

  /// Retrieves an existing job by its ID.
  ///
  /// @see QDMI_session_retrieve_job_by_id
  int retrieveJobById(const char* jobId, QDMI_Job* job);

  /// Frees the job associated with the device.
  ///
  /// @see QDMI_job_free
  void freeJob(QDMI_Job job);

  /// Queries a device property.
  ///
  /// @see QDMI_device_query_device_property
  int queryDeviceProperty(QDMI_Device_Property prop, size_t size, void* value,
                          size_t* sizeRet) const;

  /// Queries a site property.
  ///
  /// @see QDMI_device_query_site_property
  int querySiteProperty(QDMI_Site site, QDMI_Site_Property prop, size_t size,
                        void* value, size_t* sizeRet) const;

  /// Queries an operation property.
  ///
  /// @see QDMI_device_query_operation_property
  int queryOperationProperty(QDMI_Operation operation, size_t numSites,
                             const QDMI_Site* sites, size_t numParams,
                             const double* params, QDMI_Operation_Property prop,
                             size_t size, void* value, size_t* sizeRet) const;
};

/// Definition of the QDMI Job.
struct QDMI_Job_impl_d : llvm::ilist_node<QDMI_Job_impl_d> {
  friend struct QDMI_Device_impl_d;

private:
  /// The device job handle.
  QDMI_Device_Job deviceJob_ = nullptr;
  /// The device associated with the job.
  QDMI_Device device_ = nullptr;

public:
  /// Constructor for the QDMI job.
  ///
  /// This constructor initializes the job with the device job handle
  /// and the device library.
  /// @param deviceJob is the handle to the device job.
  /// @param device is the device associated with the job.
  explicit QDMI_Job_impl_d(QDMI_Device_Job deviceJob,
                           QDMI_Device device) noexcept
      : deviceJob_(deviceJob), device_(device) {}

  /// Destructor for the QDMI job.
  ///
  /// This destructor frees the device job handle using the
  /// @ref QDMI_device_job_free function from the device library.
  ~QDMI_Job_impl_d();

  /// Sets a parameter for the job.
  ///
  /// @see QDMI_job_set_parameter
  int setParameter(QDMI_Job_Parameter param, size_t size,
                   const void* value) const;

  int setPrograms(QDMI_Program_Format format, size_t count, const size_t* sizes,
                  const void* const* programs) const;

  /// Retrieves an indexed program payload.
  ///
  /// @see QDMI_job_get_program
  int getProgram(size_t programIndex, size_t size, void* data,
                 size_t* sizeRet) const;

  /// Queries an indexed program status.
  ///
  /// @see QDMI_job_get_program_status
  int getProgramStatus(size_t programIndex, QDMI_Job_Status* status) const;

  /// Queries a property of the job.
  ///
  /// @see QDMI_job_query_property
  int queryProperty(QDMI_Job_Property prop, size_t size, void* value,
                    size_t* sizeRet) const;

  /// Submits the job to the device.
  ///
  /// @see QDMI_job_submit
  [[nodiscard]] int submit() const;

  /// Cancels the job.
  ///
  /// @see QDMI_job_cancel
  [[nodiscard]] int cancel() const;

  /// Checks the status of the job.
  ///
  /// @see QDMI_job_check
  int check(QDMI_Job_Status* status) const;

  /// Waits for the job to complete but at most for the specified timeout.
  ///
  /// @see QDMI_job_wait
  [[nodiscard]] int wait(size_t timeout) const;

  /// Gets the results of the job.
  ///
  /// @see QDMI_job_get_results
  int getResults(size_t programIndex, QDMI_Job_Result result, size_t size,
                 void* data, size_t* sizeRet) const;

  /// Frees the job.
  ///
  /// @note This function just forwards to the device's @ref
  /// QDMI_Device_impl_d::freeJob function. This function is needed because the
  /// interface only provides the job handle to the @ref QDMI_job_free function
  /// and the job's device handle is private.
  void free();
};

/// Definition of the QDMI Session.
struct QDMI_Session_impl_d {
private:
  /// The status of the session.
  qdmi::SessionStatus status_ = qdmi::SessionStatus::ALLOCATED;
  /// Snapshot of devices visible when this session was allocated.
  llvm::ArrayRef<QDMI_Device> devices_;

  /// A targeted private allocation creates one device outside the shared
  /// registry.
  ///
  /// Keep that device alive for the lifetime of this session; ordinary sessions
  /// instead borrow the fixed catalog of registry-owned devices.
  std::unique_ptr<QDMI_Device_impl_d> ownedDevice_;
  QDMI_Device privateDevice_ = nullptr;

public:
  /// Constructor borrowing the fixed device catalog.
  explicit QDMI_Session_impl_d(llvm::ArrayRef<QDMI_Device> devices) noexcept;

  /// Constructor for one privately targeted device session.
  explicit QDMI_Session_impl_d(
      std::unique_ptr<QDMI_Device_impl_d> device) noexcept;

  // The targeted device view points into this object.
  QDMI_Session_impl_d(const QDMI_Session_impl_d&) = delete;
  QDMI_Session_impl_d& operator=(const QDMI_Session_impl_d&) = delete;
  QDMI_Session_impl_d(QDMI_Session_impl_d&&) = delete;
  QDMI_Session_impl_d& operator=(QDMI_Session_impl_d&&) = delete;
  ~QDMI_Session_impl_d() = default;

  /// Initializes the session.
  ///
  /// @see QDMI_session_init
  int init();

  /// Sets a parameter for the session.
  ///
  /// @see QDMI_session_set_parameter
  int setParameter(QDMI_Session_Parameter param, size_t size,
                   const void* value) const;

  /// Queries a session property.
  ///
  /// @see QDMI_session_query_session_property
  int querySessionProperty(QDMI_Session_Property prop, size_t size, void* value,
                           size_t* sizeRet) const;
};

namespace qdmi {
/// The MQT QDMI driver class.
///
/// This driver discovers configured QDMI device definitions and opens
/// their libraries. Additional definitions can be registered at runtime.
/// @note This class is a singleton that manages the QDMI libraries and
/// sessions. It is responsible for loading the libraries, allocating sessions,
/// and providing access to the devices.
/// Fallible C++ methods return owned llvm::Error or llvm::Expected results.
/// The QDMI C API reports failures with status codes.
class Driver final : public Singleton<Driver> {
  friend class Singleton;
  friend class Session;

  /// Private constructor to enforce the singleton pattern.
  Driver() = default;

  /// Called with stateMutex_ held. A failed discovery can be retried.
  llvm::Error initialize();
  bool initialized_ = false;

  /// Guards all mutable driver state below.
  mutable std::mutex stateMutex_;

  /// Wakes callers waiting for a persistent device to finish opening.
  std::condition_variable stateChanged_;

  /// Ensures that the configured client catalog is materialized once.
  std::once_flag clientCatalogOnce_;

  struct RegistryEntry {
    DeviceDefinition definition;
    std::unique_ptr<QDMI_Device_impl_d> device;
    bool opening = false;
  };
  /// Entries stay at stable addresses while the registry grows.
  llvm::StringMap<RegistryEntry> definitions_;
  llvm::SmallVector<RegistryEntry*> registrationOrder_;
  llvm::StringSet<> disabledDeviceIds_;
  /// Keys owned by definitions_; runtime registration does not extend this
  /// list.
  llvm::SmallVector<llvm::StringRef> clientDefinitionIds_;
  /// Published once; sessions borrow this immutable catalog.
  llvm::SmallVector<QDMI_Device> clientDevices_;

  /// Opens the initially configured definitions for the client API.
  void materializeClientCatalog();

  /// Opens a fresh device session with per-call overrides.
  llvm::Expected<std::unique_ptr<QDMI_Device_impl_d>>
  openFresh(std::string_view id, const DeviceSessionConfig& overrides,
            bool strict = false);

public:
  /// Returns the instance owned by this driver library.
  ///
  /// This out-of-line accessor keeps static-library consumers from
  /// instantiating separate singleton storage in different translation units.
  [[nodiscard]] static Driver& get();

  /// Registers a device definition without loading its library.
  ///
  /// @param definition The definition to validate and store.
  /// @param replace Whether an existing unopened definition may be replaced.
  /// Reports QDMI_ERROR_INVALIDARGUMENT if the definition is incomplete or its
  /// ID is already registered. Replacing an opened definition also fails.
  [[nodiscard]] llvm::Error registerDevice(DeviceDefinition definition,
                                           bool replace = false);

  /// Registers a device definition unless its ID is already present.
  ///
  /// @param definition The definition to validate and store.
  /// @returns Whether the definition was inserted.
  /// Reports QDMI_ERROR_INVALIDARGUMENT if the definition is incomplete.
  ///
  /// Existing and explicitly disabled IDs are not inserted. The
  /// complete definition is validated before checking for either condition.
  llvm::Expected<bool> registerDeviceIfAbsent(DeviceDefinition definition);

  /// Lists the stable IDs of all registered devices.
  ///
  /// @returns The enabled device IDs in deterministic registration order.
  ///
  /// This query includes runtime registrations and does not load device
  /// libraries or expose their definitions.
  [[nodiscard]] llvm::Expected<std::vector<std::string>> registeredDeviceIds();

  /// Opens the registered device with the given stable ID.
  ///
  /// @returns The existing device handle when the ID is already open.
  /// Reports QDMI_ERROR_NOTFOUND if the ID is unknown.
  /// Loading or session initialization can also fail.
  llvm::Expected<QDMI_Device> open(std::string_view id);

  /// Allocates a new session.
  ///
  /// @see QDMI_session_alloc
  int sessionAlloc(QDMI_Session* session);

  /// Allocates a strict one-device session for the private Core extension.
  int sessionAllocForDevice(std::string_view id,
                            const DeviceSessionConfig& config,
                            QDMI_Session* session);
};

} // namespace qdmi
