/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

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
 * @brief Definition of the device library.
 * @details The device library contains function pointers to the QDMI
 * device interface functions.
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

  // Default constructor
  DeviceLibrary() = default;
  // delete copy constructor and copy assignment operator
  DeviceLibrary(const DeviceLibrary&) = delete;
  DeviceLibrary& operator=(const DeviceLibrary&) = delete;
  // define move constructor and move assignment operator
  DeviceLibrary(DeviceLibrary&&) = default;
  DeviceLibrary& operator=(DeviceLibrary&&) = default;
  // destructor should be virtual to allow for polymorphic deletion
  virtual ~DeviceLibrary() = default;
};

/**
 * @brief Definition of the dynamic device library.
 * @details This class is used to load the QDMI device interface functions
 * from a dynamic library at runtime. It inherits from DeviceLibrary and
 * overrides the constructor and destructor to open and close the library.
 */
class DynamicDeviceLibrary final : public DeviceLibrary {
  /// @brief Handle to the dynamic library returned by `dlopen`.
  void* libHandle_;

  /**
   * @brief Returns a reference to the set of open library handles.
   * @details This set can be used to check whether a newly opened library
   * is already open.
   */
  static auto openLibHandles() -> std::unordered_set<void*>& {
    static std::unordered_set<void*> libHandles;
    return libHandles;
  }

public:
  /**
   * @brief Constructs a DynamicDeviceLibrary object.
   * @details This constructor loads the QDMI device interface functions
   * from the dynamic library specified by `libName` and `prefix`.
   * @param libName is the name of the dynamic library to load.
   * @param prefix is the prefix used for the function names in the library.
   */
  DynamicDeviceLibrary(const std::string& libName, const std::string& prefix);

  /**
   * @brief Destructor for the DynamicDeviceLibrary.
   * @details This destructor calls the @ref QDMI_device_finalize function if it
   * is not null and closes the dynamic library.
   */
  ~DynamicDeviceLibrary() override;
};

// Macro to define a static library class that inherits from DeviceLibrary.
// It binds all device library functions to the functions of the static library.
// @param prefix is the prefix used for the function names in the library.
#define DECLARE_STATIC_LIBRARY(prefix)                                         \
  class prefix##DeviceLibrary final : public DeviceLibrary {                   \
  public:                                                                      \
    prefix##DeviceLibrary();                                                   \
                                                                               \
    ~prefix##DeviceLibrary() override;                                         \
  };

// Call the above macro for all static libraries that we want to support.
DECLARE_STATIC_LIBRARY(MQT_NA)

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
 * @brief Definition of the QDMI Device.
 */
// Since we treat this struct as a class, we apply also the naming scheme for
// classes, i.e., an underscore at the end of member names.
struct QDMI_Device_impl_d {
private:
  /**
   * @brief The device library that provides the device interface functions.
   * @note This must be a pointer type as we need access to dynamic and static
   * libraries that are subclasses of qdmi::DeviceLibrary.
   */
  std::unique_ptr<qdmi::DeviceLibrary> library_;
  /// @brief The device session handle.
  QDMI_Device_Session deviceSession_ = nullptr;
  /**
   * @brief Map of jobs to their corresponding unique pointers of
   * QDMI_Job_impl_d objects.
   */
  std::unordered_map<QDMI_Job, std::unique_ptr<QDMI_Job_impl_d>> jobs_;

public:
  /**
   * @brief Constructor for the QDMI device.
   * @details This constructor initializes the device session and allocates
   * the device session handle.
   * @param lib is a unique pointer to the device library that provides the
   * device interface functions.
   */
  explicit QDMI_Device_impl_d(std::unique_ptr<qdmi::DeviceLibrary>&& lib);

  /**
   * @brief Destructor for the QDMI device.
   * @details This destructor frees the device session and clears the jobs map.
   */
  ~QDMI_Device_impl_d() {
    jobs_.clear();
    if (deviceSession_ != nullptr) {
      library_->device_session_free(deviceSession_);
    }
  }

  /**
   * @brief Creates a job for the device.
   * @see QDMI_device_create_job
   */
  auto createJob(QDMI_Job* job) -> int;

  /**
   * @brief Frees the job associated with the device.
   * @see QDMI_job_free
   */
  auto freeJob(QDMI_Job job) -> void;

  /**
   * @brief Queries a device property.
   * @see QDMI_device_query_device_property
   */
  auto queryDeviceProperty(QDMI_Device_Property prop, size_t size, void* value,
                           size_t* sizeRet) const -> int;

  /**
   * @brief Queries a site property.
   * @see QDMI_device_query_site_property
   */
  auto querySiteProperty(QDMI_Site site, QDMI_Site_Property prop, size_t size,
                         void* value, size_t* sizeRet) const -> int;

  /**
   * @brief Queries an operation property.
   * @see QDMI_device_query_operation_property
   */
  auto queryOperationProperty(QDMI_Operation operation, size_t numSites,
                              const QDMI_Site* sites, size_t numParams,
                              const double* params,
                              QDMI_Operation_Property prop, size_t size,
                              void* value, size_t* sizeRet) const -> int;
};

/**
 * @brief Definition of the QDMI Job.
 */
// Since we treat this struct as a class, we apply also the naming scheme for
// classes, i.e., an underscore at the end of member names.
struct QDMI_Job_impl_d {
private:
  /// @brief The device library that provides the device interface functions.
  const qdmi::DeviceLibrary* library_ = nullptr;
  /// @brief The device job handle.
  QDMI_Device_Job deviceJob_ = nullptr;
  /// @brief The device associated with the job.
  QDMI_Device device_ = nullptr;

public:
  /**
   * @brief Constructor for the QDMI job.
   * @details This constructor initializes the job with the device job handle
   * and the device library.
   * @param library is a pointer to the device library that provides the
   * device interface functions.
   * @param deviceJob is the handle to the device job.
   * @param device is the device associated with the job.
   */
  explicit QDMI_Job_impl_d(const qdmi::DeviceLibrary* library,
                           QDMI_Device_Job deviceJob, QDMI_Device device)
      : library_(library), deviceJob_(deviceJob), device_(device) {}

  /**
   * @brief Destructor for the QDMI job.
   * @details This destructor frees the device job handle using the
   * @ref QDMI_device_job_free function from the device library.
   */
  ~QDMI_Job_impl_d();

  /**
   * @brief Sets a parameter for the job.
   * @see QDMI_job_set_parameter
   */
  auto setParameter(QDMI_Job_Parameter param, size_t size,
                    const void* value) const -> int;

  /**
   * @brief Queries a property of the job.
   * @see QDMI_job_query_property
   */
  auto queryProperty(QDMI_Job_Property prop, size_t size, void* value,
                     size_t* sizeRet) const -> int;

  /**
   * @brief Submits the job to the device.
   * @see QDMI_job_submit
   */
  [[nodiscard]] auto submit() const -> int;

  /**
   * @brief Cancels the job.
   * @see QDMI_job_cancel
   */
  [[nodiscard]] auto cancel() const -> int;

  /**
   * @brief Checks the status of the job.
   * @see QDMI_job_check
   */
  auto check(QDMI_Job_Status* status) const -> int;

  /**
   * @brief Waits for the job to complete but at most for the specified
   * timeout.
   * @see QDMI_job_wait
   */
  [[nodiscard]] auto wait(size_t timeout) const -> int;

  /**
   * @brief Gets the results of the job.
   * @see QDMI_job_get_results
   */
  auto getResults(QDMI_Job_Result result, size_t size, void* data,
                  size_t* sizeRet) const -> int;

  /**
   * @brief Frees the job.
   * @note This function just forwards to the device's @ref Device::freeJob
   * function. This function is needed because the interface only provides the
   * job handle to the @ref QDMI_job_free function and the job's device handle
   * is private.
   */
  auto free() -> void;
};

/**
 * @brief Definition of the QDMI Session.
 */
// Since we treat this struct as a class, we apply also the naming scheme for
// classes, i.e., an underscore at the end of member names.
struct QDMI_Session_impl_d {
private:
  /// @brief The status of the session.
  qdmi::SessionStatus status_ = qdmi::SessionStatus::ALLOCATED;
  /// @brief A pointer to the list of all devices.
  const std::vector<std::unique_ptr<QDMI_Device_impl_d>>* devices_;

public:
  /// @brief Constructor for the QDMI session.
  explicit QDMI_Session_impl_d(
      const std::vector<std::unique_ptr<QDMI_Device_impl_d>>& devices)
      : devices_(&devices) {}

  /**
   * @brief Initializes the session.
   * @see QDMI_session_init
   */
  auto init() -> int;

  /**
   * @brief Sets a parameter for the session.
   * @see QDMI_session_set_parameter
   */
  auto setParameter(QDMI_Session_Parameter param, size_t size,
                    const void* value) const -> int;

  /**
   * @brief Queries a session property.
   * @see QDMI_session_query_session_property
   */
  auto querySessionProperty(QDMI_Session_Property prop, size_t size,
                            void* value, size_t* sizeRet) const -> int;
};

namespace qdmi {
/**
 * @brief The MQT QDMI driver class.
 * @details This driver loads all statically known and linked QDMI device
 * libraries. Additional devices can be added dynamically.
 * @note This class is a singleton that manages the QDMI libraries and
 * sessions. It is responsible for loading the libraries, allocating sessions,
 * and providing access to the devices.
 */
class Driver final {
  /// @brief Private constructor to enforce the singleton pattern.
  Driver();

  /**
   * @brief Map of sessions to their corresponding unique pointers to
   * QDMI_Session_impl_d objects.
   */
  std::unordered_map<QDMI_Session, std::unique_ptr<QDMI_Session_impl_d>>
      sessions_;

  /**
   * @brief Vector of unique pointers to QDMI_Device_impl_d objects.
   */
  std::vector<std::unique_ptr<QDMI_Device_impl_d>> devices_;

public:
  // Delete copy constructors and assignment operators to prevent copying the
  // singleton instance.
  Driver(const Driver&) = delete;
  Driver& operator=(const Driver&) = delete;

  /// @brief Returns the singleton instance.
  static auto get() -> Driver& {
    static Driver instance;
    return instance;
  }

  /// @brief Destructor for the Driver class.
  ~Driver();

  /**
   * @brief Adds a dynamic device library to the driver.
   * @param libName is the path to the dynamic library to load.
   * @param prefix is the prefix used for the device interface functions.
   */
  auto addDynamicDeviceLibrary(const std::string& libName,
                               const std::string& prefix) -> void;

  /**
   * @brief Allocates a new session.
   * @see QDMI_session_alloc
   */
  auto sessionAlloc(QDMI_Session* session) -> int;

  /**
   * @brief Frees a session.
   * @see QDMI_session_free
   */
  auto sessionFree(QDMI_Session session) -> void;
};

} // namespace qdmi
