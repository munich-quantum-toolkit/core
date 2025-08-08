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

/** @file
 * @brief The MQT QDMI device implementation for neutral atom devices.
 */

#include "mqt_na_qdmi/device.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace qdmi {
class Device final {
  /// @brief Provides access to the device name.
  std::string name;

  /// @brief The number of qubits in the device.
  size_t qubitsNum = 0;

  /// @brief The list of sites.
  std::vector<std::unique_ptr<MQT_NA_QDMI_Site_impl_d>> sites;

  /// @brief The list of operations.
  std::vector<std::unique_ptr<MQT_NA_QDMI_Operation_impl_d>> operations;

  /// @brief The list of device sessions.
  std::unordered_map<MQT_NA_QDMI_Device_Session,
                     std::unique_ptr<MQT_NA_QDMI_Device_Session_impl_d>>
      sessions;

  /// @brief Private constructor to enforce the singleton pattern.
  Device();

public:
  // Default move constructor and move assignment operator.
  Device(Device&&) = default;
  Device& operator=(Device&&) = default;
  // Delete copy constructor and assignment operator to enforce singleton.
  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;

  /// @returns the singleton instance of the Device class.
  [[nodiscard]] static Device& get() {
    static Device instance;
    return instance;
  }

  /// @brief Destructor for the Device class.
  ~Device() = default;

  /**
   * @brief Allocates a new device session.
   * @see MQT_NA_QDMI_device_session_alloc
   */
  auto sessionAlloc(MQT_NA_QDMI_Device_Session* session) -> int;

  /**
   * @brief Frees a device session.
   * @see MQT_NA_QDMI_device_session_free
   */
  auto sessionFree(MQT_NA_QDMI_Device_Session session) -> void;

  /**
   * @brief Query a device property.
   * @see MQT_NA_QDMI_device_session_query_device_property
   */
  auto queryProperty(QDMI_Device_Property prop, size_t size, void* value,
                     size_t* sizeRet) -> int;
};
} // namespace qdmi

/**
 * @brief Implementation of the MQT_NA_QDMI_Device_Session structure.
 */
struct MQT_NA_QDMI_Device_Session_impl_d {
private:
  /// The status of the session.
  enum class Status : uint8_t {
    ALLOCATED,   ///< The session has been allocated but not initialized
    INITIALIZED, ///< The session has been initialized and is ready for use
  };
  /// @brief The current status of the session.
  Status status = Status::ALLOCATED;
  /// @brief The device jobs associated with this session.
  std::unordered_map<MQT_NA_QDMI_Device_Job,
                     std::unique_ptr<MQT_NA_QDMI_Device_Job_impl_d>>
      jobs;

public:
  /**
   * @brief Initializes the device session.
   * @see MQT_NA_QDMI_device_session_init
   */
  auto init() -> int;

  /**
   * @brief Sets a parameter for the device session.
   * @see MQT_NA_QDMI_device_session_set_parameter
   */
  auto setParameter(QDMI_Device_Session_Parameter param, size_t size,
                    const void* value) const -> int;

  /**
   * @brief Create a new device job.
   * @see MQT_NA_QDMI_device_session_create_device_job
   */
  auto createDeviceJob(MQT_NA_QDMI_Device_Job* job) -> int;

  /**
   * @brief Frees the device job.
   * @see MQT_NA_QDMI_device_job_free
   */
  auto freeDeviceJob(MQT_NA_QDMI_Device_Job job) -> void;

  /**
   * @brief Forwards a query of a device property to the device.
   * @see MQT_NA_QDMI_device_session_query_device_property
   */
  auto queryDeviceProperty(QDMI_Device_Property prop, size_t size, void* value,
                           size_t* sizeRet) const -> int;

  /**
   * @brief Forwards a query of a site property to the site.
   * @see MQT_NA_QDMI_device_session_query_site_property
   */
  auto querySiteProperty(MQT_NA_QDMI_Site site, QDMI_Site_Property prop,
                         size_t size, void* value, size_t* sizeRet) const
      -> int;

  /**
   * @brief Forwards a query of an operation property to the operation.
   * @see MQT_NA_QDMI_device_session_query_operation_property
   */
  auto queryOperationProperty(MQT_NA_QDMI_Operation operation, size_t numSites,
                              const MQT_NA_QDMI_Site* sites, size_t numParams,
                              const double* params,
                              QDMI_Operation_Property prop, size_t size,
                              void* value, size_t* sizeRet) const -> int;
};

/**
 * @brief Implementation of the MQT_NA_QDMI_Device_Job structure.
 */
struct MQT_NA_QDMI_Device_Job_impl_d {
private:
  /// @brief The device session associated with the job.
  MQT_NA_QDMI_Device_Session_impl_d* session;

public:
  /// @brief Constructor for the MQT_NA_QDMI_Device_Job_impl_d.
  explicit MQT_NA_QDMI_Device_Job_impl_d(
      MQT_NA_QDMI_Device_Session_impl_d* session)
      : session(session) {}
  /**
   * @brief Frees the device job.
   * @note This function just forwards to the session's @ref freeDeviceJob
   * function. This function is needed because the interface only provides the
   * job handle to the @ref QDMI_job_free function and the job's session handle
   * is private.
   * @see QDMI_job_free
   */
  auto free() -> void;

  /**
   * @brief Sets a parameter for the job.
   * @see MQT_NA_QDMI_device_job_set_parameter
   */
  static auto setParameter(QDMI_Device_Job_Parameter param, size_t size,
                           const void* value) -> int;

  /**
   * @brief Queries a property of the job.
   * @see MQT_NA_QDMI_device_job_query_property
   */
  static auto queryProperty(QDMI_Device_Job_Property prop, size_t size,
                            void* value, size_t* sizeRet) -> int;

  /**
   * @brief Submits the job to the device.
   * @see MQT_NA_QDMI_device_job_submit
   */
  static auto submit() -> int;

  /**
   * @brief Cancels the job.
   * @see MQT_NA_QDMI_device_job_cancel
   */
  static auto cancel() -> int;

  /**
   * @brief Checks the status of the job.
   * @see MQT_NA_QDMI_device_job_check
   */
  static auto check(QDMI_Job_Status* status) -> int;

  /**
   * @brief Waits for the job to complete but at most for the specified timeout.
   * @see MQT_NA_QDMI_device_job_wait
   */
  static auto wait(size_t timeout) -> int;

  /**
   * @brief Gets the results of the job.
   * @see MQT_NA_QDMI_device_job_get_results
   */
  static auto getResults(QDMI_Job_Result result, size_t size, void* data,
                         [[maybe_unused]] size_t* sizeRet) -> int;
};

/**
 * @brief Implementation of the MQT_NA_QDMI_Device_Site structure.
 */
struct MQT_NA_QDMI_Site_impl_d {
private:
  uint64_t id = 0;       ///< Unique identifier of the site
  uint64_t moduleId = 0; ///< Identifier of the module the site belongs to
  uint64_t subModuleId =
      0;         ///< Identifier of the sub-module the site belongs to
  int64_t x = 0; ///< X coordinate of the site in the lattice
  int64_t y = 0; ///< Y coordinate of the site in the lattice
  /// @brief Collects decoherence times for the device.
  struct DecoherenceTimes {
    double t1 = 0.0; ///< T1 time in microseconds
    double t2 = 0.0; ///< T2 time in microseconds
  };
  /// @brief The decoherence times of the device.
  DecoherenceTimes decoherenceTimes{};

public:
  /// @brief Constructor for the MQT_NA_QDMI_Site_impl_d.
  MQT_NA_QDMI_Site_impl_d(uint64_t id, uint64_t module, uint64_t subModule,
                          int64_t x, int64_t y);
  /**
   * @brief Queries a property of the site.
   * @see MQT_NA_QDMI_device_session_query_site_property
   */
  auto queryProperty(QDMI_Site_Property prop, size_t size, void* value,
                     size_t* sizeRet) const -> int;
};

/**
 * @brief Implementation of the MQT_NA_QDMI_Device_Operation structure.
 */
struct MQT_NA_QDMI_Operation_impl_d {
  /// The type of operation.
  enum class Type : uint8_t {
    GlobalSingleQubit, ///< Global single-qubit operation
    GlobalMultiQubit,  ///< Global multi-qubit operation
    LocalSingleQubit,  ///< Local single-qubit operation
    LocalMultiQubit,   ///< Local multi-qubit operation
    ShuttlingLoad,     ///< Shuttling load operation
    ShuttlingMove,     ///< Shuttling move operation
    ShuttlingStore,    ///< Shuttling store operation
  };

private:
  std::string name;     ///< Name of the operation
  Type type;            ///< Type of the operation
  size_t numParameters; ///< Number of parameters for the operation
  /**
   * @brief Number of qubits involved in the operation
   * @note This number is only valid if the operation is a multi-qubit
   * operation.
   */
  size_t numQubits;
  double duration; ///< Duration of the operation in microseconds
  double fidelity; ///< Fidelity of the operation

  /**
   * @brief Checks if the operation type is a shuttling operation.
   * @param type The operation type to check.
   * @return true if the operation type is a shuttling operation, false
   * otherwise.
   */
  [[nodiscard]] static auto isShuttling(Type type) -> bool;

public:
  /// @brief Constructor for the MQT_NA_QDMI_Operation_impl_d.
  MQT_NA_QDMI_Operation_impl_d(std::string name, Type type,
                               size_t numParameters, size_t numQubits,
                               double duration, double fidelity)
      : name(std::move(name)), type(type), numParameters(numParameters),
        numQubits(numQubits), duration(duration), fidelity(fidelity) {}

  /**
   * @brief Queries a property of the operation.
   * @see MQT_NA_QDMI_device_session_query_operation_property
   */
  auto queryProperty(size_t numSites, const MQT_NA_QDMI_Site* sites,
                     size_t numParams, const double* params,
                     QDMI_Operation_Property prop, size_t size, void* value,
                     size_t* sizeRet) const -> int;
};
