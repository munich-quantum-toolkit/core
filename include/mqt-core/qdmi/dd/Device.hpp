/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

/** @file
 * @brief The MQT QDMI device implementation for its DD-based simulator.
 */

#include "dd/DDDefinitions.hpp"
#include "dd/Package.hpp"
#include "mqt_ddsim_qdmi/device.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <future>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <unordered_map>

namespace qdmi::dd {
class Device final {
  /// Provides access to the device name.
  std::string name_;

  /// The number of qubits supported by the simulator.
  size_t qubitsNum_ = 0;

  /// The status of the device.
  std::atomic<QDMI_Device_Status> status_{QDMI_DEVICE_STATUS_OFFLINE};

  /// The list of device sessions.
  std::unordered_map<MQT_DDSIM_QDMI_Device_Session,
                     std::unique_ptr<MQT_DDSIM_QDMI_Device_Session_impl_d>>
      sessions_;
  /// Mutex protecting access to sessions_.
  mutable std::mutex sessionsMutex_;

  /// RNG for generating unique IDs.
  std::mt19937_64 rng_{std::random_device{}()};
  /// Mutex protecting RNG usage.
  mutable std::mutex rngMutex_;

  /// Distribution for generating unique IDs.
  std::uniform_int_distribution<> dis_ =
      std::uniform_int_distribution<>(0, std::numeric_limits<int>::max());

  /// The number of running jobs.
  std::atomic<size_t> runningJobs_{0};

  /// @brief Private constructor to enforce the singleton pattern.
  Device();

public:
  // Default move constructor and move assignment operator.
  Device(Device&&) = delete;
  Device& operator=(Device&&) = delete;
  // Delete copy constructor and assignment operator to enforce singleton.
  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;

  /// @brief Destructor for the Device class.
  ~Device() = default;

  /// @returns the singleton instance of the Device class.
  [[nodiscard]] static auto get() -> Device&;

  /**
   * @brief Allocates a new device session.
   * @see MQT_DDSIM_QDMI_device_session_alloc
   */
  auto sessionAlloc(MQT_DDSIM_QDMI_Device_Session* session) -> QDMI_STATUS;

  /**
   * @brief Frees a device session.
   * @see MQT_DDSIM_QDMI_device_session_free
   */
  auto sessionFree(MQT_DDSIM_QDMI_Device_Session session) -> void;

  /**
   * @brief Query a device property.
   * @see MQT_DDSIM_QDMI_device_session_query_device_property
   */
  auto queryProperty(QDMI_Device_Property prop, size_t size, void* value,
                     size_t* sizeRet) const -> QDMI_STATUS;

  /// Generates a unique ID.
  auto generateUniqueID() -> int;

  /// Sets the device status.
  auto setStatus(QDMI_Device_Status status) -> void;

  /// Bumps the number of running jobs and updates the status
  auto increaseRunningJobs() -> void;

  /// Decreases the number of running jobs and updates the status
  auto decreaseRunningJobs() -> void;
};
} // namespace qdmi::dd

/**
 * @brief Implementation of the MQT_DDSIM_QDMI_Device_Session structure.
 */
struct MQT_DDSIM_QDMI_Device_Session_impl_d {
private:
  /// The status of the session.
  enum class Status : uint8_t {
    ALLOCATED,   ///< The session has been allocated but not initialized
    INITIALIZED, ///< The session has been initialized and is ready for use
  };
  /// @brief The current status of the session.
  Status status_ = Status::ALLOCATED;
  /// @brief The device jobs associated with this session.
  std::unordered_map<MQT_DDSIM_QDMI_Device_Job,
                     std::unique_ptr<MQT_DDSIM_QDMI_Device_Job_impl_d>>
      jobs_;
  /// @brief Mutex protecting access to jobs_.
  mutable std::mutex jobsMutex_;

public:
  /**
   * @brief Initializes the device session.
   * @see MQT_DDSIM_QDMI_device_session_init
   */
  auto init() -> QDMI_STATUS;

  /**
   * @brief Sets a parameter for the device session.
   * @see MQT_DDSIM_QDMI_device_session_set_parameter
   */
  auto setParameter(QDMI_Device_Session_Parameter param, size_t size,
                    const void* value) const -> QDMI_STATUS;

  /**
   * @brief Create a new device job.
   * @see MQT_DDSIM_QDMI_device_session_create_device_job
   */
  auto createDeviceJob(MQT_DDSIM_QDMI_Device_Job* job) -> QDMI_STATUS;

  /**
   * @brief Frees the device job.
   * @see MQT_DDSIM_QDMI_device_job_free
   */
  auto freeDeviceJob(MQT_DDSIM_QDMI_Device_Job job) -> void;

  /**
   * @brief Forwards a query of a device property to the device.
   * @see MQT_DDSIM_QDMI_device_session_query_device_property
   */
  auto queryDeviceProperty(QDMI_Device_Property prop, size_t size, void* value,
                           size_t* sizeRet) const -> QDMI_STATUS;

  /**
   * @brief Forwards a query of a site property to the site.
   * @see MQT_DDSIM_QDMI_device_session_query_site_property
   */
  auto querySiteProperty(MQT_DDSIM_QDMI_Site site, QDMI_Site_Property prop,
                         size_t size, void* value, size_t* sizeRet) const
      -> QDMI_STATUS;

  /**
   * @brief Forwards a query of an operation property to the operation.
   * @see MQT_DDSIM_QDMI_device_session_query_operation_property
   */
  auto queryOperationProperty(MQT_DDSIM_QDMI_Operation operation,
                              size_t numSites, const MQT_DDSIM_QDMI_Site* sites,
                              size_t numParams, const double* params,
                              QDMI_Operation_Property prop, size_t size,
                              void* value, size_t* sizeRet) const
      -> QDMI_STATUS;
};

/**
 * @brief Implementation of the MQT_DDSIM_QDMI_Device_Job structure.
 */
struct MQT_DDSIM_QDMI_Device_Job_impl_d {
private:
  /// The device session associated with the job.
  MQT_DDSIM_QDMI_Device_Session_impl_d* session_;

  /// The unique identifier of the job.
  int id_ = 0;

  /// The status of the job
  std::atomic<QDMI_Job_Status> status_{QDMI_JOB_STATUS_CREATED};

  /// The program format
  QDMI_Program_Format format_ = QDMI_PROGRAM_FORMAT_QASM3;

  /// The quantum program associated with the job
  std::string program_;

  /// The number of shots for the job
  size_t numShots_ = 1024U;

  /// Handle for the asynchronous job
  std::future<void> jobHandle_;

  /// The measurement counts for the job
  std::map<std::string, std::size_t> counts_;

  /// The DD package used for the state vector simulation
  std::unique_ptr<dd::Package> dd_;

  /// The final DD at the end of the state vector simulation
  dd::VectorDD stateVecDD_{};

  /// The state vector for the job (only available if no mid-circuit
  /// measurements are used).
  dd::CVec stateVec_;

  /// The sparse state vector for the job (only available if no mid-circuit
  /// measurements are used).
  dd::SparseCVec stateVecSparse_;

  /// One-time flags to lazily materialize vectors in a thread-safe way
  std::once_flag stateVecOnce_;
  std::once_flag stateVecSparseOnce_;

  /// Translate counts to QDMI histogram
  auto getHistogram(QDMI_Job_Result result, size_t size, void* data,
                    size_t* sizeRet) -> QDMI_STATUS;

  /// Translate the state vector DD to a dense state vector for QDMI
  auto getStateVector(size_t size, void* data, size_t* sizeRet) -> QDMI_STATUS;

  /// Translate the state vector DD to sparse representations for QDMI
  auto getSparseResults(QDMI_Job_Result result, size_t size, void* data,
                        size_t* sizeRet) -> QDMI_STATUS;

  /// Translate the state vector DD to a dense vector of probabilities for QDMI
  auto getProbabilities(size_t size, void* data, size_t* sizeRet)
      -> QDMI_STATUS;

public:
  /// Constructor for the MQT_DDSIM_QDMI_Device_Job_impl_d.
  explicit MQT_DDSIM_QDMI_Device_Job_impl_d(
      MQT_DDSIM_QDMI_Device_Session_impl_d* session)
      : session_(session), id_(qdmi::dd::Device::get().generateUniqueID()) {}
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
   * @see MQT_DDSIM_QDMI_device_job_set_parameter
   */
  auto setParameter(QDMI_Device_Job_Parameter param, size_t size,
                    const void* value) -> QDMI_STATUS;

  /**
   * @brief Queries a property of the job.
   * @see MQT_DDSIM_QDMI_device_job_query_property
   */
  auto queryProperty(QDMI_Device_Job_Property prop, size_t size, void* value,
                     size_t* sizeRet) const -> QDMI_STATUS;

  /**
   * @brief Submits the job to the device.
   * @see MQT_DDSIM_QDMI_device_job_submit
   */
  auto submit() -> QDMI_STATUS;

  /**
   * @brief Cancels the job.
   * @see MQT_DDSIM_QDMI_device_job_cancel
   */
  auto cancel() -> QDMI_STATUS;

  /**
   * @brief Checks the status of the job.
   * @see MQT_DDSIM_QDMI_device_job_check
   */
  auto check(QDMI_Job_Status* status) const -> QDMI_STATUS;

  /**
   * @brief Waits for the job to complete but at most for the specified timeout.
   * @see MQT_DDSIM_QDMI_device_job_wait
   */
  auto wait(size_t timeout) const -> QDMI_STATUS;

  /**
   * @brief Gets the results of the job.
   * @see MQT_DDSIM_QDMI_device_job_get_results
   */
  auto getResults(QDMI_Job_Result result, size_t size, void* data,
                  size_t* sizeRet) -> QDMI_STATUS;
};
