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
 * @brief The MQT QDMI device implementation for superconducting devices.
 */

#include "mqt_sc_qdmi/device.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

namespace qdmi::sc {
class Device final {
  /// @brief Provides access to the device name.
  std::string name_;

  /// @brief The number of qubits in the device.
  size_t qubitsNum_ = 0;

  /// @brief The list of sites.
  std::vector<std::unique_ptr<MQT_SC_QDMI_Site_impl_d>> sites_;

  /// @brief The list of couplings, i.e., qubit pairs
  std::vector<std::pair<MQT_SC_QDMI_Site, MQT_SC_QDMI_Site>> couplingMap_;

  /// @brief The list of operations.
  std::vector<std::unique_ptr<MQT_SC_QDMI_Operation_impl_d>> operations_;

  /// @brief The list of device sessions.
  std::unordered_map<MQT_SC_QDMI_Device_Session,
                     std::unique_ptr<MQT_SC_QDMI_Device_Session_impl_d>>
      sessions_;

  /// @brief Private constructor to enforce the singleton pattern.
  Device();

public:
  // Delete move constructor and move assignment operator.
  Device(Device&&) = delete;
  Device& operator=(Device&&) = delete;
  // Delete copy constructor and assignment operator to enforce singleton.
  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;

  /// @brief Destructor for the Device class.
  ~Device();

  /// @returns the singleton instance of the Device class.
  [[nodiscard]] static auto get() -> Device&;

  /**
   * @brief Allocates a new device session.
   * @see MQT_SC_QDMI_device_session_alloc
   */
  auto sessionAlloc(MQT_SC_QDMI_Device_Session* session) -> int;

  /**
   * @brief Frees a device session.
   * @see MQT_SC_QDMI_device_session_free
   */
  auto sessionFree(MQT_SC_QDMI_Device_Session session) -> void;

  /**
   * @brief Query a device property.
   * @see MQT_SC_QDMI_device_session_query_device_property
   */
  auto queryProperty(QDMI_Device_Property prop, size_t size, void* value,
                     size_t* sizeRet) const -> int;
};
} // namespace qdmi::sc

/**
 * @brief Implementation of the MQT_SC_QDMI_Device_Session structure.
 */
struct MQT_SC_QDMI_Device_Session_impl_d {
private:
  /// The status of the session.
  enum class Status : uint8_t {
    ALLOCATED,   ///< The session has been allocated but not initialized
    INITIALIZED, ///< The session has been initialized and is ready for use
  };
  /// @brief The current status of the session.
  Status status_ = Status::ALLOCATED;
  /// @brief The device jobs associated with this session.
  std::unordered_map<MQT_SC_QDMI_Device_Job,
                     std::unique_ptr<MQT_SC_QDMI_Device_Job_impl_d>>
      jobs_;

public:
  /**
   * @brief Initializes the device session.
   * @see MQT_SC_QDMI_device_session_init
   */
  auto init() -> int;

  /**
   * @brief Sets a parameter for the device session.
   * @see MQT_SC_QDMI_device_session_set_parameter
   */
  auto setParameter(QDMI_Device_Session_Parameter param, size_t size,
                    const void* value) const -> int;

  /**
   * @brief Create a new device job.
   * @see MQT_SC_QDMI_device_session_create_device_job
   */
  auto createDeviceJob(MQT_SC_QDMI_Device_Job* job) -> int;

  /**
   * @brief Frees the device job.
   * @see MQT_SC_QDMI_device_job_free
   */
  auto freeDeviceJob(MQT_SC_QDMI_Device_Job job) -> void;

  /**
   * @brief Forwards a query of a device property to the device.
   * @see MQT_SC_QDMI_device_session_query_device_property
   */
  auto queryDeviceProperty(QDMI_Device_Property prop, size_t size, void* value,
                           size_t* sizeRet) const -> int;

  /**
   * @brief Forwards a query of a site property to the site.
   * @see MQT_SC_QDMI_device_session_query_site_property
   */
  auto querySiteProperty(MQT_SC_QDMI_Site site, QDMI_Site_Property prop,
                         size_t size, void* value, size_t* sizeRet) const
      -> int;

  /**
   * @brief Forwards a query of an operation property to the operation.
   * @see MQT_SC_QDMI_device_session_query_operation_property
   */
  auto queryOperationProperty(MQT_SC_QDMI_Operation operation, size_t numSites,
                              const MQT_SC_QDMI_Site* sites, size_t numParams,
                              const double* params,
                              QDMI_Operation_Property prop, size_t size,
                              void* value, size_t* sizeRet) const -> int;
};

/**
 * @brief Implementation of the MQT_SC_QDMI_Device_Job structure.
 */
struct MQT_SC_QDMI_Device_Job_impl_d {
private:
  /// @brief The device session associated with the job.
  MQT_SC_QDMI_Device_Session_impl_d* session_;

public:
  /**
   * @brief Initializes a device job implementation bound to the given session.
   *
   * @param session Pointer to the owning MQT_SC_QDMI_Device_Session_impl_d. The
   * session must remain valid for the job's lifetime.
   */
  explicit MQT_SC_QDMI_Device_Job_impl_d(
      MQT_SC_QDMI_Device_Session_impl_d* session)
      : session_(session) {}
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
   * @see MQT_SC_QDMI_device_job_set_parameter
   */
  auto setParameter(QDMI_Device_Job_Parameter param, size_t size,
                    const void* value) -> int;

  /**
   * @brief Queries a property of the job.
   * @see MQT_SC_QDMI_device_job_query_property
   */
  auto queryProperty(QDMI_Device_Job_Property prop, size_t size, void* value,
                     size_t* sizeRet) -> int;

  /**
   * @brief Submits the job to the device.
   * @see MQT_SC_QDMI_device_job_submit
   */
  auto submit() -> int;

  /**
   * @brief Cancels the job.
   * @see MQT_SC_QDMI_device_job_cancel
   */
  auto cancel() -> int;

  /**
   * @brief Checks the status of the job.
   * @see MQT_SC_QDMI_device_job_check
   */
  auto check(QDMI_Job_Status* status) -> int;

  /**
   * @brief Waits for the job to complete but at most for the specified timeout.
   * @see MQT_SC_QDMI_device_job_wait
   */
  auto wait(size_t timeout) -> int;

  /**
   * @brief Gets the results of the job.
   * @see MQT_SC_QDMI_device_job_get_results
   */
  auto getResults(QDMI_Job_Result result, size_t size, void* data,
                  [[maybe_unused]] size_t* sizeRet) -> int;
};

/**
 * @brief Implementation of the MQT_SC_QDMI_Site structure.
 */
struct MQT_SC_QDMI_Site_impl_d {
  friend MQT_SC_QDMI_Operation_impl_d;

private:
  uint64_t id_ = 0; ///< Unique identifier of the site

  /**
   * @brief Initializes a site implementation with the given unique identifier.
   *
   * @param id Unique identifier for the site.
   */
  explicit MQT_SC_QDMI_Site_impl_d(uint64_t id) : id_(id) {}

public:
  /// @brief Factory function for regular sites.
  [[nodiscard]] static auto makeUniqueSite(uint64_t id)
      -> std::unique_ptr<MQT_SC_QDMI_Site_impl_d>;
  /**
   * @brief Queries a property of the site.
   * @see MQT_SC_QDMI_device_session_query_site_property
   */
  auto queryProperty(QDMI_Site_Property prop, size_t size, void* value,
                     size_t* sizeRet) const -> int;
};

/**
 * @brief Implementation of the MQT_SC_QDMI_Operation structure.
 */
struct MQT_SC_QDMI_Operation_impl_d {
private:
  std::string name_;     ///< Name of the operation
  size_t numParameters_; ///< Number of parameters for the operation
  /**
   * @brief Number of qubits involved in the operation
   */
  size_t numQubits_{};
  /**
   * @brief Storage for individual sites and site pairs.
   * @details Uses std::variant to preserve the tuple structure of the operation
   * sites:
   * - Single-qubit and zoned operations: vector<Site>
   * - Local two-qubit operations: vector<pair<Site, Site>>
   * This maintains type safety and QDMI specification compliance, which states
   * that operation sites should be "a list of tuples" for local multi-qubit
   * operations.
   */
  using SitesStorage =
      std::variant<std::vector<MQT_SC_QDMI_Site>,
                   std::vector<std::pair<MQT_SC_QDMI_Site, MQT_SC_QDMI_Site>>>;

  /// The operation's supported sites
  SitesStorage supportedSites_;
  /// @brief Constructor for a single-qubit operation.
  MQT_SC_QDMI_Operation_impl_d(std::string name, size_t numParameters,
                               const std::vector<MQT_SC_QDMI_Site>& sites);
  /// @brief Constructor for a two-qubit operation.
  MQT_SC_QDMI_Operation_impl_d(
      std::string name, size_t numParameters,
      const std::vector<std::pair<MQT_SC_QDMI_Site, MQT_SC_QDMI_Site>>& sites);

  /// @brief Sort the sites such that the occurrence of a given site can be
  /// determined in O(log n) time.
  auto sortSites() -> void;

public:
  /// @brief Factory function a single-qubit operation.
  [[nodiscard]] static auto
  makeUniqueSingleQubit(std::string name, size_t numParameters,
                        const std::vector<MQT_SC_QDMI_Site>& sites)
      -> std::unique_ptr<MQT_SC_QDMI_Operation_impl_d>;
  /// @brief Factory function a two-qubit operation.
  [[nodiscard]] static auto makeUniqueTwoQubit(
      std::string name, size_t numParameters,
      const std::vector<std::pair<MQT_SC_QDMI_Site, MQT_SC_QDMI_Site>>& sites)
      -> std::unique_ptr<MQT_SC_QDMI_Operation_impl_d>;

  /**
   * @brief Queries a property of the operation.
   * @see MQT_SC_QDMI_device_session_query_operation_property
   */
  auto queryProperty(size_t numSites, const MQT_SC_QDMI_Site* sites,
                     size_t numParams, const double* params,
                     QDMI_Operation_Property prop, size_t size, void* value,
                     size_t* sizeRet) const -> int;
};
