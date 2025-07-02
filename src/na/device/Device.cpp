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

#include "mqt_na_qdmi/device.h"

#include "Device.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {
/// The status of the session.
enum class SessionStatus : uint8_t {
  ALLOCATED,   ///< The session has been allocated but not initialized
  INITIALIZED, ///< The session has been initialized and is ready for use
};

/// The type of operation.
enum class OperationType : uint8_t {
  GLOBAL_SINGLE_QUBIT, ///< Global single-qubit operation
  GLOBAL_MULTI_QUBIT,  ///< Global multi-qubit operation
  LOCAL_SINGLE_QUBIT,  ///< Local single-qubit operation
  LOCAL_MULTI_QUBIT,   ///< Local multi-qubit operation
  SHUTTLING_LOAD,      ///< Shuttling load operation
  SHUTTLING_MOVE,      ///< Shuttling move operation
  SHUTTLING_STORE,     ///< Shuttling store operation
};

/**
 * @brief Checks if the operation type is a shuttling operation.
 * @param type The operation type to check.
 * @return true if the operation type is a shuttling operation, false
 * otherwise.
 */
[[nodiscard]] auto isShuttling(const OperationType type) -> bool {
  switch (type) {
  case OperationType::SHUTTLING_LOAD:
  case OperationType::SHUTTLING_MOVE:
  case OperationType::SHUTTLING_STORE:
    return true;
  default:
    return false;
  }
}
} // namespace

/**
 * @brief Implementation of the MQT_NA_QDMI_Device_Session structure.
 */
struct MQT_NA_QDMI_Device_Session_impl_d {
  SessionStatus status = SessionStatus::ALLOCATED;
};

/**
 * @brief Implementation of the MQT_NA_QDMI_Device_Job structure.
 */
struct MQT_NA_QDMI_Device_Job_impl_d {};

/**
 * @brief Implementation of the MQT_NA_QDMI_Device_Site structure.
 */
struct MQT_NA_QDMI_Site_impl_d {
  size_t id;        ///< Unique identifier of the site
  size_t module;    ///< Identifier of the module the site belongs to
  size_t subModule; ///< Identifier of the sub-module the site belongs to
  int64_t x;        ///< X coordinate of the site in the lattice
  int64_t y;        ///< Y coordinate of the site in the lattice
};

/**
 * @brief Implementation of the MQT_NA_QDMI_Device_Operation structure.
 */
struct MQT_NA_QDMI_Operation_impl_d {
  std::string name;     ///< Name of the operation
  OperationType type;   ///< Type of the operation
  size_t numParameters; ///< Number of parameters for the operation
  /**
   * @brief Number of qubits involved in the operation
   * @note This number is only valid if the operation is a multi-qubit
   * operation.
   */
  size_t numQubits;
  double duration; ///< Duration of the operation in microseconds
  double fidelity; ///< Fidelity of the operation
};

namespace {

/**
 * @brief Provides access to the device name.
 * @returns a reference to a static variables that stores the
 * device name.
 */
[[nodiscard]] auto name() -> std::string& {
  static std::string name;
  return name;
}

/**
 * @brief Provides access to the number of qubits in the device.
 * @returns a reference to a static variable that stores the number of qubits.
 */
[[nodiscard]] auto qubitsNum() -> size_t& {
  static size_t qubitsNum = 0;
  return qubitsNum;
}

/**
 * @brief Provides access to the lists of sites and operations.
 * @returns a reference to a static vector of unique pointers to
 * @ref MQT_NA_QDMI_Site_impl_d.
 */
[[nodiscard]] auto sites()
    -> std::vector<std::unique_ptr<MQT_NA_QDMI_Site_impl_d>>& {
  static std::vector<std::unique_ptr<MQT_NA_QDMI_Site_impl_d>> sites;
  return sites;
}

/**
 * @brief Provides access to the list of operations.
 * @returns a reference to a static vector of unique pointers to
 * @ref MQT_NA_QDMI_Operation_impl_d.
 */
[[nodiscard]] auto operations()
    -> std::vector<std::unique_ptr<MQT_NA_QDMI_Operation_impl_d>>& {
  static std::vector<std::unique_ptr<MQT_NA_QDMI_Operation_impl_d>> operations;
  return operations;
}

/// Collects decoherence times for the device.
struct DecoherenceTimes {
  double t1; ///< T1 time in microseconds
  double t2; ///< T2 time in microseconds
};

/**
 * @brief Provides access to the decoherence times of the device.
 * @details This function returns a reference to a static instance of
 * @ref DecoherenceTimes, which is used to store the decoherence times for the
 * device.
 * @returns A reference to a static instance of @ref DecoherenceTimes.
 */
[[nodiscard]] auto decoherenceTimes() -> DecoherenceTimes& {
  static DecoherenceTimes decoherenceTimes;
  return decoherenceTimes;
}

/**
 * @brief Provides access to the list of device sessions.
 * @return a reference to a static vector of unique pointers to
 * MQT_NA_QDMI_Device_Session_impl_d.
 */
[[nodiscard]] auto sessions()
    -> std::unordered_map<MQT_NA_QDMI_Device_Session,
                          std::unique_ptr<MQT_NA_QDMI_Device_Session_impl_d>>& {
  static std::unordered_map<MQT_NA_QDMI_Device_Session,
                            std::unique_ptr<MQT_NA_QDMI_Device_Session_impl_d>>
      sessions;
  return sessions;
}
} // namespace

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

int MQT_NA_QDMI_device_initialize() {
  INITIALIZE_NAME(name());
  INITIALIZE_QUBITSNUM(qubitsNum());
  INITIALIZE_SITES(sites());
  INITIALIZE_OPERATIONS(operations());
  INITIALIZE_T1(decoherenceTimes().t1);
  INITIALIZE_T2(decoherenceTimes().t2);
  return QDMI_SUCCESS;
}

int MQT_NA_QDMI_device_finalize() {
  while (!sessions().empty()) {
    MQT_NA_QDMI_device_session_free(sessions().begin()->first);
  }
  return QDMI_SUCCESS;
}

int MQT_NA_QDMI_device_session_alloc(MQT_NA_QDMI_Device_Session* session) {
  if (session == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  auto uniqueSession = std::make_unique<MQT_NA_QDMI_Device_Session_impl_d>();
  *session = sessions()
                 .emplace(uniqueSession.get(), std::move(uniqueSession))
                 .first->first;
  return QDMI_SUCCESS;
}

int MQT_NA_QDMI_device_session_init(MQT_NA_QDMI_Device_Session session) {
  if (session == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  if (session->status != SessionStatus::ALLOCATED) {
    return QDMI_ERROR_BADSTATE;
  }
  session->status = SessionStatus::INITIALIZED;
  return QDMI_SUCCESS;
}

void MQT_NA_QDMI_device_session_free(MQT_NA_QDMI_Device_Session session) {
  sessions().erase(session);
}

int MQT_NA_QDMI_device_session_set_parameter(
    MQT_NA_QDMI_Device_Session session, QDMI_Device_Session_Parameter param,
    const size_t size, const void* value) {
  if (session == nullptr || (value != nullptr && size == 0) ||
      param >= QDMI_DEVICE_SESSION_PARAMETER_MAX) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  if (session->status == SessionStatus::ALLOCATED) {
    return QDMI_ERROR_BADSTATE;
  }
  return QDMI_ERROR_NOTSUPPORTED;
}

int MQT_NA_QDMI_device_session_create_device_job(
    MQT_NA_QDMI_Device_Session session, MQT_NA_QDMI_Device_Job* job) {
  if (session == nullptr || job == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  if (session->status == SessionStatus::ALLOCATED) {
    return QDMI_ERROR_BADSTATE;
  }
  return QDMI_ERROR_NOTSUPPORTED;
}

void MQT_NA_QDMI_device_job_free(MQT_NA_QDMI_Device_Job /* unused */) {}

int MQT_NA_QDMI_device_job_set_parameter(MQT_NA_QDMI_Device_Job job,
                                         const QDMI_Device_Job_Parameter param,
                                         const size_t size, const void* value) {
  if (job == nullptr || (value != nullptr && size == 0) ||
      param >= QDMI_DEVICE_JOB_PARAMETER_MAX) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  return QDMI_ERROR_NOTSUPPORTED;
}

int MQT_NA_QDMI_device_job_query_property(
    MQT_NA_QDMI_Device_Job job, const QDMI_Device_Job_Property prop,
    const size_t size,
    // NOLINTNEXTLINE(readability-non-const-parameter)
    void* value, size_t* /* unused */) {
  if (job == nullptr || (value != nullptr && size == 0) ||
      prop >= QDMI_DEVICE_JOB_PROPERTY_MAX) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  return QDMI_ERROR_NOTSUPPORTED;
}

int MQT_NA_QDMI_device_job_submit(MQT_NA_QDMI_Device_Job job) {
  if (job == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  return QDMI_ERROR_NOTSUPPORTED;
}

int MQT_NA_QDMI_device_job_cancel(MQT_NA_QDMI_Device_Job job) {
  if (job == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  return QDMI_ERROR_NOTSUPPORTED;
}

int MQT_NA_QDMI_device_job_check(
    MQT_NA_QDMI_Device_Job job,
    // NOLINTNEXTLINE(readability-non-const-parameter)
    QDMI_Job_Status* status) {
  if (job == nullptr || status == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  return QDMI_ERROR_NOTSUPPORTED;
}

int MQT_NA_QDMI_device_job_wait(MQT_NA_QDMI_Device_Job job,
                                const size_t /* unused */) {
  if (job == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  return QDMI_ERROR_NOTSUPPORTED;
}

int MQT_NA_QDMI_device_job_get_results(MQT_NA_QDMI_Device_Job job,
                                       QDMI_Job_Result result,
                                       const size_t size, void* data,
                                       [[maybe_unused]] size_t* sizeRet) {
  if (job == nullptr || (data != nullptr && size == 0) ||
      result >= QDMI_JOB_RESULT_MAX) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  return QDMI_ERROR_NOTSUPPORTED;
}

int MQT_NA_QDMI_device_session_query_device_property(
    MQT_NA_QDMI_Device_Session session, const QDMI_Device_Property prop,
    const size_t size, void* value, size_t* sizeRet) {
  if (session == nullptr || (value != nullptr && size == 0) ||
      prop >= QDMI_DEVICE_PROPERTY_MAX) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  if (session->status != SessionStatus::INITIALIZED) {
    return QDMI_ERROR_BADSTATE;
  }
  ADD_STRING_PROPERTY(QDMI_DEVICE_PROPERTY_NAME, name().c_str(), prop, size,
                      value, sizeRet)
  // NOLINTNEXTLINE(misc-include-cleaner)
  ADD_STRING_PROPERTY(QDMI_DEVICE_PROPERTY_VERSION, MQT_CORE_VERSION, prop,
                      size, value, sizeRet)
  // NOLINTNEXTLINE(misc-include-cleaner)
  ADD_STRING_PROPERTY(QDMI_DEVICE_PROPERTY_LIBRARYVERSION, QDMI_VERSION, prop,
                      size, value, sizeRet)
  ADD_SINGLE_VALUE_PROPERTY(QDMI_DEVICE_PROPERTY_STATUS, QDMI_Device_Status,
                            QDMI_DEVICE_STATUS_IDLE, prop, size, value, sizeRet)
  ADD_SINGLE_VALUE_PROPERTY(QDMI_DEVICE_PROPERTY_QUBITSNUM, size_t, qubitsNum(),
                            prop, size, value, sizeRet)
  // This device never needs calibration
  ADD_SINGLE_VALUE_PROPERTY(QDMI_DEVICE_PROPERTY_NEEDSCALIBRATION, size_t, 0,
                            prop, size, value, sizeRet)
  ADD_LIST_PROPERTY(QDMI_DEVICE_PROPERTY_SITES, MQT_NA_QDMI_Site, sites(), prop,
                    size, value, sizeRet)
  ADD_LIST_PROPERTY(QDMI_DEVICE_PROPERTY_OPERATIONS, MQT_NA_QDMI_Operation,
                    operations(), prop, size, value, sizeRet)
  return QDMI_ERROR_NOTSUPPORTED;
}

int MQT_NA_QDMI_device_session_query_site_property(
    MQT_NA_QDMI_Device_Session session, MQT_NA_QDMI_Site site,
    const QDMI_Site_Property prop, const size_t size, void* value,
    size_t* sizeRet) {
  if (session == nullptr || site == nullptr ||
      (value != nullptr && size == 0) ||
      (prop != QDMI_SITE_PROPERTY_CUSTOM1 &&
       prop != QDMI_SITE_PROPERTY_CUSTOM2 &&
       prop != QDMI_SITE_PROPERTY_CUSTOM3 &&
       prop != QDMI_SITE_PROPERTY_CUSTOM4 && prop >= QDMI_SITE_PROPERTY_MAX)) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  ADD_SINGLE_VALUE_PROPERTY(QDMI_SITE_PROPERTY_INDEX, size_t, site->id, prop,
                            size, value, sizeRet)
  ADD_SINGLE_VALUE_PROPERTY(QDMI_SITE_PROPERTY_CUSTOM1, size_t, site->module,
                            prop, size, value, sizeRet)
  ADD_SINGLE_VALUE_PROPERTY(QDMI_SITE_PROPERTY_CUSTOM2, size_t, site->subModule,
                            prop, size, value, sizeRet)
  ADD_SINGLE_VALUE_PROPERTY(QDMI_SITE_PROPERTY_CUSTOM3, int64_t, site->x, prop,
                            size, value, sizeRet)
  ADD_SINGLE_VALUE_PROPERTY(QDMI_SITE_PROPERTY_CUSTOM4, int64_t, site->y, prop,
                            size, value, sizeRet)
  ADD_SINGLE_VALUE_PROPERTY(QDMI_SITE_PROPERTY_T1, double,
                            decoherenceTimes().t1, prop, size, value, sizeRet)
  ADD_SINGLE_VALUE_PROPERTY(QDMI_SITE_PROPERTY_T2, double,
                            decoherenceTimes().t2, prop, size, value, sizeRet)
  return QDMI_ERROR_NOTSUPPORTED;
}

int MQT_NA_QDMI_device_session_query_operation_property(
    MQT_NA_QDMI_Device_Session session, MQT_NA_QDMI_Operation operation,
    const size_t numSites, const MQT_NA_QDMI_Site* sites,
    const size_t numParams, const double* params,
    const QDMI_Operation_Property prop, const size_t size, void* value,
    size_t* sizeRet) {
  if (session == nullptr || operation == nullptr ||
      (sites != nullptr && numSites == 0) ||
      (params != nullptr && numParams == 0) ||
      (value != nullptr && size == 0) || prop >= QDMI_OPERATION_PROPERTY_MAX) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  ADD_STRING_PROPERTY(QDMI_OPERATION_PROPERTY_NAME, operation->name.c_str(),
                      prop, size, value, sizeRet)
  ADD_SINGLE_VALUE_PROPERTY(QDMI_OPERATION_PROPERTY_PARAMETERSNUM, size_t,
                            operation->numParameters, prop, size, value,
                            sizeRet)
  if (operation->type != OperationType::SHUTTLING_MOVE) {
    ADD_SINGLE_VALUE_PROPERTY(QDMI_OPERATION_PROPERTY_DURATION, double,
                              operation->duration, prop, size, value, sizeRet)
    ADD_SINGLE_VALUE_PROPERTY(QDMI_OPERATION_PROPERTY_FIDELITY, double,
                              operation->fidelity, prop, size, value, sizeRet)
  } else {
    ADD_SINGLE_VALUE_PROPERTY(QDMI_OPERATION_PROPERTY_DURATION, double, 0.0,
                              prop, size, value, sizeRet)
    ADD_SINGLE_VALUE_PROPERTY(QDMI_OPERATION_PROPERTY_FIDELITY, double, 1.0,
                              prop, size, value, sizeRet)
  }
  if (!isShuttling(operation->type)) {
    ADD_SINGLE_VALUE_PROPERTY(QDMI_OPERATION_PROPERTY_QUBITSNUM, size_t,
                              operation->numQubits, prop, size, value, sizeRet)
  }
  return QDMI_ERROR_NOTSUPPORTED;
}
