/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include <qdmi/device.h>

#include <array>
#include <atomic>
#include <cstddef>
#include <cstring>
#include <new>
#include <string>
#include <unordered_map>

struct QDMI_Child_Device_impl_d {};

struct QDMI_Operation_impl_d {
  const char* name;
  size_t qubitsNum;
  size_t parametersNum;
};

struct QDMI_Device_Session_impl_d {
  std::unordered_map<int, std::string> parameters;
  QDMI_Child_Device child = nullptr;
  bool initialized = false;
};

struct QDMI_Device_Job_impl_d {
  QDMI_Device_Session session = nullptr;
  bool retrieved = false;
  QDMI_Program_Format format = QDMI_PROGRAM_FORMAT_MAX;
};

namespace {
[[nodiscard]] auto activeSessions() -> std::atomic_size_t& {
  static std::atomic_size_t sessions = 0;
  return sessions;
}

[[nodiscard]] auto parameter(const QDMI_Device_Session_impl_d* const session,
                             const QDMI_Device_Session_Parameter key)
    -> std::string {
  if (const auto entry = session->parameters.find(key);
      entry != session->parameters.end()) {
    return entry->second;
  }
  return "<unset>";
}

[[nodiscard]] auto deviceStatus(const std::string& configuredStatus)
    -> QDMI_Device_Status {
  if (configuredStatus == "busy") {
    return QDMI_DEVICE_STATUS_BUSY;
  }
  if (configuredStatus == "offline") {
    return QDMI_DEVICE_STATUS_OFFLINE;
  }
  if (configuredStatus == "error") {
    return QDMI_DEVICE_STATUS_ERROR;
  }
  if (configuredStatus == "maintenance") {
    return QDMI_DEVICE_STATUS_MAINTENANCE;
  }
  if (configuredStatus == "calibration") {
    return QDMI_DEVICE_STATUS_CALIBRATION;
  }
  if (configuredStatus == "max") {
    return QDMI_DEVICE_STATUS_MAX;
  }
  return QDMI_DEVICE_STATUS_IDLE;
}

[[nodiscard]] auto childDeviceHandle() -> QDMI_Child_Device {
  static QDMI_Child_Device_impl_d child;
  return &child;
}

[[nodiscard]] auto customOperationHandles()
    -> const std::array<QDMI_Operation, 2>& {
  static QDMI_Operation_impl_d rotate{
      .name = "custom-rx", .qubitsNum = 1, .parametersNum = 1};
  static QDMI_Operation_impl_d controlledNot{
      .name = "custom-cx", .qubitsNum = 2, .parametersNum = 0};
  static const std::array<QDMI_Operation, 2> OPERATIONS{&rotate,
                                                        &controlledNot};
  return OPERATIONS;
}

[[nodiscard]] auto findCustomOperation(QDMI_Operation operation)
    -> const QDMI_Operation_impl_d* {
  for (auto* const handle : customOperationHandles()) {
    if (operation == handle) {
      return handle;
    }
  }
  return nullptr;
}

auto queryString(const std::string& result, const size_t size, void* value,
                 size_t* sizeRet) -> int {
  const auto required = result.size() + 1;
  if (sizeRet != nullptr) {
    *sizeRet = required;
  }
  if (value == nullptr) {
    return QDMI_SUCCESS;
  }
  if (size < required) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  std::memcpy(value, result.c_str(), required);
  return QDMI_SUCCESS;
}

template <typename T>
auto queryValue(const T& result, const size_t size, void* value,
                size_t* sizeRet) -> int {
  if (sizeRet != nullptr) {
    *sizeRet = sizeof(T);
  }
  if (value == nullptr) {
    return QDMI_SUCCESS;
  }
  if (size < sizeof(T)) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  std::memcpy(value, &result, sizeof(T));
  return QDMI_SUCCESS;
}
} // namespace

// QDMI requires these exported C symbols to use the configured device prefix.
// NOLINTBEGIN(readability-identifier-naming)
extern "C" int TEST_SESSION_QDMI_device_initialize() { return QDMI_SUCCESS; }

extern "C" int TEST_SESSION_QDMI_device_finalize() { return QDMI_SUCCESS; }

extern "C" int
TEST_SESSION_QDMI_device_session_alloc(QDMI_Device_Session* session) {
  if (session == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  // The QDMI C API transfers this allocation through an opaque raw handle.
  // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
  *session = new (std::nothrow) QDMI_Device_Session_impl_d;
  if (*session == nullptr) {
    return QDMI_ERROR_OUTOFMEM;
  }
  ++activeSessions();
  return QDMI_SUCCESS;
}

extern "C" int TEST_SESSION_QDMI_device_session_set_parameter(
    QDMI_Device_Session session, const QDMI_Device_Session_Parameter param,
    const size_t size, const void* value) {
  if (session == nullptr || (value != nullptr && size == 0)) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  if (session->initialized) {
    return QDMI_ERROR_BADSTATE;
  }
  if (param == QDMI_DEVICE_SESSION_PARAMETER_CHILDDEVICE) {
    if (value == nullptr || size != sizeof(QDMI_Child_Device)) {
      return QDMI_ERROR_INVALIDARGUMENT;
    }
    QDMI_Child_Device child = nullptr;
    std::memcpy(static_cast<void*>(&child), value, sizeof(QDMI_Child_Device));
    if (child != childDeviceHandle()) {
      return QDMI_ERROR_INVALIDARGUMENT;
    }
    session->child = child;
    return QDMI_SUCCESS;
  }
  if (value != nullptr) {
    session->parameters[param] = static_cast<const char*>(value);
  }
  return QDMI_SUCCESS;
}

extern "C" int
TEST_SESSION_QDMI_device_session_init(QDMI_Device_Session session) {
  if (session == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  if (session->initialized) {
    return QDMI_ERROR_BADSTATE;
  }
  session->initialized = true;
  return QDMI_SUCCESS;
}

extern "C" void
TEST_SESSION_QDMI_device_session_free(QDMI_Device_Session session) {
  if (session == nullptr) {
    return;
  }
  --activeSessions();
  // This releases the opaque handle allocated by device_session_alloc.
  // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
  delete session;
}

extern "C" int TEST_SESSION_QDMI_device_session_query_device_property(
    QDMI_Device_Session session, const QDMI_Device_Property prop,
    const size_t size, void* value, size_t* sizeRet) {
  if (session == nullptr || !session->initialized) {
    return QDMI_ERROR_BADSTATE;
  }
  if (prop == QDMI_DEVICE_PROPERTY_CHILDDEVICES) {
    if (session->child != nullptr ||
        parameter(session, QDMI_DEVICE_SESSION_PARAMETER_CUSTOM5) !=
            "with-child") {
      return QDMI_ERROR_NOTSUPPORTED;
    }
    constexpr auto required = sizeof(QDMI_Child_Device);
    if (sizeRet != nullptr) {
      *sizeRet = required;
    }
    if (value == nullptr) {
      return QDMI_SUCCESS;
    }
    if (size < required) {
      return QDMI_ERROR_INVALIDARGUMENT;
    }
    auto* const child = childDeviceHandle();
    std::memcpy(value, static_cast<const void*>(&child),
                sizeof(QDMI_Child_Device));
    return QDMI_SUCCESS;
  }
  if (prop == QDMI_DEVICE_PROPERTY_CUSTOM1) {
    const auto& operations = customOperationHandles();
    const auto required = operations.size() * sizeof(QDMI_Operation);
    if (sizeRet != nullptr) {
      *sizeRet = required;
    }
    if (value == nullptr) {
      return QDMI_SUCCESS;
    }
    if (size < required) {
      return QDMI_ERROR_INVALIDARGUMENT;
    }
    std::memcpy(value, static_cast<const void*>(operations.data()), required);
    return QDMI_SUCCESS;
  }
  if (prop == QDMI_DEVICE_PROPERTY_CUSTOM2) {
    if (sizeRet != nullptr) {
      *sizeRet = 0;
    }
    return QDMI_SUCCESS;
  }
  if (prop == QDMI_DEVICE_PROPERTY_CUSTOM3) {
    if (sizeRet != nullptr) {
      *sizeRet = sizeof(QDMI_Operation) + 1;
    }
    return value == nullptr ? QDMI_SUCCESS : QDMI_ERROR_INVALIDARGUMENT;
  }
  if (prop == QDMI_DEVICE_PROPERTY_STATUS) {
    return queryValue(
        deviceStatus(parameter(session, QDMI_DEVICE_SESSION_PARAMETER_CUSTOM4)),
        size, value, sizeRet);
  }
  if (prop != QDMI_DEVICE_PROPERTY_NAME) {
    return QDMI_ERROR_NOTSUPPORTED;
  }
  if (session->child != nullptr) {
    return queryString("child;active=" +
                           std::to_string(activeSessions().load()),
                       size, value, sizeRet);
  }
  const auto name =
      "base=" + parameter(session, QDMI_DEVICE_SESSION_PARAMETER_BASEURL) +
      ";token=" + parameter(session, QDMI_DEVICE_SESSION_PARAMETER_TOKEN) +
      ";custom1=" + parameter(session, QDMI_DEVICE_SESSION_PARAMETER_CUSTOM1) +
      ";custom2=" + parameter(session, QDMI_DEVICE_SESSION_PARAMETER_CUSTOM2) +
      ";active=" + std::to_string(activeSessions().load());
  return queryString(name, size, value, sizeRet);
}

extern "C" int TEST_SESSION_QDMI_device_session_query_site_property(
    QDMI_Device_Session /*session*/, QDMI_Site /*site*/,
    QDMI_Site_Property /*property*/, size_t /*size*/, void* /*value*/,
    size_t* /*sizeRet*/) {
  return QDMI_ERROR_NOTSUPPORTED;
}

extern "C" int TEST_SESSION_QDMI_device_session_query_operation_property(
    QDMI_Device_Session session, QDMI_Operation operation, size_t /*numSites*/,
    const QDMI_Site* /*sites*/, size_t /*numParams*/, const double* /*params*/,
    const QDMI_Operation_Property property, const size_t size, void* value,
    size_t* sizeRet) {
  if (session == nullptr || !session->initialized) {
    return QDMI_ERROR_BADSTATE;
  }
  const auto* const customOperation = findCustomOperation(operation);
  if (customOperation == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  if (property == QDMI_OPERATION_PROPERTY_NAME) {
    return queryString(customOperation->name, size, value, sizeRet);
  }
  if (property == QDMI_OPERATION_PROPERTY_QUBITSNUM) {
    return queryValue(customOperation->qubitsNum, size, value, sizeRet);
  }
  if (property == QDMI_OPERATION_PROPERTY_PARAMETERSNUM) {
    return queryValue(customOperation->parametersNum, size, value, sizeRet);
  }
  return QDMI_ERROR_NOTSUPPORTED;
}

extern "C" int
TEST_SESSION_QDMI_device_session_create_device_job(QDMI_Device_Session session,
                                                   QDMI_Device_Job* job) {
  if (session == nullptr || !session->initialized || job == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  // The QDMI C API transfers this allocation through an opaque raw handle.
  // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
  *job = new (std::nothrow) QDMI_Device_Job_impl_d{.session = session};
  return *job == nullptr ? QDMI_ERROR_OUTOFMEM : QDMI_SUCCESS;
}

extern "C" int TEST_SESSION_QDMI_device_session_retrieve_device_job_by_id(
    QDMI_Device_Session session, const char* jobId, QDMI_Device_Job* job) {
  if (session == nullptr || !session->initialized || jobId == nullptr ||
      *jobId == '\0' || job == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  if (std::strcmp(jobId, "session-job") != 0) {
    return QDMI_ERROR_NOTFOUND;
  }
  // The QDMI C API transfers this allocation through an opaque raw handle.
  // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
  *job = new (std::nothrow)
      QDMI_Device_Job_impl_d{.session = session, .retrieved = true};
  return *job == nullptr ? QDMI_ERROR_OUTOFMEM : QDMI_SUCCESS;
}

extern "C" int TEST_SESSION_QDMI_device_job_set_parameter(
    QDMI_Device_Job job, const QDMI_Device_Job_Parameter parameter,
    const size_t size, const void* value) {
  if (job == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  if (job->retrieved) {
    return QDMI_ERROR_BADSTATE;
  }
  if (parameter == QDMI_DEVICE_JOB_PARAMETER_PROGRAMFORMAT) {
    if (value == nullptr || size != sizeof(job->format)) {
      return QDMI_ERROR_INVALIDARGUMENT;
    }
    std::memcpy(&job->format, value, size);
  }
  if (parameter == QDMI_DEVICE_JOB_PARAMETER_SHOTSNUM &&
      job->format == QDMI_PROGRAM_FORMAT_CUSTOM1) {
    return QDMI_ERROR_NOTSUPPORTED;
  }
  return QDMI_SUCCESS;
}

extern "C" int TEST_SESSION_QDMI_device_job_query_property(
    QDMI_Device_Job job, const QDMI_Device_Job_Property prop, const size_t size,
    void* value, size_t* sizeRet) {
  if (job == nullptr || job->session == nullptr ||
      (prop != QDMI_DEVICE_JOB_PROPERTY_ID &&
       prop != QDMI_DEVICE_JOB_PROPERTY_QUEUEPOSITION)) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  if (prop == QDMI_DEVICE_JOB_PROPERTY_QUEUEPOSITION) {
    return QDMI_ERROR_NOTSUPPORTED;
  }
  return queryString("session-job", size, value, sizeRet);
}

extern "C" int TEST_SESSION_QDMI_device_job_submit(QDMI_Device_Job job) {
  if (job == nullptr || job->session == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  return job->retrieved ? QDMI_ERROR_BADSTATE : QDMI_SUCCESS;
}

extern "C" int TEST_SESSION_QDMI_device_job_cancel(QDMI_Device_Job /*job*/) {
  return QDMI_ERROR_NOTSUPPORTED;
}

extern "C" int TEST_SESSION_QDMI_device_job_check(QDMI_Device_Job /*job*/,
                                                  QDMI_Job_Status* /*status*/) {
  return QDMI_ERROR_NOTSUPPORTED;
}

extern "C" int TEST_SESSION_QDMI_device_job_wait(QDMI_Device_Job /*job*/,
                                                 size_t /*timeout*/) {
  return QDMI_ERROR_NOTSUPPORTED;
}

extern "C" int TEST_SESSION_QDMI_device_job_get_results(
    QDMI_Device_Job /*job*/, QDMI_Job_Result /*result*/, size_t /*size*/,
    void* /*value*/, size_t* /*sizeRet*/) {
  return QDMI_ERROR_NOTSUPPORTED;
}

extern "C" void TEST_SESSION_QDMI_device_job_free(QDMI_Device_Job job) {
  // This releases the opaque handle allocated by
  // device_session_create_device_job.
  // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
  delete job;
}
// NOLINTEND(readability-identifier-naming)
