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
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

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
  bool submitted = false;
  std::string id = "session-job";
  QDMI_Program_Format format{};
  size_t shots = 0U;
  std::vector<std::byte> program;
};

namespace {
[[nodiscard]] auto activeSessions() -> std::atomic_size_t& {
  static std::atomic_size_t sessions = 0;
  return sessions;
}

[[nodiscard]] auto freedJobs() -> std::atomic_size_t& {
  static std::atomic_size_t jobs = 0;
  return jobs;
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
      .name = "custom-rx",
      .qubitsNum = 1,
      .parametersNum = 1,
  };
  static QDMI_Operation_impl_d controlledNot{
      .name = "custom-cx",
      .qubitsNum = 2,
      .parametersNum = 0,
  };
  static const std::array<QDMI_Operation, 2> OPERATIONS{
      &rotate,
      &controlledNot,
  };
  return OPERATIONS;
}

[[nodiscard]] auto findCustomOperation(QDMI_Operation operation)
    -> const QDMI_Operation_impl_d* {
  for (const auto* const handle : customOperationHandles()) {
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

auto queryBytes(const std::span<const std::byte> result, const size_t size,
                void* value, size_t* sizeRet) -> int {
  if (sizeRet != nullptr) {
    *sizeRet = result.size();
  }
  if (value == nullptr) {
    return QDMI_SUCCESS;
  }
  if (size < result.size()) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  if (!result.empty()) {
    std::memcpy(value, result.data(), result.size());
  }
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
    const auto* const child = childDeviceHandle();
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
  if (prop == QDMI_DEVICE_PROPERTY_CUSTOM5) {
    return queryValue(freedJobs().load(), size, value, sizeRet);
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

extern "C" int TEST_SESSION_QDMI_device_session_query_program_features(
    QDMI_Device_Session /*session*/, const QDMI_Program_Format* /*format*/,
    size_t /*size*/, void* /*value*/, size_t* /*sizeRet*/) {
  return QDMI_ERROR_NOTSUPPORTED;
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
  if (std::strcmp(jobId, "session-job") != 0 &&
      std::strcmp(jobId, "malformed-text-job") != 0) {
    return QDMI_ERROR_NOTFOUND;
  }
  // The QDMI C API transfers this allocation through an opaque raw handle.
  // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
  auto* retrieved = new (std::nothrow)
      QDMI_Device_Job_impl_d{.session = session, .retrieved = true};
  if (retrieved == nullptr) {
    return QDMI_ERROR_OUTOFMEM;
  }
  retrieved->id = jobId;
  retrieved->format = {
      .version = QDMI_MAKE_VERSION(3U, 0U, 0U),
      .encoding = QDMI_PROGRAM_ENCODING_TEXT,
      .id = "openqasm",
      .profile = "",
  };
  retrieved->program = {
      std::byte{'x'},
      std::byte{0},
      std::byte{'y'},
      std::byte{0},
  };
  *job = retrieved;
  return QDMI_SUCCESS;
}

extern "C" int TEST_SESSION_QDMI_device_job_set_parameter(
    QDMI_Device_Job job, const QDMI_Device_Job_Parameter parameter,
    const size_t size, const void* value) {
  if (job == nullptr || (value != nullptr && size == 0U)) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  if (job->retrieved || job->submitted) {
    return QDMI_ERROR_BADSTATE;
  }
  if (parameter == QDMI_DEVICE_JOB_PARAMETER_PROGRAMFORMAT) {
    if (value == nullptr || size != sizeof(QDMI_Program_Format)) {
      return QDMI_ERROR_INVALIDARGUMENT;
    }
    job->format = *static_cast<const QDMI_Program_Format*>(value);
    return QDMI_SUCCESS;
  }
  if (parameter == QDMI_DEVICE_JOB_PARAMETER_PROGRAM) {
    if (value == nullptr) {
      return QDMI_SUCCESS;
    }
    const std::span bytes{static_cast<const std::byte*>(value), size};
    job->program.assign(bytes.begin(), bytes.end());
    return QDMI_SUCCESS;
  }
  if (parameter == QDMI_DEVICE_JOB_PARAMETER_SHOTSNUM) {
    if (job->format.encoding == QDMI_PROGRAM_ENCODING_BINARY) {
      return QDMI_ERROR_NOTSUPPORTED;
    }
    if (value == nullptr || size != sizeof(size_t)) {
      return QDMI_ERROR_INVALIDARGUMENT;
    }
    job->shots = *static_cast<const size_t*>(value);
    return QDMI_SUCCESS;
  }
  return QDMI_ERROR_NOTSUPPORTED;
}

extern "C" int TEST_SESSION_QDMI_device_job_query_property(
    QDMI_Device_Job job, const QDMI_Device_Job_Property prop, const size_t size,
    void* value, size_t* sizeRet) {
  if (job == nullptr || job->session == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  if (prop == QDMI_DEVICE_JOB_PROPERTY_QUEUEPOSITION) {
    return QDMI_ERROR_NOTSUPPORTED;
  }
  if (prop == QDMI_DEVICE_JOB_PROPERTY_ID) {
    return queryString(job->id, size, value, sizeRet);
  }
  if (prop == QDMI_DEVICE_JOB_PROPERTY_SHOTSNUM && !job->retrieved) {
    return queryValue(job->shots, size, value, sizeRet);
  }
  if (prop == QDMI_DEVICE_JOB_PROPERTY_PROGRAMFORMAT) {
    return queryValue(job->format, size, value, sizeRet);
  }
  if (prop == QDMI_DEVICE_JOB_PROPERTY_PROGRAM) {
    return queryBytes(job->program, size, value, sizeRet);
  }
  return QDMI_ERROR_NOTSUPPORTED;
}

extern "C" int TEST_SESSION_QDMI_device_job_submit(QDMI_Device_Job job) {
  if (job == nullptr || job->session == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  if (job->retrieved || job->submitted) {
    return QDMI_ERROR_BADSTATE;
  }
  if (job->program.empty()) {
    return QDMI_ERROR_BADSTATE;
  }
  job->submitted = true;
  return QDMI_SUCCESS;
}

extern "C" int TEST_SESSION_QDMI_device_job_cancel(QDMI_Device_Job /*job*/) {
  return QDMI_ERROR_NOTSUPPORTED;
}

extern "C" int TEST_SESSION_QDMI_device_job_check(QDMI_Device_Job job,
                                                  QDMI_Job_Status* status) {
  if (job == nullptr || status == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  *status = job->submitted || job->retrieved ? QDMI_JOB_STATUS_DONE
                                             : QDMI_JOB_STATUS_CREATED;
  return QDMI_SUCCESS;
}

extern "C" int TEST_SESSION_QDMI_device_job_wait(QDMI_Device_Job job,
                                                 size_t /*timeout*/) {
  return job != nullptr && (job->submitted || job->retrieved)
             ? QDMI_SUCCESS
             : QDMI_ERROR_BADSTATE;
}

extern "C" int TEST_SESSION_QDMI_device_job_get_results(
    QDMI_Device_Job job, const QDMI_Job_Result result, const size_t size,
    void* value, size_t* sizeRet) {
  if (job == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  if (!job->retrieved && !job->submitted) {
    return QDMI_ERROR_BADSTATE;
  }
  if (result == QDMI_JOB_RESULT_SHOTS) {
    return queryString("10,01", size, value, sizeRet);
  }
  if (result == QDMI_JOB_RESULT_HIST_KEYS) {
    constexpr std::array MALFORMED_KEYS{
        std::byte{'1'},
        std::byte{0},
        std::byte{'0'},
        std::byte{0},
    };
    return queryBytes(MALFORMED_KEYS, size, value, sizeRet);
  }
  if (result == QDMI_JOB_RESULT_HIST_VALUES) {
    constexpr std::array<size_t, 2> VALUES{1U, 1U};
    return queryBytes(std::as_bytes(std::span{VALUES}), size, value, sizeRet);
  }
  if (result == QDMI_JOB_RESULT_PROGRAMOUTPUT) {
    return queryBytes(job->program, size, value, sizeRet);
  }
  return QDMI_ERROR_NOTSUPPORTED;
}

extern "C" void TEST_SESSION_QDMI_device_job_free(QDMI_Device_Job job) {
  if (job != nullptr) {
    freedJobs().fetch_add(1, std::memory_order_relaxed);
  }
  // This releases the opaque handle allocated by
  // device_session_create_device_job.
  // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
  delete job;
}
// NOLINTEND(readability-identifier-naming)
