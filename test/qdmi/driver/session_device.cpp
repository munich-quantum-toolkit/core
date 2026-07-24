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

#include <atomic>
#include <cstddef>
#include <cstring>
#include <new>
#include <string>
#include <unordered_map>

struct QDMI_Child_Device_impl_d {};

struct QDMI_Device_Session_impl_d {
  std::unordered_map<int, std::string> parameters;
  QDMI_Child_Device child = nullptr;
  bool initialized = false;
};

struct QDMI_Device_Job_impl_d {
  QDMI_Device_Session session = nullptr;
};

namespace {
std::atomic_size_t activeSessions = 0;

[[nodiscard]] auto parameter(const QDMI_Device_Session session,
                             const QDMI_Device_Session_Parameter key)
    -> std::string {
  if (const auto entry = session->parameters.find(key);
      entry != session->parameters.end()) {
    return entry->second;
  }
  return "<unset>";
}

[[nodiscard]] auto childDeviceHandle() -> QDMI_Child_Device {
  static QDMI_Child_Device_impl_d child;
  return &child;
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
} // namespace

extern "C" int TEST_SESSION_QDMI_device_initialize() { return QDMI_SUCCESS; }

extern "C" int TEST_SESSION_QDMI_device_finalize() { return QDMI_SUCCESS; }

extern "C" int
TEST_SESSION_QDMI_device_session_alloc(QDMI_Device_Session* session) {
  if (session == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  *session = new (std::nothrow) QDMI_Device_Session_impl_d;
  if (*session == nullptr) {
    return QDMI_ERROR_OUTOFMEM;
  }
  ++activeSessions;
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
    std::memcpy(&child, value, sizeof(child));
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
  if (session != nullptr) {
    --activeSessions;
    delete session;
  }
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
    const auto child = childDeviceHandle();
    std::memcpy(value, &child, required);
    return QDMI_SUCCESS;
  }
  if (prop != QDMI_DEVICE_PROPERTY_NAME) {
    return QDMI_ERROR_NOTSUPPORTED;
  }
  if (session->child != nullptr) {
    return queryString("child;active=" + std::to_string(activeSessions.load()),
                       size, value, sizeRet);
  }
  const auto name =
      "base=" + parameter(session, QDMI_DEVICE_SESSION_PARAMETER_BASEURL) +
      ";token=" + parameter(session, QDMI_DEVICE_SESSION_PARAMETER_TOKEN) +
      ";custom1=" + parameter(session, QDMI_DEVICE_SESSION_PARAMETER_CUSTOM1) +
      ";custom2=" + parameter(session, QDMI_DEVICE_SESSION_PARAMETER_CUSTOM2) +
      ";active=" + std::to_string(activeSessions.load());
  return queryString(name, size, value, sizeRet);
}

extern "C" int TEST_SESSION_QDMI_device_session_query_site_property(
    QDMI_Device_Session, QDMI_Site, QDMI_Site_Property, size_t, void*,
    size_t*) {
  return QDMI_ERROR_NOTSUPPORTED;
}

extern "C" int TEST_SESSION_QDMI_device_session_query_operation_property(
    QDMI_Device_Session, QDMI_Operation, size_t, const QDMI_Site*, size_t,
    const double*, QDMI_Operation_Property, size_t, void*, size_t*) {
  return QDMI_ERROR_NOTSUPPORTED;
}

extern "C" int
TEST_SESSION_QDMI_device_session_create_device_job(QDMI_Device_Session session,
                                                   QDMI_Device_Job* job) {
  if (session == nullptr || !session->initialized || job == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  *job = new (std::nothrow) QDMI_Device_Job_impl_d{session};
  return *job == nullptr ? QDMI_ERROR_OUTOFMEM : QDMI_SUCCESS;
}

extern "C" int TEST_SESSION_QDMI_device_job_set_parameter(
    QDMI_Device_Job job, QDMI_Device_Job_Parameter, size_t, const void*) {
  return job == nullptr ? QDMI_ERROR_INVALIDARGUMENT : QDMI_SUCCESS;
}

extern "C" int TEST_SESSION_QDMI_device_job_query_property(
    QDMI_Device_Job job, const QDMI_Device_Job_Property prop, const size_t size,
    void* value, size_t* sizeRet) {
  if (job == nullptr || job->session == nullptr ||
      prop != QDMI_DEVICE_JOB_PROPERTY_ID) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  return queryString("session-job", size, value, sizeRet);
}

extern "C" int TEST_SESSION_QDMI_device_job_submit(QDMI_Device_Job job) {
  if (job == nullptr || job->session == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  return QDMI_SUCCESS;
}

extern "C" int TEST_SESSION_QDMI_device_job_cancel(QDMI_Device_Job) {
  return QDMI_ERROR_NOTSUPPORTED;
}

extern "C" int TEST_SESSION_QDMI_device_job_check(QDMI_Device_Job,
                                                  QDMI_Job_Status*) {
  return QDMI_ERROR_NOTSUPPORTED;
}

extern "C" int TEST_SESSION_QDMI_device_job_wait(QDMI_Device_Job, size_t) {
  return QDMI_ERROR_NOTSUPPORTED;
}

extern "C" int TEST_SESSION_QDMI_device_job_get_results(QDMI_Device_Job,
                                                        QDMI_Job_Result, size_t,
                                                        void*, size_t*) {
  return QDMI_ERROR_NOTSUPPORTED;
}

extern "C" void TEST_SESSION_QDMI_device_job_free(QDMI_Device_Job job) {
  delete job;
}
