/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include <qdmi/client.h>

#include <cstdint>

#ifdef TEST_FULL_CLIENT
#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <new>
#include <span>
#include <string>
#include <string_view>

#ifndef _WIN32
/// NOLINTNEXTLINE(modernize-deprecated-headers)
#include <stdlib.h>
#endif

struct QDMI_Site_impl_d {
  size_t index = 0;
};

struct QDMI_Operation_impl_d {};

struct QDMI_Session_impl_d;

struct QDMI_Device_impl_d {
  QDMI_Session_impl_d* session;
  QDMI_Site_impl_d site;
  QDMI_Operation_impl_d operation;
};

struct QDMI_Session_impl_d {
  std::string token;
  bool initialized = false;
  QDMI_Device_impl_d device{.session = this};
};

namespace {
constexpr auto DEVICE_ID = "test.fake.client";
constexpr auto FAIL_ALLOCATION = "MQT_CORE_QDMI_FAKE_FAIL_ALLOCATION";

[[nodiscard]] auto failAllocation() -> bool {
  const auto* value = std::getenv(FAIL_ALLOCATION);
  if (value == nullptr || std::string_view{value} != "1") {
    return false;
  }
#ifdef _WIN32
  static_cast<void>(_putenv_s(FAIL_ALLOCATION, "0"));
#else
  static_cast<void>(setenv(FAIL_ALLOCATION, "0", 1));
#endif
  return true;
}

auto queryString(const std::string_view result, const size_t size, void* value,
                 size_t* sizeRet) -> int {
  const auto required = result.size() + 1U;
  if (sizeRet != nullptr) {
    *sizeRet = required;
  }
  if (value == nullptr) {
    return QDMI_SUCCESS;
  }
  if (size < required) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  std::memcpy(value, result.data(), result.size());
  const std::span output{static_cast<char*>(value), size};
  output[result.size()] = '\0';
  return QDMI_SUCCESS;
}

template <class T>
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
  std::memcpy(value, static_cast<const void*>(std::addressof(result)),
              sizeof(T));
  return QDMI_SUCCESS;
}
} // namespace
#endif

/// NOLINTBEGIN(readability-identifier-naming, readability-named-parameter)
uint32_t QDMI_driver_get_client_abi_version() { return TEST_CLIENT_ABI; }

#ifdef TEST_FULL_CLIENT
int QDMI_session_alloc(QDMI_Session* session) {
  if (session == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  *session = nullptr;
  if (failAllocation()) {
    return QDMI_ERROR_OUTOFMEM;
  }
  /// NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
  *session = new (std::nothrow) QDMI_Session_impl_d;
  return *session == nullptr ? QDMI_ERROR_OUTOFMEM : QDMI_SUCCESS;
}

int QDMI_session_set_parameter(QDMI_Session session,
                               const QDMI_Session_Parameter param,
                               const size_t size, const void* value) {
  if (session == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  if (session->initialized) {
    return QDMI_ERROR_BADSTATE;
  }
  if (param != QDMI_SESSION_PARAMETER_TOKEN) {
    return QDMI_ERROR_NOTSUPPORTED;
  }
  if (value == nullptr) {
    return QDMI_SUCCESS;
  }
  const std::string_view text{static_cast<const char*>(value), size};
  if (text.empty() || text.back() != '\0' ||
      text.substr(0U, text.size() - 1U).find('\0') != std::string_view::npos) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  try {
    session->token.assign(text.data(), text.size() - 1U);
  } catch (const std::bad_alloc&) {
    return QDMI_ERROR_OUTOFMEM;
  }
  return QDMI_SUCCESS;
}

int QDMI_session_init(QDMI_Session session) {
  if (session == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  if (session->initialized) {
    return QDMI_ERROR_BADSTATE;
  }
  session->initialized = true;
  return QDMI_SUCCESS;
}

int QDMI_session_query_session_property(QDMI_Session session,
                                        const QDMI_Session_Property prop,
                                        const size_t size, void* value,
                                        size_t* sizeRet) {
  if (session == nullptr || !session->initialized) {
    return QDMI_ERROR_BADSTATE;
  }
  if (prop != QDMI_SESSION_PROPERTY_DEVICES) {
    return QDMI_ERROR_NOTSUPPORTED;
  }
  if (session->token == "odd-size") {
    if (sizeRet != nullptr) {
      *sizeRet = sizeof(QDMI_Device) + 1U;
    }
    return value == nullptr ? QDMI_SUCCESS : QDMI_ERROR_INVALIDARGUMENT;
  }
  QDMI_Device device = &session->device;
  return queryValue(device, size, value, sizeRet);
}

void QDMI_session_free(QDMI_Session session) {
  /// NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
  delete session;
}

int QDMI_device_query_device_property(QDMI_Device device,
                                      const QDMI_Device_Property prop,
                                      const size_t size, void* value,
                                      size_t* sizeRet) {
  if (device == nullptr || device->session == nullptr ||
      !device->session->initialized) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  if (prop == QDMI_DEVICE_PROPERTY_ID) {
    return queryString(DEVICE_ID, size, value, sizeRet);
  }
  if (prop == QDMI_DEVICE_PROPERTY_NAME) {
    return queryString(device->session->token, size, value, sizeRet);
  }
  if (prop == QDMI_DEVICE_PROPERTY_QUBITSNUM) {
    constexpr size_t qubits = 1U;
    return queryValue(qubits, size, value, sizeRet);
  }
  if (prop == QDMI_DEVICE_PROPERTY_SITES) {
    if (device->session->token == "odd-device-size") {
      if (sizeRet != nullptr) {
        *sizeRet = sizeof(QDMI_Site) + 1U;
      }
      return value == nullptr ? QDMI_SUCCESS : QDMI_ERROR_INVALIDARGUMENT;
    }
    QDMI_Site site = &device->site;
    return queryValue(site, size, value, sizeRet);
  }
  if (prop == QDMI_DEVICE_PROPERTY_OPERATIONS) {
    QDMI_Operation operation = &device->operation;
    return queryValue(operation, size, value, sizeRet);
  }
  return QDMI_ERROR_NOTSUPPORTED;
}

int QDMI_device_query_site_property(QDMI_Device device, QDMI_Site site,
                                    const QDMI_Site_Property prop,
                                    const size_t size, void* value,
                                    size_t* sizeRet) {
  if (device == nullptr || site != &device->site) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  if (prop != QDMI_SITE_PROPERTY_INDEX) {
    return QDMI_ERROR_NOTSUPPORTED;
  }
  return queryValue(site->index, size, value, sizeRet);
}

int QDMI_device_query_operation_property(
    QDMI_Device device, QDMI_Operation operation, size_t, const QDMI_Site*,
    size_t, const double*, const QDMI_Operation_Property prop,
    const size_t size, void* value, size_t* sizeRet) {
  if (device == nullptr || operation != &device->operation) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  if (prop == QDMI_OPERATION_PROPERTY_SITES &&
      device->session->token == "odd-operation-size") {
    if (sizeRet != nullptr) {
      *sizeRet = sizeof(QDMI_Site) + 1U;
    }
    return value == nullptr ? QDMI_SUCCESS : QDMI_ERROR_INVALIDARGUMENT;
  }
  return QDMI_ERROR_NOTSUPPORTED;
}

int QDMI_device_create_job(QDMI_Device, QDMI_Job* job) {
  if (job != nullptr) {
    *job = nullptr;
  }
  return QDMI_ERROR_NOTSUPPORTED;
}

int QDMI_session_retrieve_job_by_id(QDMI_Device, const char*, QDMI_Job* job) {
  if (job != nullptr) {
    *job = nullptr;
  }
  return QDMI_ERROR_NOTSUPPORTED;
}

int QDMI_job_set_parameter(QDMI_Job, QDMI_Job_Parameter, size_t, const void*) {
  return QDMI_ERROR_NOTSUPPORTED;
}

int QDMI_job_query_property(QDMI_Job, QDMI_Job_Property, size_t, void*,
                            size_t*) {
  return QDMI_ERROR_NOTSUPPORTED;
}

int QDMI_job_submit(QDMI_Job) { return QDMI_ERROR_NOTSUPPORTED; }

int QDMI_job_cancel(QDMI_Job) { return QDMI_ERROR_NOTSUPPORTED; }

int QDMI_job_check(QDMI_Job, QDMI_Job_Status*) {
  return QDMI_ERROR_NOTSUPPORTED;
}

int QDMI_job_wait(QDMI_Job, size_t) { return QDMI_ERROR_NOTSUPPORTED; }

int QDMI_job_get_results(QDMI_Job, QDMI_Job_Result, size_t, void*,
                         size_t*) {
  return QDMI_ERROR_NOTSUPPORTED;
}

void QDMI_job_free(QDMI_Job) {}
#endif
/// NOLINTEND(readability-identifier-naming, readability-named-parameter)
