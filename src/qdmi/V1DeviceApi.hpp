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

#include <qdmi/device.h>

#include <filesystem>
#include <string>

namespace qdmi::detail {

/// One loaded and initialized QDMI v1.3 device implementation.
struct V1DeviceApi {
  V1DeviceApi() = default;
  V1DeviceApi(const std::filesystem::path& library, const std::string& prefix);
  ~V1DeviceApi();

  V1DeviceApi(const V1DeviceApi&) = delete;
  V1DeviceApi& operator=(const V1DeviceApi&) = delete;
  V1DeviceApi(V1DeviceApi&&) = delete;
  V1DeviceApi& operator=(V1DeviceApi&&) = delete;

  // Keep the QDMI names so every stored signature visibly corresponds to the
  // selected ABI declaration.
  // NOLINTBEGIN(readability-identifier-naming)
  decltype(QDMI_device_session_alloc)* device_session_alloc = nullptr;
  decltype(QDMI_device_session_init)* device_session_init = nullptr;
  decltype(QDMI_device_session_free)* device_session_free = nullptr;
  decltype(QDMI_device_session_set_parameter)* device_session_set_parameter =
      nullptr;
  decltype(QDMI_device_session_create_device_job)*
      device_session_create_device_job = nullptr;
  decltype(QDMI_device_job_free)* device_job_free = nullptr;
  decltype(QDMI_device_job_set_parameter)* device_job_set_parameter = nullptr;
  decltype(QDMI_device_job_query_property)* device_job_query_property = nullptr;
  decltype(QDMI_device_job_submit)* device_job_submit = nullptr;
  decltype(QDMI_device_job_cancel)* device_job_cancel = nullptr;
  decltype(QDMI_device_job_check)* device_job_check = nullptr;
  decltype(QDMI_device_job_wait)* device_job_wait = nullptr;
  decltype(QDMI_device_job_get_results)* device_job_get_results = nullptr;
  decltype(QDMI_device_session_query_device_property)*
      device_session_query_device_property = nullptr;
  decltype(QDMI_device_session_query_site_property)*
      device_session_query_site_property = nullptr;
  decltype(QDMI_device_session_query_operation_property)*
      device_session_query_operation_property = nullptr;
  // NOLINTEND(readability-identifier-naming)

private:
  void* library_ = nullptr;
  decltype(QDMI_device_finalize)* finalize_ = nullptr;
  bool initialized_ = false;
};

} // namespace qdmi::detail
