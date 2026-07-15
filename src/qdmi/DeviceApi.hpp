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

#include "qdmi/DeviceManager.hpp"

#include <qdmi/device.h>

#include <cstddef>
#include <filesystem>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace qdmi::detail {

template <class Function> struct ContextFunction;

/// Add the immutable adapter context to an existing QDMI function signature.
template <class Result, class... Args> struct ContextFunction<Result(Args...)> {
  using type = Result (*)(const void*, Args...);
};

template <class Function>
using ContextFunctionT = typename ContextFunction<Function>::type;

/// Private context/function table for the single supported QDMI ABI.
struct DeviceApi {
  /// Operations consumed by the object model. Every function receives the
  /// matching immutable context as its first argument.
  struct Functions {
    QDMI_Device_Session (*openSession)(const void*, const SessionParameters&,
                                       QDMI_Child_Device) = nullptr;
    void (*closeSession)(const void*, QDMI_Device_Session) noexcept = nullptr;
    QDMI_Device_Job (*createJob)(const void*, QDMI_Device_Session) = nullptr;
    void (*freeJob)(const void*, QDMI_Device_Job) noexcept = nullptr;
    ContextFunctionT<decltype(QDMI_device_job_set_parameter)> setJobParameter =
        nullptr;
    ContextFunctionT<decltype(QDMI_device_job_query_property)>
        queryJobProperty = nullptr;
    void (*submitJob)(const void*, QDMI_Device_Job) = nullptr;
    void (*cancelJob)(const void*, QDMI_Device_Job) = nullptr;
    QDMI_Job_Status (*checkJob)(const void*, QDMI_Device_Job) = nullptr;
    bool (*waitJob)(const void*, QDMI_Device_Job, size_t) = nullptr;
    ContextFunctionT<decltype(QDMI_device_job_get_results)> getJobResult =
        nullptr;
    ContextFunctionT<decltype(QDMI_device_session_query_device_property)>
        queryDevice = nullptr;
    ContextFunctionT<decltype(QDMI_device_session_query_site_property)>
        querySite = nullptr;
    ContextFunctionT<decltype(QDMI_device_session_query_operation_property)>
        queryOperation = nullptr;
  };

  std::shared_ptr<const void> context;
  Functions functions;
};

struct SessionState {
  std::shared_ptr<const DeviceApi> api;
  QDMI_Device_Session session = nullptr;
  std::shared_ptr<SessionState> parent;

  SessionState(std::shared_ptr<const DeviceApi> deviceApi,
               const SessionParameters& sessionParameters,
               QDMI_Child_Device child,
               std::shared_ptr<SessionState> parentSession);
  ~SessionState();
  SessionState(const SessionState&) = delete;
  SessionState& operator=(const SessionState&) = delete;
  SessionState(SessionState&&) = delete;
  SessionState& operator=(SessionState&&) = delete;
};

struct DeviceState {
  std::shared_ptr<SessionState> lifetime;
  std::shared_ptr<const DeviceApi> api;
  QDMI_Device_Session session = nullptr;
  SessionParameters parameters;
  std::vector<std::shared_ptr<DeviceState>> children;

  DeviceState(std::shared_ptr<const DeviceApi> deviceApi,
              const SessionParameters& sessionParameters,
              QDMI_Child_Device child = nullptr,
              std::shared_ptr<SessionState> parentSession = nullptr);
  ~DeviceState();
  DeviceState(const DeviceState&) = delete;
  DeviceState& operator=(const DeviceState&) = delete;
  DeviceState(DeviceState&&) = delete;
  DeviceState& operator=(DeviceState&&) = delete;
};

struct JobState {
  std::shared_ptr<const DeviceState> device;
  QDMI_Device_Job job = nullptr;

  explicit JobState(std::shared_ptr<const DeviceState> deviceState);
  ~JobState();
  JobState(const JobState&) = delete;
  JobState& operator=(const JobState&) = delete;
  JobState(JobState&&) = delete;
  JobState& operator=(JobState&&) = delete;
};

/** Test-only/private construction access for ABI adapters. */
struct DeviceFactory {
  [[nodiscard]] static auto create(std::shared_ptr<const DeviceApi> api,
                                   const SessionParameters& parameters = {})
      -> Device {
    return Device(std::make_shared<DeviceState>(std::move(api), parameters));
  }
};

[[nodiscard]] auto makeV1DeviceApi(const std::filesystem::path& library,
                                   const std::string& prefix)
    -> std::shared_ptr<const DeviceApi>;

} // namespace qdmi::detail
