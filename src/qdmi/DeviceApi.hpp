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

/** Private, ABI-neutral interface used by the public QDMI object model. */
class DeviceApi {
public:
  DeviceApi() = default;
  DeviceApi(const DeviceApi&) = delete;
  DeviceApi& operator=(const DeviceApi&) = delete;
  DeviceApi(DeviceApi&&) = delete;
  DeviceApi& operator=(DeviceApi&&) = delete;
  virtual ~DeviceApi() = default;

  [[nodiscard]] virtual auto
  openSession(const SessionParameters& parameters,
              QDMI_Child_Device child = nullptr) const
      -> QDMI_Device_Session = 0;
  virtual void closeSession(QDMI_Device_Session session) const noexcept = 0;

  [[nodiscard]] virtual auto createJob(QDMI_Device_Session session) const
      -> QDMI_Device_Job = 0;
  virtual void freeJob(QDMI_Device_Job job) const noexcept = 0;
  [[nodiscard]] virtual auto
  setJobParameter(QDMI_Device_Job job, QDMI_Device_Job_Parameter parameter,
                  size_t size, const void* value) const -> int = 0;
  [[nodiscard]] virtual auto queryJobProperty(QDMI_Device_Job job,
                                              QDMI_Device_Job_Property property,
                                              size_t size, void* value,
                                              size_t* sizeRet) const -> int = 0;
  virtual void submitJob(QDMI_Device_Job job) const = 0;
  virtual void cancelJob(QDMI_Device_Job job) const = 0;
  [[nodiscard]] virtual auto checkJob(QDMI_Device_Job job) const
      -> QDMI_Job_Status = 0;
  [[nodiscard]] virtual auto waitJob(QDMI_Device_Job job, size_t timeout) const
      -> bool = 0;
  [[nodiscard]] virtual auto getJobResult(QDMI_Device_Job job,
                                          QDMI_Job_Result result, size_t size,
                                          void* data, size_t* sizeRet) const
      -> int = 0;

  [[nodiscard]] virtual auto queryDevice(QDMI_Device_Session session,
                                         QDMI_Device_Property property,
                                         size_t size, void* value,
                                         size_t* sizeRet) const -> int = 0;
  [[nodiscard]] virtual auto querySite(QDMI_Device_Session session,
                                       QDMI_Site site,
                                       QDMI_Site_Property property, size_t size,
                                       void* value, size_t* sizeRet) const
      -> int = 0;
  [[nodiscard]] virtual auto
  queryOperation(QDMI_Device_Session session, QDMI_Operation operation,
                 size_t numSites, const QDMI_Site* sites, size_t numParams,
                 const double* params, QDMI_Operation_Property property,
                 size_t size, void* value, size_t* sizeRet) const -> int = 0;
};

struct SessionState {
  std::shared_ptr<DeviceApi> api;
  QDMI_Device_Session session = nullptr;
  std::shared_ptr<SessionState> parent;

  SessionState(std::shared_ptr<DeviceApi> deviceApi,
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
  std::shared_ptr<DeviceApi> api;
  QDMI_Device_Session session = nullptr;
  SessionParameters parameters;
  std::vector<std::shared_ptr<DeviceState>> children;

  DeviceState(std::shared_ptr<DeviceApi> deviceApi,
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
  std::shared_ptr<DeviceState> device;
  QDMI_Device_Job job = nullptr;

  explicit JobState(std::shared_ptr<DeviceState> deviceState);
  ~JobState();
  JobState(const JobState&) = delete;
  JobState& operator=(const JobState&) = delete;
  JobState(JobState&&) = delete;
  JobState& operator=(JobState&&) = delete;
};

/** Test-only/private construction access for ABI adapters. */
struct DeviceFactory {
  [[nodiscard]] static auto create(std::shared_ptr<DeviceApi> api,
                                   const SessionParameters& parameters = {})
      -> Device {
    return Device(std::make_shared<DeviceState>(std::move(api), parameters));
  }
};

[[nodiscard]] auto makeV1DeviceApi(const std::filesystem::path& library,
                                   const std::string& prefix)
    -> std::shared_ptr<DeviceApi>;

} // namespace qdmi::detail
