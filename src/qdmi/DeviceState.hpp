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

#include "DeviceApi.hpp"
#include "qdmi/Device.hpp"
#include "qdmi/SessionConfig.hpp"

#include <qdmi/device.h>

#include <memory>
#include <utility>
#include <vector>

namespace qdmi::detail {

/// Owns one native QDMI session and keeps its implementation loaded.
///
/// A child session retains its parent because QDMI child handles originate
/// from the parent session.
struct SessionState {
  std::shared_ptr<const DeviceApi> api;
  QDMI_Device_Session session = nullptr;
  std::shared_ptr<SessionState> parent;

  SessionState(std::shared_ptr<const DeviceApi> deviceApi,
               const DeviceSessionConfig& sessionParameters,
               QDMI_Child_Device child,
               std::shared_ptr<SessionState> parentSession);
  ~SessionState();
  SessionState(const SessionState&) = delete;
  SessionState& operator=(const SessionState&) = delete;
  SessionState(SessionState&&) = delete;
  SessionState& operator=(SessionState&&) = delete;
};

/// Shared state retained by a Device and every Site or Operation derived from
/// it. Jobs retain the complete DeviceState separately.
struct DeviceState {
  std::shared_ptr<SessionState> sessionState;
  DeviceSessionConfig parameters;
  std::vector<std::shared_ptr<DeviceState>> children;

  DeviceState(std::shared_ptr<const DeviceApi> deviceApi,
              const DeviceSessionConfig& sessionParameters,
              QDMI_Child_Device child = nullptr,
              std::shared_ptr<SessionState> parentSession = nullptr);
  DeviceState(const DeviceState&) = delete;
  DeviceState& operator=(const DeviceState&) = delete;
  DeviceState(DeviceState&&) = delete;
  DeviceState& operator=(DeviceState&&) = delete;

  [[nodiscard]] const DeviceApi& api() const { return *sessionState->api; }
  [[nodiscard]] QDMI_Device_Session session() const {
    return sessionState->session;
  }
};

struct JobState {
  std::shared_ptr<const DeviceState> device;
  QDMI_Device_Job job = nullptr;

  explicit JobState(std::shared_ptr<const DeviceState> deviceState);
  JobState(std::shared_ptr<const DeviceState> deviceState,
           QDMI_Device_Job existingJob);
  ~JobState();
  JobState(const JobState&) = delete;
  JobState& operator=(const JobState&) = delete;
  JobState(JobState&&) = delete;
  JobState& operator=(JobState&&) = delete;
};

/// Test-only construction access for scripted QDMI implementations.
struct DeviceFactory {
  [[nodiscard]] static Device
  create(std::shared_ptr<const DeviceApi> api,
         const DeviceSessionConfig& parameters = {}) {
    return Device(std::make_shared<DeviceState>(std::move(api), parameters));
  }
};

} // namespace qdmi::detail
