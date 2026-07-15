/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "DeviceState.hpp"

#include "qdmi/DeviceRegistry.hpp"
#include "qdmi/common/Common.hpp"

#include <qdmi/device.h>
#include <spdlog/spdlog.h>

#include <cstddef>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace qdmi::detail {
namespace {
[[nodiscard]] auto openSession(const DeviceApi& api,
                               const SessionParameters& parameters,
                               const QDMI_Child_Device child)
    -> QDMI_Device_Session {
  QDMI_Device_Session session = nullptr;
  throwIfError(api.device_session_alloc(&session),
               "Allocating QDMI device session");
  try {
    const auto set = [&api, session](const std::optional<std::string>& value,
                                     const QDMI_Device_Session_Parameter key) {
      if (!value) {
        return;
      }
      const auto result = api.device_session_set_parameter(
          session, key, value->size() + 1, value->c_str());
      if (result == QDMI_ERROR_NOTSUPPORTED) {
        SPDLOG_INFO("QDMI device session parameter {} is not supported",
                    qdmi::toString(key));
        return;
      }
      throwIfError(result, "Setting QDMI device session parameter " +
                               std::string(qdmi::toString(key)));
    };
    set(parameters.baseUrl, QDMI_DEVICE_SESSION_PARAMETER_BASEURL);
    set(parameters.token, QDMI_DEVICE_SESSION_PARAMETER_TOKEN);
    if (parameters.authFile) {
      set(parameters.authFile->string(),
          QDMI_DEVICE_SESSION_PARAMETER_AUTHFILE);
    }
    set(parameters.authUrl, QDMI_DEVICE_SESSION_PARAMETER_AUTHURL);
    set(parameters.username, QDMI_DEVICE_SESSION_PARAMETER_USERNAME);
    set(parameters.password, QDMI_DEVICE_SESSION_PARAMETER_PASSWORD);
    set(parameters.custom1, QDMI_DEVICE_SESSION_PARAMETER_CUSTOM1);
    set(parameters.custom2, QDMI_DEVICE_SESSION_PARAMETER_CUSTOM2);
    set(parameters.custom3, QDMI_DEVICE_SESSION_PARAMETER_CUSTOM3);
    set(parameters.custom4, QDMI_DEVICE_SESSION_PARAMETER_CUSTOM4);
    set(parameters.custom5, QDMI_DEVICE_SESSION_PARAMETER_CUSTOM5);
    if (child != nullptr) {
      throwIfError(api.device_session_set_parameter(
                       session, QDMI_DEVICE_SESSION_PARAMETER_CHILDDEVICE,
                       sizeof(QDMI_Child_Device),
                       static_cast<const void*>(&child)),
                   "Selecting QDMI child device");
    }
    throwIfError(api.device_session_init(session),
                 "Initializing QDMI device session");
    return session;
  } catch (...) {
    api.device_session_free(session);
    throw;
  }
}
} // namespace

SessionState::SessionState(std::shared_ptr<const DeviceApi> deviceApi,
                           const SessionParameters& sessionParameters,
                           QDMI_Child_Device child,
                           std::shared_ptr<SessionState> parentSession)
    : api(std::move(deviceApi)),
      session(openSession(*api, sessionParameters, child)),
      parent(std::move(parentSession)) {}

SessionState::~SessionState() {
  if (session != nullptr) {
    api->device_session_free(session);
  }
}

DeviceState::DeviceState(std::shared_ptr<const DeviceApi> deviceApi,
                         const SessionParameters& sessionParameters,
                         QDMI_Child_Device child,
                         std::shared_ptr<SessionState> parentSession)
    : lifetime(std::make_shared<SessionState>(deviceApi, sessionParameters,
                                              child, std::move(parentSession))),
      api(std::move(deviceApi)), session(lifetime->session),
      parameters(sessionParameters) {
  try {
    size_t size = 0;
    const auto result = api->device_session_query_device_property(
        session, QDMI_DEVICE_PROPERTY_CHILDDEVICES, 0, nullptr, &size);
    if (result == QDMI_ERROR_NOTSUPPORTED) {
      return;
    }
    throwIfError(result, "Querying QDMI child devices");
    if (size % sizeof(QDMI_Child_Device) != 0) {
      throw std::runtime_error("QDMI device returned an invalid child list");
    }
    std::vector<QDMI_Child_Device> handles(size / sizeof(QDMI_Child_Device));
    if (size != 0) {
      throwIfError(api->device_session_query_device_property(
                       session, QDMI_DEVICE_PROPERTY_CHILDDEVICES, size,
                       static_cast<void*>(handles.data()), nullptr),
                   "Querying QDMI child devices");
    }
    children.reserve(handles.size());
    for (auto* handle : handles) {
      children.emplace_back(
          std::make_shared<DeviceState>(api, parameters, handle, lifetime));
    }
  } catch (...) {
    children.clear();
    lifetime.reset();
    session = nullptr;
    throw;
  }
}

DeviceState::~DeviceState() {
  children.clear();
  lifetime.reset();
}

JobState::JobState(std::shared_ptr<const DeviceState> deviceState)
    : device(std::move(deviceState)) {
  throwIfError(
      device->api->device_session_create_device_job(device->session, &job),
      "Creating QDMI device job");
}

JobState::~JobState() {
  if (job != nullptr) {
    device->api->device_job_free(job);
  }
}
} // namespace qdmi::detail
