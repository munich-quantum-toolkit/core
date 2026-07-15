/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "DeviceApi.hpp"

#include "qdmi/common/Common.hpp"

#include <qdmi/device.h>

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace qdmi::detail {
SessionState::SessionState(std::shared_ptr<const DeviceApi> deviceApi,
                           const SessionParameters& sessionParameters,
                           QDMI_Child_Device child,
                           std::shared_ptr<SessionState> parentSession)
    : api(std::move(deviceApi)),
      session(api->functions.openSession(api->context.get(), sessionParameters,
                                         child)),
      parent(std::move(parentSession)) {}

SessionState::~SessionState() {
  api->functions.closeSession(api->context.get(), session);
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
    const auto result = api->functions.queryDevice(
        api->context.get(), session, QDMI_DEVICE_PROPERTY_CHILDDEVICES, 0,
        nullptr, &size);
    if (result == QDMI_ERROR_NOTSUPPORTED) {
      return;
    }
    throwIfError(result, "Querying QDMI child devices");
    if (size % sizeof(QDMI_Child_Device) != 0) {
      throw std::runtime_error("QDMI device returned an invalid child list");
    }
    std::vector<QDMI_Child_Device> handles(size / sizeof(QDMI_Child_Device));
    if (size != 0) {
      throwIfError(api->functions.queryDevice(
                       api->context.get(), session,
                       QDMI_DEVICE_PROPERTY_CHILDDEVICES, size,
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
    : device(std::move(deviceState)),
      job(device->api->functions.createJob(device->api->context.get(),
                                           device->session)) {}

JobState::~JobState() {
  device->api->functions.freeJob(device->api->context.get(), job);
}
} // namespace qdmi::detail
