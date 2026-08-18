/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "qdmi/DeviceManager.hpp"

#include "DefaultDeviceRegistry.hpp"
#include "DeviceApi.hpp"
#include "DeviceState.hpp"
#include "qdmi/Device.hpp"
#include "qdmi/DeviceRegistry.hpp"
#include "qdmi/SessionConfig.hpp"

#include <algorithm>
#include <exception>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace qdmi {
namespace {
template <class T>
void overlayValue(std::optional<T>& target, const std::optional<T>& source) {
  if (source) {
    target = source;
  }
}

void overlay(DeviceSessionConfig& target, const DeviceSessionConfig& source) {
  overlayValue(target.baseUrl, source.baseUrl);
  overlayValue(target.token, source.token);
  overlayValue(target.authFile, source.authFile);
  overlayValue(target.authUrl, source.authUrl);
  overlayValue(target.username, source.username);
  overlayValue(target.password, source.password);
  overlayValue(target.deviceConfiguration, source.deviceConfiguration);
  overlayValue(target.custom1, source.custom1);
  overlayValue(target.custom2, source.custom2);
  overlayValue(target.custom3, source.custom3);
  overlayValue(target.custom4, source.custom4);
  overlayValue(target.custom5, source.custom5);
}
} // namespace

DeviceManager::DeviceManager()
    : registry_(detail::snapshotDefaultDeviceRegistry()) {}

DeviceManager::DeviceManager(DeviceRegistry registry)
    : registry_(std::move(registry)) {}

Device DeviceManager::open(const std::string_view id,
                           const DeviceSessionConfig& sessionOverrides) const {
  const auto& available = registry_.definitions();
  const auto definition =
      std::ranges::find(available, id, &DeviceDefinition::id);
  if (definition == available.end()) {
    throw std::out_of_range("Unknown QDMI device ID '" + std::string(id) + "'");
  }
  const auto library =
      detail::loadDeviceApi(definition->library, definition->prefix);
  auto parameters = definition->session;
  overlay(parameters, sessionOverrides);
  return Device(std::make_shared<detail::DeviceState>(library, parameters));
}

OpenAllResult
DeviceManager::openAll(const DeviceSessionConfig& sessionOverrides) const {
  OpenAllResult result;
  for (const auto& definition : definitions()) {
    try {
      result.devices.emplace(definition.id,
                             open(definition.id, sessionOverrides));
    } catch (const std::exception& error) {
      result.errors.emplace(definition.id, error.what());
    }
  }
  return result;
}

Device openDevice(const std::string_view id,
                  const DeviceSessionConfig& sessionOverrides) {
  return DeviceManager{}.open(id, sessionOverrides);
}
} // namespace qdmi
