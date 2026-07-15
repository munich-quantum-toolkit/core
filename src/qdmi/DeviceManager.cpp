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

#include "DeviceApi.hpp"

#include <algorithm>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
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

void overlay(SessionParameters& target, const SessionParameters& source) {
  overlayValue(target.baseUrl, source.baseUrl);
  overlayValue(target.token, source.token);
  overlayValue(target.authFile, source.authFile);
  overlayValue(target.authUrl, source.authUrl);
  overlayValue(target.username, source.username);
  overlayValue(target.password, source.password);
  overlayValue(target.custom1, source.custom1);
  overlayValue(target.custom2, source.custom2);
  overlayValue(target.custom3, source.custom3);
  overlayValue(target.custom4, source.custom4);
  overlayValue(target.custom5, source.custom5);
}
} // namespace

struct DeviceManager::Impl {
  explicit Impl(DeviceRegistry initialRegistry)
      : registry(std::move(initialRegistry)) {}

  DeviceRegistry registry;
  mutable std::mutex registryMutex;
  std::mutex libraryMutex;
  // Non-owning cache: opened device state owns the immutable API/context.
  std::map<std::string, std::weak_ptr<const detail::DeviceApi>> libraries;

  [[nodiscard]] Device
  openDefinition(const DeviceDefinition& definition,
                 const SessionParameters& sessionOverrides) {
    const auto key = definition.library.string() + "\n" + definition.prefix;
    std::shared_ptr<const detail::DeviceApi> library;
    {
      const std::scoped_lock lock(libraryMutex);
      library = libraries[key].lock();
      if (!library) {
        library =
            detail::makeV1DeviceApi(definition.library, definition.prefix);
        libraries[key] = library;
      }
    }
    auto parameters = definition.session;
    overlay(parameters, sessionOverrides);
    return Device(std::make_shared<detail::DeviceState>(library, parameters));
  }
};

DeviceManager::DeviceManager(const ConfigOptions& options)
    : DeviceManager(DeviceRegistry(options)) {}

DeviceManager::DeviceManager(DeviceRegistry registry)
    : impl_(std::make_unique<Impl>(std::move(registry))) {}

DeviceManager::~DeviceManager() = default;
DeviceManager::DeviceManager(DeviceManager&&) noexcept = default;
DeviceManager& DeviceManager::operator=(DeviceManager&&) noexcept = default;

std::vector<DeviceDefinition> DeviceManager::definitions() const {
  const std::scoped_lock lock(impl_->registryMutex);
  return impl_->registry.definitions();
}

void DeviceManager::registerDevice(DeviceDefinition definition,
                                   const bool replace) {
  const std::scoped_lock lock(impl_->registryMutex);
  impl_->registry.registerDevice(std::move(definition), replace);
}

bool DeviceManager::unregisterDevice(const std::string_view id) {
  const std::scoped_lock lock(impl_->registryMutex);
  return impl_->registry.unregisterDevice(id);
}

Device DeviceManager::open(const std::string_view id,
                           const SessionParameters& sessionOverrides) {
  DeviceDefinition definition;
  {
    const std::scoped_lock lock(impl_->registryMutex);
    const auto& definitions = impl_->registry.definitions();
    const auto found =
        std::ranges::find(definitions, id, &DeviceDefinition::id);
    if (found == definitions.end()) {
      throw std::out_of_range("No QDMI device is registered with id '" +
                              std::string(id) + "'");
    }
    definition = *found;
  }
  return impl_->openDefinition(definition, sessionOverrides);
}

OpenAllResult
DeviceManager::openAll(const SessionParameters& sessionOverrides) {
  OpenAllResult result;
  for (const auto& definition : definitions()) {
    try {
      result.devices.emplace(
          definition.id, impl_->openDefinition(definition, sessionOverrides));
    } catch (const std::exception& error) {
      result.errors.emplace(definition.id, error.what());
    }
  }
  return result;
}
} // namespace qdmi
