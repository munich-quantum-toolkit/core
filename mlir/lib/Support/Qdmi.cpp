/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Support/Qdmi.h"

#include "fomac/FoMaC.hpp"
#include "qdmi/driver/Driver.hpp"

#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Support/LLVM.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <memory>
#include <optional>
#include <string>

namespace {
template <typename T>
void getJsonStringIfExists(const nlohmann::json& jsonObj,
                           const std::string& field,
                           T& target) {
    if (jsonObj.contains(field) && jsonObj[field].is_string()) {
        target = jsonObj[field].get<std::string>();
    }
}
} // namespace

mlir::qdmi::impl::Config mlir::qdmi::impl::Config::read(StringRef path) {
  using json = nlohmann::json;

  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
      llvm::MemoryBuffer::getFile(path);
  if (!fileOrErr) {
    llvm::errs() << "Failed to open QDMI config file: " << path << "\n";
    return {};
  }

  json j;
  try {
    std::string content(fileOrErr->get()->getBufferStart(),
                        fileOrErr->get()->getBufferSize());
    j = json::parse(content);
  } catch (const json::parse_error& e) {
    llvm::errs() << "Failed to parse QDMI config file '" << path
                 << "': " << e.what() << "\n";
    return {};
  } catch (const std::exception& e) {
    llvm::errs() << "Error reading QDMI config file '" << path
                 << "': " << e.what() << "\n";
    return {};
  }

  Config config;

  if (j.contains("session")) {
    getJsonStringIfExists(j["session"], "token", config.session.token);
    getJsonStringIfExists(j["session"], "authFile", config.session.authFile);
    getJsonStringIfExists(j["session"], "authUrl", config.session.authUrl);
    getJsonStringIfExists(j["session"], "username", config.session.username);
    getJsonStringIfExists(j["session"], "password", config.session.password);
    getJsonStringIfExists(j["session"], "projectId", config.session.projectId);
    getJsonStringIfExists(j["session"], "custom1", config.session.custom1);
    getJsonStringIfExists(j["session"], "custom2", config.session.custom2);
    getJsonStringIfExists(j["session"], "custom3", config.session.custom3);
    getJsonStringIfExists(j["session"], "custom4", config.session.custom4);
    getJsonStringIfExists(j["session"], "custom5", config.session.custom5);
  }

  // Parse devices array
  if (j.contains("devices") && j["devices"].is_array()) {
    for (const auto& deviceJson : j["devices"]) {
      if (!deviceJson.contains("libName") ||
          !deviceJson["libName"].is_string() ||
          !deviceJson.contains("prefix") || !deviceJson["prefix"].is_string()) {
        continue; // Skip malformed device entries.
      }

      impl::DeviceConfig device;
      device.libName = deviceJson["libName"].get<std::string>();
      device.prefix = deviceJson["prefix"].get<std::string>();

      // Parse deviceConfig
      if (deviceJson.contains("deviceConfig") &&
          deviceJson["deviceConfig"].is_object()) {
        const auto& dc = deviceJson["deviceConfig"];
        getJsonStringIfExists(dc, "baseUrl", device.deviceConfig.baseUrl);
        getJsonStringIfExists(dc, "token", device.deviceConfig.token);
        getJsonStringIfExists(dc, "authFile", device.deviceConfig.authFile);
        getJsonStringIfExists(dc, "authUrl", device.deviceConfig.authUrl);
        getJsonStringIfExists(dc, "username", device.deviceConfig.username);
        getJsonStringIfExists(dc, "password", device.deviceConfig.password);
        getJsonStringIfExists(dc, "custom1", device.deviceConfig.custom1);
        getJsonStringIfExists(dc, "custom2", device.deviceConfig.custom2);
        getJsonStringIfExists(dc, "custom3", device.deviceConfig.custom3);
        getJsonStringIfExists(dc, "custom4", device.deviceConfig.custom4);
        getJsonStringIfExists(dc, "custom5", device.deviceConfig.custom5);
      }

      config.devices.push_back(device);
    }
  }

  return config;
}

void mlir::qdmi::listAvailableDevices(fomac::Session& session,
                                      llvm::raw_ostream& os) {
  os << "Available QDMI devices:\n";
  for (const auto& dev : session.getDevices()) {
    os << '\t' << dev.getName() << '\n';
  }
}

std::shared_ptr<fomac::Device> mlir::qdmi::getDevice(fomac::Session& session,
                                                     StringRef name) {
  const auto devices = session.getDevices();
  const auto it = std::ranges::find_if(
      devices, [&](const auto& dev) { return dev.getName() == name; });
  return it != devices.end() ? std::make_shared<fomac::Device>(*it) : nullptr;
}

fomac::Session mlir::qdmi::prepareSession(StringRef configPath) {
  const auto config = impl::Config::read(configPath);

  // Load dynamic device libraries from config.
  for (const auto& device : config.devices) {
    ::qdmi::Driver::get().addDynamicDeviceLibrary(device.libName, device.prefix,
                                                  device.deviceConfig);
  }

  return fomac::Session(config.session);
}

fomac::Session mlir::qdmi::prepareSession() { return fomac::Session(); }
