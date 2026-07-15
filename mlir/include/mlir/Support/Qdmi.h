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

#include "fomac/FoMaC.hpp"
#include "qdmi/driver/Driver.hpp"

#include <llvm/Support/raw_ostream.h>
#include <mlir/Support/LLVM.h>

#include <memory>

namespace mlir::qdmi {
namespace impl {

struct DeviceConfig {
  /// Library name/path
  std::string libName;
  /// Prefix for function names
  std::string prefix;
  /// Device session configuration
  ::qdmi::DeviceSessionConfig deviceConfig;
};

struct Config {
  /// Read and parse the given QDMI JSON config file.
  /// The function expects the following JSON structure, where some attributes
  /// may be omitted depending on the use-case.
  /// {
  ///   "session": {
  ///     "token": "...",
  ///     "authFile": "...",
  ///     "authUrl": "...",
  ///     "username": "...",
  ///     "password": "...",
  ///     "projectId": "...",
  ///     "custom1": "...",
  ///     "custom2": "...",
  ///     "custom3": "...",
  ///     "custom4": "...",
  ///     "custom5": "..."
  ///   },
  ///   "devices": [
  ///     {
  ///       "libName": "...",
  ///       "prefix": "...",
  ///       "deviceConfig": {
  ///         "baseUrl": "...",
  ///         "token": "...",
  ///         "authFile": "...",
  ///         "authUrl": "...",
  ///         "username": "...",
  ///         "password": "...",
  ///         "custom1": "...",
  ///         "custom2": "...",
  ///         "custom3": "...",
  ///         "custom4": "...",
  ///         "custom5": "..."
  ///       }
  ///     }
  ///   ]
  /// }
  static Config read(StringRef path);

  fomac::SessionConfig session;
  SmallVector<DeviceConfig, 0> devices;
};
} // namespace impl

/// Return parameterized session.
/// Applies the provided session config and loads the specified QDMI devices.
fomac::Session prepareSession(StringRef configPath);

/// Return unparameterized session.
/// Applies default session config and does not load any dynamic QDMI devices.
fomac::Session prepareSession();

/// Output a list of all available QDMI devices to @p os.
void listAvailableDevices(fomac::Session& session,
                          llvm::raw_ostream& os = llvm::outs());

/// Find a QDMI device with the given name and return its FoMaC class.
std::shared_ptr<fomac::Device> getDevice(fomac::Session& session,
                                         StringRef name);
} // namespace mlir::qdmi
