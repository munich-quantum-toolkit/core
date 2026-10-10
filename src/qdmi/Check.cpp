/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "qdmi/QDMI.hpp"
#include "qdmi/common/Common.hpp"

#include "qdmi/constants.h"

#include <cstdio>
#include <filesystem>
#include <iostream>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace {
constexpr std::string_view USAGE =
    "Usage: mqt-core-qdmi-check --device ID [--manifest PATH]...\n"
    "Probe whether a device registered with MQT Core's QDMI driver is "
    "operational.\n"
    "Exit codes: 0 available, 1 failed, 2 invalid arguments.\n";
} // namespace

#ifdef _WIN32
int wmain(const int argc, wchar_t** argv) try {
  std::vector<std::string> utf8Arguments;
  for (const auto* argument : std::span(argv, static_cast<size_t>(argc))) {
    utf8Arguments.emplace_back(qdmi::detail::pathToString(argument));
  }
  const auto arguments = std::span(utf8Arguments);
  constexpr auto* sink = "NUL";
#else
int main(const int argc, char** argv) try {
  const auto arguments = std::span(argv, static_cast<size_t>(argc));
  constexpr auto* sink = "/dev/null";
#endif
  if (argc == 2 && std::string_view(arguments[1]) == "--help") {
    std::cout << USAGE;
    return 0;
  }
  std::string id;
  std::vector<std::filesystem::path> manifests;
  for (size_t i = 1; i < arguments.size(); ++i) {
    const std::string_view option(arguments[i]);
    if (++i == arguments.size()) {
      std::cerr << USAGE;
      return 2;
    }
    const std::string_view value(arguments[i]);
    if (option == "--device" && id.empty() && !value.empty()) {
      id = value;
    } else if (option == "--manifest" && !value.empty()) {
      manifests.emplace_back(qdmi::detail::pathFromString(value));
    } else {
      std::cerr << USAGE;
      return 2;
    }
  }
  if (id.empty()) {
    std::cerr << USAGE;
    return 2;
  }
  // Device diagnostics can contain credentials, including during discovery.
  // The C runtime retains ownership of these reopened standard streams.
  // NOLINTBEGIN(cppcoreguidelines-owning-memory)
  if (std::freopen(sink, "w", stdout) == nullptr ||
      std::freopen(sink, "w", stderr) == nullptr) {
    return 1;
  }
  // NOLINTEND(cppcoreguidelines-owning-memory)
  for (const auto& manifest : manifests) {
    try {
      qdmi::builtin_driver::addManifest(manifest);
    } catch (...) {
      // Match Python discovery's best-effort handling of installed manifests.
      continue;
    }
  }
  const auto device = qdmi::builtin_driver::openDevice(id);
  const auto status = device.getStatus();
  return status == QDMI_DEVICE_STATUS_IDLE || status == QDMI_DEVICE_STATUS_BUSY
             ? 0
             : 1;
} catch (...) {
  return 1;
}
