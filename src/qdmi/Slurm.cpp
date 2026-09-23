/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "qdmi/Slurm.hpp"

#include "qdmi/Client.hpp"

#include "DeviceAvailability.hpp"

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cstddef>
#include <cstdlib>
#include <iterator>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>

namespace qdmi::slurm {
namespace {

[[nodiscard]] auto parseLicense(const std::string_view licenseSpec)
    -> std::string {
  if (licenseSpec.empty()) {
    throw std::runtime_error(
        "SLURM_JOB_LICENSES is not set or empty; no QDMI device license is "
        "available");
  }
  if (std::ranges::any_of(licenseSpec, [](const unsigned char character) {
        return std::isspace(character) != 0;
      })) {
    throw std::runtime_error("SLURM_JOB_LICENSES must not contain whitespace");
  }
  if (licenseSpec.find(',') != std::string::npos) {
    throw std::runtime_error(
        "SLURM_JOB_LICENSES uses a compound AND expression; exactly one QDMI "
        "device license is required");
  }
  if (licenseSpec.find('|') != std::string::npos) {
    throw std::runtime_error(
        "SLURM_JOB_LICENSES uses a compound OR expression; exactly one QDMI "
        "device license is required");
  }
  if (licenseSpec.find('@') != std::string::npos) {
    throw std::runtime_error(
        "A remote Slurm license cannot select a QDMI device");
  }

  const auto countSeparator = licenseSpec.find(':');
  if (countSeparator != std::string::npos &&
      licenseSpec.find(':', countSeparator + 1) != std::string::npos) {
    throw std::runtime_error("SLURM_JOB_LICENSES contains a malformed license");
  }
  const auto deviceId = licenseSpec.substr(0, countSeparator);
  if (deviceId.empty()) {
    throw std::runtime_error("SLURM_JOB_LICENSES contains a malformed license");
  }

  if (countSeparator != std::string::npos) {
    const auto countText = licenseSpec.substr(countSeparator + 1);
    if (countText.empty()) {
      throw std::runtime_error(
          "SLURM_JOB_LICENSES contains a malformed license count");
    }
    size_t count = 0;
    const char* const countBegin = countText.data();
    const auto countLength = static_cast<std::ptrdiff_t>(countText.size());
    const char* const countEnd = std::next(countBegin, countLength);
    const auto [parsedEnd, error] =
        std::from_chars(countBegin, countEnd, count);
    if (error != std::errc{} || parsedEnd != countEnd) {
      throw std::runtime_error(
          "SLURM_JOB_LICENSES contains an invalid license count");
    }
    if (count != 1) {
      throw std::runtime_error(
          "A QDMI device job must request exactly one Slurm license");
    }
  }

  return std::string(deviceId);
}

} // namespace

Device openDeviceFromLicense() {
  /// The job can modify its environment. Use this value only to select a
  /// registered device; the provider or operating system must authorize access.
  const auto* const environmentValue = std::getenv("SLURM_JOB_LICENSES");
  const std::string licenseSpec =
      environmentValue == nullptr ? std::string{} : environmentValue;
  const auto deviceId = parseLicense(licenseSpec);
  auto device = [&] {
    try {
      return Session::openDevice(deviceId);
    } catch (const std::out_of_range&) {
      throw std::runtime_error("Slurm license '" + deviceId +
                               "' is not a registered QDMI device ID");
    }
  }();
  detail::checkDeviceAvailability(device, deviceId,
                                  "SLURM_JOB_LICENSES names QDMI device ");
  return device;
}

} // namespace qdmi::slurm
