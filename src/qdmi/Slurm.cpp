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

#include "qdmi/QDMI.hpp"

#include "qdmi/constants.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <string_view>

namespace qdmi::slurm {
namespace {

[[nodiscard]] auto statusName(const QDMI_Device_Status status)
    -> std::string_view {
  switch (status) {
  case QDMI_DEVICE_STATUS_OFFLINE:
    return "OFFLINE";
  case QDMI_DEVICE_STATUS_IDLE:
    return "IDLE";
  case QDMI_DEVICE_STATUS_BUSY:
    return "BUSY";
  case QDMI_DEVICE_STATUS_ERROR:
    return "ERROR";
  case QDMI_DEVICE_STATUS_MAINTENANCE:
    return "MAINTENANCE";
  case QDMI_DEVICE_STATUS_CALIBRATION:
    return "CALIBRATION";
  case QDMI_DEVICE_STATUS_MAX:
    return "UNKNOWN";
  }
  return "UNKNOWN";
}

[[nodiscard]] auto parseLicense(std::string_view licenseSpec) -> std::string {
  if (licenseSpec.ends_with(":1")) {
    licenseSpec.remove_suffix(2);
  }
  if (licenseSpec.empty() ||
      licenseSpec.find_first_of(":,@|") != std::string_view::npos ||
      std::ranges::any_of(licenseSpec, [](const unsigned char character) {
        return std::isspace(character) != 0;
      })) {
    throw std::runtime_error("SLURM_JOB_LICENSES must name exactly one local "
                             "QDMI device license (ID or ID:1)");
  }
  const auto ids = builtin_driver::registeredDeviceIds();
  if (std::ranges::find(ids, licenseSpec) == ids.end()) {
    throw std::runtime_error("Slurm license '" + std::string(licenseSpec) +
                             "' is not a registered QDMI device ID");
  }
  return std::string(licenseSpec);
}

} // namespace

Device openDeviceFromLicense() {
  // The job can modify its environment. Use this value only to select a
  // registered device; the device implementation or operating system must
  // authorize access.
  const auto* const environmentValue = std::getenv("SLURM_JOB_LICENSES");
  const std::string licenseSpec =
      environmentValue == nullptr ? std::string{} : environmentValue;
  const auto deviceId = parseLicense(licenseSpec);
  auto device = builtin_driver::openDevice(deviceId);
  const auto status = device.getStatus();
  if (status != QDMI_DEVICE_STATUS_IDLE && status != QDMI_DEVICE_STATUS_BUSY) {
    throw std::runtime_error("SLURM_JOB_LICENSES names QDMI device '" +
                             deviceId + "' with status " +
                             std::string(statusName(status)));
  }
  return device;
}

} // namespace qdmi::slurm
