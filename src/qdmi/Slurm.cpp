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
#include "qdmi/common/Common.hpp"

#include "qdmi/constants.h"

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cstddef>
#include <cstdlib>
#include <iterator>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <variant>

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

[[nodiscard]] auto parseLicense(const std::string_view licenseSpec)
    -> Result<std::string> {
  if (licenseSpec.empty()) {
    return Error{
        .status = QDMI_ERROR_BADSTATE,
        .message =
            "SLURM_JOB_LICENSES is not set or empty; no QDMI device license is "
            "available",
    };
  }
  if (std::ranges::any_of(licenseSpec, [](const unsigned char character) {
        return std::isspace(character) != 0;
      })) {
    return Error{
        .status = QDMI_ERROR_BADSTATE,
        .message = "SLURM_JOB_LICENSES must not contain whitespace",
    };
  }
  if (licenseSpec.find(',') != std::string::npos) {
    return Error{
        .status = QDMI_ERROR_BADSTATE,
        .message = "SLURM_JOB_LICENSES uses a compound AND "
                   "expression; exactly one QDMI "
                   "device license is required",
    };
  }
  if (licenseSpec.find('|') != std::string::npos) {
    return Error{
        .status = QDMI_ERROR_BADSTATE,
        .message = "SLURM_JOB_LICENSES uses a compound OR expression; "
                   "exactly one QDMI "
                   "device license is required",
    };
  }
  if (licenseSpec.find('@') != std::string::npos) {
    return Error{
        .status = QDMI_ERROR_BADSTATE,
        .message = "A remote Slurm license cannot select a QDMI device",
    };
  }

  const auto countSeparator = licenseSpec.find(':');
  if (countSeparator != std::string::npos &&
      licenseSpec.find(':', countSeparator + 1) != std::string::npos) {
    return Error{
        .status = QDMI_ERROR_BADSTATE,
        .message = "SLURM_JOB_LICENSES contains a malformed license",
    };
  }
  const auto deviceId = licenseSpec.substr(0, countSeparator);
  if (deviceId.empty()) {
    return Error{
        .status = QDMI_ERROR_BADSTATE,
        .message = "SLURM_JOB_LICENSES contains a malformed license",
    };
  }

  if (countSeparator != std::string::npos) {
    const auto countText = licenseSpec.substr(countSeparator + 1);
    if (countText.empty()) {
      return Error{
          .status = QDMI_ERROR_BADSTATE,
          .message = "SLURM_JOB_LICENSES contains a malformed license count",
      };
    }
    size_t count = 0;
    const char* const countBegin = countText.data();
    const auto countLength = static_cast<std::ptrdiff_t>(countText.size());
    const char* const countEnd = std::next(countBegin, countLength);
    const auto [parsedEnd, error] =
        std::from_chars(countBegin, countEnd, count);
    if (error != std::errc{} || parsedEnd != countEnd) {
      return Error{
          .status = QDMI_ERROR_BADSTATE,
          .message = "SLURM_JOB_LICENSES contains an invalid license count",
      };
    }
    if (count != 1) {
      return Error{
          .status = QDMI_ERROR_BADSTATE,
          .message = "A QDMI device job must request exactly one Slurm license",
      };
    }
  }

  return std::string(deviceId);
}

} // namespace

Result<Device> openDeviceFromLicense() {
  /// The job can modify its environment. The provider must authorize access.
  const auto* environmentValue = std::getenv("SLURM_JOB_LICENSES");
  auto id = parseLicense(environmentValue == nullptr ? "" : environmentValue);
  if (auto* error = std::get_if<Error>(&id)) {
    return std::move(*error);
  }
  const auto& deviceId = std::get<0>(id);
  auto result = Session::openDevice(deviceId);
  if (auto* error = std::get_if<Error>(&result)) {
    if (error->status == QDMI_ERROR_OUTOFRANGE) {
      error->status = QDMI_ERROR_BADSTATE;
      error->message =
          "Slurm license '" + deviceId + "' is not a registered QDMI device ID";
    }
    return std::move(*error);
  }
  auto& device = std::get<0>(result);
  auto statusResult = device.getStatus();
  if (auto* error = std::get_if<Error>(&statusResult)) {
    return std::move(*error);
  }
  const auto status = std::get<0>(statusResult);
  if (status != QDMI_DEVICE_STATUS_IDLE && status != QDMI_DEVICE_STATUS_BUSY) {
    return Error{
        .status = QDMI_ERROR_BADSTATE,
        .message = "SLURM_JOB_LICENSES names QDMI device '" + deviceId +
                   "' with status " + std::string(statusName(status)),
    };
  }
  return std::move(device);
}

} // namespace qdmi::slurm
