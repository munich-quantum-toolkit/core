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

#include "qdmi/Client.hpp"

#include "qdmi/constants.h"

#include <stdexcept>
#include <string>
#include <string_view>

namespace qdmi::detail {

[[nodiscard]] inline auto statusName(const QDMI_Device_Status status)
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

inline void checkDeviceAvailability(const Device& device, const std::string& id,
                                    const std::string_view context) {
  const auto status = device.getStatus();
  if (status != QDMI_DEVICE_STATUS_IDLE && status != QDMI_DEVICE_STATUS_BUSY) {
    throw std::runtime_error(std::string(context) + "'" + id +
                             "' with status " +
                             std::string(statusName(status)));
  }
}

} // namespace qdmi::detail
