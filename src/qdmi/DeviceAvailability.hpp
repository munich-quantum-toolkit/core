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

inline void checkDeviceAvailability(const Device& device, const std::string& id,
                                    const std::string_view context) {
  std::string_view status = "UNKNOWN";
  switch (device.getStatus()) {
  case QDMI_DEVICE_STATUS_IDLE:
  case QDMI_DEVICE_STATUS_BUSY:
    return;
  case QDMI_DEVICE_STATUS_OFFLINE:
    status = "OFFLINE";
    break;
  case QDMI_DEVICE_STATUS_ERROR:
    status = "ERROR";
    break;
  case QDMI_DEVICE_STATUS_MAINTENANCE:
    status = "MAINTENANCE";
    break;
  case QDMI_DEVICE_STATUS_CALIBRATION:
    status = "CALIBRATION";
    break;
  case QDMI_DEVICE_STATUS_MAX:
    break;
  }
  throw std::runtime_error(std::string(context) + "'" + id + "' with status " +
                           std::string(status));
}

} // namespace qdmi::detail
