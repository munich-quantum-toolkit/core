/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "qdmi/common/Common.hpp"
#include "qdmi/driver/Driver.hpp"
#include "qdmi/driver/DriverExtension.hpp"

#include "DeviceRegistry.hpp"

#include "qdmi/client.h"

#include "mlir/Support/LogicalResult.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

uint32_t QDMI_driver_get_client_abi_version() {
  return QDMI_CLIENT_ABI_VERSION;
}

/// The private C ABI fixes these exported symbol names.
/// NOLINTBEGIN(readability-identifier-naming)
extern "C" QDMI_DRIVER_EXPORT int
MQT_CORE_QDMI_driver_add_manifest_v1(const char* const manifestPath) {
  if (manifestPath == nullptr || *manifestPath == '\0') {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  try {
    return qdmi::detail::stageDeviceManifest(
        qdmi::detail::pathFromString(manifestPath));
  } catch (const std::bad_alloc&) {
    return QDMI_ERROR_OUTOFMEM;
  } catch (...) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
}

extern "C" QDMI_DRIVER_EXPORT int MQT_CORE_QDMI_driver_registered_device_ids_v1(
    const size_t size, char* const ids, size_t* const sizeRet) {
  if ((ids == nullptr && sizeRet == nullptr) || (ids != nullptr && size == 0)) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  try {
    std::string buffer;
    std::vector<std::string> idsValue;
    const auto status = qdmi::invokeStatus([&]() -> mlir::LogicalResult {
      auto result = qdmi::Driver::get().registeredDeviceIds();
      if (mlir::failed(result)) {
        return mlir::failure();
      }
      idsValue = std::move(*result);
      return mlir::success();
    });
    if (status != QDMI_SUCCESS) {
      return status;
    }
    for (const auto& id : idsValue) {
      buffer.append(id).push_back('\0');
    }
    if (sizeRet != nullptr) {
      *sizeRet = buffer.size();
    }
    if (ids != nullptr) {
      if (size < buffer.size()) {
        return QDMI_ERROR_INVALIDARGUMENT;
      }
      std::ranges::copy(buffer, ids);
    }
    return QDMI_SUCCESS;
  } catch (const std::bad_alloc&) {
    return QDMI_ERROR_OUTOFMEM;
  } catch (...) {
    return QDMI_ERROR_FATAL;
  }
}

extern "C" QDMI_DRIVER_EXPORT int
MQT_CORE_QDMI_driver_session_alloc_for_device_v1(
    const char* const deviceId, const size_t deviceSessionJsonSize,
    const char* const deviceSessionJson, QDMI_Session* const session) {
  if (session == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  *session = nullptr;
  if (deviceId == nullptr || *deviceId == '\0' ||
      ((deviceSessionJson == nullptr) != (deviceSessionJsonSize == 0))) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  qdmi::DeviceSessionConfig config;
  if (const auto status = qdmi::detail::parseDeviceSessionJson(
          deviceSessionJson, deviceSessionJsonSize, config);
      status != QDMI_SUCCESS) {
    return status;
  }
  try {
    return qdmi::Driver::get().sessionAllocForDevice(deviceId, config, session);
  } catch (const std::bad_alloc&) {
    return QDMI_ERROR_OUTOFMEM;
  } catch (const std::invalid_argument&) {
    return QDMI_ERROR_INVALIDARGUMENT;
  } catch (...) {
    return QDMI_ERROR_FATAL;
  }
}
/// NOLINTEND(readability-identifier-naming)

int QDMI_session_alloc(QDMI_Session* session) {
  if (session == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  *session = nullptr;
  try {
    return qdmi::Driver::get().sessionAlloc(session);
  } catch (const std::bad_alloc&) {
    return QDMI_ERROR_OUTOFMEM;
  } catch (...) {
    return QDMI_ERROR_FATAL;
  }
}
