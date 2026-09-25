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

#include "qdmi/client.h"

#include <cstddef>

/// These names are fixed by the private C ABI.
/// NOLINTBEGIN(readability-identifier-naming)

extern "C" {
/// Register a device manifest before the MQT Core QDMI driver is initialized.
QDMI_DRIVER_EXPORT int MQT_CORE_QDMI_driver_add_manifest_v1(const char* path);
/// Allocate an independent session for one configured stable ID.
QDMI_DRIVER_EXPORT int MQT_CORE_QDMI_driver_session_alloc_for_device_v1(
    const char* id, size_t size, const char* parameters, QDMI_Session* session);
}
/// NOLINTEND(readability-identifier-naming)
