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

// These names are fixed by the private C ABI.
// NOLINTBEGIN(readability-identifier-naming)

extern "C" {
/// Register a device manifest before the builtin driver reads its catalogue.
///
/// @param path Path to a manifest file; must not be null or empty.
/// @return QDMI_SUCCESS on registration, QDMI_ERROR_INVALIDARGUMENT for an
/// invalid path or manifest, or QDMI_ERROR_OUTOFMEM on allocation failure.
QDMI_DRIVER_EXPORT int MQT_CORE_QDMI_driver_add_manifest_v1(const char* path);
/// Return enabled stable IDs without loading device libraries.
///
/// The buffer contains consecutive NUL-terminated IDs. An empty catalogue
/// requires zero bytes.
///
/// @param size Capacity of @p ids in bytes; ignored if @p ids is null.
/// @param ids Output buffer, or null to query the required size.
/// @param sizeRet Output for the required byte size. Must be nonnull when
/// @p ids is null.
/// @return QDMI_SUCCESS on success, QDMI_ERROR_INVALIDARGUMENT for invalid
/// arguments or a short buffer, QDMI_ERROR_OUTOFMEM on allocation failure,
/// or QDMI_ERROR_FATAL on another failure.
QDMI_DRIVER_EXPORT int
MQT_CORE_QDMI_driver_registered_device_ids_v1(size_t size, char* ids,
                                              size_t* sizeRet);
/// Allocate an independent session for one configured stable ID.
///
/// @param id Configured stable device ID; must not be null or empty.
/// @param size Byte size of @p parameters, or zero when no overrides are set.
/// @param parameters JSON overrides for this device session, or null when
/// @p size is zero. The values override settings in the manifest.
/// @param session Output session handle; must not be null. Set to null on
/// error.
/// @return QDMI_SUCCESS on success, QDMI_ERROR_INVALIDARGUMENT for invalid
/// arguments or overrides, QDMI_ERROR_OUTOFMEM on allocation failure, or a
/// status from opening the selected device.
QDMI_DRIVER_EXPORT int MQT_CORE_QDMI_driver_session_alloc_for_device_v1(
    const char* id, size_t size, const char* parameters, QDMI_Session* session);
}
// NOLINTEND(readability-identifier-naming)
