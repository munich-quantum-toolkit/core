/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "qdmi/client.h"

#include <cstdint>

// Only the ABI query is exported, to exercise validation before allocation.
// NOLINTNEXTLINE(readability-identifier-naming)
uint32_t QDMI_driver_get_client_abi_version() {
#ifdef TEST_INCOMPATIBLE_DRIVER
  return (1U << 22U) | (5U << 12U);
#else
  // A different patch must pass ABI validation before the missing-symbol check.
  return (1U << 22U) | (4U << 12U) | 99U;
#endif
}
