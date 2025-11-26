/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

#include <cstdint>

namespace mqt::ir::opt::decomposition {
/**
 * Largest number that will be assumed as zero for the euler decompositions.
 */
static constexpr auto DEFAULT_ATOL = 1e-12;

/**
 * EulerBasis for a euler decomposition.
 *
 * @note only the following bases are supported for now: ZYZ, ZXZ and XZX
 */
enum class EulerBasis : std::uint8_t {
  U3 = 0,
  U321 = 1,
  U = 2,
  PSX = 3,
  U1X = 4,
  RR = 5,
  ZYZ = 6,
  ZXZ = 7,
  XZX = 8,
  XYX = 9,
  ZSXX = 10,
  ZSX = 11,
};
} // namespace mqt::ir::opt::decomposition
