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

#include "ir/operations/OpType.hpp"

#include <cstdint>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/ErrorHandling.h>

namespace mlir::qco::decomposition {
/**
 * Largest number that will be assumed as zero for the euler decompositions.
 */
static constexpr auto DEFAULT_ATOL = 1e-12;

/**
 * EulerBasis for a euler decomposition.
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

[[nodiscard]] inline llvm::SmallVector<qc::OpType, 3>
getGateTypesForEulerBasis(EulerBasis eulerBasis) {
  switch (eulerBasis) {
  case EulerBasis::ZYZ:
    return {qc::RZ, qc::RY};
  case EulerBasis::ZXZ:
    return {qc::RZ, qc::RX};
  case EulerBasis::XZX:
    return {qc::RX, qc::RZ};
  case EulerBasis::XYX:
    return {qc::RX, qc::RY};
  case EulerBasis::U3:
    [[fallthrough]];
  case EulerBasis::U321:
    [[fallthrough]];
  case EulerBasis::U:
    return {qc::U};
  case EulerBasis::RR:
    return {qc::R};
  case EulerBasis::ZSXX:
    [[fallthrough]];
  case EulerBasis::ZSX:
    return {qc::RZ, qc::SX};
  case EulerBasis::PSX:
    [[fallthrough]];
  case EulerBasis::U1X:
    return {qc::RZ, qc::P};
  }
  llvm::reportFatalInternalError(
      "Unsupported euler basis for translation to gate types");
}

} // namespace mlir::qco::decomposition
