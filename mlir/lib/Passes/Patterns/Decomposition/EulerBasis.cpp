/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Passes/Decomposition/EulerBasis.h"

#include "ir/operations/OpType.hpp"

#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/ErrorHandling.h>

namespace mlir::qco::decomposition {

[[nodiscard]] llvm::SmallVector<qc::OpType, 3>
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
