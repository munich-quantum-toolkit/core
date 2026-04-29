/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QCO/Transforms/Decomposition/EulerBasis.h"

#include "mlir/Dialect/QCO/Transforms/Decomposition/GateKind.h"

#include <llvm/Support/ErrorHandling.h>
#include <mlir/Support/LLVM.h>

namespace mlir::qco::decomposition {

[[nodiscard]] SmallVector<GateKind>
getGateTypesForEulerBasis(EulerBasis eulerBasis) {
  switch (eulerBasis) {
  case EulerBasis::ZYZ:
    // Z-Y-Z style decompositions only emit `rz` and `ry`.
    return {GateKind::RZ, GateKind::RY};
  case EulerBasis::ZXZ:
    // Z-X-Z and X-Z-X share the same two-axis alphabet with swapped roles.
    return {GateKind::RZ, GateKind::RX};
  case EulerBasis::XZX:
    return {GateKind::RX, GateKind::RZ};
  case EulerBasis::XYX:
    return {GateKind::RX, GateKind::RY};
  case EulerBasis::U3:
    [[fallthrough]];
  case EulerBasis::U321:
    [[fallthrough]];
  case EulerBasis::U:
    // All U variants collapse to a single `u` operation at emission time.
    return {GateKind::U};
  case EulerBasis::ZSX:
    // `ZSX` only emits `rz` and `sx`.
    return {GateKind::RZ, GateKind::SX};
  case EulerBasis::ZSXX:
    // `ZSXX` additionally allows a bare `x` when the middle rotation is
    // +/- pi, staying within the `{rz, sx, x}` alphabet.
    return {GateKind::RZ, GateKind::SX, GateKind::X};
  }
  llvm::reportFatalInternalError(
      "Unsupported euler basis for translation to gate types");
}

} // namespace mlir::qco::decomposition
