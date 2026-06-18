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

#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/ErrorHandling.h>

namespace mlir::qco::decomposition {

[[nodiscard]] llvm::SmallVector<GateKind, 3>
getGateTypesForEulerBasis(GateEulerBasis eulerBasis) {
  switch (eulerBasis) {
  case GateEulerBasis::ZYZ:
    // Z-Y-Z style decompositions only emit `rz` and `ry`.
    return {GateKind::RZ, GateKind::RY};
  case GateEulerBasis::ZXZ:
    // Z-X-Z and X-Z-X share the same two-axis alphabet with swapped roles.
    return {GateKind::RZ, GateKind::RX};
  case GateEulerBasis::XZX:
    return {GateKind::RX, GateKind::RZ};
  case GateEulerBasis::XYX:
    return {GateKind::RX, GateKind::RY};
  case GateEulerBasis::U3:
    [[fallthrough]];
  case GateEulerBasis::U321:
    [[fallthrough]];
  case GateEulerBasis::U:
    // All U variants collapse to a single `u` operation at emission time.
    return {GateKind::U};
  case GateEulerBasis::ZSX:
    // `ZSX` only emits `rz` and `sx`.
    return {GateKind::RZ, GateKind::SX};
  case GateEulerBasis::ZSXX:
    // `ZSXX` additionally allows a bare `X` when the middle rotation is
    // +/- pi, staying within the `{rz, sx, x}` alphabet.
    return {GateKind::RZ, GateKind::SX, GateKind::X};
  }
  llvm::reportFatalInternalError(
      "Unsupported euler basis for translation to gate types");
}

} // namespace mlir::qco::decomposition
