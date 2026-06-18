/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QCO/Transforms/NativeSynthesis/Policy.h"

#include "mlir/Dialect/QCO/IR/QCOInterfaces.h"
#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QCO/Transforms/NativeSynthesis/Types.h"

#include <llvm/ADT/STLExtras.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/ErrorHandling.h>
#include <mlir/IR/Operation.h>

#include <optional>

namespace mlir::qco::native_synth {

bool usesCxEntangler(const NativeProfileSpec& spec) {
  return llvm::is_contained(spec.entanglerBases, EntanglerBasis::Cx);
}

bool usesCzEntangler(const NativeProfileSpec& spec) {
  return llvm::is_contained(spec.entanglerBases, EntanglerBasis::Cz);
}

/// Map a single-qubit `UnitaryOpInterface` op to the `NativeGateKind` that
/// must appear in the menu for the op to be a no-op.
static std::optional<NativeGateKind>
singleQubitNativeGateKind(UnitaryOpInterface op) {
  Operation* raw = op.getOperation();
  if (llvm::isa<UOp>(raw)) {
    return NativeGateKind::U;
  }
  if (llvm::isa<XOp>(raw)) {
    return NativeGateKind::X;
  }
  if (llvm::isa<SXOp>(raw)) {
    return NativeGateKind::Sx;
  }
  if (llvm::isa<RZOp, POp>(raw)) {
    // `p` is a Z-rotation primitive for menu purposes.
    return NativeGateKind::Rz;
  }
  if (llvm::isa<RXOp>(raw)) {
    return NativeGateKind::Rx;
  }
  if (llvm::isa<RYOp>(raw)) {
    return NativeGateKind::Ry;
  }
  if (llvm::isa<ROp>(raw)) {
    return NativeGateKind::R;
  }
  return std::nullopt;
}

bool allowsSingleQubitOp(UnitaryOpInterface op, const NativeProfileSpec& spec) {
  if (llvm::isa<BarrierOp, GPhaseOp>(op.getOperation())) {
    return true;
  }
  const auto gate = singleQubitNativeGateKind(op);
  return gate && spec.allowedGates.contains(*gate);
}

/// True when `decomposeTo*` should run instead of folding to a constant `2×2`
/// matrix: trivial `Id`/`P`, dynamic-angle ops the matrix path cannot close
/// over, and (for ZSXX with direct Rx) `Rx`/`Ry`/`R`. Static angles still use
/// matrix + Euler.
bool canDirectlyDecomposeToZSXX(Operation* op, bool supportsDirectRx) {
  if (llvm::isa<IdOp, POp>(op)) {
    return true;
  }
  return supportsDirectRx && llvm::isa<RXOp, RYOp, ROp>(op);
}

bool canDirectlyDecomposeToU3(Operation* op) {
  return llvm::isa<IdOp, RXOp, RYOp, RZOp, POp, U2Op, ROp, UOp>(op);
}

bool canDirectlyDecomposeToR(Operation* op) {
  return llvm::isa<IdOp, ROp, RXOp, RYOp>(op);
}

bool canDirectlyDecomposeToAxisPair(Operation* op, AxisPair axisPair) {
  if (llvm::isa<IdOp>(op)) {
    return true;
  }
  switch (axisPair) {
  case AxisPair::RxRz:
    // `p` on an Rx/Rz axis pair folds directly to `rz(theta)`.
    return llvm::isa<RXOp, RZOp, POp>(op);
  case AxisPair::RxRy:
    // No cheap symbolic lowering of `p` without `rz` available.
    return llvm::isa<RXOp, RYOp>(op);
  case AxisPair::RyRz:
    return llvm::isa<RYOp, RZOp, POp>(op);
  }
  llvm_unreachable("unknown axis pair");
}

} // namespace mlir::qco::native_synth
