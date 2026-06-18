/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QCO/Transforms/NativeSynthesis/Utils.h"

#include "mlir/Dialect/QCO/IR/QCOInterfaces.h"
#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QCO/Utils/Matrix.h"

#include <llvm/ADT/APFloat.h>
#include <llvm/Support/Casting.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>

#include <cmath>
#include <complex>
#include <cstddef>
#include <optional>

namespace mlir::qco::native_synth {

Value createF64Const(IRRewriter& rewriter, Location loc, double value) {
  return arith::ConstantFloatOp::create(rewriter, loc, rewriter.getF64Type(),
                                        llvm::APFloat(value))
      .getResult();
}

std::optional<double> getConstantF64(Value value) {
  if (auto constant = value.getDefiningOp<arith::ConstantFloatOp>()) {
    if (auto floatAttr = llvm::dyn_cast<FloatAttr>(constant.getValue())) {
      return floatAttr.getValueAsDouble();
    }
  }
  return std::nullopt;
}

void emitGPhaseIfNonTrivial(IRRewriter& rewriter, Location loc, double phase) {
  constexpr double epsilon = 1e-12;
  if (std::abs(phase) > epsilon) {
    GPhaseOp::create(rewriter, loc, createF64Const(rewriter, loc, phase));
  }
}

bool getBlockTwoQubitMatrix(Operation* op, Matrix4x4& matrix) {
  if (llvm::isa<BarrierOp, GPhaseOp>(op)) {
    return false;
  }
  if (auto ctrl = llvm::dyn_cast<CtrlOp>(op)) {
    if (ctrl.getNumControls() != 1 || ctrl.getNumTargets() != 1) {
      return false;
    }
    auto* body = ctrl.getBodyUnitary(0).getOperation();
    if (llvm::isa<XOp>(body)) {
      // CX matrix in the same 4x4 basis layout as ``getUnitaryMatrix4x4``.
      matrix = Matrix4x4::fromElements(1, 0, 0, 0, //
                                       0, 1, 0, 0, //
                                       0, 0, 0, 1, //
                                       0, 0, 1, 0);
      return true;
    }
    if (llvm::isa<ZOp>(body)) {
      matrix = Matrix4x4::identity();
      matrix(3, 3) = -1.0;
      return true;
    }
    return false;
  }
  auto unitary = llvm::dyn_cast<UnitaryOpInterface>(op);
  if (!unitary || !unitary.isTwoQubit()) {
    return false;
  }
  Matrix4x4 raw;
  if (!unitary.getUnitaryMatrix4x4(raw)) {
    return false;
  }
  matrix = raw;
  return true;
}

} // namespace mlir::qco::native_synth
