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
#include "mlir/Dialect/QCO/Transforms/Decomposition/Gate.h"
#include "mlir/Dialect/QCO/Transforms/Decomposition/GateKind.h"
#include "mlir/Dialect/QCO/Transforms/Decomposition/GateSequence.h"
#include "mlir/Dialect/QCO/Utils/Matrix.h"

#include <llvm/ADT/APFloat.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Casting.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LogicalResult.h>

#include <Eigen/LU>
#include <cmath>
#include <complex>
#include <optional>

namespace mlir::qco::native_synth {

Eigen::Matrix2cd toEigen(const Matrix2x2& matrix) {
  Eigen::Matrix2cd out;
  for (std::size_t row = 0; row < 2; ++row) {
    for (std::size_t col = 0; col < 2; ++col) {
      out(static_cast<Eigen::Index>(row), static_cast<Eigen::Index>(col)) =
          matrix(row, col);
    }
  }
  return out;
}

Eigen::Matrix4cd toEigen(const Matrix4x4& matrix) {
  Eigen::Matrix4cd out;
  for (std::size_t row = 0; row < 4; ++row) {
    for (std::size_t col = 0; col < 4; ++col) {
      out(static_cast<Eigen::Index>(row), static_cast<Eigen::Index>(col)) =
          matrix(row, col);
    }
  }
  return out;
}

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

bool isEquivalentUpToGlobalPhase(const Eigen::Matrix4cd& lhs,
                                 const Eigen::Matrix4cd& rhs, double atol) {
  const auto overlap = (rhs.adjoint() * lhs).trace();
  if (std::abs(overlap) <= atol) {
    return false;
  }
  const auto factor = overlap / std::abs(overlap);
  return lhs.isApprox(factor * rhs, atol);
}

void normalizeToSU4(Eigen::Matrix4cd& matrix) {
  using namespace std::complex_literals;
  const std::complex<double> det = matrix.determinant();
  // Project `matrix` into SU(4) by dividing out the fourth root of its
  // determinant (det(SU(N)) == 1). `|det|^{-1/4}` fixes the magnitude and
  // `exp(-i * arg(det) / 4)` removes the global phase so the Weyl
  // decomposition downstream operates on a special-unitary input.
  if (std::abs(det) > 1e-16) {
    matrix *=
        std::pow(std::abs(det), -0.25) * std::exp(1i * (-std::arg(det) / 4.0));
  }
}

bool getNormalizedTwoQubitMatrix(UnitaryOpInterface unitary,
                                 Eigen::Matrix4cd& matrix) {
  Matrix4x4 raw;
  if (!unitary.getUnitaryMatrix4x4(raw)) {
    return false;
  }
  matrix = toEigen(raw);
  normalizeToSU4(matrix);
  return true;
}

bool getBlockTwoQubitMatrix(Operation* op, Eigen::Matrix4cd& matrix) {
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
      matrix << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0;
      return true;
    }
    if (llvm::isa<ZOp>(body)) {
      matrix = Eigen::Matrix4cd::Identity();
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
  matrix = toEigen(raw);
  return true;
}

/// Emit a single-qubit gate from a decomposition gate, threading `target` and
/// recording the inserted op (if any) in `insertedOps` so the caller can roll
/// back on failure.
static LogicalResult
emitSingleQubitStep(IRRewriter& rewriter, Location loc,
                    const decomposition::Gate& gate, Value& target,
                    llvm::SmallVectorImpl<Operation*>& insertedOps) {
  const auto emitConst = [&](double v) {
    auto constant = arith::ConstantFloatOp::create(
        rewriter, loc, rewriter.getF64Type(), llvm::APFloat(v));
    insertedOps.push_back(constant);
    return constant.getResult();
  };
  const auto record = [&](auto op) {
    insertedOps.push_back(op.getOperation());
    return op;
  };
  switch (gate.type) {
  case decomposition::GateKind::I:
    return success();
  case decomposition::GateKind::U:
    if (gate.parameter.size() != 3) {
      return failure();
    }
    target =
        record(UOp::create(rewriter, loc, target, emitConst(gate.parameter[0]),
                           emitConst(gate.parameter[1]),
                           emitConst(gate.parameter[2])))
            .getOutputQubit(0);
    return success();
  case decomposition::GateKind::U2:
    if (gate.parameter.size() != 2) {
      return failure();
    }
    target =
        record(U2Op::create(rewriter, loc, target, emitConst(gate.parameter[0]),
                            emitConst(gate.parameter[1])))
            .getOutputQubit(0);
    return success();
  case decomposition::GateKind::SX:
    target = record(SXOp::create(rewriter, loc, target)).getOutputQubit(0);
    return success();
  case decomposition::GateKind::X:
    target = record(XOp::create(rewriter, loc, target)).getOutputQubit(0);
    return success();
  case decomposition::GateKind::RX:
    if (gate.parameter.size() != 1) {
      return failure();
    }
    target = record(RXOp::create(rewriter, loc, target,
                                 emitConst(gate.parameter[0])))
                 .getOutputQubit(0);
    return success();
  case decomposition::GateKind::RY:
    if (gate.parameter.size() != 1) {
      return failure();
    }
    target = record(RYOp::create(rewriter, loc, target,
                                 emitConst(gate.parameter[0])))
                 .getOutputQubit(0);
    return success();
  case decomposition::GateKind::RZ:
    if (gate.parameter.size() != 1) {
      return failure();
    }
    target = record(RZOp::create(rewriter, loc, target,
                                 emitConst(gate.parameter[0])))
                 .getOutputQubit(0);
    return success();
  default:
    return failure();
  }
}

/// Erase all ops tracked in `insertedOps` in reverse insertion order.
static void
rollbackInsertedOps(IRRewriter& rewriter,
                    llvm::SmallVectorImpl<Operation*>& insertedOps) {
  for (Operation* op : llvm::reverse(insertedOps)) {
    rewriter.eraseOp(op);
  }
  insertedOps.clear();
}

LogicalResult
emitTwoQubitGateSequenceAtLoc(IRRewriter& rewriter, Location loc, Value qubit0,
                              Value qubit1,
                              const decomposition::TwoQubitGateSequence& seq,
                              Value& outQubit0, Value& outQubit1) {
  llvm::SmallVector<Operation*, 16> insertedOps;
  for (const auto& gate : seq.gates) {
    if (gate.qubitId.size() == 1) {
      Value& target = (gate.qubitId[0] == 0) ? qubit0 : qubit1;
      if (failed(
              emitSingleQubitStep(rewriter, loc, gate, target, insertedOps))) {
        rollbackInsertedOps(rewriter, insertedOps);
        return failure();
      }
      continue;
    }

    if (gate.qubitId.size() != 2) {
      rollbackInsertedOps(rewriter, insertedOps);
      return failure();
    }

    if (gate.type == decomposition::GateKind::RZZ) {
      if (gate.parameter.size() != 1) {
        rollbackInsertedOps(rewriter, insertedOps);
        return failure();
      }
      const decomposition::QubitId a = gate.qubitId[0];
      const decomposition::QubitId b = gate.qubitId[1];
      if (a + b != 1) {
        rollbackInsertedOps(rewriter, insertedOps);
        return failure();
      }
      const Value va = (a == 0) ? qubit0 : qubit1;
      const Value vb = (b == 0) ? qubit0 : qubit1;
      Value thetaVal = createF64Const(rewriter, loc, gate.parameter[0]);
      insertedOps.push_back(thetaVal.getDefiningOp());
      auto rzz = RZZOp::create(rewriter, loc, va, vb, thetaVal);
      insertedOps.push_back(rzz.getOperation());
      qubit0 = (gate.qubitId[0] == 0) ? rzz.getOutputQubit(0)
                                      : rzz.getOutputQubit(1);
      qubit1 = (gate.qubitId[0] == 1) ? rzz.getOutputQubit(0)
                                      : rzz.getOutputQubit(1);
      continue;
    }

    if (gate.type != decomposition::GateKind::X &&
        gate.type != decomposition::GateKind::Z) {
      rollbackInsertedOps(rewriter, insertedOps);
      return failure();
    }

    const decomposition::QubitId controlId = gate.qubitId[0];
    const decomposition::QubitId targetId = gate.qubitId[1];
    const Value controlIn = (controlId == 0) ? qubit0 : qubit1;
    const Value targetIn = (targetId == 0) ? qubit0 : qubit1;

    auto ctrlOp = CtrlOp::create(
        rewriter, loc, ValueRange{controlIn}, ValueRange{targetIn},
        [&](ValueRange targetArgs) -> llvm::SmallVector<Value> {
          if (gate.type == decomposition::GateKind::X) {
            return {
                XOp::create(rewriter, loc, targetArgs[0]).getOutputQubit(0)};
          }
          return {ZOp::create(rewriter, loc, targetArgs[0]).getOutputQubit(0)};
        });
    // Erasing the `CtrlOp` also removes its nested body op.
    insertedOps.push_back(ctrlOp.getOperation());
    const Value controlOut = ctrlOp.getOutputControl(0);
    const Value targetOut = ctrlOp.getOutputTarget(0);
    Value next0 = qubit0;
    Value next1 = qubit1;
    if (controlId == 0) {
      next0 = controlOut;
    } else {
      next1 = controlOut;
    }
    if (targetId == 0) {
      next0 = targetOut;
    } else {
      next1 = targetOut;
    }
    qubit0 = next0;
    qubit1 = next1;
  }

  outQubit0 = qubit0;
  outQubit1 = qubit1;
  return success();
}

LogicalResult
emitTwoQubitGateSequence(IRRewriter& rewriter, Operation* op, Value qubit0,
                         Value qubit1,
                         const decomposition::TwoQubitGateSequence& seq) {
  Value outQubit0;
  Value outQubit1;
  if (failed(emitTwoQubitGateSequenceAtLoc(
          rewriter, op->getLoc(), qubit0, qubit1, seq, outQubit0, outQubit1))) {
    return failure();
  }
  // Match `seq.getUnitaryMatrix()` / `PassTwoQubitWindows` materialization:
  // residual phase from Weyl + basis decomposition is not represented as 2q
  // ops in `seq.gates`.
  if (seq.hasGlobalPhase()) {
    emitGPhaseIfNonTrivial(rewriter, op->getLoc(), seq.globalPhase);
  }
  rewriter.replaceOp(op, ValueRange{outQubit0, outQubit1});
  return success();
}

} // namespace mlir::qco::native_synth
