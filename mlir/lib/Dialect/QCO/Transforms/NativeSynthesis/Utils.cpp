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

#include "mlir/Dialect/QCO/IR/QCOOps.h"

#include <Eigen/LU>
#include <llvm/ADT/APFloat.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/BuiltinAttributes.h>

#include <cmath>
#include <complex>

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
  if (std::abs(det) > 1e-16) {
    matrix *=
        std::pow(std::abs(det), -0.25) * std::exp(1i * (-std::arg(det) / 4.0));
  }
}

bool getNormalizedTwoQubitMatrix(UnitaryOpInterface unitary,
                                 Eigen::Matrix4cd& matrix) {
  if (!unitary.getUnitaryMatrix4x4(matrix)) {
    return false;
  }
  normalizeToSU4(matrix);
  return true;
}

bool getBlockTwoQubitMatrix(Operation* op, Eigen::Matrix4cd& matrix) {
  if (isa<BarrierOp, GPhaseOp>(op)) {
    return false;
  }
  if (auto ctrl = dyn_cast<CtrlOp>(op)) {
    if (ctrl.getNumControls() != 1 || ctrl.getNumTargets() != 1) {
      return false;
    }
    auto* body = ctrl.getBodyUnitary().getOperation();
    if (isa<XOp>(body)) {
      // CX matrix in the same 4x4 basis layout as ``getUnitaryMatrix4x4``.
      matrix << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0;
      return true;
    }
    if (isa<ZOp>(body)) {
      matrix = Eigen::Matrix4cd::Identity();
      matrix(3, 3) = -1.0;
      return true;
    }
    return false;
  }
  auto unitary = dyn_cast<UnitaryOpInterface>(op);
  if (!unitary || !unitary.isTwoQubit()) {
    return false;
  }
  return unitary.getUnitaryMatrix4x4(matrix);
}

namespace {

/// Emit a single-qubit gate from a decomposition gate, threading `target`.
/// Returns `failure()` if the gate kind/parameter count is unsupported.
LogicalResult emitSingleQubitStep(IRRewriter& rewriter, Location loc,
                                  const decomposition::Gate& gate,
                                  Value& target) {
  const auto emitConst = [&](double v) {
    return createF64Const(rewriter, loc, v);
  };
  switch (gate.type) {
  case decomposition::GateKind::U:
    if (gate.parameter.size() != 3) {
      return failure();
    }
    // EulerDecomposition emits `U` with parameters = {lambda, phi, theta}
    // whereas `UOp` takes (theta, phi, lambda); reorder accordingly.
    target =
        UOp::create(rewriter, loc, target, emitConst(gate.parameter[2]),
                    emitConst(gate.parameter[1]), emitConst(gate.parameter[0]))
            .getOutputQubit(0);
    return success();
  case decomposition::GateKind::SX:
    target = SXOp::create(rewriter, loc, target).getOutputQubit(0);
    return success();
  case decomposition::GateKind::X:
    target = XOp::create(rewriter, loc, target).getOutputQubit(0);
    return success();
  case decomposition::GateKind::RX:
    if (gate.parameter.size() != 1) {
      return failure();
    }
    target = RXOp::create(rewriter, loc, target, emitConst(gate.parameter[0]))
                 .getOutputQubit(0);
    return success();
  case decomposition::GateKind::RY:
    if (gate.parameter.size() != 1) {
      return failure();
    }
    target = RYOp::create(rewriter, loc, target, emitConst(gate.parameter[0]))
                 .getOutputQubit(0);
    return success();
  case decomposition::GateKind::RZ:
    if (gate.parameter.size() != 1) {
      return failure();
    }
    target = RZOp::create(rewriter, loc, target, emitConst(gate.parameter[0]))
                 .getOutputQubit(0);
    return success();
  default:
    return failure();
  }
}

} // namespace

LogicalResult
emitTwoQubitGateSequenceAtLoc(IRRewriter& rewriter, Location loc, Value qubit0,
                              Value qubit1,
                              const decomposition::TwoQubitGateSequence& seq,
                              Value& outQubit0, Value& outQubit1) {
  for (const auto& gate : seq.gates) {
    if (gate.qubitId.size() == 1) {
      Value& target = (gate.qubitId[0] == 0) ? qubit0 : qubit1;
      if (failed(emitSingleQubitStep(rewriter, loc, gate, target))) {
        return failure();
      }
      continue;
    }

    const bool isCxOrCz =
        gate.qubitId.size() == 2 && (gate.type == decomposition::GateKind::X ||
                                     gate.type == decomposition::GateKind::Z);
    if (!isCxOrCz) {
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
  rewriter.replaceOp(op, ValueRange{outQubit0, outQubit1});
  return success();
}

} // namespace mlir::qco::native_synth
