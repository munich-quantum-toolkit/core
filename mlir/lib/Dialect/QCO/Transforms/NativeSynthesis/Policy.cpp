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
#include "mlir/Dialect/QCO/Transforms/Decomposition/EulerBasis.h"
#include "mlir/Dialect/QCO/Transforms/Decomposition/EulerDecomposition.h"
#include "mlir/Dialect/QCO/Transforms/Decomposition/GateKind.h"
#include "mlir/Dialect/QCO/Transforms/Decomposition/GateSequence.h"
#include "mlir/Dialect/QCO/Transforms/NativeSynthesis/NativeSpec.h"
#include "mlir/Dialect/QCO/Transforms/NativeSynthesis/Types.h"
#include "mlir/Dialect/QCO/Transforms/NativeSynthesis/Utils.h"
#include "mlir/Dialect/QCO/Utils/Matrix.h"

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/ErrorHandling.h>
#include <mlir/IR/Operation.h>

#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <optional>

namespace mlir::qco::native_synth {

bool areValidScoreWeights(const ScoreWeights& weights) {
  return std::isfinite(weights.twoQ) && std::isfinite(weights.oneQ) &&
         std::isfinite(weights.depth) && weights.twoQ >= 0.0 &&
         weights.oneQ >= 0.0 && weights.depth >= 0.0;
}

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

CandidateMetrics
computeGateSequenceMetrics(const decomposition::QubitGateSequence& seq) {
  CandidateMetrics metrics;
  // Per-qubit depth counters used as a mini scheduler: single-qubit gates
  // advance only their own wire's counter, while two-qubit gates act as a
  // *sync barrier* and advance both wires to `1 + max(...)`. This mirrors a
  // simple ASAP scheduling model where entangling gates force alignment of
  // the two wires they touch.
  llvm::SmallVector<unsigned, 2> qubitDepths(2, 0);
  for (const auto& gate : seq.gates) {
    if (gate.qubitId.size() == 2) {
      ++metrics.numTwoQ;
      const unsigned q0 = gate.qubitId[0];
      const unsigned q1 = gate.qubitId[1];
      const unsigned neededSize = std::max(q0, q1) + 1;
      if (neededSize > qubitDepths.size()) {
        qubitDepths.resize(neededSize, 0);
      }
      const auto gateDepth = std::max(qubitDepths[q0], qubitDepths[q1]) + 1;
      qubitDepths[q0] = gateDepth;
      qubitDepths[q1] = gateDepth;
      metrics.depth = std::max(metrics.depth, gateDepth);
    } else if (gate.qubitId.size() == 1) {
      ++metrics.numOneQ;
      const unsigned q = gate.qubitId[0];
      if (q >= qubitDepths.size()) {
        qubitDepths.resize(q + 1, 0);
      }
      const auto gateDepth = qubitDepths[q] + 1;
      qubitDepths[q] = gateDepth;
      metrics.depth = std::max(metrics.depth, gateDepth);
    }
  }
  return metrics;
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

CandidateMetrics
estimateDirectSingleQubitMetrics(Operation* op,
                                 const SingleQubitEmitterSpec& emitter) {
  if (llvm::isa<IdOp>(op)) {
    return {};
  }
  // ZSXX + direct Rx: `ry`/`r` use a three-gate `rz * rx * rz` sandwich; other
  // direct paths emit a single native op.
  const bool threeGate = emitter.mode == SingleQubitMode::ZSXX &&
                         emitter.supportsDirectRx && llvm::isa<RYOp, ROp>(op);
  const unsigned count = threeGate ? 3U : 1U;
  return {.numOneQ = count, .numTwoQ = 0, .depth = count};
}

std::optional<CandidateMetrics>
estimateMatrixSingleQubitMetrics(UnitaryOpInterface unitary,
                                 const SingleQubitEmitterSpec& emitter) {
  if (!unitary.isSingleQubit()) {
    return std::nullopt;
  }
  Matrix2x2 raw;
  if (!unitary.getUnitaryMatrix2x2(raw)) {
    return std::nullopt;
  }
  const Eigen::Matrix2cd matrix = toEigen(raw);

  const auto countNonIdentity =
      [](const decomposition::QubitGateSequence& seq) {
        CandidateMetrics metrics;
        for (const auto& gate : seq.gates) {
          if (gate.type != decomposition::GateKind::I) {
            ++metrics.numOneQ;
          }
        }
        metrics.depth = metrics.numOneQ;
        return metrics;
      };

  switch (emitter.mode) {
  case SingleQubitMode::ZSXX:
    return computeGateSequenceMetrics(
        decomposition::EulerDecomposition::generateCircuit(
            decomposition::GateEulerBasis::ZSXX, matrix, /*simplify=*/true,
            std::nullopt));
  case SingleQubitMode::U3:
    return CandidateMetrics{.numOneQ = 1, .numTwoQ = 0, .depth = 1};
  case SingleQubitMode::R:
    return countNonIdentity(decomposition::EulerDecomposition::generateCircuit(
        decomposition::GateEulerBasis::XYX, matrix, /*simplify=*/true,
        std::nullopt));
  case SingleQubitMode::AxisPair: {
    const auto bases = getEulerBasesForAxisPair(emitter.axisPair);
    if (bases.empty()) {
      return std::nullopt;
    }
    return countNonIdentity(decomposition::EulerDecomposition::generateCircuit(
        bases.front(), matrix, /*simplify=*/true, std::nullopt));
  }
  }
  llvm_unreachable("unknown single-qubit mode");
}

} // namespace mlir::qco::native_synth
