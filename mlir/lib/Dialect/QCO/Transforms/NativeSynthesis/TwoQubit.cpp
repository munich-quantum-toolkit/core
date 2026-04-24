/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QCO/Transforms/NativeSynthesis/TwoQubit.h"

#include "mlir/Dialect/QCO/IR/QCOInterfaces.h"
#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QCO/Transforms/Decomposition/BasisDecomposer.h"
#include "mlir/Dialect/QCO/Transforms/Decomposition/EulerBasis.h"
#include "mlir/Dialect/QCO/Transforms/Decomposition/Gate.h"
#include "mlir/Dialect/QCO/Transforms/Decomposition/GateKind.h"
#include "mlir/Dialect/QCO/Transforms/Decomposition/GateSequence.h"
#include "mlir/Dialect/QCO/Transforms/Decomposition/WeylDecomposition.h"
#include "mlir/Dialect/QCO/Transforms/NativeSynthesis/Policy.h"
#include "mlir/Dialect/QCO/Transforms/NativeSynthesis/Types.h"
#include "mlir/Dialect/QCO/Transforms/NativeSynthesis/Utils.h"

#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Casting.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LogicalResult.h>

#include <algorithm>
#include <cstdint>
#include <numbers>
#include <optional>
#include <tuple>
#include <utility>

namespace mlir::qco::native_synth {

constexpr double PI = std::numbers::pi;
constexpr double HALF_PI = PI / 2.0;

/// Whether the given single-qubit emitter can lower a decomposition-IR gate
/// of `kind` (an intermediate from Euler/Weyl, *not* a `NativeGateKind`) to a
/// native output sequence.
static bool
emitterHandlesDecompositionGate(const SingleQubitEmitterSpec& emitter,
                                decomposition::GateKind kind) {
  if (kind == decomposition::GateKind::I) {
    return true;
  }
  switch (emitter.mode) {
  case SingleQubitMode::ZSXX:
    return kind == decomposition::GateKind::RZ ||
           kind == decomposition::GateKind::SX ||
           kind == decomposition::GateKind::X;
  case SingleQubitMode::U3:
    return kind == decomposition::GateKind::U;
  case SingleQubitMode::R:
    return kind == decomposition::GateKind::RX ||
           kind == decomposition::GateKind::RY ||
           kind == decomposition::GateKind::X ||
           kind == decomposition::GateKind::Y;
  case SingleQubitMode::AxisPair:
    switch (emitter.axisPair) {
    case AxisPair::RxRz:
      return kind == decomposition::GateKind::RX ||
             kind == decomposition::GateKind::RZ ||
             kind == decomposition::GateKind::X ||
             kind == decomposition::GateKind::Z;
    case AxisPair::RxRy:
      return kind == decomposition::GateKind::RX ||
             kind == decomposition::GateKind::RY ||
             kind == decomposition::GateKind::X ||
             kind == decomposition::GateKind::Y;
    case AxisPair::RyRz:
      return kind == decomposition::GateKind::RY ||
             kind == decomposition::GateKind::RZ ||
             kind == decomposition::GateKind::Y ||
             kind == decomposition::GateKind::Z;
    }
    break;
  }
  return false;
}

/// Check that a single decomposition gate is allowed by the profile menu.
static bool menuAllows(const decomposition::Gate& gate,
                       const NativeProfileSpec& spec) {
  if (gate.qubitId.size() == 1) {
    return std::ranges::any_of(spec.singleQubitEmitters,
                               [&gate](const SingleQubitEmitterSpec& emitter) {
                                 return emitterHandlesDecompositionGate(
                                     emitter, gate.type);
                               });
  }
  if (gate.qubitId.size() == 2) {
    switch (gate.type) {
    case decomposition::GateKind::X:
      return usesCxEntangler(spec);
    case decomposition::GateKind::Z:
      return usesCzEntangler(spec);
    case decomposition::GateKind::RZZ:
      return spec.allowRzz;
    default:
      return false;
    }
  }
  return false;
}

/// Whether `emitter` can lower the single-qubit `op` directly.
static bool emitterHasDirectLowering(Operation* op,
                                     const SingleQubitEmitterSpec& emitter) {
  switch (emitter.mode) {
  case SingleQubitMode::ZSXX:
    return canDirectlyDecomposeToZSXX(op, emitter.supportsDirectRx);
  case SingleQubitMode::U3:
    return canDirectlyDecomposeToU3(op);
  case SingleQubitMode::R:
    return canDirectlyDecomposeToR(op);
  case SingleQubitMode::AxisPair:
    return canDirectlyDecomposeToAxisPair(op, emitter.axisPair);
  }
  return false;
}

bool gateSequenceFitsMenu(const decomposition::TwoQubitGateSequence& seq,
                          const NativeProfileSpec& spec) {
  return std::ranges::all_of(seq.gates,
                             [&spec](const decomposition::Gate& gate) {
                               return menuAllows(gate, spec);
                             });
}

std::optional<decomposition::TwoQubitGateSequence>
decomposeTwoQubitFromMatrix(const Eigen::Matrix4cd& matrix,
                            EntanglerBasis entangler,
                            decomposition::EulerBasis eulerBasis,
                            std::optional<std::uint8_t> numBasisUses) {
  // Basis-gate qubit ids align with `getBlockTwoQubitMatrix` / CX layout.
  const decomposition::Gate basisGate{
      .type = entangler == EntanglerBasis::Cz ? decomposition::GateKind::Z
                                              : decomposition::GateKind::X,
      .qubitId = {0, 1},
  };
  auto decomposer =
      decomposition::TwoQubitBasisDecomposer::create(basisGate, 1.0);
  auto weyl =
      decomposition::TwoQubitWeylDecomposition::create(matrix, std::nullopt);
  return decomposer.twoQubitDecompose(
      weyl, llvm::SmallVector<decomposition::EulerBasis>{eulerBasis},
      std::nullopt, /*approximate=*/false, numBasisUses);
}

llvm::SmallVector<SynthesisCandidate<SingleQubitRewritePlan>>
collectSingleQubitCandidates(UnitaryOpInterface unitary,
                             const NativeProfileSpec& spec) {
  llvm::SmallVector<SynthesisCandidate<SingleQubitRewritePlan>> candidates;
  Operation* op = unitary.getOperation();
  unsigned enumerationIndex = 0;
  const auto addCandidate = [&](CandidateClass klass, CandidateMetrics metrics,
                                SingleQubitRewriteStrategy strategy,
                                const SingleQubitEmitterSpec& emitter) {
    candidates.push_back(SynthesisCandidate<SingleQubitRewritePlan>{
        .candidateClass = klass,
        .metrics = metrics,
        .enumerationIndex = enumerationIndex++,
        .payload =
            SingleQubitRewritePlan{.strategy = strategy, .emitter = emitter},
    });
  };
  for (const auto& emitter : spec.singleQubitEmitters) {
    if (emitterHasDirectLowering(op, emitter)) {
      addCandidate(CandidateClass::DirectSingleQ,
                   estimateDirectSingleQubitMetrics(op, emitter),
                   SingleQubitRewriteStrategy::Direct, emitter);
    }
    if (auto matrixMetrics =
            estimateMatrixSingleQubitMetrics(unitary, emitter)) {
      addCandidate(CandidateClass::MatrixSingleQ, *matrixMetrics,
                   SingleQubitRewriteStrategy::MatrixFallback, emitter);
    }
  }
  return candidates;
}

/// Try every `numBasisUses` in `{0, 1, 2, 3}` for the `(entangler, emitter,
/// basis)` triple, running the Weyl-based basis decomposer for each. Any
/// resulting gate sequence that both matches `targetMatrix` up to global
/// phase AND stays inside the native menu is appended to `candidates`.
static void tryAddTwoQubitBasisCandidatesForEmitterBasis(
    llvm::SmallVector<SynthesisCandidate<TwoQubitRewritePlan>, 0>& candidates,
    unsigned& enumerationIndex, const Eigen::Matrix4cd& targetMatrix,
    const NativeProfileSpec& spec, EntanglerBasis entangler,
    const SingleQubitEmitterSpec& emitter, decomposition::EulerBasis basis) {
  // An arbitrary 2-qubit unitary can always be realized using at most three
  // copies of any fixed (non-diagonal) entangler plus local gates -- this is
  // a consequence of the KAK/Weyl decomposition. Trying all four candidate
  // counts (0..3) and scoring them with the gate-sequence metric lets the
  // outer pass pick the cheapest realization for the particular target
  // unitary (e.g. local unitaries collapse to 0 entanglers, SWAP uses 3).
  for (std::uint8_t numBasisUses = 0; numBasisUses <= 3; ++numBasisUses) {
    auto seq = decomposeTwoQubitFromMatrix(targetMatrix, entangler, basis,
                                           numBasisUses);
    // Two independent checks: `isEquivalentUpToGlobalPhase` verifies the
    // numerical decomposition actually reproduces the target; `fitsMenu`
    // verifies every emitted gate kind is in the backend native set. Both
    // are required because the decomposer can legitimately produce an
    // accurate sequence that still contains non-native gates (e.g. when the
    // requested emitter supports fewer axes than the target unitary needs).
    if (!seq ||
        !isEquivalentUpToGlobalPhase(seq->getUnitaryMatrix(), targetMatrix) ||
        !gateSequenceFitsMenu(*seq, spec)) {
      continue;
    }
    candidates.push_back(SynthesisCandidate<TwoQubitRewritePlan>{
        .candidateClass = CandidateClass::TwoQubitBasisRewrite,
        .metrics = computeGateSequenceMetrics(*seq),
        .enumerationIndex = enumerationIndex++,
        .payload = {.sequence = *seq,
                    .emitter = emitter,
                    .entanglerBasis = entangler},
    });
  }
}

llvm::SmallVector<SynthesisCandidate<TwoQubitRewritePlan>, 0>
collectTwoQubitBasisCandidatesFromMatrix(const Eigen::Matrix4cd& targetMatrix,
                                         const NativeProfileSpec& spec) {
  llvm::SmallVector<SynthesisCandidate<TwoQubitRewritePlan>, 0> candidates;
  if (spec.entanglerBases.empty()) {
    return candidates;
  }
  unsigned enumerationIndex = 0;
  for (const auto entangler : spec.entanglerBases) {
    for (const auto& emitter : spec.singleQubitEmitters) {
      for (const auto basis : emitter.eulerBases) {
        tryAddTwoQubitBasisCandidatesForEmitterBasis(
            candidates, enumerationIndex, targetMatrix, spec, entangler,
            emitter, basis);
      }
    }
  }
  return candidates;
}

CandidateMetrics xxPlusMinusYyRzzRewriteScoringMetrics() {
  // Tallies for `rewriteXXPlusMinusYYViaRxxRyy` (identical for `XXPlusYY` and
  // `XXMinusYY`): leading/final `rz` on `q0` (2) + `ryy` via `rzz` (four 1q +
  // one `rzz`) + `rxx` via `rzz` (four `(rz, sx, rz)` per wire around each
  // `rzz`, i.e. twelve 1q + one `rzz`).
  constexpr unsigned numOneQ = 18;
  constexpr unsigned numTwoQ = 2;
  constexpr unsigned depth = 10;
  return {.numOneQ = numOneQ, .numTwoQ = numTwoQ, .depth = depth};
}

llvm::SmallVector<SynthesisCandidate<TwoQubitRewritePlan>, 0>
collectTwoQubitBasisCandidates(UnitaryOpInterface unitary,
                               const NativeProfileSpec& spec) {
  Eigen::Matrix4cd target;
  if (!getNormalizedTwoQubitMatrix(unitary, target)) {
    return {};
  }
  return collectTwoQubitBasisCandidatesFromMatrix(target, spec);
}

LogicalResult rewriteXXPlusMinusYYViaRxxRyy(IRRewriter& rewriter,
                                            Operation* op) {
  rewriter.setInsertionPoint(op);
  const auto loc = op->getLoc();
  const auto constF = [&](double v) {
    return createF64Const(rewriter, loc, v);
  };
  const auto half = [&](Value v) -> Value {
    if (auto c = getConstantF64(v)) {
      return constF(*c * 0.5);
    }
    return arith::MulFOp::create(rewriter, loc, v, constF(0.5)).getResult();
  };
  const auto neg = [&](Value v) -> Value {
    if (auto c = getConstantF64(v)) {
      return constF(-*c);
    }
    return arith::NegFOp::create(rewriter, loc, v).getResult();
  };
  const auto emitH = [&](Value q) -> Value {
    auto rz0 = RZOp::create(rewriter, loc, q, constF(HALF_PI));
    auto sx = SXOp::create(rewriter, loc, rz0.getOutputQubit(0));
    return RZOp::create(rewriter, loc, sx.getOutputQubit(0), constF(HALF_PI))
        .getOutputQubit(0);
  };
  // Realize `Rxx(theta)` as `(H ⊗ H) * Rzz(theta) * (H ⊗ H)`: Hadamard
  // conjugation maps the Z axis to X on each qubit, and the tensor-product
  // identity `(H ⊗ H) * ZZ * (H ⊗ H) == XX` lifts that to the entangler.
  const auto emitRxxViaRzz = [&](Value q0, Value q1,
                                 Value theta) -> std::pair<Value, Value> {
    q0 = emitH(q0);
    q1 = emitH(q1);
    auto rzz = RZZOp::create(rewriter, loc, q0, q1, theta);
    q0 = rzz.getOutputQubit(0);
    q1 = rzz.getOutputQubit(1);
    return {emitH(q0), emitH(q1)};
  };
  // Realize `Ryy(theta)` as `(Rx(-pi/2) ⊗ Rx(-pi/2)) * Rzz(theta) *
  // (Rx(pi/2) ⊗ Rx(pi/2))`: Rx(pi/2) maps Z to Y on each qubit, so the
  // conjugation transports `ZZ` to `YY` just like the Hadamard sandwich
  // above maps it to `XX`.
  const auto emitRyyViaRzz = [&](Value q0, Value q1,
                                 Value theta) -> std::pair<Value, Value> {
    auto rx0 = RXOp::create(rewriter, loc, q0, constF(HALF_PI));
    auto rx1 = RXOp::create(rewriter, loc, q1, constF(HALF_PI));
    auto rzz = RZZOp::create(rewriter, loc, rx0.getOutputQubit(0),
                             rx1.getOutputQubit(0), theta);
    auto rxb0 =
        RXOp::create(rewriter, loc, rzz.getOutputQubit(0), constF(-HALF_PI));
    auto rxb1 =
        RXOp::create(rewriter, loc, rzz.getOutputQubit(1), constF(-HALF_PI));
    return {rxb0.getOutputQubit(0), rxb1.getOutputQubit(0)};
  };

  // `XXPlusYY(theta, beta)` and `XXMinusYY(theta, beta)` both act as
  //   Rz(-beta) on q0 -> entangling core -> Rz(+beta) on q0,
  // but differ in the entangling core:
  //   XXPlusYY:  exp(-i * theta/4 * (XX + YY))  == Ryy(theta/2) * Rxx(theta/2)
  //   XXMinusYY: exp(-i * theta/4 * (XX - YY))  == Rxx(theta/2) * Ryy(-theta/2)
  // (XX and YY commute, so the two multiplication orders produce identical
  // unitaries; the distinct order and sign below are what makes `XXMinusYY`
  // the "minus" variant and must be preserved even though an order flip
  // alone would also compile.)
  if (auto xxPlus = llvm::dyn_cast<XXPlusYYOp>(op)) {
    Value q0 = xxPlus.getInputQubit(0);
    Value q1 = xxPlus.getInputQubit(1);
    q0 = RZOp::create(rewriter, loc, q0, neg(xxPlus.getBeta()))
             .getOutputQubit(0);
    const auto halfTheta = half(xxPlus.getTheta());
    std::tie(q0, q1) = emitRyyViaRzz(q0, q1, halfTheta);
    std::tie(q0, q1) = emitRxxViaRzz(q0, q1, halfTheta);
    q0 = RZOp::create(rewriter, loc, q0, xxPlus.getBeta()).getOutputQubit(0);
    rewriter.replaceOp(op, ValueRange{q0, q1});
    return success();
  }
  if (auto xxMinus = llvm::dyn_cast<XXMinusYYOp>(op)) {
    Value q0 = xxMinus.getInputQubit(0);
    Value q1 = xxMinus.getInputQubit(1);
    q0 = RZOp::create(rewriter, loc, q0, neg(xxMinus.getBeta()))
             .getOutputQubit(0);
    const auto halfTheta = half(xxMinus.getTheta());
    std::tie(q0, q1) = emitRxxViaRzz(q0, q1, halfTheta);
    std::tie(q0, q1) = emitRyyViaRzz(q0, q1, neg(halfTheta));
    q0 = RZOp::create(rewriter, loc, q0, xxMinus.getBeta()).getOutputQubit(0);
    rewriter.replaceOp(op, ValueRange{q0, q1});
    return success();
  }
  return failure();
}

} // namespace mlir::qco::native_synth
