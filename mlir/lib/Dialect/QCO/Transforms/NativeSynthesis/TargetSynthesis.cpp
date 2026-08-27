/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Compiler/Target.h"
#include "mlir/Dialect/MQT/Transforms/GlobalPhaseNormalization.h"
#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QCO/IR/QCOInterfaces.h"
#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QCO/Transforms/Decomposition/Euler.h"
#include "mlir/Dialect/QCO/Transforms/Decomposition/Weyl.h"
#include "mlir/Dialect/QCO/Transforms/Passes.h"
#include "mlir/Dialect/QCO/Utils/Matrix.h"
#include "mlir/Dialect/QTensor/IR/QTensorOps.h"

#include <llvm/ADT/STLExtras.h>
#include <llvm/Support/ErrorHandling.h>
#include <mlir/Dialect/Arith/IR/Arith.h> // IWYU pragma: keep (Passes.h.inc)
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/Interfaces/FunctionInterfaces.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Support/TypeID.h>
#include <mlir/Support/WalkResult.h>

#include <cassert>
#include <cstddef>
#include <memory>
#include <optional>

namespace mlir::qco {

using decomposition::decomposeUnitary2QWeyl;
using decomposition::emitUnitary2QWeyl;

namespace {

/** Composed unitary and metadata for a fusable two-qubit run. */
struct FusableTwoQubitRun {
  SmallVector<Operation*, 8> ops; ///< Members in program order.
  Matrix4x4 composed = Matrix4x4::identity();
  unsigned numTwoQ = 0; ///< Number of two-qubit members (entanglers consumed).
  Value tailA;          ///< Current output wires of the run's tail.
  Value tailB;
};

} // namespace

// --- Run membership ------------------------------------------------------- //

/// Whether `op` is nested under a modifier body. Such unitaries are handled
/// through their shell op, so the top-level walk skips them.
static bool isExcludedFromTopLevelUnitaryWalk(Operation* op) {
  return op->getParentOfType<CtrlOp>() || op->getParentOfType<InvOp>() ||
         op->getParentOfType<PowOp>();
}

/// Whether `op` is a unitary shell the pass may rewrite at top level.
static bool isWalkableUnitaryShell(Operation* op) {
  return !isa<BarrierOp, GPhaseOp>(op) &&
         !isExcludedFromTopLevelUnitaryWalk(op);
}

/// Builds the constant 4x4 matrix for a two-qubit op (bare or single-target
/// `CtrlOp`). Returns false for a `CtrlOp` that is not
/// single-control/single-target, or an op whose matrix is not known at compile
/// time.
static bool assignTwoQubitOpMatrix(Operation* op, Matrix4x4& matrix) {
  if (auto ctrl = dyn_cast<CtrlOp>(op)) {
    if (ctrl.getNumControls() != 1 || ctrl.getNumTargets() != 1) {
      return false;
    }
    return cast<UnitaryOpInterface>(ctrl.getOperation())
        .getUnitaryMatrix4x4(matrix);
  }
  auto unitary = cast<UnitaryOpInterface>(op);
  assert(unitary.isTwoQubit() &&
         "only two-qubit unitary shells are passed to assignTwoQubitOpMatrix");
  return unitary.getUnitaryMatrix4x4(matrix);
}

/// Return the constant matrix when `unitary` is a single-qubit run member.
static std::optional<Matrix2x2>
oneQubitRunMemberMatrix(UnitaryOpInterface unitary) {
  if (!unitary || !unitary.isSingleQubit() ||
      !isWalkableUnitaryShell(unitary.getOperation())) {
    return std::nullopt;
  }
  Matrix2x2 matrix;
  if (!unitary.getUnitaryMatrix2x2(matrix)) {
    return std::nullopt;
  }
  return matrix;
}

/// Return the constant matrix when `unitary` is a two-qubit run member.
static std::optional<Matrix4x4>
twoQubitRunMemberMatrix(UnitaryOpInterface unitary) {
  if (!unitary || !unitary.isTwoQubit() ||
      !isWalkableUnitaryShell(unitary.getOperation())) {
    return std::nullopt;
  }
  Matrix4x4 matrix;
  if (!assignTwoQubitOpMatrix(unitary.getOperation(), matrix)) {
    return std::nullopt;
  }
  return matrix;
}

// --- Wire navigation ------------------------------------------------------ //

/// The sole walkable one- or two-qubit consumer of `wire`, or a null interface.
/// `wire` is single-use by qubit linearity.
static UnitaryOpInterface uniqueUnitaryUser(Value wire) {
  assert(wire.hasOneUse() &&
         "qubit values are single-use, so a run tail has exactly one user");
  auto unitary = dyn_cast<UnitaryOpInterface>(*wire.user_begin());
  if (!unitary || !isWalkableUnitaryShell(unitary.getOperation()) ||
      (!unitary.isSingleQubit() && !unitary.isTwoQubit())) {
    return {};
  }
  return unitary;
}

/// Traces `wire` upstream through single-qubit gates to the two-qubit run
/// member terminating the chain, or `nullptr` if the chain is broken.
static Operation* twoQubitGateAtEndOfOneQChain(Value wire) {
  Value cur = wire;
  while (Operation* def = cur.getDefiningOp()) {
    auto unitary = dyn_cast<UnitaryOpInterface>(def);
    if (!unitary) {
      return nullptr;
    }
    if (unitary.isTwoQubit()) {
      return twoQubitRunMemberMatrix(unitary) ? def : nullptr;
    }
    if (!oneQubitRunMemberMatrix(unitary)) {
      return nullptr;
    }
    cur = unitary.getInputQubit(0);
  }
  return nullptr;
}

/// Whether both input wires of `op` come from one earlier two-qubit run, making
/// `op` a continuation of that run rather than a fresh run start.
static bool feedsFromSameTwoQubitRun(UnitaryOpInterface op) {
  Value in0 = op.getInputQubit(0);
  Value in1 = op.getInputQubit(1);
  assert(in0.hasOneUse() && in1.hasOneUse() &&
         "qubit values are single-use, so a run member consumes each input "
         "exactly once");
  Operation* gate0 = twoQubitGateAtEndOfOneQChain(in0);
  Operation* gate1 = twoQubitGateAtEndOfOneQChain(in1);
  return gate0 != nullptr && gate0 == gate1;
}

// --- Run scanning --------------------------------------------------------- //

/// Appends a two-qubit gate to `run`, composing its matrix. No-op unless both
/// of `op`'s inputs are the run's current tail wires (in either order), keeping
/// the run confined to a single pair of wires.
static void absorbTwoQubitIntoRun(FusableTwoQubitRun& run,
                                  UnitaryOpInterface op,
                                  const Matrix4x4& opMatrix) {
  Value in0 = op.getInputQubit(0);
  Value in1 = op.getInputQubit(1);
  size_t id0 = 0;
  size_t id1 = 1;
  if (in0 == run.tailA && in1 == run.tailB) {
    run.tailA = op.getOutputQubit(0);
    run.tailB = op.getOutputQubit(1);
  } else if (in0 == run.tailB && in1 == run.tailA) {
    id0 = 1;
    id1 = 0;
    run.tailA = op.getOutputQubit(1);
    run.tailB = op.getOutputQubit(0);
  } else {
    llvm_unreachable(
        "a unique user of both tail wires connects to both of them");
  }
  run.composed.premultiplyBy(opMatrix.reorderForQubits(id0, id1));
  run.ops.push_back(op.getOperation());
  ++run.numTwoQ;
}

/// Appends a single-qubit gate on run wire `wireIndex` (0 = A, 1 = B).
static void absorbOneQubitIntoRun(FusableTwoQubitRun& run,
                                  UnitaryOpInterface op,
                                  const Matrix2x2& opMatrix,
                                  unsigned wireIndex) {
  run.composed.premultiplyBy(opMatrix.embedInTwoQubit(wireIndex));
  run.ops.push_back(op.getOperation());
  (wireIndex == 0 ? run.tailA : run.tailB) = op.getOutputQubit(0);
}

/// Walks forward from `head`, composing the run's matrix and metadata. Absorbs
/// a following two-qubit gate when it keeps both run wires together, otherwise
/// the single-qubit gate first in program order; stops at the first boundary
/// that would split the run's two wires.
static FusableTwoQubitRun scanFusableTwoQubitRun(UnitaryOpInterface head,
                                                 const Matrix4x4& headMatrix) {
  FusableTwoQubitRun run;
  run.composed = headMatrix;
  run.tailA = head.getOutputQubit(0);
  run.tailB = head.getOutputQubit(1);
  run.ops.push_back(head.getOperation());
  run.numTwoQ = 1;

  while (true) {
    UnitaryOpInterface nextOnA = uniqueUnitaryUser(run.tailA);
    UnitaryOpInterface nextOnB = uniqueUnitaryUser(run.tailB);
    const bool sameOp =
        nextOnA && nextOnB && nextOnA.getOperation() == nextOnB.getOperation();

    if (sameOp && nextOnA.isTwoQubit()) {
      const auto matrix = twoQubitRunMemberMatrix(nextOnA);
      if (!matrix) {
        break;
      }
      absorbTwoQubitIntoRun(run, nextOnA, *matrix);
      continue;
    }

    const auto matrixA =
        !sameOp ? oneQubitRunMemberMatrix(nextOnA) : std::nullopt;
    const auto matrixB =
        !sameOp ? oneQubitRunMemberMatrix(nextOnB) : std::nullopt;
    const bool aSingle = matrixA.has_value();
    const bool bSingle = matrixB.has_value();
    if (aSingle && bSingle && nextOnA->getBlock() != nextOnB->getBlock()) {
      break;
    }
    if (aSingle && (!bSingle || nextOnA->isBeforeInBlock(nextOnB))) {
      absorbOneQubitIntoRun(run, nextOnA, *matrixA, /*wireIndex=*/0);
      continue;
    }
    if (bSingle) {
      absorbOneQubitIntoRun(run, nextOnB, *matrixB, /*wireIndex=*/1);
      continue;
    }
    break;
  }
  return run;
}

/// Erases all run members, successors first so each is dead when erased.
static void eraseFusableRun(RewriterBase& rewriter,
                            const FusableTwoQubitRun& run) {
  for (Operation* member : llvm::reverse(run.ops)) {
    rewriter.eraseOp(member);
  }
}

/// Fuses a maximal constant run only when generic resynthesis strictly reduces
/// its two-qubit operation count.
static bool fuseTwoQubitGateRun(IRRewriter& rewriter, UnitaryOpInterface head,
                                const Matrix4x4& headMatrix,
                                const CompilerTarget::SynthesisBasis basis) {
  FusableTwoQubitRun run = scanFusableTwoQubitRun(head, headMatrix);
  if (run.ops.size() < 2) {
    return false;
  }

  const auto native = decomposeUnitary2QWeyl(run.composed, basis.entangler);
  if (native.numBasisUses >= run.numTwoQ) {
    return false;
  }

  auto firstOp = cast<UnitaryOpInterface>(run.ops.front());
  rewriter.setInsertionPoint(firstOp);
  const auto synthesized =
      emitUnitary2QWeyl(rewriter, firstOp.getLoc(), firstOp.getInputQubit(0),
                        firstOp.getInputQubit(1), native, basis);
  decomposition::emitGPhaseIfNeeded(rewriter, firstOp.getLoc(),
                                    synthesized.globalPhase);
  rewriter.replaceAllUsesWith(run.tailA, synthesized.qubit0);
  rewriter.replaceAllUsesWith(run.tailB, synthesized.qubit1);
  eraseFusableRun(rewriter, run);
  return true;
}

static bool requiresTargetSynthesis(Operation* operation,
                                    const CompilerTarget& target) {
  return !target.supports(operation);
}

namespace {

struct SynthesisPlan {
  Operation* firstNeed = nullptr;
  Operation* matrixUnavailable = nullptr;
  SmallVector<Operation*> operations;
};

} // namespace

static SynthesisPlan planTargetSynthesis(Operation* root,
                                         const CompilerTarget& target) {
  SynthesisPlan plan;
  root->walk([&](Operation* operation) {
    auto unitary = dyn_cast<UnitaryOpInterface>(operation);
    if (!unitary || !isWalkableUnitaryShell(operation) ||
        (unitary.getNumQubits() != 1 && unitary.getNumQubits() != 2)) {
      return WalkResult::advance();
    }
    if (!requiresTargetSynthesis(operation, target)) {
      return WalkResult::advance();
    }
    if (plan.firstNeed == nullptr) {
      plan.firstNeed = operation;
    }

    if (unitary.isSingleQubit()) {
      Matrix2x2 matrix;
      if (unitary.getUnitaryMatrix2x2(matrix) ||
          decomposition::canSynthesizeParameterizedUnitary1Q(operation)) {
        plan.operations.emplace_back(operation);
        return WalkResult::advance();
      }
    } else {
      Matrix4x4 matrix;
      if (assignTwoQubitOpMatrix(operation, matrix)) {
        plan.operations.emplace_back(operation);
        return WalkResult::advance();
      }
    }
    plan.matrixUnavailable = operation;
    return WalkResult::interrupt();
  });
  return plan;
}

static void lowerTargetOperation(IRRewriter& rewriter, UnitaryOpInterface op,
                                 const CompilerTarget::SynthesisBasis basis) {
  Operation* const operation = op.getOperation();
  rewriter.setInsertionPoint(operation);
  if (op.isSingleQubit()) {
    Matrix2x2 matrix;
    if (!op.getUnitaryMatrix2x2(matrix)) {
      decomposition::synthesizeParameterizedUnitary1Q(rewriter, operation,
                                                      basis.singleQubit);
      return;
    }
    const auto synthesized = decomposition::synthesizeUnitary1QEuler(
        rewriter, operation->getLoc(), op.getInputQubit(0), matrix,
        /*runSize=*/1, /*hasNonBasisGate=*/true, basis.singleQubit);
    if (!synthesized) {
      llvm::reportFatalInternalError(
          "target single-qubit basis failed to synthesize a unitary matrix");
    }
    decomposition::emitGPhaseIfNeeded(rewriter, operation->getLoc(),
                                      synthesized->globalPhase);
    rewriter.replaceOp(operation, synthesized->qubit);
    return;
  }

  Matrix4x4 matrix;
  assignTwoQubitOpMatrix(operation, matrix);
  Value input0;
  Value input1;
  if (auto ctrl = dyn_cast<CtrlOp>(operation)) {
    input0 = ctrl.getInputControl(0);
    input1 = ctrl.getInputTarget(0);
  } else {
    input0 = op.getInputQubit(0);
    input1 = op.getInputQubit(1);
  }

  const auto native = decomposeUnitary2QWeyl(matrix, basis.entangler);
  const auto synthesized = emitUnitary2QWeyl(rewriter, operation->getLoc(),
                                             input0, input1, native, basis);
  decomposition::emitGPhaseIfNeeded(rewriter, operation->getLoc(),
                                    synthesized.globalPhase);
  rewriter.replaceOp(operation,
                     ValueRange{synthesized.qubit0, synthesized.qubit1});
}

static LogicalResult fuseTwoQubitGates(ModuleOp moduleOp) {
  constexpr CompilerTarget::SynthesisBasis basis{
      .singleQubit = CompilerTarget::SingleQubitBasis::U,
      .entangler = CompilerTarget::GateKind::CZ};

  SmallVector<Operation*> runHeads;
  moduleOp.walk([&](Operation* operation) {
    auto unitary = dyn_cast<UnitaryOpInterface>(operation);
    const auto matrix = twoQubitRunMemberMatrix(unitary);
    if (matrix && !feedsFromSameTwoQubitRun(unitary)) {
      runHeads.emplace_back(operation);
    }
  });

  bool changed = false;
  IRRewriter rewriter(moduleOp.getContext());
  for (Operation* operation : runHeads) {
    auto unitary = cast<UnitaryOpInterface>(operation);
    const auto matrix = twoQubitRunMemberMatrix(unitary);
    if (matrix) {
      changed |= fuseTwoQubitGateRun(rewriter, unitary, *matrix, basis);
    }
  }
  if (!changed) {
    return success();
  }
  return mlir::mqt::normalizeGlobalPhases(moduleOp);
}

namespace {

struct FuseTwoQubitGatesPass final
    : PassWrapper<FuseTwoQubitGatesPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FuseTwoQubitGatesPass)

  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<QCODialect, arith::ArithDialect>();
  }

protected:
  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    if (failed(fuseTwoQubitGates(moduleOp))) {
      signalPassFailure();
    }
  }
};

struct TargetNativeSynthesisPass final
    : PassWrapper<TargetNativeSynthesisPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TargetNativeSynthesisPass)

  explicit TargetNativeSynthesisPass(const CompilerTarget& targetIn)
      : target(targetIn) {}

  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<QCODialect, arith::ArithDialect, math::MathDialect>();
  }

protected:
  void runOnOperation() override {
    if (!target.hasExplicitOperations()) {
      return;
    }
    ModuleOp moduleOp = getOperation();
    const auto plan = planTargetSynthesis(moduleOp, target);
    if (plan.firstNeed == nullptr) {
      return;
    }

    const auto targetBasis = target.synthesisBasis();
    if (!targetBasis) {
      plan.firstNeed->emitError()
          << "target-native synthesis cannot lower operation '"
          << plan.firstNeed->getName()
          << "': the target has no usable synthesis basis";
      signalPassFailure();
      return;
    }
    if (plan.matrixUnavailable != nullptr) {
      plan.matrixUnavailable->emitError()
          << "target-native synthesis cannot lower operation '"
          << plan.matrixUnavailable->getName()
          << "': its unitary matrix is not available at compile time";
      signalPassFailure();
      return;
    }

    IRRewriter rewriter(&getContext());
    for (Operation* operation : plan.operations) {
      lowerTargetOperation(rewriter, cast<UnitaryOpInterface>(operation),
                           *targetBasis);
    }
    if (failed(mlir::mqt::normalizeGlobalPhases(moduleOp))) {
      signalPassFailure();
    }
  }

  CompilerTarget target;
};

struct VerifyTargetConformancePass final
    : PassWrapper<VerifyTargetConformancePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VerifyTargetConformancePass)

  explicit VerifyTargetConformancePass(const CompilerTarget& targetIn)
      : target(targetIn) {}

protected:
  void runOnOperation() override {
    WalkResult result = getOperation()->walk([&](Operation* operation) {
      if (auto function = dyn_cast<FunctionOpInterface>(operation);
          function &&
          llvm::any_of(function.getArgumentTypes(), [](const auto type) {
            if (isa<QubitType>(type)) {
              return true;
            }
            const auto tensor = dyn_cast<RankedTensorType>(type);
            return tensor && isa<QubitType>(tensor.getElementType());
          })) {
        function.emitError()
            << "target conformance requires quantum function inputs to be "
               "assigned to qco.static target sites";
        return WalkResult::interrupt();
      }
      if (auto staticOp = dyn_cast<StaticOp>(operation)) {
        const auto site =
            static_cast<CompilerTarget::SiteId>(staticOp.getIndex());
        if (target.vertexForSite(site)) {
          return WalkResult::advance();
        }
        staticOp.emitError() << "target does not contain static site " << site;
        return WalkResult::interrupt();
      }
      if (isa<AllocOp, qtensor::AllocOp>(operation)) {
        operation->emitError()
            << "target conformance requires qubits to be assigned to "
               "qco.static target sites";
        return WalkResult::interrupt();
      }

      size_t arity = 1;
      size_t parameterCount = 0;
      if (auto unitary = dyn_cast<UnitaryOpInterface>(operation)) {
        if (isExcludedFromTopLevelUnitaryWalk(operation)) {
          return WalkResult::advance();
        }
        arity = unitary.getNumQubits();
        parameterCount = unitary.getNumParams();
      } else if (!isa<MeasureOp, ResetOp>(operation)) {
        return WalkResult::advance();
      }

      if (target.supports(operation)) {
        return WalkResult::advance();
      }

      auto diagnostic = operation->emitError()
                        << "target does not support operation '"
                        << operation->getName() << "' with arity " << arity
                        << " and " << parameterCount << " parameter(s)";
      return WalkResult::interrupt();
    });
    if (result.wasInterrupted()) {
      signalPassFailure();
    }
  }

  CompilerTarget target;
};

} // namespace

std::unique_ptr<Pass> createFuseTwoQubitGates() {
  return std::make_unique<FuseTwoQubitGatesPass>();
}

std::unique_ptr<Pass>
createTargetNativeSynthesis(const CompilerTarget& target) {
  return std::make_unique<TargetNativeSynthesisPass>(target);
}

std::unique_ptr<Pass>
createVerifyTargetConformance(const CompilerTarget& target) {
  return std::make_unique<VerifyTargetConformancePass>(target);
}

} // namespace mlir::qco
