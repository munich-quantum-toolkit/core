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
#include "mlir/Dialect/MQT/IR/MQTDialect.h"
#include "mlir/Dialect/MQT/Transforms/GlobalPhaseNormalization.h"
#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QCO/IR/QCOInterfaces.h"
#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QCO/Transforms/Decomposition/Euler.h"
#include "mlir/Dialect/QCO/Transforms/Decomposition/Weyl.h"
#include "mlir/Dialect/QCO/Transforms/Passes.h"
#include "mlir/Dialect/QCO/Utils/Matrix.h"
#include "mlir/Dialect/QTensor/IR/QTensorOps.h"

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/Support/ErrorHandling.h>
#include <mlir/Dialect/Arith/IR/Arith.h> // IWYU pragma: keep (Passes.h.inc)
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/IRMapping.h>
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

#include <array>
#include <cassert>
#include <cstddef>
#include <memory>
#include <optional>
#include <utility>

namespace mlir::qco {

using decomposition::decomposeUnitary2QWeyl;
using decomposition::emitUnitary2QWeyl;

namespace {

/// Composed unitary and metadata for a fusable two-qubit run.
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
                                CompilerTarget::SynthesisBasis basis) {
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

namespace {

using SiteId = CompilerTarget::SiteId;
using SiteMap = DenseMap<Value, std::optional<SiteId>>;

} // namespace

static SmallVector<Value> getQubitValues(ValueRange values) {
  return llvm::to_vector(llvm::make_filter_range(
      values, [](Value value) { return isa<QubitType>(value.getType()); }));
}

static bool joinSite(Value value, std::optional<SiteId> site, SiteMap& sites) {
  const auto [position, inserted] = sites.try_emplace(value, site);
  if (inserted || !position->second) {
    return inserted;
  }
  if (!site || *position->second != *site) {
    position->second.reset();
    return true;
  }
  return false;
}

static bool propagateSites(ValueRange inputs, ValueRange outputs,
                           SiteMap& sites) {
  bool changed = false;
  const auto inputQubits = getQubitValues(inputs);
  const auto outputQubits = getQubitValues(outputs);
  for (const auto [input, output] :
       llvm::zip_equal(inputQubits, outputQubits)) {
    if (const auto found = sites.find(input); found != sites.end()) {
      changed |= joinSite(output, found->second, sites);
    }
  }
  return changed;
}

static bool propagateBranchSites(ValueRange inputs,
                                 MutableArrayRef<Region> regions,
                                 ValueRange results, SiteMap& sites) {
  bool changed = false;
  for (Region& region : regions) {
    changed |= propagateSites(inputs, region.getArguments(), sites);
    changed |= propagateSites(region.front().getTerminator()->getOperands(),
                              results, sites);
  }
  return changed;
}

static bool propagateForSites(scf::ForOp op, SiteMap& sites) {
  bool changed = propagateSites(op.getInits(), op.getRegionIterArgs(), sites);
  auto yield = cast<scf::YieldOp>(op.getBody()->getTerminator());
  changed |= propagateSites(yield.getResults(), op.getRegionIterArgs(), sites);
  changed |= propagateSites(op.getRegionIterArgs(), op.getResults(), sites);
  return changed;
}

static bool propagateWhileSites(scf::WhileOp op, SiteMap& sites) {
  bool changed = propagateSites(op.getInits(), op.getBeforeArguments(), sites);
  auto afterYield = cast<scf::YieldOp>(op.getAfterBody()->getTerminator());
  changed |=
      propagateSites(afterYield.getResults(), op.getBeforeArguments(), sites);
  auto condition = cast<scf::ConditionOp>(op.getBeforeBody()->getTerminator());
  changed |= propagateSites(condition.getArgs(), op.getAfterArguments(), sites);
  changed |= propagateSites(condition.getArgs(), op.getResults(), sites);
  return changed;
}

static SiteMap collectStaticSites(Operation* root) {
  SiteMap sites;
  bool changed = false;
  do {
    changed = false;
    root->walk([&](Operation* operation) {
      if (auto staticOp = dyn_cast<StaticOp>(operation)) {
        changed |= joinSite(staticOp.getQubit(), staticOp.getIndex(), sites);
      } else if (auto unitary = dyn_cast<UnitaryOpInterface>(operation)) {
        changed |= propagateSites(unitary.getInputQubits(),
                                  unitary.getOutputQubits(), sites);
      } else if (auto reset = dyn_cast<ResetOp>(operation)) {
        changed |=
            propagateSites(reset.getQubitIn(), reset.getQubitOut(), sites);
      } else if (auto measure = dyn_cast<MeasureOp>(operation)) {
        changed |=
            propagateSites(measure.getQubitIn(), measure.getQubitOut(), sites);
      } else if (auto ifOp = dyn_cast<IfOp>(operation)) {
        changed |= propagateBranchSites(ifOp.getQubits(), ifOp->getRegions(),
                                        ifOp.getResults(), sites);
      } else if (auto switchOp = dyn_cast<IndexSwitchOp>(operation)) {
        changed |=
            propagateBranchSites(switchOp.getTargets(), switchOp->getRegions(),
                                 switchOp.getResults(), sites);
      } else if (auto forOp = dyn_cast<scf::ForOp>(operation)) {
        changed |= propagateForSites(forOp, sites);
      } else if (auto whileOp = dyn_cast<scf::WhileOp>(operation)) {
        changed |= propagateWhileSites(whileOp, sites);
      }
    });
  } while (changed);
  return sites;
}

static std::optional<SmallVector<SiteId, 2>>
getOperationSites(Operation* operation, const SiteMap& sites) {
  SmallVector<Value, 2> qubits;
  if (auto unitary = dyn_cast<UnitaryOpInterface>(operation)) {
    llvm::append_range(qubits, unitary.getInputQubits());
  } else if (auto reset = dyn_cast<ResetOp>(operation)) {
    qubits.emplace_back(reset.getQubitIn());
  } else if (auto measure = dyn_cast<MeasureOp>(operation)) {
    qubits.emplace_back(measure.getQubitIn());
  } else {
    return std::nullopt;
  }

  SmallVector<SiteId, 2> result;
  result.reserve(qubits.size());
  for (Value qubit : qubits) {
    const auto found = sites.find(qubit);
    if (found == sites.end()) {
      return std::nullopt;
    }
    result.emplace_back(found->second.value_or(-1));
  }
  return result;
}

static bool
supportsAtPossibleSites(Operation* operation, const CompilerTarget& target,
                        const std::optional<SmallVector<SiteId, 2>>& sites) {
  if (!sites) {
    return target.supports(operation);
  }
  if (!llvm::is_contained(*sites, SiteId{-1})) {
    return target.supports(operation, *sites);
  }
  return sites->size() == 1 &&
         llvm::all_of(target.siteIds(), [&](const SiteId site) {
           return target.supports(operation, ArrayRef<SiteId>{site});
         });
}

/// Normalize relative phase effects and discard only the unobservable global
/// phase of an entry point when the target cannot represent it.
static LogicalResult prepareGlobalPhases(ModuleOp moduleOp,
                                         const CompilerTarget& target) {
  if (failed(mqt::normalizeGlobalPhases(moduleOp))) {
    return failure();
  }
  if (target.supportsOperation("gphase", 0, 1)) {
    return success();
  }
  auto entryPoint = mqt::getEntryPoint(moduleOp);
  if (!entryPoint) {
    return success();
  }
  for (auto& block : entryPoint.getBody()) {
    for (auto phase : llvm::make_early_inc_range(block.getOps<GPhaseOp>())) {
      phase.erase();
    }
  }
  return success();
}

namespace {

struct PlannedOperation {
  Operation* operation;
  bool reverseEntangler = false;
  bool reorderOperands = false;
};

} // namespace

static bool isOperandSwapInvariant(UnitaryOpInterface unitary) {
  Operation* operation = unitary.getOperation();
  if (isa<SWAPOp, iSWAPOp, RXXOp, RYYOp, RZZOp>(operation)) {
    return true;
  }
  auto controlled = dyn_cast<CtrlOp>(operation);
  return controlled && controlled.getNumControls() == 1 &&
         controlled.getNumTargets() == 1 &&
         controlled.getNumBodyUnitaries() == 1 &&
         isa<ZOp>(controlled.getBodyUnitary(0).getOperation());
}

static FailureOr<SmallVector<PlannedOperation>> planTargetSynthesis(
    Operation* root, const CompilerTarget& target,
    const std::optional<CompilerTarget::SynthesisBasis>& targetBasis) {
  SmallVector<PlannedOperation> plan;
  const auto sites = collectStaticSites(root);
  const auto result = root->walk([&](Operation* operation) {
    auto unitary = dyn_cast<UnitaryOpInterface>(operation);
    if (!unitary || !isWalkableUnitaryShell(operation) ||
        (unitary.getNumQubits() != 1 && unitary.getNumQubits() != 2)) {
      return WalkResult::advance();
    }
    auto operationSites = getOperationSites(operation, sites);
    if (supportsAtPossibleSites(operation, target, operationSites)) {
      return WalkResult::advance();
    }
    const bool sitesKnown =
        !operationSites || !llvm::is_contained(*operationSites, SiteId{-1});
    if (operationSites && unitary.isTwoQubit() && sitesKnown &&
        isOperandSwapInvariant(unitary)) {
      const std::array reverseSites{(*operationSites)[1], (*operationSites)[0]};
      if (target.supports(operation, reverseSites)) {
        plan.emplace_back(
            PlannedOperation{.operation = operation, .reorderOperands = true});
        return WalkResult::advance();
      }
    }

    bool matrixAvailable = false;
    if (unitary.isSingleQubit()) {
      Matrix2x2 matrix;
      matrixAvailable =
          unitary.getUnitaryMatrix2x2(matrix) ||
          decomposition::canSynthesizeParameterizedUnitary1Q(operation);
    } else {
      Matrix4x4 matrix;
      matrixAvailable = assignTwoQubitOpMatrix(operation, matrix);
    }
    if (!matrixAvailable) {
      operation->emitError()
          << "target-native synthesis cannot lower operation '"
          << operation->getName()
          << "': its unitary matrix is not available at compile time";
      return WalkResult::interrupt();
    }
    if (!targetBasis) {
      operation->emitError()
          << "target-native synthesis cannot lower operation '"
          << operation->getName()
          << "': the target has no usable synthesis basis";
      return WalkResult::interrupt();
    }
    if (unitary.isTwoQubit() && !sitesKnown) {
      operation->emitError()
          << "no supported synthesis-basis placement is known for its "
             "static sites";
      return WalkResult::interrupt();
    }
    bool reverseEntangler = false;
    if (operationSites && unitary.isTwoQubit() &&
        !target.supports(targetBasis->entangler, *operationSites)) {
      const std::array reverseSites{(*operationSites)[1], (*operationSites)[0]};
      if (!target.supports(targetBasis->entangler, reverseSites)) {
        operation->emitError()
            << "no supported synthesis-basis placement is known for its "
               "static sites";
        return WalkResult::interrupt();
      }
      reverseEntangler = true;
    }
    plan.emplace_back(PlannedOperation{operation, reverseEntangler});
    return WalkResult::advance();
  });
  if (result.wasInterrupted()) {
    return failure();
  }
  return plan;
}

static void reorderTwoQubitOperation(IRRewriter& rewriter,
                                     UnitaryOpInterface unitary) {
  IRMapping mapping;
  mapping.map(unitary.getInputQubit(0), unitary.getInputQubit(1));
  mapping.map(unitary.getInputQubit(1), unitary.getInputQubit(0));
  rewriter.setInsertionPoint(unitary);
  auto reordered = cast<UnitaryOpInterface>(
      rewriter.clone(*unitary.getOperation(), mapping));
  rewriter.replaceOp(
      unitary.getOperation(),
      ValueRange{reordered.getOutputQubit(1), reordered.getOutputQubit(0)});
}

static void lowerTargetOperation(IRRewriter& rewriter, UnitaryOpInterface op,
                                 CompilerTarget::SynthesisBasis basis,
                                 bool reverseEntangler) {
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

  if (reverseEntangler) {
    matrix = matrix.reorderForQubits(1, 0);
    std::swap(input0, input1);
  }
  const auto native = decomposeUnitary2QWeyl(matrix, basis.entangler);
  const auto synthesized = emitUnitary2QWeyl(rewriter, operation->getLoc(),
                                             input0, input1, native, basis);
  decomposition::emitGPhaseIfNeeded(rewriter, operation->getLoc(),
                                    synthesized.globalPhase);
  if (reverseEntangler) {
    rewriter.replaceOp(operation,
                       ValueRange{synthesized.qubit1, synthesized.qubit0});
  } else {
    rewriter.replaceOp(operation,
                       ValueRange{synthesized.qubit0, synthesized.qubit1});
  }
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
    if (target.nativeOperationsKind() ==
        CompilerTarget::NativeOperations::Kind::Unrestricted) {
      return;
    }
    ModuleOp moduleOp = getOperation();
    const auto targetBasis = target.synthesisBasis();
    bool hasGlobalPhases = false;
    moduleOp.walk([&](GPhaseOp) { hasGlobalPhases = true; });

    if (hasGlobalPhases) {
      OwningOpRef<ModuleOp> normalized = cast<ModuleOp>(moduleOp->clone());
      if (failed(prepareGlobalPhases(*normalized, target))) {
        signalPassFailure();
        return;
      }
      if (failed(planTargetSynthesis(*normalized, target, targetBasis))) {
        signalPassFailure();
        return;
      }
    }
    if (failed(prepareGlobalPhases(moduleOp, target))) {
      signalPassFailure();
      return;
    }
    auto plan = planTargetSynthesis(moduleOp, target, targetBasis);
    if (failed(plan)) {
      signalPassFailure();
      return;
    }
    if (plan->empty()) {
      return;
    }

    IRRewriter rewriter(&getContext());
    for (const auto& action : *plan) {
      auto unitary = cast<UnitaryOpInterface>(action.operation);
      if (action.reorderOperands) {
        reorderTwoQubitOperation(rewriter, unitary);
      } else {
        lowerTargetOperation(rewriter, unitary, *targetBasis,
                             action.reverseEntangler);
      }
    }
    if (failed(prepareGlobalPhases(moduleOp, target))) {
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
    const auto sites = collectStaticSites(getOperation());
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

      const auto operationSites = getOperationSites(operation, sites);
      if (supportsAtPossibleSites(operation, target, operationSites)) {
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
