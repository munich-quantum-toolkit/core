/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/Compiler/Target.h"
#include "mqt/Compiler/TargetEnvironment.h"
#include "mqt/Dialect/MQT/IR/MQTDialect.h"
#include "mqt/Dialect/MQT/Transforms/GlobalPhaseNormalization.h"
#include "mqt/Dialect/QCO/IR/QCODialect.h"
#include "mqt/Dialect/QCO/IR/QCOInterfaces.h"
#include "mqt/Dialect/QCO/IR/QCOOps.h"
#include "mqt/Dialect/QCO/Transforms/Decomposition/Euler.h"
#include "mqt/Dialect/QCO/Transforms/Decomposition/Weyl.h"
#include "mqt/Dialect/QCO/Transforms/Passes.h"
#include "mqt/Dialect/QCO/Utils/Matrix.h"
#include "mqt/Dialect/QTensor/IR/QTensorOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h" // IWYU pragma: keep (Passes.h.inc)
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Iterators.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/TypeID.h"
#include "mlir/Support/WalkResult.h"
#include "mlir/Transforms/FoldUtils.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"

#include <array>
#include <cassert>
#include <cstddef>
#include <memory>
#include <optional>
#include <utility>

namespace mlir::qco {

using decomposition::decomposeUnitary2QWeyl;
using decomposition::emitUnitary2QWeyl;

#define GEN_PASS_DEF_TARGETNATIVESYNTHESIS
#define GEN_PASS_DEF_VERIFYTARGETCONFORMANCE
#include "mqt/Dialect/QCO/Transforms/Passes.h.inc"

namespace {

/// Composed unitary and metadata for a fusable two-qubit run.
struct FusableTwoQubitRun {
  SmallVector<Operation*, 8> ops; ///< Members in dependency order.
  Matrix4x4 composed = Matrix4x4::identity();
  size_t numTwoQ = 0; ///< Number of two-qubit members.
  Value tailA;        ///< Current output wires of the run's tail.
  Value tailB;
};

/// Reuse the last numerical decomposition across gates on different wires.
///
/// The target basis is fixed for the pass; no SSA values or locations are kept.
struct LastTwoQubitDecomposition {
  Matrix4x4 matrix;
  std::optional<decomposition::TwoQubitNativeDecomposition> native;

  const std::optional<decomposition::TwoQubitNativeDecomposition>&
  get(const Matrix4x4& nextMatrix, CompilerTarget::GateKind entangler) {
    if (!native || matrix.data != nextMatrix.data) {
      matrix = nextMatrix;
      native = decomposeUnitary2QWeyl(matrix, entangler);
    }
    return native;
  }
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

/// Multi-target control bodies lack a supported operand-to-matrix mapping.
static bool assignTwoQubitOpMatrix(UnitaryOpInterface op, Matrix4x4& matrix) {
  return (!isa<CtrlOp>(op) || op.getNumControls() == 1) &&
         op.getUnitaryMatrix4x4(matrix);
}

/// Return the constant matrix when `unitary` is a single-qubit run member.
static std::optional<Matrix2x2>
oneQubitRunMemberMatrix(UnitaryOpInterface unitary) {
  if (!unitary || !unitary.isSingleQubit() ||
      !isWalkableUnitaryShell(unitary.getOperation())) {
    return std::nullopt;
  }
  return unitary.getUnitaryMatrix<Matrix2x2>();
}

/// Return the constant matrix when `unitary` is a two-qubit run member.
static std::optional<Matrix4x4>
twoQubitRunMemberMatrix(UnitaryOpInterface unitary) {
  if (!unitary || !unitary.isTwoQubit() ||
      !isWalkableUnitaryShell(unitary.getOperation())) {
    return std::nullopt;
  }
  Matrix4x4 matrix;
  if (!assignTwoQubitOpMatrix(unitary, matrix)) {
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
/// single-qubit gates on either wire; stops at the first boundary
/// that would split the run's two wires.
static FusableTwoQubitRun scanFusableTwoQubitRun(UnitaryOpInterface head,
                                                 const Matrix4x4& headMatrix) {
  FusableTwoQubitRun run;
  run.composed = headMatrix;
  run.tailA = head.getOutputQubit(0);
  run.tailB = head.getOutputQubit(1);
  run.ops.push_back(head.getOperation());
  run.numTwoQ = 1;

  UnitaryOpInterface cachedOpA;
  UnitaryOpInterface cachedOpB;
  std::optional<Matrix2x2> matrixA;
  std::optional<Matrix2x2> matrixB;
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

    // Only the consumed wire advances; retain the other pending matrix.
    if (!sameOp && nextOnA.getOperation() != cachedOpA.getOperation()) {
      cachedOpA = nextOnA;
      matrixA = oneQubitRunMemberMatrix(nextOnA);
    }
    if (!sameOp && nextOnB.getOperation() != cachedOpB.getOperation()) {
      cachedOpB = nextOnB;
      matrixB = oneQubitRunMemberMatrix(nextOnB);
    }
    const bool aSingle = !sameOp && matrixA.has_value();
    const bool bSingle = !sameOp && matrixB.has_value();
    if (aSingle && bSingle && nextOnA->getBlock() != nextOnB->getBlock()) {
      break;
    }
    // Gates on distinct wires commute. Comparing their order would rescan the
    // whole block after each preceding fusion invalidates its order cache.
    if (aSingle) {
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

namespace {

using SiteId = CompilerTarget::SiteId;
/// A missing site denotes a placed operand whose address is selected at
/// runtime.
using SiteMap = DenseMap<Value, std::optional<SiteId>>;

} // namespace

static SmallVector<Value> getQubitValues(ValueRange values,
                                         bool includeTensors = false) {
  return llvm::filter_to_vector(values, [includeTensors](Value value) {
    auto type = value.getType();
    auto tensor = dyn_cast<RankedTensorType>(type);
    return isa<QubitType>(type) || (includeTensors && tensor &&
                                    isa<QubitType>(tensor.getElementType()));
  });
}

/// Propagate exact sites, rejecting unknown inputs or inconsistent joins.
static LogicalResult propagateSites(ValueRange inputs, ValueRange outputs,
                                    SiteMap& sites, bool indexed) {
  auto inputQubits = getQubitValues(inputs, indexed);
  auto outputQubits = getQubitValues(outputs, indexed);
  if (inputQubits.size() != outputQubits.size()) {
    return failure();
  }
  for (auto [input, output] : llvm::zip_equal(inputQubits, outputQubits)) {
    auto found = sites.find(input);
    if (found == sites.end()) {
      return failure();
    }
    const auto site = found->second;
    const auto [position, inserted] = sites.try_emplace(output, site);
    if (!inserted && position->second != site) {
      return failure();
    }
  }
  return success();
}

/// Visit each region once. Branches must agree and loop backedges must retain
/// the entry sites; neither rule is implied by all-to-all placement.
static FailureOr<SiteMap> collectStaticSites(Operation* root, bool indexed) {
  SiteMap sites;
  auto result = root->walk([&](Operation* operation, const WalkStage& stage) {
    const auto propagate = [&](ValueRange inputs, ValueRange outputs) {
      if (succeeded(propagateSites(inputs, outputs, sites, indexed))) {
        return WalkResult::advance();
      }
      operation->emitError("target compilation requires known, consistent "
                           "static sites across branches and loop backedges");
      return WalkResult::interrupt();
    };
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
          << "target compilation requires quantum function inputs to be "
             "assigned to qco.static target sites";
      return WalkResult::interrupt();
    }
    if (isa<AllocOp, qtensor::AllocOp>(operation)) {
      operation->emitError()
          << "target compilation requires qubits to be assigned to "
             "qco.static target sites";
      return WalkResult::interrupt();
    }
    if (auto staticOp = dyn_cast<StaticOp>(operation)) {
      sites.try_emplace(staticOp.getQubit(),
                        indexed ? std::nullopt
                                : std::optional<SiteId>(staticOp.getIndex()));
    } else if (indexed && isa<qtensor::FromElementsOp, qtensor::ExtractOp,
                              qtensor::InsertOp>(operation)) {
      for (auto input : getQubitValues(operation->getOperands(), true)) {
        if (!sites.contains(input)) {
          operation->emitError("indexed tensor operands must originate from "
                               "placed qubits");
          return WalkResult::interrupt();
        }
      }
      for (auto output : operation->getResults()) {
        sites.try_emplace(output, std::nullopt);
      }
    } else if (isa<UnitaryOpInterface, ResetOp, MeasureOp>(operation)) {
      if (propagate(operation->getOperands(), operation->getResults())
              .wasInterrupted()) {
        return WalkResult::interrupt();
      }
      return WalkResult::skip();
    } else if (isa<IfOp, IndexSwitchOp, scf::ForOp, scf::WhileOp>(operation)) {
      if (!stage.isBeforeAllRegions()) {
        auto& region = operation->getRegion(stage.getNextRegion() - 1);
        auto yielded = region.front().getTerminator()->getOperands();
        if (isa<scf::ForOp>(operation) &&
            propagate(yielded, region.getArguments()).wasInterrupted()) {
          return WalkResult::interrupt();
        }
        auto outputs = isa<scf::WhileOp>(operation) && stage.isAfterRegion(1)
                           ? ValueRange(operation->getRegion(0).getArguments())
                           : ValueRange(operation->getResults());
        if (propagate(yielded, outputs).wasInterrupted()) {
          return WalkResult::interrupt();
        }
      }
      if (!stage.isAfterAllRegions()) {
        auto& region = operation->getRegion(stage.getNextRegion());
        if (!region.hasOneBlock()) {
          operation->emitError("target compilation requires single-block "
                               "structured control flow");
          return WalkResult::interrupt();
        }
        auto inputs =
            isa<scf::WhileOp>(operation) && stage.isBeforeRegion(1)
                ? operation->getRegion(0).front().getTerminator()->getOperands()
                : operation->getOperands();
        return propagate(inputs, region.getArguments());
      }
    }
    return WalkResult::advance();
  });
  if (result.wasInterrupted()) {
    return failure();
  }
  return sites;
}

/// Collection has validated the inputs of every unitary, reset, and measure.
static SmallVector<SiteId, 2> getOperationSites(Operation* operation,
                                                const SiteMap& sites) {
  SmallVector<SiteId, 2> result;
  for (Value qubit : getQubitValues(operation->getOperands())) {
    result.push_back(*sites.at(qubit));
  }
  return result;
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

static bool isOperandSwapInvariant(UnitaryOpInterface unitary) {
  Operation* operation = unitary.getOperation();
  if (isa<SWAPOp, iSWAPOp, RXXOp, RYYOp, RZZOp>(operation)) {
    return true;
  }
  if (auto exchange = dyn_cast<XXPlusYYOp>(operation)) {
    return matchPattern(exchange.getBeta(), m_AnyZeroFloat());
  }
  auto controlled = dyn_cast<CtrlOp>(operation);
  return controlled && controlled.getNumControls() == 1 &&
         controlled.getNumTargets() == 1 &&
         controlled.getNumBodyUnitaries() == 1 &&
         isa<ZOp>(controlled.getBodyUnitary(0).getOperation());
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

static bool canReverseNativeOperation(UnitaryOpInterface op,
                                      const CompilerTarget& target,
                                      std::optional<ArrayRef<SiteId>> sites) {
  return sites && op.isTwoQubit() && isOperandSwapInvariant(op) &&
         target.supports(op.getOperation(),
                         std::array{(*sites)[1], (*sites)[0]});
}

static LogicalResult synthesizeTargetOperation(
    IRRewriter& rewriter, UnitaryOpInterface op, const CompilerTarget& target,
    const std::optional<CompilerTarget::SynthesisBasis>& basis,
    std::optional<ArrayRef<SiteId>> sites,
    LastTwoQubitDecomposition& lastDecomposition) {
  Operation* const operation = op.getOperation();
  if (sites ? target.supports(operation, *sites) : target.supports(operation)) {
    return success();
  }
  if (canReverseNativeOperation(op, target, sites)) {
    reorderTwoQubitOperation(rewriter, op);
    return success();
  }
  const auto unsupported = [&](StringRef reason) -> LogicalResult {
    return operation->emitError()
           << "target-native synthesis cannot lower operation '"
           << operation->getName() << "': " << reason;
  };
  if (!basis) {
    return unsupported("the target has no usable synthesis basis");
  }
  rewriter.setInsertionPoint(operation);
  if (op.isSingleQubit()) {
    Matrix2x2 matrix;
    if (!op.getUnitaryMatrix2x2(matrix)) {
      if (!decomposition::canSynthesizeParameterizedUnitary1Q(operation)) {
        return unsupported(
            "its unitary matrix is not available at compile time");
      }
      decomposition::synthesizeParameterizedUnitary1Q(rewriter, operation,
                                                      basis->singleQubit);
      return success();
    }
    const auto synthesized = decomposition::synthesizeUnitary1QEuler(
        rewriter, operation->getLoc(), op.getInputQubit(0), matrix,
        /*runSize=*/1, /*hasNonBasisGate=*/true, basis->singleQubit);
    if (!synthesized) {
      llvm::reportFatalInternalError(
          "target single-qubit basis failed to synthesize a unitary matrix");
    }
    decomposition::emitGPhaseIfNeeded(rewriter, operation->getLoc(),
                                      synthesized->globalPhase);
    rewriter.replaceOp(operation, synthesized->qubit);
    return success();
  }

  if (!basis->entangler) {
    return unsupported("the target has no usable two-qubit entangler");
  }
  Matrix4x4 matrix;
  if (!assignTwoQubitOpMatrix(op, matrix)) {
    return unsupported("its unitary matrix is not available at compile time");
  }
  const bool reverseEntangler =
      sites && !target.supports(*basis->entangler, *sites);
  if (reverseEntangler &&
      !target.supports(*basis->entangler,
                       std::array{(*sites)[1], (*sites)[0]})) {
    return operation->emitError()
           << "no supported synthesis-basis placement is known for its "
              "static sites";
  }
  Value input0 = op.getInputQubit(0);
  Value input1 = op.getInputQubit(1);

  if (reverseEntangler) {
    matrix = matrix.reorderForQubits(1, 0);
    std::swap(input0, input1);
  }
  const auto& native = lastDecomposition.get(matrix, *basis->entangler);
  if (!native) {
    return unsupported(
        "its unitary matrix could not be numerically decomposed");
  }
  const auto synthesized = emitUnitary2QWeyl(rewriter, operation->getLoc(),
                                             input0, input1, *native, *basis);
  decomposition::emitGPhaseIfNeeded(rewriter, operation->getLoc(),
                                    synthesized.globalPhase);
  if (reverseEntangler) {
    rewriter.replaceOp(operation,
                       ValueRange{synthesized.qubit1, synthesized.qubit0});
  } else {
    rewriter.replaceOp(operation,
                       ValueRange{synthesized.qubit0, synthesized.qubit1});
  }
  return success();
}

/// Compare with individual native lowering, stopping once fusion wins.
static bool reducesNativeCost(const FusableTwoQubitRun& run, size_t fusedCost,
                              const CompilerTarget& target,
                              CompilerTarget::GateKind entangler,
                              const SiteMap* sites,
                              LastTwoQubitDecomposition& lastDecomposition) {
  size_t cost = 0;
  for (Operation* operation : run.ops) {
    auto unitary = cast<UnitaryOpInterface>(operation);
    if (!unitary.isTwoQubit()) {
      continue;
    }
    auto siteValues = sites != nullptr ? getOperationSites(operation, *sites)
                                       : SmallVector<SiteId, 2>{};
    const auto operationSites =
        sites != nullptr ? std::optional<ArrayRef<SiteId>>(siteValues)
                         : std::nullopt;
    if ((sites != nullptr ? target.supports(operation, siteValues)
                          : target.supports(operation)) ||
        canReverseNativeOperation(unitary, target, operationSites)) {
      ++cost;
    } else {
      Matrix4x4 matrix;
      if (!assignTwoQubitOpMatrix(unitary, matrix)) {
        return false;
      }
      if (sites != nullptr && !target.supports(entangler, siteValues)) {
        matrix = matrix.reorderForQubits(1, 0);
      }
      const auto& native = lastDecomposition.get(matrix, entangler);
      if (!native) {
        return false;
      }
      cost += native->numBasisUses;
    }
    if (cost > fusedCost) {
      return true;
    }
  }
  return false;
}

/// Fuses a constant run only when resynthesis reduces its two-qubit cost.
/// Without a target, the original operation count is a conservative bound.
static bool fuseTwoQubitGateRun(IRRewriter& rewriter, UnitaryOpInterface head,
                                const Matrix4x4& headMatrix,
                                CompilerTarget::SynthesisBasis basis,
                                const CompilerTarget* target,
                                const SiteMap* sites,
                                LastTwoQubitDecomposition& lastDecomposition,
                                bool shrinkOnly) {
  auto run = scanFusableTwoQubitRun(head, headMatrix);
  if (run.ops.size() < 2) {
    return false;
  }
  bool reverseEntangler = false;
  if (sites != nullptr) {
    const auto headSites = getOperationSites(head, *sites);
    reverseEntangler = !target->supports(*basis.entangler, headSites);
    if (reverseEntangler &&
        !target->supports(*basis.entangler,
                          std::array{headSites[1], headSites[0]})) {
      return false;
    }
  }
  const auto native = decomposeUnitary2QWeyl(
      reverseEntangler ? run.composed.reorderForQubits(1, 0) : run.composed,
      *basis.entangler);
  if (!native || (shrinkOnly && native->numBasisUses >= run.numTwoQ) ||
      (target != nullptr
           ? !reducesNativeCost(run, native->numBasisUses, *target,
                                *basis.entangler, sites, lastDecomposition)
           : native->numBasisUses >= run.numTwoQ)) {
    return false;
  }

  Value input0 = head.getInputQubit(0);
  Value input1 = head.getInputQubit(1);
  if (reverseEntangler) {
    std::swap(input0, input1);
  }
  rewriter.setInsertionPoint(head);
  const auto synthesized = emitUnitary2QWeyl(rewriter, head.getLoc(), input0,
                                             input1, *native, basis);
  decomposition::emitGPhaseIfNeeded(rewriter, head.getLoc(),
                                    synthesized.globalPhase);
  rewriter.replaceAllUsesWith(run.tailA, reverseEntangler ? synthesized.qubit1
                                                          : synthesized.qubit0);
  rewriter.replaceAllUsesWith(run.tailB, reverseEntangler ? synthesized.qubit0
                                                          : synthesized.qubit1);
  eraseFusableRun(rewriter, run);
  return true;
}

static bool fuseTwoQubitGates(IRRewriter& rewriter, ModuleOp moduleOp,
                              CompilerTarget::SynthesisBasis basis,
                              const CompilerTarget* target = nullptr,
                              const SiteMap* sites = nullptr,
                              bool shrinkOnly = false) {
  bool changed = false;
  LastTwoQubitDecomposition lastDecomposition;
  /// A run's successors have already been visited when its head erases them.
  moduleOp->walk<WalkOrder::PostOrder, ReverseIterator>(
      [&](Operation* operation) {
        auto unitary = dyn_cast<UnitaryOpInterface>(operation);
        const auto matrix = twoQubitRunMemberMatrix(unitary);
        if (matrix && !feedsFromSameTwoQubitRun(unitary)) {
          changed |=
              fuseTwoQubitGateRun(rewriter, unitary, *matrix, basis, target,
                                  sites, lastDecomposition, shrinkOnly);
        }
      });
  return changed;
}

namespace {

struct FuseTwoQubitGatesPass final
    : PassWrapper<FuseTwoQubitGatesPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FuseTwoQubitGatesPass)

  FuseTwoQubitGatesPass() = default;
  explicit FuseTwoQubitGatesPass(const CompilerTarget& target)
      : target_(target) {}

  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<QCODialect, arith::ArithDialect>();
  }

protected:
  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    const auto basis =
        target_ ? target_->synthesisBasis()
                : std::optional{CompilerTarget::SynthesisBasis{
                      .singleQubit = CompilerTarget::SingleQubitBasis::U,
                      .entangler = CompilerTarget::GateKind::CZ,
                  }};
    if (!basis || !basis->entangler) {
      return;
    }
    IRRewriter rewriter(&getContext());
    if (fuseTwoQubitGates(rewriter, moduleOp, *basis,
                          target_ ? &*target_ : nullptr, nullptr, true) &&
        failed(mlir::mqt::normalizeGlobalPhases(moduleOp))) {
      signalPassFailure();
    }
  }

private:
  std::optional<CompilerTarget> target_;
};

/// Track generated wire sites and defer folding until builders finish.
class SynthesisListener final : public OpBuilder::Listener {
public:
  SynthesisListener(MLIRContext* context, SiteMap& sites)
      : folder_(context), sites_(sites) {}

  void notifyOperationInserted(Operation* operation,
                               OpBuilder::InsertPoint previous) override {
    if (!previous.isSet()) {
      if (isa<UnitaryOpInterface>(operation)) {
        for (auto [input, output] :
             llvm::zip_equal(getQubitValues(operation->getOperands()),
                             getQubitValues(operation->getResults()))) {
          /// Modifier builders also insert detached bodies with unplaced args.
          if (auto found = sites_.find(input); found != sites_.end()) {
            const auto site = found->second;
            sites_.insert_or_assign(output, site);
          }
        }
      }
      if (auto constant = dyn_cast<arith::ConstantOp>(operation)) {
        pending_.push_back(constant);
      }
    }
  }

  void foldPending() {
    for (auto constant : pending_) {
      folder_.insertKnownConstant(constant, constant.getValue());
    }
    pending_.clear();
  }

private:
  OperationFolder folder_;
  SiteMap& sites_;
  SmallVector<arith::ConstantOp> pending_;
};

struct TargetNativeSynthesisPass final
    : impl::TargetNativeSynthesisBase<TargetNativeSynthesisPass> {

protected:
  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    const auto& environment = getAnalysis<TargetEnvironmentAnalysis>();
    if (!environment) {
      moduleOp.emitError()
          << "target-native synthesis requires a valid mqt.target_env: "
          << environment.error();
      signalPassFailure();
      return;
    }
    const CompilerTarget& target = environment.environment().target();
    const auto targetBasis = target.synthesisBasis();
    if (failed(prepareGlobalPhases(moduleOp, target))) {
      signalPassFailure();
      return;
    }
    const bool indexed = environment.environment().supportsIndexedQubits();
    auto sites = collectStaticSites(moduleOp, indexed);
    if (failed(sites)) {
      signalPassFailure();
      return;
    }

    SynthesisListener listener(&getContext(), *sites);
    IRRewriter rewriter(&getContext(), &listener);
    if (targetBasis && targetBasis->entangler) {
      fuseTwoQubitGates(rewriter, moduleOp, *targetBasis, &target,
                        indexed ? nullptr : &*sites);
    }
    listener.foldPending();
    LastTwoQubitDecomposition lastDecomposition;
    /// Rewrite users before producers so each unvisited operation retains its
    /// original operands and their collected sites.
    const auto result = moduleOp->walk<WalkOrder::PostOrder, ReverseIterator>(
        [&](Operation* operation) {
          auto unitary = dyn_cast<UnitaryOpInterface>(operation);
          if (!unitary || !isWalkableUnitaryShell(operation) ||
              (!unitary.isSingleQubit() && !unitary.isTwoQubit())) {
            return WalkResult::advance();
          }
          auto operationSites = indexed ? SmallVector<SiteId, 2>{}
                                        : getOperationSites(operation, *sites);
          const auto synthesized = synthesizeTargetOperation(
              rewriter, unitary, target, targetBasis,
              indexed ? std::nullopt
                      : std::optional<ArrayRef<SiteId>>(operationSites),
              lastDecomposition);
          listener.foldPending();
          return failed(synthesized) ? WalkResult::interrupt()
                                     : WalkResult::advance();
        });
    if (result.wasInterrupted()) {
      signalPassFailure();
      return;
    }
    if (failed(prepareGlobalPhases(moduleOp, target))) {
      signalPassFailure();
    }
  }
};

struct VerifyTargetConformancePass final
    : impl::VerifyTargetConformanceBase<VerifyTargetConformancePass> {

protected:
  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    const auto& environment = getAnalysis<TargetEnvironmentAnalysis>();
    if (!environment) {
      moduleOp.emitError()
          << "target conformance requires a valid mqt.target_env: "
          << environment.error();
      signalPassFailure();
      return;
    }
    const CompilerTarget& target = environment.environment().target();
    const bool indexed = environment.environment().supportsIndexedQubits();
    auto sites = collectStaticSites(moduleOp, indexed);
    if (failed(sites)) {
      signalPassFailure();
      return;
    }
    WalkResult result = moduleOp->walk([&](Operation* operation) {
      if (auto staticOp = dyn_cast<StaticOp>(operation)) {
        const auto site =
            static_cast<CompilerTarget::SiteId>(staticOp.getIndex());
        if (target.vertexForSite(site)) {
          return WalkResult::advance();
        }
        staticOp.emitError() << "target does not contain static site " << site;
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

      auto operationSites = indexed ? SmallVector<SiteId, 2>{}
                                    : getOperationSites(operation, *sites);
      if (indexed ? target.supports(operation)
                  : target.supports(operation, operationSites)) {
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
};

} // namespace

std::unique_ptr<Pass> createFuseTwoQubitGates() {
  return std::make_unique<FuseTwoQubitGatesPass>();
}

std::unique_ptr<Pass> createFuseTwoQubitGates(const CompilerTarget& target) {
  return std::make_unique<FuseTwoQubitGatesPass>(target);
}

} // namespace mlir::qco
