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
#include "mqt/Dialect/QCO/Transforms/NativeSynthesis/NativeCost.h"
#include "mqt/Dialect/QCO/Transforms/Passes.h"
#include "mqt/Dialect/QCO/Utils/Matrix.h"
#include "mqt/Dialect/QTensor/IR/QTensorOps.h"
#include "mqt/Support/RandomSeed.h"

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
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/TypeID.h"
#include "mlir/Support/WalkResult.h"
#include "mlir/Transforms/FoldUtils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/xxhash.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <span>
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

std::optional<bool> NativeCostAnalysis::nativeOrientation(
    UnitaryOpInterface operation, const CompilerTarget& target, Sites sites) {
  if (sites ? target.supports(operation.getOperation(), *sites)
            : target.supports(operation.getOperation())) {
    return false;
  }
  if (sites && operation.isTwoQubit() && isOperandSwapInvariant(operation) &&
      target.supports(operation.getOperation(),
                      std::array{(*sites)[1], (*sites)[0]})) {
    return true;
  }
  return std::nullopt;
}

std::optional<bool>
NativeCostAnalysis::entanglerOrientation(const CompilerTarget& target,
                                         CompilerTarget::GateKind entangler,
                                         Sites sites) {
  if (!sites || target.supports(entangler, *sites)) {
    return false;
  }
  if (target.supports(entangler, std::array{(*sites)[1], (*sites)[0]})) {
    return true;
  }
  return std::nullopt;
}

/// Hashes only accelerate lookup. Bitwise equality below keeps collisions,
/// signed zeros, and non-finite numerical failures from producing false hits.
static uint64_t matrixHash(const Matrix4x4& matrix,
                           CompilerTarget::GateKind entangler) {
  return llvm::xxh3_64bits(reinterpret_cast<const uint8_t*>(matrix.data.data()),
                           sizeof(matrix.data)) ^
         static_cast<uint64_t>(entangler);
}

static bool sameMatrix(const Matrix4x4& a, const Matrix4x4& b) {
  const auto aBytes = std::as_bytes(std::span(a.data));
  const auto bBytes = std::as_bytes(std::span(b.data));
  return std::memcmp(aBytes.data(), bBytes.data(), aBytes.size()) == 0;
}

const std::optional<uint8_t>*
NativeCostTable::lookup(const Matrix4x4& matrix,
                        CompilerTarget::GateKind entangler,
                        uint64_t hash) const {
  const auto [begin, end] = index_.equal_range(hash);
  for (auto it = begin; it != end; ++it) {
    const auto& entry = entries_[it->second];
    if (entry.entangler == entangler && sameMatrix(entry.matrix, matrix)) {
      return &entry.count;
    }
  }
  return nullptr;
}

std::unique_ptr<const NativeCostTable>
NativeCostTable::precompute(Operation* root, CompilerTarget::GateKind entangler,
                            uint64_t seed) {
  constexpr size_t capacity = 1024;
  auto result = std::make_unique<NativeCostTable>();
  result->seed_ = seed;
  const auto add = [&](const Matrix4x4& matrix) {
    if (result->entries_.size() == capacity) {
      return;
    }
    const auto hash = matrixHash(matrix, entangler);
    if (result->lookup(matrix, entangler, hash) != nullptr) {
      return;
    }
    const auto native = decomposeUnitary2QWeyl(matrix, entangler, seed);
    result->index_.emplace(hash, result->entries_.size());
    result->entries_.push_back({
        .matrix = matrix,
        .entangler = entangler,
        .count = native ? std::optional(native->numBasisUses) : std::nullopt,
    });
  };
  const auto swap = SWAPOp::getUnitaryMatrix();
  const auto addOrientations = [&](const Matrix4x4& matrix) {
    for (const bool reverse : {false, true}) {
      const auto ordered = reverse ? matrix.reorderForQubits(1, 0) : matrix;
      add(ordered);
      add(swap * ordered);
      add(ordered * swap);
    }
  };
  add(Matrix4x4::identity());
  add(swap);
  root->walk([&](Operation* op) {
    if (result->entries_.size() == capacity) {
      return WalkResult::interrupt();
    }
    auto unitary = dyn_cast<UnitaryOpInterface>(op);
    const auto matrix = twoQubitRunMemberMatrix(unitary);
    if (!matrix) {
      return WalkResult::advance();
    }
    addOrientations(*matrix);
    if (!feedsFromSameTwoQubitRun(unitary)) {
      addOrientations(scanFusableTwoQubitRun(unitary, *matrix).composed);
    }
    return WalkResult::advance();
  });
  return result;
}

const std::optional<decomposition::TwoQubitNativeDecomposition>&
NativeCostAnalysis::decompose(const Matrix4x4& matrix,
                              CompilerTarget::GateKind entangler) {
  if (!decompositions_.empty()) {
    const auto& entry = decompositions_[lastDecomposition_];
    if (entry.entangler == entangler && entry.matrix.data == matrix.data) {
      return entry.native;
    }
  }
  const auto hash = matrixHash(matrix, entangler);
  for (size_t i = 0; i < decompositionHashes_.size(); ++i) {
    const auto& entry = decompositions_[i];
    if (decompositionHashes_[i] == hash && entry.entangler == entangler &&
        sameMatrix(entry.matrix, matrix)) {
      lastDecomposition_ = i;
      return entry.native;
    }
  }
  if (decompositions_.empty()) {
    decompositions_.reserve(CACHE_SIZE);
    decompositionHashes_.reserve(CACHE_SIZE);
  }
  DecompositionEntry entry{
      .matrix = matrix,
      .entangler = entangler,
      .native = decomposeUnitary2QWeyl(matrix, entangler, seed_),
  };
  if (decompositions_.size() < CACHE_SIZE) {
    lastDecomposition_ = decompositions_.size();
    decompositions_.push_back(std::move(entry));
    decompositionHashes_.push_back(hash);
  } else {
    lastDecomposition_ = nextDecomposition_;
    decompositions_[nextDecomposition_] = std::move(entry);
    decompositionHashes_[nextDecomposition_] = hash;
    nextDecomposition_ = (nextDecomposition_ + 1) % CACHE_SIZE;
  }
  return decompositions_[lastDecomposition_].native;
}

std::optional<uint8_t>
NativeCostAnalysis::count(const Matrix4x4& matrix,
                          CompilerTarget::GateKind entangler) {
  if (lastCount_ && lastCount_->entangler == entangler &&
      lastCount_->matrix.data == matrix.data) {
    return lastCount_->count;
  }
  const auto hash = matrixHash(matrix, entangler);
  if (shared_->seed_ == seed_) {
    if (const auto* cached = shared_->lookup(matrix, entangler, hash)) {
      lastCount_ = {.matrix = matrix, .entangler = entangler, .count = *cached};
      return *cached;
    }
  }
  for (size_t i = 0; i < countHashes_.size(); ++i) {
    const auto& entry = counts_[i];
    if (countHashes_[i] == hash && entry.entangler == entangler &&
        sameMatrix(entry.matrix, matrix)) {
      lastCount_ = entry;
      return entry.count;
    }
  }
  const auto native = decomposeUnitary2QWeyl(matrix, entangler, seed_);
  lastCount_ = {
      .matrix = matrix,
      .entangler = entangler,
      .count = native ? std::optional(native->numBasisUses) : std::nullopt,
  };
  if (counts_.empty()) {
    counts_.reserve(CACHE_SIZE);
    countHashes_.reserve(CACHE_SIZE);
  }
  if (counts_.size() < CACHE_SIZE) {
    counts_.push_back(*lastCount_);
    countHashes_.push_back(hash);
  } else {
    counts_[nextCount_] = *lastCount_;
    countHashes_[nextCount_] = hash;
    nextCount_ = (nextCount_ + 1) % CACHE_SIZE;
  }
  return lastCount_->count;
}

std::optional<size_t>
NativeCostAnalysis::operationCost(UnitaryOpInterface operation,
                                  const CompilerTarget& target, Sites sites) {
  if (!operation.isSingleQubit() && !operation.isTwoQubit()) {
    return std::nullopt;
  }
  if (nativeOrientation(operation, target, sites)) {
    return operation.isTwoQubit() ? 1 : 0;
  }
  if (!target.synthesisBasis()) {
    return std::nullopt;
  }
  if (operation.isSingleQubit()) {
    if (operation.getUnitaryMatrix<Matrix2x2>() ||
        decomposition::canSynthesizeParameterizedUnitary1Q(
            operation.getOperation())) {
      return 0;
    }
    return std::nullopt;
  }
  Matrix4x4 matrix;
  if (!assignTwoQubitOpMatrix(operation, matrix)) {
    return std::nullopt;
  }
  return matrixCost(matrix, target, sites);
}

std::optional<size_t>
NativeCostAnalysis::matrixCost(const Matrix4x4& matrix,
                               const CompilerTarget& target, Sites sites) {
  const auto basis = target.synthesisBasis();
  if (!basis || !basis->entangler) {
    return std::nullopt;
  }
  const auto reverse = entanglerOrientation(target, *basis->entangler, sites);
  if (!reverse) {
    return std::nullopt;
  }
  const auto ordered = *reverse ? matrix.reorderForQubits(1, 0) : matrix;
  if (shared_ != nullptr) {
    return count(ordered, *basis->entangler);
  }
  const auto& native = decompose(ordered, *basis->entangler);
  return native ? std::optional<size_t>(native->numBasisUses) : std::nullopt;
}

size_t NativeCostAnalysis::runCost(const Matrix4x4& matrix, size_t separateCost,
                                   const CompilerTarget& target, Sites sites) {
  const auto fused = matrixCost(matrix, target, sites);
  return fused && *fused < separateCost ? *fused : separateCost;
}

std::optional<size_t>
NativeCostAnalysis::swapCost(const CompilerTarget& target,
                             ArrayRef<CompilerTarget::SiteId> sites) {
  if (target.supportsOperation("swap", 2, 0, sites) ||
      target.supportsOperation("swap", 2, 0, std::array{sites[1], sites[0]})) {
    return 1;
  }
  return matrixCost(SWAPOp::getUnitaryMatrix(), target, sites);
}

NativeCostTracker::NativeCostTracker(const CompilerTarget& target,
                                     uint64_t seed,
                                     const NativeCostTable* shared)
    : target_(target), analysis_(seed, shared), runs_(target.numSites()),
      partners_(target.numSites(), target.numSites()),
      depths_(target.numSites()) {}

void NativeCostTracker::charge(size_t cost, size_t a, size_t b) {
  count_ += cost;
  if (cost != 0) {
    depths_[a] = depths_[b] = std::max(depths_[a], depths_[b]) + cost;
    depth_ = std::max(depth_, depths_[a]);
  }
}

size_t NativeCostTracker::pendingCost(size_t a, size_t b) {
  const auto& run = runs_[a];
  /// Emission preserves a lone native gate, even if its matrix is local.
  return run.canFuse ? analysis_.runCost(run.matrix, run.separateCost, target_,
                                         std::array{
                                             target_.siteForVertex(a),
                                             target_.siteForVertex(b),
                                         })
                     : run.separateCost;
}

void NativeCostTracker::flush(size_t vertex) {
  if (partners_[vertex] == partners_.size()) {
    return;
  }
  const size_t partner = partners_[vertex];
  const auto [a, b] = std::minmax(vertex, partner);
  charge(pendingCost(a, b), a, b);
  partners_[a] = partners_[b] = partners_.size();
}

void NativeCostTracker::flush() {
  for (size_t vertex = 0; vertex < partners_.size(); ++vertex) {
    flush(vertex);
  }
}

void NativeCostTracker::appendPair(const Matrix4x4& matrix, size_t cost,
                                   size_t a, size_t b) {
  const auto ordered = a < b ? matrix : matrix.reorderForQubits(1, 0);
  if (partners_[a] != b) {
    flush(a);
    flush(b);
    runs_[std::min(a, b)] = {.matrix = ordered, .separateCost = cost};
    partners_[a] = b;
    partners_[b] = a;
    return;
  }
  auto& run = runs_[std::min(a, b)];
  run.matrix.premultiplyBy(ordered);
  run.separateCost += cost;
  run.canFuse = true;
}

void NativeCostTracker::append(Operation* operation,
                               ArrayRef<size_t> vertices) {
  if (!available_ || cancellations_.erase(operation)) {
    return;
  }
  SmallVector<CompilerTarget::SiteId, 2> sites;
  for (size_t vertex : vertices) {
    sites.push_back(target_.siteForVertex(vertex));
  }
  auto unitary = dyn_cast<UnitaryOpInterface>(operation);
  if (!unitary || isa<BarrierOp>(operation)) {
    size_t depth = 0;
    for (size_t vertex : vertices) {
      flush(vertex);
      depth = std::max(depth, depths_[vertex]);
    }
    for (size_t vertex : vertices) {
      depths_[vertex] = depth;
    }
    if (isa<MeasureOp, ResetOp>(operation) &&
        !target_.supports(operation, sites)) {
      available_ = false;
    }
    return;
  }
  /// Adjacent inverses must not split an earlier pending run on another pair.
  /// Their shared wires keep both gates together in the routing traversal.
  const auto twoQubitMatrix = twoQubitRunMemberMatrix(unitary);
  if (twoQubitMatrix) {
    auto next = uniqueUnitaryUser(unitary.getOutputQubit(0));
    if (next && next == uniqueUnitaryUser(unitary.getOutputQubit(1))) {
      if (auto inverse = twoQubitRunMemberMatrix(next)) {
        if (next.getInputQubit(0) != unitary.getOutputQubit(0)) {
          inverse = inverse->reorderForQubits(1, 0);
        }
        if ((*inverse * *twoQubitMatrix).isApprox(Matrix4x4::identity())) {
          cancellations_.insert(next.getOperation());
          return;
        }
      }
    }
  }
  const auto cost = analysis_.operationCost(unitary, target_, sites);
  if (!cost) {
    available_ = false;
    return;
  }
  if (unitary.isSingleQubit()) {
    const size_t vertex = vertices.front();
    const auto matrix = oneQubitRunMemberMatrix(unitary);
    if (!matrix) {
      flush(vertex);
    } else if (const size_t partner = partners_[vertex];
               partner != partners_.size()) {
      auto& run = runs_[std::min(vertex, partner)];
      run.matrix.premultiplyBy(
          matrix->embedInTwoQubit(vertex < partner ? 0 : 1));
      run.canFuse = true;
    }
    return;
  }
  const size_t a = vertices[0];
  const size_t b = vertices[1];
  if (twoQubitMatrix) {
    appendPair(*twoQubitMatrix, *cost, a, b);
  } else {
    flush(a);
    flush(b);
    charge(*cost, a, b);
  }
}

void NativeCostTracker::appendSwap(size_t a, size_t b) {
  if (!available_) {
    return;
  }
  const auto cost = analysis_.swapCost(target_, std::array{
                                                    target_.siteForVertex(a),
                                                    target_.siteForVertex(b),
                                                });
  if (!cost) {
    available_ = false;
    return;
  }
  appendPair(SWAPOp::getUnitaryMatrix(), *cost, a, b);
}

void NativeCostTracker::merge(NativeCostTracker& child) {
  child.flush();
  count_ += child.count_;
  depth_ = std::max(depth_, child.depth_);
  available_ = available_ && child.available_;
}

std::optional<std::pair<size_t, size_t>> NativeCostTracker::score() {
  flush();
  return available_ ? std::optional(std::pair{count_, depth_}) : std::nullopt;
}

int64_t NativeCostTracker::swapCostAdjustment(size_t a, size_t b,
                                              size_t standaloneCost) {
  if (!available_ || partners_[a] != b) {
    return 0;
  }
  const auto [first, second] = std::minmax(a, b);
  const std::array sites{
      target_.siteForVertex(first),
      target_.siteForVertex(second),
  };
  const auto& run = runs_[first];
  const size_t before = pendingCost(first, second);
  const size_t after =
      analysis_.runCost(SWAPOp::getUnitaryMatrix() * run.matrix,
                        run.separateCost + standaloneCost, target_, sites);
  return static_cast<int64_t>(after) - static_cast<int64_t>(before) -
         static_cast<int64_t>(standaloneCost);
}

static LogicalResult synthesizeTargetOperation(
    IRRewriter& rewriter, UnitaryOpInterface op, const CompilerTarget& target,
    const std::optional<CompilerTarget::SynthesisBasis>& basis,
    std::optional<ArrayRef<SiteId>> sites, NativeCostAnalysis& analysis) {
  Operation* const operation = op.getOperation();
  if (const auto reverse = analysis.nativeOrientation(op, target, sites)) {
    if (*reverse) {
      reorderTwoQubitOperation(rewriter, op);
    }
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
  if (auto controlled = dyn_cast<CtrlOp>(operation);
      controlled && basis->singleQubit == CompilerTarget::SingleQubitBasis::U &&
      controlled.getNumTargets() == 1 &&
      controlled.getNumBodyUnitaries() == 1 &&
      isa<U2Op>(controlled.getBodyUnitary(0).getOperation())) {
    /// Canonicalization may shorten a native controlled U(pi/2, phi, lambda)
    /// to U2. Restore its native form before attempting matrix synthesis.
    decomposition::synthesizeParameterizedUnitary1Q(
        rewriter, controlled.getBodyUnitary(0).getOperation(),
        basis->singleQubit);
    if (sites ? target.supports(operation, *sites)
              : target.supports(operation)) {
      return success();
    }
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
  const auto direction =
      analysis.entanglerOrientation(target, *basis->entangler, sites);
  if (!direction) {
    return operation->emitError()
           << "no supported synthesis-basis placement is known for its "
              "static sites";
  }
  const bool reverseEntangler = *direction;
  Value input0 = op.getInputQubit(0);
  Value input1 = op.getInputQubit(1);

  if (reverseEntangler) {
    matrix = matrix.reorderForQubits(1, 0);
    std::swap(input0, input1);
  }
  const auto& native = analysis.decompose(matrix, *basis->entangler);
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
                              const SiteMap* sites,
                              NativeCostAnalysis& analysis) {
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
    const auto individual =
        analysis.operationCost(unitary, target, operationSites);
    if (!individual) {
      return false;
    }
    cost += *individual;
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
                                NativeCostAnalysis& analysis, bool shrinkOnly) {
  auto run = scanFusableTwoQubitRun(head, headMatrix);
  if (run.ops.size() < 2) {
    return false;
  }
  bool reverseEntangler = false;
  if (sites != nullptr) {
    const auto headSites = getOperationSites(head, *sites);
    const auto direction =
        analysis.entanglerOrientation(*target, *basis.entangler, headSites);
    if (!direction) {
      return false;
    }
    reverseEntangler = *direction;
  }
  const auto native = analysis.decompose(
      reverseEntangler ? run.composed.reorderForQubits(1, 0) : run.composed,
      *basis.entangler);
  if (!native || (shrinkOnly && native->numBasisUses >= run.numTwoQ) ||
      (target != nullptr ? !reducesNativeCost(run, native->numBasisUses,
                                              *target, sites, analysis)
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
                              bool shrinkOnly = false, uint64_t seed = 2023) {
  bool changed = false;
  NativeCostAnalysis analysis(compilationSeed(moduleOp, seed));
  /// A run's successors have already been visited when its head erases them.
  moduleOp->walk<WalkOrder::PostOrder, ReverseIterator>(
      [&](Operation* operation) {
        auto unitary = dyn_cast<UnitaryOpInterface>(operation);
        const auto matrix = twoQubitRunMemberMatrix(unitary);
        if (matrix && !feedsFromSameTwoQubitRun(unitary)) {
          changed |= fuseTwoQubitGateRun(rewriter, unitary, *matrix, basis,
                                         target, sites, analysis, shrinkOnly);
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
  using TargetNativeSynthesisBase::TargetNativeSynthesisBase;

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
    if (targetBasis &&
        targetBasis->singleQubit != CompilerTarget::SingleQubitBasis::U) {
      RewritePatternSet patterns(&getContext());
      decomposition::populateParameterizedSingleQubitRunCompositionPatterns(
          patterns, targetBasis->singleQubit, &target);
      decomposition::populateFuseSingleQubitUnitaryRunsPatterns(
          patterns, targetBasis->singleQubit, /*skipControlledBodies=*/true,
          &target);
      if (failed(applyPatternsGreedily(moduleOp, std::move(patterns)))) {
        signalPassFailure();
        return;
      }
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
                        indexed ? nullptr : &*sites, false, seed);
    }
    listener.foldPending();
    NativeCostAnalysis analysis(compilationSeed(moduleOp, seed));
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
              analysis);
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

void populateTargetNativeSynthesisPipeline(OpPassManager& pm) {
  /// Placement consumes allocations; native synthesis normalizes phases.
  pm.addPass(createCanonicalizerPass(
      GreedyRewriteConfig{}.setMaxIterations(GreedyRewriteConfig::kNoLimit)));
  /// Reuse unchanged classical reads before native synthesis splits their uses.
  pm.addPass(createCSEPass());
  pm.addPass(createRemoveDeadValuesPass());
  pm.addPass(createTargetNativeSynthesis());
  pm.addPass(createCSEPass());
  pm.addPass(createVerifyTargetConformance());
}

} // namespace mlir::qco
