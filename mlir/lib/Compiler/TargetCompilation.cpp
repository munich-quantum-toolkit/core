/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Compiler/TargetCompilation.h"

#include "mlir/Compiler/ProgramFormat.h"
#include "mlir/Compiler/Target.h"
#include "mlir/Dialect/CBit/IR/CBitOps.h"
#include "mlir/Dialect/MQT/IR/MQTAttributes.h"
#include "mlir/Dialect/MQT/IR/MQTDialect.h"
#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QCO/IR/QCOInterfaces.h"
#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QCO/Transforms/Mapping/Mapping.h"
#include "mlir/Dialect/QCO/Transforms/Passes.h"
#include "mlir/Dialect/QTensor/IR/QTensorDialect.h"
#include "mlir/Dialect/QTensor/IR/QTensorOps.h"
#include "mlir/Support/Passes.h"

#include <llvm/ADT/APInt.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SetVector.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/ErrorHandling.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Utils/StaticValueUtils.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/Dominance.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/Matchers.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Region.h>
#include <mlir/IR/TypeUtilities.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/IR/Visitors.h>
#include <mlir/Interfaces/CallInterfaces.h>
#include <mlir/Interfaces/ControlFlowInterfaces.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Rewrite/FrozenRewritePatternSet.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Support/TypeID.h>
#include <mlir/Support/WalkResult.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <mlir/Transforms/Passes.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <utility>

namespace mlir {

[[nodiscard]] static constexpr llvm::StringRef
programFormatName(const ProgramFormat format) {
  switch (format) {
  case ProgramFormat::QCImport:
    return "qc-import";
  case ProgramFormat::QCO:
    return "qco";
  case ProgramFormat::QCOOptimized:
    return "qco-optimized";
  case ProgramFormat::QC:
    return "qc";
  case ProgramFormat::OpenQASM3:
    return "openqasm3";
  case ProgramFormat::Jeff:
    return "jeff";
  case ProgramFormat::QIRBase:
    return "qir-base";
  case ProgramFormat::QIRAdaptive:
    return "qir-adaptive";
  }
  llvm_unreachable("unknown program format");
}

[[nodiscard]] static constexpr llvm::StringRef
programFeatureName(const ProgramFeature feature) {
  switch (feature) {
  case ProgramFeature::MidCircuitMeasurement:
    return "mid-circuit-measurement";
  case ProgramFeature::MeasuredQubitReuse:
    return "measured-qubit-reuse";
  case ProgramFeature::MeasurementResultUse:
    return "measurement-result-use";
  case ProgramFeature::BooleanComputation:
    return "boolean-computation";
  case ProgramFeature::IntegerComputation:
    return "integer-computation";
  case ProgramFeature::FloatComputation:
    return "float-computation";
  case ProgramFeature::ForwardBranching:
    return "forward-branching";
  case ProgramFeature::CountedIteration:
    return "counted-iteration";
  case ProgramFeature::ConditionalLoop:
    return "conditional-loop";
  case ProgramFeature::MultiwayBranching:
    return "multiway-branching";
  }
  llvm_unreachable("unknown program feature");
}

[[nodiscard]] static bool isExternalPayload(const ProgramFormat format) {
  return format == ProgramFormat::OpenQASM3 ||
         format == ProgramFormat::QIRBase ||
         format == ProgramFormat::QIRAdaptive;
}

static void appendImmediateNestedTypes(const Type type,
                                       SmallVectorImpl<Type>& worklist) {
  type.walkImmediateSubElements(
      [](Attribute) {},
      [&](const Type nested) { worklist.emplace_back(nested); });
  // Mutable identified LLVM structs do not expose their body through generic
  // immutable storage traversal on every supported MLIR version.
  if (const auto structure = dyn_cast<LLVM::LLVMStructType>(type)) {
    llvm::append_range(worklist, structure.getBody());
  }
}

struct TypeSummary {
  bool hasQuantumState = false;
  bool hasCBitRegister = false;
  bool hasBoolean = false;
  bool hasInteger = false;
  bool hasFloat = false;

  [[nodiscard]] bool hasClassicalComputation() const {
    return hasBoolean || hasInteger || hasFloat;
  }
};

class TypeSummaryCache final {
public:
  [[nodiscard]] const TypeSummary& get(const Type root) {
    if (const auto cached = summaries.find(root); cached != summaries.end()) {
      return cached->second;
    }

    llvm::DenseMap<Type, TypeSummary> discovered;
    llvm::DenseMap<Type, SmallVector<Type>> parents;
    SmallVector<Type> discoveryWorklist{root};
    while (!discoveryWorklist.empty()) {
      const Type type = discoveryWorklist.pop_back_val();
      if (summaries.contains(type) || discovered.contains(type)) {
        continue;
      }

      TypeSummary summary = getDirectSummary(type);
      SmallVector<Type> nestedTypes;
      appendImmediateNestedTypes(type, nestedTypes);
      for (const Type nested : nestedTypes) {
        if (const auto cached = summaries.find(nested);
            cached != summaries.end()) {
          merge(summary, cached->second);
          continue;
        }
        parents[nested].emplace_back(type);
        discoveryWorklist.emplace_back(nested);
      }
      discovered.try_emplace(type, summary);
    }

    SmallVector<Type> propagationWorklist;
    propagationWorklist.reserve(discovered.size());
    for (const auto& [type, summary] : discovered) {
      static_cast<void>(summary);
      propagationWorklist.emplace_back(type);
    }
    while (!propagationWorklist.empty()) {
      const Type child = propagationWorklist.pop_back_val();
      const TypeSummary childSummary = discovered.lookup(child);
      for (const Type parent : parents.lookup(child)) {
        if (merge(discovered[parent], childSummary)) {
          propagationWorklist.emplace_back(parent);
        }
      }
    }

    for (const auto& [type, summary] : discovered) {
      summaries.try_emplace(type, summary);
    }
    return summaries.find(root)->second;
  }

private:
  [[nodiscard]] static TypeSummary getDirectSummary(const Type type) {
    TypeSummary summary;
    summary.hasQuantumState = isa<qco::QubitType>(type);
    // RegisterType is generated into and publicly provided by CBitOps.h.
    // NOLINTNEXTLINE(misc-include-cleaner)
    summary.hasCBitRegister = isa<cbit::RegisterType>(type);
    if (const auto integer = dyn_cast<IntegerType>(type)) {
      summary.hasBoolean = integer.getWidth() == 1U;
      summary.hasInteger = integer.getWidth() != 1U;
    } else {
      summary.hasInteger = isa<IndexType>(type);
    }
    summary.hasFloat = isa<FloatType>(type);
    return summary;
  }

  [[nodiscard]] static bool merge(TypeSummary& result,
                                  const TypeSummary& other) {
    const TypeSummary previous = result;
    result.hasQuantumState |= other.hasQuantumState;
    result.hasCBitRegister |= other.hasCBitRegister;
    result.hasBoolean |= other.hasBoolean;
    result.hasInteger |= other.hasInteger;
    result.hasFloat |= other.hasFloat;
    return result.hasQuantumState != previous.hasQuantumState ||
           result.hasCBitRegister != previous.hasCBitRegister ||
           result.hasBoolean != previous.hasBoolean ||
           result.hasInteger != previous.hasInteger ||
           result.hasFloat != previous.hasFloat;
  }

  llvm::DenseMap<Type, TypeSummary> summaries;
};

namespace {

struct AttachTargetEnvironmentPass final
    : PassWrapper<AttachTargetEnvironmentPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AttachTargetEnvironmentPass)

  AttachTargetEnvironmentPass(const CompilerTarget& targetIn,
                              const ProgramFormat formatIn)
      : target(targetIn), format(formatIn) {}

  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<mqt::MQTDialect>();
  }

protected:
  void runOnOperation() override {
    if (!isTargetCompilationFormat(format)) {
      getOperation().emitError()
          << "target compilation requires QCOOptimized, QC, OpenQASM3, or QIR "
             "output";
      signalPassFailure();
      return;
    }
    const auto profiles = target.executionProfiles();
    const auto* const profile = target.executionProfile(format);
    if (profiles && isExternalPayload(format) && profile == nullptr) {
      getOperation().emitError()
          << "compiler target does not report support for selected program "
             "format '"
          << programFormatName(format) << "'";
      signalPassFailure();
      return;
    }

    SmallVector<llvm::StringRef> features;
    if (profile != nullptr) {
      features.reserve(profile->features().size());
      for (const ProgramFeature feature : profile->features()) {
        features.emplace_back(programFeatureName(feature));
      }
    }
    const bool optionalFeaturesKnown = profile != nullptr
                                           ? profile->optionalFeaturesKnown()
                                           : profiles.has_value();
    getOperation()->setAttr(
        mqt::TargetEnvAttr::getOperationAttributeName(),
        mqt::TargetEnvAttr::get(&getContext(), programFormatName(format),
                                features, optionalFeaturesKnown));
  }

private:
  CompilerTarget target;
  ProgramFormat format;
};

constexpr uint64_t MAX_LEGALIZED_LOOP_CLONES = 4096U;
constexpr uint64_t MAX_LEGALIZED_LOOP_WORK = 65536U;
constexpr size_t MAX_CBIT_ANALYSIS_STEPS = 4096U;
constexpr size_t MAX_FEEDBACK_ANALYSIS_STEPS = 4096U;

[[nodiscard]] std::optional<llvm::APInt>
getExactConstantTripCount(scf::ForOp loop) {
  llvm::APInt lower;
  llvm::APInt upper;
  llvm::APInt step;
  if (!matchPattern(loop.getLowerBound(), m_ConstantInt(&lower)) ||
      !matchPattern(loop.getUpperBound(), m_ConstantInt(&upper)) ||
      !matchPattern(loop.getStep(), m_ConstantInt(&step)) || step.isZero()) {
    return std::nullopt;
  }

  const unsigned width =
      std::max({lower.getBitWidth(), upper.getBitWidth(), step.getBitWidth()});
  if (width == std::numeric_limits<unsigned>::max()) {
    return std::nullopt;
  }
  const unsigned extendedWidth = width + 1U;
  llvm::APInt lowerExtended;
  llvm::APInt upperExtended;
  llvm::APInt stepExtended;
  if (loop.getUnsignedCmp()) {
    lowerExtended = lower.zextOrTrunc(extendedWidth);
    upperExtended = upper.zextOrTrunc(extendedWidth);
    stepExtended = step.zextOrTrunc(extendedWidth);
    if (!lowerExtended.ult(upperExtended)) {
      return llvm::APInt::getZero(extendedWidth);
    }
  } else {
    if (!step.isStrictlyPositive()) {
      return llvm::APInt::getZero(extendedWidth);
    }
    lowerExtended = lower.sextOrTrunc(extendedWidth);
    upperExtended = upper.sextOrTrunc(extendedWidth);
    stepExtended = step.sextOrTrunc(extendedWidth);
    if (!lowerExtended.slt(upperExtended)) {
      return llvm::APInt::getZero(extendedWidth);
    }
  }

  const llvm::APInt distance = upperExtended - lowerExtended;
  llvm::APInt tripCount = distance.udiv(stepExtended);
  if (!distance.urem(stepExtended).isZero()) {
    ++tripCount;
  }
  return tripCount;
}

[[nodiscard]] std::optional<uint64_t>
getCappedStaticTripCount(scf::ForOp loop) {
  std::optional<llvm::APInt> tripCount = getExactConstantTripCount(loop);
  if (!tripCount) {
    tripCount = loop.getStaticTripCount();
  }
  if (!tripCount) {
    return std::nullopt;
  }
  return tripCount->getLimitedValue(MAX_LEGALIZED_LOOP_CLONES + 1U);
}

[[nodiscard]] bool equalNonnegativeValues(const llvm::APInt& lhs,
                                          const llvm::APInt& rhs) {
  const unsigned width = std::max(lhs.getBitWidth(), rhs.getBitWidth());
  return lhs.zextOrTrunc(width) == rhs.zextOrTrunc(width);
}

struct VerifyStaticLoopTripCountsPass final
    : PassWrapper<VerifyStaticLoopTripCountsPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VerifyStaticLoopTripCountsPass)

protected:
  void runOnOperation() override {
    func::FuncOp entryPoint = mqt::getEntryPoint(getOperation());
    if (!entryPoint) {
      getOperation().emitError("no program entry point found");
      signalPassFailure();
      return;
    }
    const WalkResult result = entryPoint.walk([&](scf::ForOp loop) {
      const std::optional<llvm::APInt> exact = getExactConstantTripCount(loop);
      const std::optional<llvm::APInt> reported = loop.getStaticTripCount();
      if (!exact || !reported || equalNonnegativeValues(*exact, *reported)) {
        return WalkResult::advance();
      }
      loop.emitError()
          << "target compilation refuses to canonicalize a constant loop "
             "whose static trip count overflows MLIR's analysis";
      return WalkResult::interrupt();
    });
    if (result.wasInterrupted()) {
      signalPassFailure();
    }
  }
};

[[nodiscard]] uint64_t cappedLoopCloneAdd(const uint64_t lhs,
                                          const uint64_t rhs) {
  if (lhs > MAX_LEGALIZED_LOOP_CLONES ||
      rhs > MAX_LEGALIZED_LOOP_CLONES - lhs) {
    return MAX_LEGALIZED_LOOP_CLONES + 1U;
  }
  return lhs + rhs;
}

[[nodiscard]] uint64_t cappedLoopCloneMultiply(const uint64_t lhs,
                                               const uint64_t rhs) {
  if (lhs == 0U || rhs == 0U) {
    return 0U;
  }
  if (lhs > MAX_LEGALIZED_LOOP_CLONES ||
      rhs > MAX_LEGALIZED_LOOP_CLONES / lhs) {
    return MAX_LEGALIZED_LOOP_CLONES + 1U;
  }
  return lhs * rhs;
}

[[nodiscard]] uint64_t countNestedOperationsCapped(scf::ForOp loop) {
  uint64_t count = 0U;
  for (Region& region : loop->getRegions()) {
    const WalkResult result = region.walk([&](Operation*) {
      count = cappedLoopCloneAdd(count, 1U);
      return count > MAX_LEGALIZED_LOOP_CLONES ? WalkResult::interrupt()
                                               : WalkResult::advance();
    });
    if (result.wasInterrupted()) {
      break;
    }
  }
  return count;
}

void emitLoopCloneLimitError(scf::ForOp loop, const uint64_t tripCount) {
  loop.emitError()
      << "target legalization refuses to unroll constant loop with "
      << (tripCount > MAX_LEGALIZED_LOOP_CLONES ? "at least " : "") << tripCount
      << " iterations because the aggregate expansion would clone more than "
      << MAX_LEGALIZED_LOOP_CLONES << " operations";
}

void emitLoopWorkLimitError(scf::ForOp loop) {
  loop.emitError()
      << "target legalization refuses further constant-loop unrolling because "
         "total loop-cloning work would exceed "
      << MAX_LEGALIZED_LOOP_WORK << " operations";
}

void collectOutermostConstantLoops(Operation* operation,
                                   SmallVectorImpl<scf::ForOp>& loops) {
  operation->walk<WalkOrder::PreOrder>([&](scf::ForOp loop) {
    if (getExactConstantTripCount(loop) || loop.getStaticTripCount()) {
      loops.emplace_back(loop);
    }
    return WalkResult::skip();
  });
}

[[nodiscard]] uint64_t countOperations(Operation* operation) {
  uint64_t count = 0U;
  operation->walk([&](Operation*) {
    if (count != std::numeric_limits<uint64_t>::max()) {
      ++count;
    }
  });
  return count;
}

[[nodiscard]] uint64_t getLiveLoopExpansion(Operation* operation,
                                            const uint64_t originalOperations) {
  const uint64_t currentOperations = countOperations(operation);
  return currentOperations > originalOperations
             ? currentOperations - originalOperations
             : 0U;
}

[[nodiscard]] bool isLegalizationControl(Operation* operation) {
  return isa<qco::IfOp, qco::IndexSwitchOp, scf::IfOp, scf::IndexSwitchOp>(
      operation);
}

void collectLegalizationRoots(ArrayRef<scf::ForOp> loops, Operation* boundary,
                              SmallVectorImpl<Operation*>& roots) {
  llvm::DenseSet<Operation*> seen;
  for (const scf::ForOp loop : loops) {
    for (Operation* operation = loop;
         operation != nullptr && operation != boundary;
         operation = operation->getParentOp()) {
      if ((operation == loop || isLegalizationControl(operation)) &&
          seen.insert(operation).second) {
        roots.emplace_back(operation);
      }
    }
  }
}

struct ConstantLoopLegalizationState {
  uint64_t clonedOperations = 0U;
  uint64_t totalClonedOperations = 0U;
  uint64_t rewrittenLoops = 0U;
};

struct LegalizeConstantLoopPattern final : OpRewritePattern<scf::ForOp> {
  LegalizeConstantLoopPattern(MLIRContext* context,
                              ConstantLoopLegalizationState* stateIn)
      : OpRewritePattern(context), state(stateIn) {}

  LogicalResult matchAndRewrite(scf::ForOp loop,
                                PatternRewriter& rewriter) const override {
    const std::optional<uint64_t> tripCount = getCappedStaticTripCount(loop);
    if (!tripCount) {
      return failure();
    }

    if (*tripCount > 1U) {
      const uint64_t loopClones = cappedLoopCloneMultiply(
          countNestedOperationsCapped(loop), *tripCount - 1U);
      const uint64_t totalClones =
          cappedLoopCloneAdd(state->clonedOperations, loopClones);
      if (totalClones > MAX_LEGALIZED_LOOP_CLONES) {
        return failure();
      }
      if (state->totalClonedOperations > MAX_LEGALIZED_LOOP_WORK ||
          loopClones > MAX_LEGALIZED_LOOP_WORK - state->totalClonedOperations) {
        return failure();
      }
      state->clonedOperations = totalClones;
      state->totalClonedOperations += loopClones;
    }

    ++state->rewrittenLoops;

    if (*tripCount == 0U) {
      rewriter.replaceOp(loop, loop.getInitArgs());
      return success();
    }

    if (*tripCount == 1U) {
      Block* const body = loop.getBody();
      Operation* const terminator = body->getTerminator();
      SmallVector<Value> blockArguments{loop.getLowerBound()};
      llvm::append_range(blockArguments, loop.getInitArgs());
      rewriter.inlineBlockBefore(body, loop, blockArguments);
      SmallVector<Value> yieldedValues(terminator->getOperands());
      rewriter.eraseOp(terminator);
      rewriter.replaceOp(loop, yieldedValues);
      return success();
    }

    rewriter.setInsertionPoint(loop);
    SmallVector<Value> carriedValues(loop.getInitArgs());
    Value inductionValue = loop.getLowerBound();
    for (uint64_t iteration = 0U; iteration < *tripCount; ++iteration) {
      IRMapping mapping;
      mapping.map(loop.getInductionVar(), inductionValue);
      mapping.map(loop.getRegionIterArgs(), carriedValues);
      for (Operation& operation : loop.getBody()->without_terminator()) {
        rewriter.clone(operation, mapping);
      }

      SmallVector<Value> nextCarriedValues;
      nextCarriedValues.reserve(carriedValues.size());
      for (const Value yielded :
           loop.getBody()->getTerminator()->getOperands()) {
        nextCarriedValues.emplace_back(mapping.lookupOrDefault(yielded));
      }
      carriedValues = std::move(nextCarriedValues);
      if (iteration + 1U < *tripCount) {
        inductionValue = rewriter.createOrFold<arith::AddIOp>(
            loop.getLoc(), inductionValue, loop.getStep());
      }
    }
    rewriter.replaceOp(loop, carriedValues);
    return success();
  }

private:
  ConstantLoopLegalizationState* state;
};

struct LegalizeCountedIterationPass final
    : PassWrapper<LegalizeCountedIterationPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LegalizeCountedIterationPass)

  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<arith::ArithDialect, mqt::MQTDialect, scf::SCFDialect>();
  }

protected:
  void runOnOperation() override {
    const auto environment = getOperation()->getAttrOfType<mqt::TargetEnvAttr>(
        mqt::TargetEnvAttr::getOperationAttributeName());
    if (!environment || environment.supports(programFeatureName(
                            ProgramFeature::CountedIteration))) {
      return;
    }

    func::FuncOp entryPoint = mqt::getEntryPoint(getOperation());
    if (!entryPoint) {
      getOperation().emitError("no program entry point found");
      signalPassFailure();
      return;
    }

    const uint64_t originalOperations = countOperations(entryPoint);
    ConstantLoopLegalizationState state;
    while (true) {
      SmallVector<scf::ForOp> initialLoops;
      collectOutermostConstantLoops(entryPoint, initialLoops);
      if (initialLoops.empty()) {
        return;
      }
      SmallVector<Operation*> initialLoopOperations;
      initialLoopOperations.reserve(initialLoops.size());
      for (scf::ForOp loop : initialLoops) {
        initialLoopOperations.emplace_back(loop.getOperation());
      }

      state.clonedOperations =
          getLiveLoopExpansion(entryPoint, originalOperations);
      const uint64_t initialExpansion = state.clonedOperations;
      const uint64_t initialRewrittenLoops = state.rewrittenLoops;

      RewritePatternSet patterns(&getContext());
      patterns.add<LegalizeConstantLoopPattern>(&getContext(), &state);
      qco::IfOp::getCanonicalizationPatterns(patterns, &getContext());
      qco::IndexSwitchOp::getCanonicalizationPatterns(patterns, &getContext());
      scf::IfOp::getCanonicalizationPatterns(patterns, &getContext());
      scf::IndexSwitchOp::getCanonicalizationPatterns(patterns, &getContext());

      SmallVector<Operation*> legalizationRoots;
      collectLegalizationRoots(initialLoops, entryPoint, legalizationRoots);
      if (failed(applyOpPatternsGreedily(
              legalizationRoots, FrozenRewritePatternSet(std::move(patterns)),
              GreedyRewriteConfig{}.setScope(&entryPoint.getBody())))) {
        entryPoint.emitError(
            "target legalization failed to reach a fixed point");
        signalPassFailure();
        return;
      }

      SmallVector<scf::ForOp> residualLoops;
      collectOutermostConstantLoops(entryPoint, residualLoops);
      if (residualLoops.empty()) {
        return;
      }

      const uint64_t residualExpansion =
          getLiveLoopExpansion(entryPoint, originalOperations);
      bool sameLoops = initialLoopOperations.size() == residualLoops.size();
      for (size_t i = 0U; sameLoops && i < residualLoops.size(); ++i) {
        sameLoops = initialLoopOperations[i] == residualLoops[i].getOperation();
      }
      if (state.rewrittenLoops != initialRewrittenLoops || !sameLoops ||
          residualExpansion < initialExpansion) {
        continue;
      }

      state.clonedOperations = residualExpansion;
      scf::ForOp residual = residualLoops.front();
      const std::optional<uint64_t> tripCount =
          getCappedStaticTripCount(residual);
      const uint64_t loopClones =
          tripCount && *tripCount > 1U
              ? cappedLoopCloneMultiply(countNestedOperationsCapped(residual),
                                        *tripCount - 1U)
              : 0U;
      if (tripCount && cappedLoopCloneAdd(state.clonedOperations, loopClones) >
                           MAX_LEGALIZED_LOOP_CLONES) {
        emitLoopCloneLimitError(residual, *tripCount);
      } else if (tripCount &&
                 (state.totalClonedOperations > MAX_LEGALIZED_LOOP_WORK ||
                  loopClones >
                      MAX_LEGALIZED_LOOP_WORK - state.totalClonedOperations)) {
        emitLoopWorkLimitError(residual);
      } else {
        residual.emitError(
            "target legalization failed to process a constant loop");
      }
      signalPassFailure();
      return;
    }
  }
};

struct ProgramRequirement {
  ProgramFeature feature;
  Operation* operation;
};

class ProgramRequirements final {
public:
  explicit ProgramRequirements(TypeSummaryCache& typeSummariesIn)
      : typeSummaries(typeSummariesIn) {}

  [[nodiscard]] LogicalResult collect(func::FuncOp entryPoint) {
    indexQuantumEvolution(entryPoint);
    collectMidCircuitMeasurements(entryPoint.getBody(), false);
    const WalkResult result = entryPoint.walk([&](Operation* operation) {
      if (auto measurement = dyn_cast<qco::MeasureOp>(operation)) {
        collectMeasurementRequirements(measurement);
      }

      if (auto ifOp = dyn_cast<qco::IfOp>(operation)) {
        add(ProgramFeature::ForwardBranching, operation);
        return failed(
                   collectFeedbackRequirements(operation, ifOp.getCondition()))
                   ? WalkResult::interrupt()
                   : WalkResult::advance();
      }
      if (auto ifOp = dyn_cast<scf::IfOp>(operation)) {
        add(ProgramFeature::ForwardBranching, operation);
        return failed(
                   collectFeedbackRequirements(operation, ifOp.getCondition()))
                   ? WalkResult::interrupt()
                   : WalkResult::advance();
      }
      if (isa<scf::ForOp>(operation)) {
        add(ProgramFeature::CountedIteration, operation);
        return WalkResult::advance();
      }
      if (auto whileOp = dyn_cast<scf::WhileOp>(operation)) {
        add(ProgramFeature::ConditionalLoop, operation);
        for (Block& block : whileOp.getBefore()) {
          auto condition = dyn_cast<scf::ConditionOp>(block.getTerminator());
          if (!condition || failed(collectFeedbackRequirements(
                                operation, condition.getCondition()))) {
            return WalkResult::interrupt();
          }
        }
        return WalkResult::advance();
      }
      if (auto switchOp = dyn_cast<qco::IndexSwitchOp>(operation)) {
        add(ProgramFeature::MultiwayBranching, operation);
        return failed(collectFeedbackRequirements(operation, switchOp.getArg()))
                   ? WalkResult::interrupt()
                   : WalkResult::advance();
      }
      if (auto switchOp = dyn_cast<scf::IndexSwitchOp>(operation)) {
        add(ProgramFeature::MultiwayBranching, operation);
        return failed(collectFeedbackRequirements(operation, switchOp.getArg()))
                   ? WalkResult::interrupt()
                   : WalkResult::advance();
      }
      if (isRuntimeClassicalComputation(operation)) {
        addComputationFeatures(operation);
        return WalkResult::advance();
      }
      if (hasUnmodeledClassicalResult(operation)) {
        operation->emitError()
            << "target compilation cannot classify runtime classical "
               "producer '"
            << operation->getName() << "'";
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    return result.wasInterrupted() ? failure() : success();
  }

  [[nodiscard]] ArrayRef<ProgramRequirement> get() const {
    return requirements;
  }

private:
  struct CBitAliasComponent {
    llvm::SetVector<Value> aliases;
    SmallVector<cbit::LoadOp> loads;
    SmallVector<cbit::LoadOp> dynamicLoads;
    llvm::DenseMap<int64_t, SmallVector<cbit::LoadOp>> constantLoads;
  };

  void add(const ProgramFeature feature, Operation* operation) {
    if (llvm::none_of(requirements, [&](const ProgramRequirement requirement) {
          return requirement.feature == feature;
        })) {
      requirements.push_back({feature, operation});
    }
  }

  [[nodiscard]] bool isClassicalComputationType(const Type type) {
    return typeSummaries.get(type).hasClassicalComputation();
  }

  [[nodiscard]] bool isRuntimeClassicalComputation(Operation* operation) {
    if (operation->hasTrait<OpTrait::ConstantLike>()) {
      return false;
    }
    const llvm::StringRef dialect = operation->getName().getDialectNamespace();
    if (dialect == "arith" || dialect == "math") {
      return true;
    }
    if (dialect != "llvm" || operation->hasTrait<OpTrait::IsTerminator>() ||
        !isMemoryEffectFree(operation)) {
      return false;
    }
    const auto isClassical = [&](const Type type) {
      return isClassicalComputationType(type);
    };
    return llvm::any_of(operation->getOperandTypes(), isClassical) ||
           llvm::any_of(operation->getResultTypes(), isClassical);
  }

  [[nodiscard]] bool hasUnmodeledClassicalResult(Operation* operation) {
    if (operation->hasTrait<OpTrait::ConstantLike>() ||
        isa<qco::MeasureOp, cbit::LoadOp, qco::IfOp, qco::IndexSwitchOp,
            scf::IfOp, scf::ForOp, scf::WhileOp, scf::IndexSwitchOp,
            CallOpInterface>(operation)) {
      return false;
    }
    return llvm::any_of(operation->getResultTypes(), [&](const Type type) {
      return isClassicalComputationType(type);
    });
  }

  void addComputationFeatures(Operation* operation) {
    bool hasBoolean = false;
    bool hasFloat = false;
    bool hasWiderInteger = false;
    const auto addType = [&](const Type type) {
      const TypeSummary& summary = typeSummaries.get(type);
      hasBoolean |= summary.hasBoolean;
      hasWiderInteger |= summary.hasInteger;
      hasFloat |= summary.hasFloat;
    };
    llvm::for_each(operation->getOperandTypes(), addType);
    llvm::for_each(operation->getResultTypes(), addType);
    if (hasBoolean) {
      add(ProgramFeature::BooleanComputation, operation);
    }
    if (hasWiderInteger) {
      add(ProgramFeature::IntegerComputation, operation);
    }
    if (hasFloat) {
      add(ProgramFeature::FloatComputation, operation);
    }
  }

  [[nodiscard]] static bool isQuantumEvolution(Operation* operation) {
    return isa<qco::UnitaryOpInterface, qco::ResetOp, qco::AllocOp,
               qtensor::AllocOp>(operation);
  }

  void indexQuantumEvolution(func::FuncOp entryPoint) {
    entryPoint.walk<WalkOrder::PostOrder>([&](Operation* operation) {
      bool containsEvolution = isQuantumEvolution(operation);
      for (Region& region : operation->getRegions()) {
        for (Block& block : region) {
          containsEvolution |= llvm::any_of(block, [&](Operation& nested) {
            return quantumEvolutionSubtrees.contains(&nested);
          });
        }
      }
      if (containsEvolution) {
        quantumEvolutionSubtrees.insert(operation);
      }
    });
  }

  void collectMidCircuitMeasurements(Region& region,
                                     const bool continuationHasEvolution) {
    for (Block& block : region) {
      bool laterEvolution = continuationHasEvolution;
      for (Operation& operation : llvm::reverse(block)) {
        const bool isLoop = isa<scf::ForOp, scf::WhileOp>(operation);
        const bool nestedContinuation =
            laterEvolution ||
            (isLoop && quantumEvolutionSubtrees.contains(&operation));
        for (Region& nested : operation.getRegions()) {
          collectMidCircuitMeasurements(nested, nestedContinuation);
        }
        if (auto measurement = dyn_cast<qco::MeasureOp>(operation);
            measurement && laterEvolution) {
          add(ProgramFeature::MidCircuitMeasurement, measurement);
        }
        laterEvolution |= quantumEvolutionSubtrees.contains(&operation);
      }
    }
  }

  void appendQuantumResults(Operation* operation,
                            SmallVectorImpl<Value>& worklist) {
    for (const Value result : operation->getResults()) {
      if (typeSummaries.get(result.getType()).hasQuantumState) {
        worklist.emplace_back(result);
      }
    }
  }

  [[nodiscard]] static bool
  forwardRegionValueUse(OpOperand& use, SmallVectorImpl<Value>& worklist) {
    Operation* const user = use.getOwner();
    const unsigned operandNumber = use.getOperandNumber();
    if (isa<qco::YieldOp>(user)) {
      Operation* const parent = user->getParentOp();
      if (operandNumber < parent->getNumResults()) {
        worklist.emplace_back(parent->getResult(operandNumber));
      }
      return true;
    }

    if (isa<scf::YieldOp>(user)) {
      Operation* const parent = user->getParentOp();
      if (auto forOp = dyn_cast<scf::ForOp>(parent)) {
        if (operandNumber < forOp->getNumResults()) {
          worklist.emplace_back(forOp->getResult(operandNumber));
        }
        if (operandNumber < forOp.getRegionIterArgs().size()) {
          worklist.emplace_back(forOp.getRegionIterArgs()[operandNumber]);
        }
        return true;
      }
      if (auto whileOp = dyn_cast<scf::WhileOp>(parent)) {
        if (operandNumber < whileOp.getBeforeArguments().size()) {
          worklist.emplace_back(whileOp.getBeforeArguments()[operandNumber]);
        }
        return true;
      }
      if (operandNumber < parent->getNumResults()) {
        worklist.emplace_back(parent->getResult(operandNumber));
      }
      return true;
    }

    if (isa<scf::ConditionOp>(user)) {
      auto whileOp = dyn_cast<scf::WhileOp>(user->getParentOp());
      if (!whileOp || operandNumber == 0U) {
        return false;
      }
      const unsigned argumentNumber = operandNumber - 1U;
      if (argumentNumber < whileOp->getNumResults()) {
        worklist.emplace_back(whileOp->getResult(argumentNumber));
      }
      if (argumentNumber < whileOp.getAfterArguments().size()) {
        worklist.emplace_back(whileOp.getAfterArguments()[argumentNumber]);
      }
      return true;
    }

    if (auto forOp = dyn_cast<scf::ForOp>(user)) {
      const OperandRange initArgs = forOp.getInitArgs();
      if (!initArgs.empty() &&
          operandNumber >= initArgs.getBeginOperandIndex()) {
        const unsigned argumentNumber =
            operandNumber - initArgs.getBeginOperandIndex();
        if (argumentNumber < initArgs.size()) {
          worklist.emplace_back(forOp.getRegionIterArgs()[argumentNumber]);
          worklist.emplace_back(forOp->getResult(argumentNumber));
          return true;
        }
      }
    }

    if (auto whileOp = dyn_cast<scf::WhileOp>(user)) {
      const OperandRange inits = whileOp.getInits();
      if (!inits.empty() && operandNumber >= inits.getBeginOperandIndex()) {
        const unsigned argumentNumber =
            operandNumber - inits.getBeginOperandIndex();
        if (argumentNumber < inits.size()) {
          worklist.emplace_back(whileOp.getBeforeArguments()[argumentNumber]);
          worklist.emplace_back(whileOp->getResult(argumentNumber));
          return true;
        }
      }
    }

    return false;
  }

  static void appendRegisterAliasSources(const Value value,
                                         SmallVectorImpl<Value>& worklist) {
    if (const auto argument = dyn_cast<BlockArgument>(value)) {
      static_cast<void>(appendBlockArgumentSources(argument, worklist));
      return;
    }

    const auto result = dyn_cast<OpResult>(value);
    if (!result) {
      return;
    }
    Operation* const definition = result.getDefiningOp();
    const unsigned resultNumber = result.getResultNumber();
    if (auto select = dyn_cast<arith::SelectOp>(definition)) {
      worklist.emplace_back(select.getTrueValue());
      worklist.emplace_back(select.getFalseValue());
      return;
    }
    if (auto forOp = dyn_cast<scf::ForOp>(definition)) {
      if (resultNumber < forOp.getInitArgs().size()) {
        worklist.emplace_back(forOp.getInitArgs()[resultNumber]);
      }
      if (resultNumber < forOp.getBody()->getTerminator()->getNumOperands()) {
        worklist.emplace_back(
            forOp.getBody()->getTerminator()->getOperand(resultNumber));
      }
      return;
    }
    if (auto whileOp = dyn_cast<scf::WhileOp>(definition)) {
      for (Block& block : whileOp.getBefore()) {
        Operation* const terminator = block.getTerminator();
        if (resultNumber + 1U < terminator->getNumOperands()) {
          worklist.emplace_back(terminator->getOperand(resultNumber + 1U));
        }
      }
      return;
    }
    if (isa<qco::IfOp, qco::IndexSwitchOp, scf::IfOp, scf::IndexSwitchOp>(
            definition)) {
      for (Region& region : definition->getRegions()) {
        if (!region.empty() &&
            resultNumber < region.front().getTerminator()->getNumOperands()) {
          worklist.emplace_back(
              region.front().getTerminator()->getOperand(resultNumber));
        }
      }
    }
  }

  [[nodiscard]] static llvm::SetVector<Value>
  collectCBitRegisterAliases(const Value root) {
    llvm::SetVector<Value> aliases;
    SmallVector<Value> worklist{root};
    while (!worklist.empty()) {
      const Value value = worklist.pop_back_val();
      if (!isa<cbit::RegisterType>(value.getType()) || !aliases.insert(value)) {
        continue;
      }
      appendRegisterAliasSources(value, worklist);
      for (OpOperand& use : value.getUses()) {
        if (auto select = dyn_cast<arith::SelectOp>(use.getOwner());
            select && use.getOperandNumber() != 0U) {
          worklist.emplace_back(select.getResult());
          continue;
        }
        static_cast<void>(forwardRegionValueUse(use, worklist));
      }
    }
    return aliases;
  }

  [[nodiscard]] const CBitAliasComponent&
  getCBitAliasComponent(const Value root) {
    if (const auto cached = cbitAliasComponentIds.find(root);
        cached != cbitAliasComponentIds.end()) {
      return cbitAliasComponents[cached->second];
    }

    CBitAliasComponent aliasComponent;
    aliasComponent.aliases = collectCBitRegisterAliases(root);
    const size_t component = cbitAliasComponents.size();
    for (const Value alias : aliasComponent.aliases) {
      cbitAliasComponentIds.try_emplace(alias, component);
      for (Operation* const user : alias.getUsers()) {
        auto load = dyn_cast<cbit::LoadOp>(user);
        if (!load) {
          continue;
        }
        aliasComponent.loads.emplace_back(load);
        if (const std::optional<int64_t> index =
                getConstantIntValue(load.getIndex())) {
          aliasComponent.constantLoads[*index].emplace_back(load);
        } else {
          aliasComponent.dynamicLoads.emplace_back(load);
        }
      }
    }
    cbitAliasComponents.emplace_back(std::move(aliasComponent));
    return cbitAliasComponents.back();
  }

  [[nodiscard]] static bool cbitIndicesMayAlias(const Value lhs,
                                                const Value rhs) {
    const auto lhsIndex = getConstantIntValue(lhs);
    const auto rhsIndex = getConstantIntValue(rhs);
    return !lhsIndex || !rhsIndex || *lhsIndex == *rhsIndex;
  }

  [[nodiscard]] static bool cbitIndicesMustAlias(const Value lhs,
                                                 const Value rhs) {
    if (lhs == rhs) {
      return true;
    }
    const auto lhsIndex = getConstantIntValue(lhs);
    const auto rhsIndex = getConstantIntValue(rhs);
    return lhsIndex && rhsIndex && *lhsIndex == *rhsIndex;
  }

  [[nodiscard]] bool consumeCBitObservationStep() {
    if (cbitObservationChecks >= MAX_CBIT_ANALYSIS_STEPS) {
      return false;
    }
    ++cbitObservationChecks;
    return true;
  }

  [[nodiscard]] FailureOr<Value> getSingleCBitRegisterRoot(const Value root) {
    if (const auto cached = cbitSingleRootCache.find(root);
        cached != cbitSingleRootCache.end()) {
      return cached->second;
    }

    llvm::SetVector<Value> roots;
    llvm::DenseSet<Value> visited;
    SmallVector<Value> worklist{root};
    while (!worklist.empty()) {
      if (!consumeCBitObservationStep()) {
        return failure();
      }
      const Value value = worklist.pop_back_val();
      if (!isa<cbit::RegisterType>(value.getType()) ||
          !visited.insert(value).second) {
        continue;
      }
      SmallVector<Value> sources;
      appendRegisterAliasSources(value, sources);
      if (sources.empty()) {
        roots.insert(value);
        if (roots.size() > 1U) {
          cbitSingleRootCache.try_emplace(root, Value{});
          return Value{};
        }
        continue;
      }
      llvm::append_range(worklist, sources);
    }
    const Value singleRoot = roots.size() == 1U ? roots.front() : Value{};
    cbitSingleRootCache.try_emplace(root, singleRoot);
    return singleRoot;
  }

  [[nodiscard]] bool cbitRegistersMustAlias(const Value lhs, const Value rhs) {
    if (lhs == rhs) {
      return true;
    }
    const FailureOr<Value> lhsRoot = getSingleCBitRegisterRoot(lhs);
    const FailureOr<Value> rhsRoot = getSingleCBitRegisterRoot(rhs);
    if (failed(lhsRoot) || failed(rhsRoot)) {
      return false;
    }
    return *lhsRoot && *rhsRoot && *lhsRoot == *rhsRoot;
  }

  [[nodiscard]] static bool isPotentiallyRepeatingLoop(Operation* operation) {
    if (isa<scf::WhileOp>(operation)) {
      return true;
    }
    if (auto forOp = dyn_cast<scf::ForOp>(operation)) {
      std::optional<llvm::APInt> tripCount = getExactConstantTripCount(forOp);
      if (!tripCount) {
        tripCount = forOp.getStaticTripCount();
      }
      return !tripCount || tripCount->getLimitedValue(2U) > 1U;
    }
    return false;
  }

  [[nodiscard]] bool isNestedInPotentiallyRepeatingLoop(Operation* operation) {
    for (Operation* parent = operation->getParentOp(); parent != nullptr;
         parent = parent->getParentOp()) {
      if (!consumeCBitObservationStep()) {
        return true;
      }
      if (isPotentiallyRepeatingLoop(parent)) {
        return true;
      }
    }
    return false;
  }

  struct CBitOperationRelation {
    std::optional<bool> lhsBeforeRhs;
    bool sharesPotentiallyRepeatingLoop = false;
  };

  [[nodiscard]] FailureOr<CBitOperationRelation>
  analyzeCBitOperationRelation(Operation* lhs, Operation* rhs) {
    SmallVector<Operation*> lhsAncestors;
    SmallVector<Operation*> rhsAncestors;
    for (Operation* current = lhs; current != nullptr;
         current = current->getParentOp()) {
      if (!consumeCBitObservationStep()) {
        return failure();
      }
      lhsAncestors.emplace_back(current);
    }
    for (Operation* current = rhs; current != nullptr;
         current = current->getParentOp()) {
      if (!consumeCBitObservationStep()) {
        return failure();
      }
      rhsAncestors.emplace_back(current);
    }

    CBitOperationRelation relation;
    llvm::DenseMap<Block*, Operation*> rhsByBlock;
    for (Operation* ancestor : rhsAncestors) {
      rhsByBlock.try_emplace(ancestor->getBlock(), ancestor);
    }
    for (Operation* ancestor : lhsAncestors) {
      const auto rhsAncestor = rhsByBlock.find(ancestor->getBlock());
      if (rhsAncestor != rhsByBlock.end() && ancestor != rhsAncestor->second) {
        relation.lhsBeforeRhs = ancestor->isBeforeInBlock(rhsAncestor->second);
        break;
      }
    }

    llvm::DenseSet<Operation*> rhsProperAncestors;
    for (Operation* ancestor : llvm::ArrayRef(rhsAncestors).drop_front()) {
      rhsProperAncestors.insert(ancestor);
    }
    for (Operation* ancestor : llvm::ArrayRef(lhsAncestors).drop_front()) {
      if (rhsProperAncestors.contains(ancestor) &&
          isPotentiallyRepeatingLoop(ancestor)) {
        relation.sharesPotentiallyRepeatingLoop = true;
        break;
      }
    }
    return relation;
  }

  [[nodiscard]] bool mayLoadObserveStore(cbit::StoreOp store,
                                         cbit::LoadOp load) {
    const FailureOr<CBitOperationRelation> relation =
        analyzeCBitOperationRelation(store, load);
    if (failed(relation)) {
      return true;
    }
    if (relation->lhsBeforeRhs) {
      return *relation->lhsBeforeRhs ||
             relation->sharesPotentiallyRepeatingLoop;
    }
    return !dominance.properlyDominates(load.getOperation(),
                                        store.getOperation()) ||
           relation->sharesPotentiallyRepeatingLoop;
  }

  [[nodiscard]] bool isDefiniteKillingStore(Operation* operation,
                                            cbit::StoreOp source) {
    auto store = dyn_cast<cbit::StoreOp>(operation);
    return store && cbitRegistersMustAlias(source.getReg(), store.getReg()) &&
           cbitIndicesMustAlias(source.getIndex(), store.getIndex());
  }

  [[nodiscard]] bool isDefinitelyKilledBeforeLoad(cbit::StoreOp store,
                                                  cbit::LoadOp load) {
    if (store->getBlock() != load->getBlock()) {
      return false;
    }

    for (Operation* candidate = store->getNextNode(); candidate != nullptr;
         candidate = candidate->getNextNode()) {
      if (!consumeCBitObservationStep()) {
        return false;
      }
      if (candidate == load.getOperation()) {
        return false;
      }
      if (isDefiniteKillingStore(candidate, store)) {
        return true;
      }
    }
    if (!isNestedInPotentiallyRepeatingLoop(store)) {
      return false;
    }
    for (Operation& candidate : *store->getBlock()) {
      if (!consumeCBitObservationStep()) {
        return false;
      }
      if (&candidate == load.getOperation()) {
        return false;
      }
      if (isDefiniteKillingStore(&candidate, store)) {
        return true;
      }
    }
    return false;
  }

  [[nodiscard]] bool hasPotentiallyObservingLoad(cbit::StoreOp store) {
    const CBitAliasComponent& aliasComponent =
        getCBitAliasComponent(store.getReg());
    const auto observesStore = [&](cbit::LoadOp load) {
      // Once the bounded proof budget is exhausted, conservatively require
      // measurement-result support instead of spending unbounded time.
      if (!consumeCBitObservationStep()) {
        return true;
      }
      return mayLoadObserveStore(store, load) &&
             !isDefinitelyKilledBeforeLoad(store, load);
    };

    const std::optional<int64_t> storeIndex =
        getConstantIntValue(store.getIndex());
    if (!storeIndex) {
      return llvm::any_of(aliasComponent.loads, observesStore);
    }
    if (llvm::any_of(aliasComponent.dynamicLoads, observesStore)) {
      return true;
    }
    if (const auto constantLoads =
            aliasComponent.constantLoads.find(*storeIndex);
        constantLoads != aliasComponent.constantLoads.end()) {
      return llvm::any_of(constantLoads->second, observesStore);
    }
    return false;
  }

  [[nodiscard]] bool forwardMeasuredQubitUse(OpOperand& use,
                                             SmallVectorImpl<Value>& worklist) {
    Operation* const user = use.getOwner();
    if (isa<qco::SinkOp, func::ReturnOp, qtensor::DeallocOp>(user)) {
      return true;
    }
    if (forwardRegionValueUse(use, worklist)) {
      return true;
    }

    if (isa<qtensor::ExtractOp, qtensor::InsertOp>(user)) {
      appendQuantumResults(user, worklist);
      return true;
    }
    return false;
  }

  [[nodiscard]] Operation* findMeasuredQubitReuse(const Value root) {
    SmallVector<Value> worklist{root};
    llvm::DenseSet<Value> visited;
    while (!worklist.empty()) {
      const Value value = worklist.pop_back_val();
      if (!visited.insert(value).second) {
        continue;
      }
      for (OpOperand& use : value.getUses()) {
        if (!forwardMeasuredQubitUse(use, worklist)) {
          return use.getOwner();
        }
      }
    }
    return nullptr;
  }

  [[nodiscard]] bool
  forwardMeasurementResultUse(OpOperand& use,
                              SmallVectorImpl<Value>& worklist) {
    Operation* const user = use.getOwner();
    if (isa<func::ReturnOp>(user)) {
      return true;
    }
    if (auto store = dyn_cast<cbit::StoreOp>(user)) {
      return !hasPotentiallyObservingLoad(store);
    }
    return forwardRegionValueUse(use, worklist);
  }

  [[nodiscard]] Operation* findMeasurementResultRuntimeUse(const Value root) {
    SmallVector<Value> worklist{root};
    llvm::DenseSet<Value> visited;
    while (!worklist.empty()) {
      const Value value = worklist.pop_back_val();
      if (!visited.insert(value).second) {
        continue;
      }
      for (OpOperand& use : value.getUses()) {
        if (!forwardMeasurementResultUse(use, worklist)) {
          return use.getOwner();
        }
      }
    }
    return nullptr;
  }

  void collectMeasurementRequirements(qco::MeasureOp measurement) {
    if (Operation* const user =
            findMeasuredQubitReuse(measurement.getQubitOut())) {
      add(ProgramFeature::MeasuredQubitReuse, user);
      add(ProgramFeature::MidCircuitMeasurement, measurement);
    }

    if (Operation* const user =
            findMeasurementResultRuntimeUse(measurement.getResult())) {
      add(ProgramFeature::MeasurementResultUse, user);
      add(ProgramFeature::MidCircuitMeasurement, measurement);
    }
  }

  [[nodiscard]] static bool
  appendBlockArgumentSources(const BlockArgument argument,
                             SmallVectorImpl<Value>& worklist) {
    Block* const block = argument.getOwner();
    Operation* const parent = block->getParentOp();
    const unsigned argumentNumber = argument.getArgNumber();

    if (auto forOp = dyn_cast<scf::ForOp>(parent)) {
      if (argumentNumber == 0U) {
        worklist.emplace_back(forOp.getLowerBound());
        worklist.emplace_back(forOp.getUpperBound());
        worklist.emplace_back(forOp.getStep());
        return true;
      }
      const unsigned iterArgumentNumber = argumentNumber - 1U;
      if (iterArgumentNumber >= forOp.getInitArgs().size()) {
        return false;
      }
      worklist.emplace_back(forOp.getInitArgs()[iterArgumentNumber]);
      Operation* const terminator = block->getTerminator();
      if (iterArgumentNumber < terminator->getNumOperands()) {
        worklist.emplace_back(terminator->getOperand(iterArgumentNumber));
      }
      return true;
    }

    if (auto whileOp = dyn_cast<scf::WhileOp>(parent)) {
      if (block->getParent() == &whileOp.getBefore()) {
        if (argumentNumber >= whileOp.getInits().size()) {
          return false;
        }
        worklist.emplace_back(whileOp.getInits()[argumentNumber]);
        for (Block& afterBlock : whileOp.getAfter()) {
          Operation* const terminator = afterBlock.getTerminator();
          if (argumentNumber < terminator->getNumOperands()) {
            worklist.emplace_back(terminator->getOperand(argumentNumber));
          }
        }
        return true;
      }
      if (block->getParent() == &whileOp.getAfter()) {
        for (Block& beforeBlock : whileOp.getBefore()) {
          Operation* const terminator = beforeBlock.getTerminator();
          if (argumentNumber + 1U < terminator->getNumOperands()) {
            worklist.emplace_back(terminator->getOperand(argumentNumber + 1U));
          }
        }
        return true;
      }
    }
    return false;
  }

  [[nodiscard]] bool consumeCBitProvenanceStep() {
    if (cbitProvenanceSteps >= MAX_CBIT_ANALYSIS_STEPS) {
      return false;
    }
    ++cbitProvenanceSteps;
    return true;
  }

  [[nodiscard]] bool
  hasPotentiallyAliasingNestedStore(Operation* root,
                                    const llvm::SetVector<Value>& aliases,
                                    const Value index) {
    bool blocksProof = false;
    const WalkResult result = root->walk([&](Operation* nested) {
      if (!consumeCBitProvenanceStep()) {
        blocksProof = true;
        return WalkResult::interrupt();
      }
      auto store = dyn_cast<cbit::StoreOp>(nested);
      if (store && aliases.contains(store.getReg()) &&
          cbitIndicesMayAlias(store.getIndex(), index)) {
        blocksProof = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    return blocksProof || result.wasInterrupted();
  }

  [[nodiscard]] static bool
  isModeledCBitProvenanceParent(Operation* operation) {
    return isa<qco::IfOp, qco::IndexSwitchOp, scf::IfOp, scf::ForOp,
               scf::WhileOp, scf::IndexSwitchOp>(operation);
  }

  [[nodiscard]] bool
  appendKnownCBitLoadSource(cbit::LoadOp load,
                            SmallVectorImpl<Value>& worklist) {
    const llvm::SetVector<Value>& aliases =
        getCBitAliasComponent(load.getReg()).aliases;
    Operation* cursor = load.getOperation();
    while (true) {
      for (Operation* candidate = cursor->getPrevNode(); candidate != nullptr;
           candidate = candidate->getPrevNode()) {
        if (!consumeCBitProvenanceStep()) {
          return false;
        }
        if (auto store = dyn_cast<cbit::StoreOp>(candidate);
            store && aliases.contains(store.getReg())) {
          if (!cbitIndicesMayAlias(store.getIndex(), load.getIndex())) {
            continue;
          }
          if (cbitRegistersMustAlias(store.getReg(), load.getReg()) &&
              cbitIndicesMustAlias(store.getIndex(), load.getIndex())) {
            worklist.emplace_back(store.getValue());
            return true;
          }
          return false;
        }
        if (auto allocation = dyn_cast<cbit::AllocOp>(candidate);
            allocation && aliases.contains(allocation.getResult())) {
          return false;
        }
        if (isa<cbit::LoadOp, arith::SelectOp>(candidate)) {
          continue;
        }
        if (isModeledCBitProvenanceParent(candidate)) {
          if (hasPotentiallyAliasingNestedStore(candidate, aliases,
                                                load.getIndex())) {
            return false;
          }
          continue;
        }
        if (candidate->getNumRegions() != 0 ||
            llvm::any_of(candidate->getOperands(),
                         [&aliases](const Value operand) {
                           return aliases.contains(operand);
                         })) {
          return false;
        }
      }

      Operation* const parent = cursor->getBlock()->getParentOp();
      if (parent == nullptr || isa<func::FuncOp>(parent) ||
          !isModeledCBitProvenanceParent(parent) ||
          !cursor->getBlock()->getParent()->hasOneBlock()) {
        return false;
      }
      if (isPotentiallyRepeatingLoop(parent) &&
          hasPotentiallyAliasingNestedStore(parent, aliases, load.getIndex())) {
        return false;
      }
      cursor = parent;
    }
  }

  [[nodiscard]] LogicalResult collectFeedbackRequirements(Operation* consumer,
                                                          const Value root) {
    if (verifiedFeedbackRoots.contains(root)) {
      return success();
    }
    SmallVector<Value> worklist{root};
    llvm::DenseSet<Value> visited;
    bool sawMeasurement = false;

    while (!worklist.empty()) {
      const Value value = worklist.pop_back_val();
      if (!visited.insert(value).second || matchPattern(value, m_Constant())) {
        continue;
      }
      if (feedbackAnalysisSteps >= MAX_FEEDBACK_ANALYSIS_STEPS) {
        consumer->emitError()
            << "target compilation cannot prove measurement-feedback "
               "semantics after "
            << MAX_FEEDBACK_ANALYSIS_STEPS << " producer steps";
        return failure();
      }
      ++feedbackAnalysisSteps;

      Operation* const definition = value.getDefiningOp();
      if (definition == nullptr) {
        if (const auto argument = dyn_cast<BlockArgument>(value);
            argument && appendBlockArgumentSources(argument, worklist)) {
          continue;
        }
        consumer->emitError()
            << "target compilation requires runtime control condition for '"
            << consumer->getName()
            << "' to be derived from a measurement result";
        return failure();
      }

      if (auto measurement = dyn_cast<qco::MeasureOp>(definition);
          measurement && value == measurement.getResult()) {
        sawMeasurement = true;
        add(ProgramFeature::MidCircuitMeasurement, measurement);
        add(ProgramFeature::MeasurementResultUse, consumer);
        continue;
      }

      if (isa<CallOpInterface>(definition)) {
        definition->emitError()
            << "target compilation cannot verify reachable function call '"
            << definition->getName() << "'";
        return failure();
      }

      if (isa<qco::IfOp, scf::IfOp>(definition)) {
        add(ProgramFeature::ForwardBranching, definition);
        const auto result = cast<OpResult>(value);
        for (Region& region : definition->getRegions()) {
          if (!region.empty() &&
              result.getResultNumber() <
                  region.front().getTerminator()->getNumOperands()) {
            worklist.emplace_back(region.front().getTerminator()->getOperand(
                result.getResultNumber()));
          }
        }
        if (auto ifOp = dyn_cast<qco::IfOp>(definition)) {
          worklist.emplace_back(ifOp.getCondition());
        } else {
          worklist.emplace_back(cast<scf::IfOp>(definition).getCondition());
        }
        continue;
      }

      if (isa<qco::IndexSwitchOp, scf::IndexSwitchOp>(definition)) {
        add(ProgramFeature::MultiwayBranching, definition);
        const auto result = cast<OpResult>(value);
        for (Region& region : definition->getRegions()) {
          if (!region.empty() &&
              result.getResultNumber() <
                  region.front().getTerminator()->getNumOperands()) {
            worklist.emplace_back(region.front().getTerminator()->getOperand(
                result.getResultNumber()));
          }
        }
        if (auto switchOp = dyn_cast<qco::IndexSwitchOp>(definition)) {
          worklist.emplace_back(switchOp.getArg());
        } else {
          worklist.emplace_back(cast<scf::IndexSwitchOp>(definition).getArg());
        }
        continue;
      }

      if (auto forOp = dyn_cast<scf::ForOp>(definition)) {
        add(ProgramFeature::CountedIteration, definition);
        worklist.emplace_back(forOp.getLowerBound());
        worklist.emplace_back(forOp.getUpperBound());
        worklist.emplace_back(forOp.getStep());
        const unsigned resultNumber = cast<OpResult>(value).getResultNumber();
        if (resultNumber < forOp.getInitArgs().size()) {
          worklist.emplace_back(forOp.getInitArgs()[resultNumber]);
        }
        Operation* const terminator = forOp.getBody()->getTerminator();
        if (resultNumber < terminator->getNumOperands()) {
          worklist.emplace_back(terminator->getOperand(resultNumber));
        }
        continue;
      }

      if (auto whileOp = dyn_cast<scf::WhileOp>(definition)) {
        add(ProgramFeature::ConditionalLoop, definition);
        const unsigned resultNumber = cast<OpResult>(value).getResultNumber();
        for (Block& block : whileOp.getBefore()) {
          Operation* const terminator = block.getTerminator();
          if (terminator->getNumOperands() != 0U) {
            worklist.emplace_back(terminator->getOperand(0));
          }
          if (resultNumber + 1U < terminator->getNumOperands()) {
            worklist.emplace_back(terminator->getOperand(resultNumber + 1U));
          }
        }
        continue;
      }

      if (auto load = dyn_cast<cbit::LoadOp>(definition);
          load && appendKnownCBitLoadSource(load, worklist)) {
        continue;
      }

      if (isRuntimeClassicalComputation(definition)) {
        addComputationFeatures(definition);
        llvm::append_range(worklist, definition->getOperands());
        continue;
      }

      consumer->emitError()
          << "target compilation cannot prove measurement-feedback semantics "
             "through condition producer '"
          << definition->getName() << "'";
      return failure();
    }

    if (!sawMeasurement) {
      consumer->emitError()
          << "target compilation requires runtime control condition for '"
          << consumer->getName() << "' to be derived from a measurement result";
      return failure();
    }
    verifiedFeedbackRoots.insert(root);
    return success();
  }

  SmallVector<ProgramRequirement> requirements;
  llvm::DenseSet<Operation*> quantumEvolutionSubtrees;
  SmallVector<CBitAliasComponent> cbitAliasComponents;
  llvm::DenseMap<Value, size_t> cbitAliasComponentIds;
  llvm::DenseMap<Value, Value> cbitSingleRootCache;
  size_t cbitObservationChecks = 0U;
  size_t cbitProvenanceSteps = 0U;
  llvm::DenseSet<Value> verifiedFeedbackRoots;
  size_t feedbackAnalysisSteps = 0U;
  DominanceInfo dominance;
  TypeSummaryCache& typeSummaries;
};

[[nodiscard]] LogicalResult
verifyProgramRequirements(const ArrayRef<ProgramRequirement> requirements,
                          const mqt::TargetEnvAttr environment) {
  for (const ProgramRequirement requirement : requirements) {
    const llvm::StringRef name = programFeatureName(requirement.feature);
    if (environment.supports(name)) {
      continue;
    }
    auto diagnostic =
        requirement.operation->emitError()
        << "selected '" << environment.getFormat().getValue()
        << "' execution profile does not support program feature '" << name
        << "' required by '" << requirement.operation->getName() << "'";
    if (!environment.getOptionalFeaturesKnown()) {
      diagnostic << "; optional feature support is unknown";
    }
    return failure();
  }
  return success();
}

class TargetLegality final {
public:
  TargetLegality(func::FuncOp entryPoint, TypeSummaryCache& typeSummariesIn)
      : typeSummaries(typeSummariesIn) {
    llvm::DenseMap<Region*, size_t> regionDepths;
    for (Region& region : entryPoint->getRegions()) {
      regionDepths.try_emplace(&region, 0U);
    }
    entryPoint.getBody().walk<WalkOrder::PreOrder>([&](Operation* operation) {
      const size_t childDepth =
          regionDepths.lookup(operation->getParentRegion()) + 1U;
      for (Region& region : operation->getRegions()) {
        regionDepths.try_emplace(&region, childDepth);
      }
    });

    constexpr size_t noQuantumDefinition = std::numeric_limits<size_t>::max();
    llvm::DenseMap<Operation*, size_t> minimumQuantumDefinitionDepths;
    entryPoint.getBody().walk<WalkOrder::PostOrder>([&](Operation* operation) {
      size_t nestedMinimum = noQuantumDefinition;
      for (Region& region : operation->getRegions()) {
        for (Block& block : region) {
          for (Operation& child : block) {
            nestedMinimum =
                std::min(nestedMinimum,
                         minimumQuantumDefinitionDepths.find(&child)->second);
          }
        }
      }

      if (nestedMinimum <= regionDepths.lookup(operation->getParentRegion())) {
        quantumCaptureOperations.insert(operation);
      }

      size_t subtreeMinimum = nestedMinimum;
      for (Value operand : operation->getOperands()) {
        if (typeSummaries.get(operand.getType()).hasQuantumState) {
          subtreeMinimum = std::min(
              subtreeMinimum, regionDepths.lookup(operand.getParentRegion()));
        }
      }
      minimumQuantumDefinitionDepths.try_emplace(operation, subtreeMinimum);
    });
  }

  [[nodiscard]] bool verifyStructuredControl(Operation* operation) const {
    return verifyStructuredQuantumState(operation);
  }

  [[nodiscard]] static bool verifyQubitIndex(Operation* operation,
                                             const Value index) {
    if (matchPattern(index, m_Constant())) {
      return true;
    }
    operation->emitError() << "target compilation cannot lower '"
                           << operation->getName()
                           << "' with a dynamic qubit index";
    return false;
  }

  [[nodiscard]] bool verifyUnknown(Operation* operation) {
    if (isa<CallOpInterface>(operation)) {
      operation->emitError()
          << "target compilation cannot verify reachable function call '"
          << operation->getName() << "'";
      return false;
    }
    if (isa<BranchOpInterface, RegionBranchOpInterface>(operation)) {
      operation->emitError()
          << "target compilation cannot lower classical-control construct '"
          << operation->getName() << "'";
      return false;
    }
    if (operation->getNumRegions() != 0U &&
        !isModeledRegionOperation(operation)) {
      operation->emitError()
          << "target compilation cannot verify regions of unmodeled operation '"
          << operation->getName() << "'";
      return false;
    }
    const auto hasQuantumState = [&](const Type type) {
      return typeSummaries.get(type).hasQuantumState;
    };
    if ((llvm::any_of(operation->getOperandTypes(), hasQuantumState) ||
         llvm::any_of(operation->getResultTypes(), hasQuantumState)) &&
        !isModeledQuantumStateOperation(operation)) {
      operation->emitError()
          << "target compilation cannot lower quantum state carried through "
             "unmodeled operation '"
          << operation->getName() << "'";
      return false;
    }
    const auto hasCBitRegister = [&](const Type type) {
      return typeSummaries.get(type).hasCBitRegister;
    };
    if ((llvm::any_of(operation->getOperandTypes(), hasCBitRegister) ||
         llvm::any_of(operation->getResultTypes(), hasCBitRegister)) &&
        !isModeledCBitRegisterOperation(operation)) {
      operation->emitError()
          << "target compilation cannot lower classical-bit register carried "
             "through unmodeled operation '"
          << operation->getName() << "'";
      return false;
    }
    return true;
  }

private:
  [[nodiscard]] static bool isModeledRegionOperation(Operation* operation) {
    return isa<func::FuncOp>(operation) ||
           operation->getName().getDialectNamespace() ==
               qco::QCODialect::getDialectNamespace();
  }

  [[nodiscard]] static bool
  isModeledQuantumStateOperation(Operation* operation) {
    const llvm::StringRef dialect = operation->getName().getDialectNamespace();
    return dialect == qco::QCODialect::getDialectNamespace() ||
           dialect == qtensor::QTensorDialect::getDialectNamespace() ||
           isa<func::ReturnOp, scf::ConditionOp, scf::YieldOp>(operation);
  }

  [[nodiscard]] static bool
  isModeledCBitRegisterOperation(Operation* operation) {
    return isa<arith::SelectOp, cbit::AllocOp, cbit::LoadOp, cbit::StoreOp,
               func::ReturnOp, qco::YieldOp, scf::ConditionOp, scf::YieldOp>(
        operation);
  }

  [[nodiscard]] bool verifyStructuredQuantumState(Operation* operation) const {
    const auto isQubitTensor = [&](const Type type) {
      const auto tensor = dyn_cast<TensorType>(type);
      return tensor &&
             typeSummaries.get(tensor.getElementType()).hasQuantumState;
    };
    if (llvm::any_of(operation->getOperandTypes(), isQubitTensor) ||
        llvm::any_of(operation->getResultTypes(), isQubitTensor)) {
      operation->emitError()
          << "target compilation cannot lower quantum tensor state carried "
             "through classical-control construct '"
          << operation->getName() << "'";
      return false;
    }

    const auto isQuantumAggregate = [&](const Type type) {
      return !isa<qco::QubitType>(type) &&
             typeSummaries.get(type).hasQuantumState;
    };
    if (llvm::any_of(operation->getOperandTypes(), isQuantumAggregate) ||
        llvm::any_of(operation->getResultTypes(), isQuantumAggregate)) {
      operation->emitError()
          << "target compilation cannot lower aggregate quantum state carried "
             "through classical-control construct '"
          << operation->getName() << "'";
      return false;
    }

    if (quantumCaptureOperations.contains(operation)) {
      operation->emitError()
          << "target compilation cannot lower quantum state captured by "
             "classical-control construct '"
          << operation->getName() << "'";
      return false;
    }

    if (isa<scf::IfOp, scf::IndexSwitchOp>(operation) &&
        (llvm::any_of(operation->getOperandTypes(),
                      [&](const Type type) {
                        return typeSummaries.get(type).hasQuantumState;
                      }) ||
         llvm::any_of(operation->getResultTypes(), [&](const Type type) {
           return typeSummaries.get(type).hasQuantumState;
         }))) {
      operation->emitError()
          << "target compilation cannot lower quantum state carried through "
             "generic classical-control construct '"
          << operation->getName() << "'";
      return false;
    }
    return true;
  }

  llvm::DenseSet<Operation*> quantumCaptureOperations;
  TypeSummaryCache& typeSummaries;
};

struct VerifyTargetEnvironmentPass final
    : PassWrapper<VerifyTargetEnvironmentPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VerifyTargetEnvironmentPass)

protected:
  void runOnOperation() override {
    const auto environment = getOperation()->getAttrOfType<mqt::TargetEnvAttr>(
        mqt::TargetEnvAttr::getOperationAttributeName());
    if (!environment) {
      getOperation().emitError("target compilation requires mqt.target_env");
      signalPassFailure();
      return;
    }
    func::FuncOp entryPoint = mqt::getEntryPoint(getOperation());
    if (!entryPoint) {
      getOperation().emitError("no program entry point found");
      signalPassFailure();
      return;
    }

    TypeSummaryCache typeSummaries;
    ProgramRequirements requirements(typeSummaries);
    if (failed(requirements.collect(entryPoint)) ||
        failed(verifyProgramRequirements(requirements.get(), environment))) {
      signalPassFailure();
      return;
    }

    TargetLegality legality(entryPoint, typeSummaries);
    ConversionTarget target(getContext());
    RewritePatternSet patterns(&getContext());

    target.addDynamicallyLegalOp<qco::IfOp>([&](qco::IfOp operation) {
      return legality.verifyStructuredControl(operation);
    });
    target.addDynamicallyLegalOp<scf::IfOp>([&](scf::IfOp operation) {
      return legality.verifyStructuredControl(operation);
    });
    target.addDynamicallyLegalOp<scf::ForOp>([&](scf::ForOp operation) {
      return legality.verifyStructuredControl(operation);
    });
    target.addDynamicallyLegalOp<scf::WhileOp>([&](scf::WhileOp operation) {
      return legality.verifyStructuredControl(operation);
    });
    target.addDynamicallyLegalOp<qco::IndexSwitchOp>(
        [&](qco::IndexSwitchOp operation) {
          return legality.verifyStructuredControl(operation);
        });
    target.addDynamicallyLegalOp<scf::IndexSwitchOp>(
        [&](scf::IndexSwitchOp operation) {
          return legality.verifyStructuredControl(operation);
        });
    target.addDynamicallyLegalOp<qtensor::ExtractOp>(
        [](qtensor::ExtractOp operation) {
          return TargetLegality::verifyQubitIndex(operation,
                                                  operation.getIndex());
        });
    target.addDynamicallyLegalOp<qtensor::InsertOp>(
        [](qtensor::InsertOp operation) {
          return TargetLegality::verifyQubitIndex(operation,
                                                  operation.getIndex());
        });
    target.markUnknownOpDynamicallyLegal([&](Operation* operation) {
      return legality.verifyUnknown(operation);
    });

    if (failed(
            applyPartialConversion(entryPoint, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

void populateTargetCompilationPipeline(OpPassManager& pm,
                                       const CompilerTarget& target) {
  populateTargetCompilationPipeline(pm, target, ProgramFormat::QCOOptimized);
}

void populateTargetCompilationPipeline(OpPassManager& pm,
                                       const CompilerTarget& target,
                                       const ProgramFormat format) {
  pm.addPass(std::make_unique<AttachTargetEnvironmentPass>(target, format));
  pm.addPass(createSCCPPass());
  pm.addPass(std::make_unique<VerifyStaticLoopTripCountsPass>());
  populateQCOCleanupPipeline(pm);
  pm.addPass(std::make_unique<LegalizeCountedIterationPass>());
  pm.addPass(createSCCPPass());
  pm.addPass(std::make_unique<VerifyStaticLoopTripCountsPass>());
  populateQCOCleanupPipeline(pm);
  pm.addPass(std::make_unique<VerifyTargetEnvironmentPass>());
  populateDecomposeMultiControlledPipeline(pm, 3);
  populateDefaultQCOOptimizationPipeline(pm);
  pm.addPass(qco::createFuseTwoQubitGates());
  pm.addPass(qco::createMappingPass(target, qco::MappingPassOptions{}));
  pm.addPass(std::make_unique<VerifyStaticLoopTripCountsPass>());
  populateQCOCleanupPipeline(pm);
  pm.addPass(qco::createTargetNativeSynthesis(target));
  pm.addPass(createCSEPass());
  pm.addPass(createRemoveDeadValuesPass());
  pm.addPass(qco::createVerifyTargetConformance(target));
}

} // namespace mlir
