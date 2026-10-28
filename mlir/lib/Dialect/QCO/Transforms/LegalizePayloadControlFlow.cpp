/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/Compiler/TargetEnvironment.h"
#include "mqt/Dialect/MQT/IR/MQTAttributes.h"
#include "mqt/Dialect/MQT/IR/MQTDialect.h"
#include "mqt/Dialect/MQT/Transforms/Passes.h"
#include "mqt/Dialect/QCO/IR/QCOOps.h"
#include "mqt/Dialect/QCO/QCOUtils.h"
#include "mqt/Dialect/QCO/Transforms/Passes.h"
#include "mqt/Dialect/QTensor/IR/QTensorDialect.h"
#include "mqt/Dialect/QTensor/IR/QTensorOps.h"

#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/WalkResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/RegionUtils.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>

namespace mlir::qco {

#define GEN_PASS_DEF_LEGALIZECONTROLFLOW
#define GEN_PASS_DEF_UNROLLLOOPSFORPAYLOAD
#include "mqt/Dialect/QCO/Transforms/Passes.h.inc"

/// Require literal bounds before expanding a counted loop.
static std::optional<llvm::APInt> getExactConstantTripCount(scf::ForOp loop) {
  if (!getConstantAPIntValue(getAsOpFoldResult(loop.getLowerBound())) ||
      !getConstantAPIntValue(getAsOpFoldResult(loop.getUpperBound())) ||
      !getConstantAPIntValue(getAsOpFoldResult(loop.getStep()))) {
    return std::nullopt;
  }
  return loop.getStaticTripCount();
}

/// Check the integer arithmetic used by MLIR's full-loop unroller.
static bool canUseFullUnroll(scf::ForOp loop, uint64_t iterations) {
  const auto constant = [&](Value value) -> std::optional<int64_t> {
    const auto result = getConstantAPIntValue(getAsOpFoldResult(value));
    if (!result) {
      return std::nullopt;
    }
    if (loop.getUnsignedCmp()) {
      if (result->first.getActiveBits() > 63) {
        return std::nullopt;
      }
      return static_cast<int64_t>(result->first.getZExtValue());
    }
    return result->first.trySExtValue();
  };
  const auto lowerBound = constant(loop.getLowerBound());
  const auto upperBound = constant(loop.getUpperBound());
  const auto step = constant(loop.getStep());
  if (!lowerBound || !upperBound || !step || *step <= 0) {
    return false;
  }
  int64_t scaledStep = 0;
  int64_t unrolledUpperBound = 0;
  if (llvm::MulOverflow(*step, static_cast<int64_t>(iterations), scaledStep) !=
          0 ||
      llvm::AddOverflow(*lowerBound, scaledStep, unrolledUpperBound) != 0) {
    return false;
  }
  auto type = dyn_cast<IntegerType>(loop.getInductionVar().getType());
  return !type || (loop.getUnsignedCmp()
                       ? llvm::isUIntN(type.getWidth(),
                                       static_cast<uint64_t>(scaledStep))
                       : llvm::isIntN(type.getWidth(), scaledStep));
}

/// Expand a constant counted loop within a cumulative cloning budget.
///
/// Requires literal bounds and an exact trip count greater than one. Diagnose
/// unsupported arithmetic or excessive expansion before changing the loop.
static LogicalResult unrollStaticLoop(scf::ForOp loop,
                                      const llvm::APInt& tripCount,
                                      uint64_t maxOperations,
                                      uint64_t& clonedOperations) {
  IRRewriter rewriter(loop.getContext());
  uint64_t bodyOperations = 0;
  Operation* terminator = loop.getBody()->getTerminator();
  loop.getRegion().walk(
      [&](Operation* operation) { bodyOperations += operation != terminator; });
  const bool emptyBody = bodyOperations == 0;
  bodyOperations = std::max(bodyOperations, uint64_t{1});
  if ((tripCount - 1)
          .ugt((maxOperations - clonedOperations) / bodyOperations)) {
    return loop.emitError() << "full unrolling would clone more than "
                            << maxOperations << " loop-body operations";
  }
  if (tripCount.getActiveBits() > 64 ||
      !canUseFullUnroll(loop, tripCount.getZExtValue())) {
    return loop.emitError(
        "cannot safely apply MLIR full unrolling to these loop bounds");
  }
  clonedOperations += bodyOperations * (tripCount.getZExtValue() - 1);
  if (emptyBody) {
    // LLVM skips terminator-only bodies; remove this constant when it supports
    // unrolling them. Caller cleanup removes it after remapping carried values.
    rewriter.setInsertionPointToStart(loop.getBody());
    arith::ConstantIndexOp::create(rewriter, loop.getLoc(), 0);
  }
  if (failed(loopUnrollFull(loop))) {
    return loop.emitError("failed to fully unroll a static counted loop");
  }
  return success();
}

namespace {

// Bound generated nesting to keep recursive IR processing manageable.
constexpr uint64_t MAX_SWITCH_EXPANSION_DEPTH = 256U;

enum class ControlFeature : uint8_t {
  ForwardBranching,
  CountedIteration,
  ConditionalLoop,
  MultiwayBranching,
  Count,
};

struct CapabilityGroup {
  bool usable = false;
  std::optional<uint64_t> maxNestingDepth;
  std::optional<uint64_t> maxIterationCount;
  std::optional<uint64_t> maxCaseCount;
};

class PayloadControlSupport {
public:
  [[nodiscard]] static std::optional<PayloadControlSupport>
  read(ModuleOp moduleOp, const TargetEnvironmentAnalysis& analysis,
       StringRef specification) {
    std::optional<PayloadSpecification> explicitPayload;
    if (!specification.empty()) {
      auto attribute = dyn_cast_or_null<mqt::PayloadSpecAttr>(
          parseAttribute(specification, moduleOp.getContext()));
      auto payload = PayloadSpecification::create(attribute);
      if (!payload) {
        moduleOp.emitError()
            << "invalid payload-spec: " << llvm::toString(payload.takeError());
        return std::nullopt;
      }
      explicitPayload = std::move(*payload);
    } else if (!analysis) {
      moduleOp.emitError()
          << "payload control-flow legalization requires a valid "
             "mqt.target_env: "
          << analysis.error();
      return std::nullopt;
    }

    PayloadControlSupport support;
    const auto& payload = explicitPayload
                              ? *explicitPayload
                              : analysis.environment().payloadSpecification();
    support.indexedQubits =
        !explicitPayload && analysis.environment().supportsIndexedQubits();
    for (const ProgramCapability& capability : payload.capabilities()) {
      const auto feature =
          llvm::StringSwitch<std::optional<ControlFeature>>(capability.id)
              .Case(ProgramCapability::FORWARD_BRANCHING,
                    ControlFeature::ForwardBranching)
              .Case(ProgramCapability::COUNTED_ITERATION,
                    ControlFeature::CountedIteration)
              .Case(ProgramCapability::CONDITIONAL_LOOP,
                    ControlFeature::ConditionalLoop)
              .Case(ProgramCapability::MULTIWAY_BRANCHING,
                    ControlFeature::MultiwayBranching)
              .Default(std::nullopt);
      if (!feature) {
        continue;
      }

      CapabilityGroup& group = support.get(*feature);
      if (capability.value != 0U) {
        continue;
      }

      group = CapabilityGroup{.usable = true};
      for (const ProgramConstraint& constraint : capability.constraints) {
        applyConstraint(group, *feature, constraint.id, constraint.value);
      }
    }
    return support;
  }

  bool indexedQubits = false;

  [[nodiscard]] const CapabilityGroup& get(const ControlFeature feature) const {
    return groups[static_cast<size_t>(feature)];
  }

  [[nodiscard]] bool coversDepth(const ControlFeature feature,
                                 Operation* operation) const {
    const CapabilityGroup& group = get(feature);
    return group.usable && (!group.maxNestingDepth ||
                            controlDepth(operation) <= *group.maxNestingDepth);
  }

  [[nodiscard]] bool
  coversIteration(const ControlFeature feature, Operation* operation,
                  const std::optional<llvm::APInt>& tripCount) const {
    if (!coversDepth(feature, operation)) {
      return false;
    }
    const CapabilityGroup& group = get(feature);
    if (!group.maxIterationCount) {
      return true;
    }
    return tripCount && tripCount->ule(*group.maxIterationCount);
  }

  [[nodiscard]] bool coversMultiwayBranching(Operation* operation,
                                             const uint64_t caseCount) const {
    if (!coversDepth(ControlFeature::MultiwayBranching, operation)) {
      return false;
    }
    const CapabilityGroup& group = get(ControlFeature::MultiwayBranching);
    return !group.maxCaseCount || caseCount <= *group.maxCaseCount;
  }

  /// Check expansion of a switch with at least one explicit case.
  [[nodiscard]] bool canLowerSwitch(Operation* operation,
                                    uint64_t caseCount) const {
    const auto& group = get(ControlFeature::ForwardBranching);
    const auto depth = controlDepth(operation);
    const auto maximum =
        std::min(MAX_SWITCH_EXPANSION_DEPTH,
                 group.maxNestingDepth.value_or(MAX_SWITCH_EXPANSION_DEPTH));
    if (!group.usable || depth > maximum || caseCount > maximum - depth + 1U) {
      return false;
    }
    // Region zero is the default; explicit case i gains i enclosing branches.
    for (auto [index, region] : llvm::enumerate(operation->getRegions())) {
      const auto extraDepth = index == 0 ? caseCount - 1U : index - 1U;
      if (region
              .walk<WalkOrder::PreOrder>([&](Operation* nested) {
                return isStructuredControl(nested) &&
                               controlDepth(nested) >
                                   MAX_SWITCH_EXPANSION_DEPTH - extraDepth
                           ? WalkResult::interrupt()
                           : WalkResult::advance();
              })
              .wasInterrupted()) {
        return false;
      }
    }
    return true;
  }

  [[nodiscard]] static uint64_t controlDepth(Operation* operation) {
    uint64_t depth = 1U;
    for (Operation* parent = operation->getParentOp(); parent != nullptr;
         parent = parent->getParentOp()) {
      if (isStructuredControl(parent)) {
        ++depth;
      }
    }
    return depth;
  }

private:
  [[nodiscard]] static bool isStructuredControl(Operation* operation) {
    return isa<IfOp, IndexSwitchOp, scf::IfOp, scf::IndexSwitchOp, scf::ForOp,
               scf::WhileOp>(operation);
  }

  [[nodiscard]] CapabilityGroup& get(const ControlFeature feature) {
    return groups[static_cast<size_t>(feature)];
  }

  static void applyConstraint(CapabilityGroup& group,
                              const ControlFeature feature,
                              const llvm::StringRef id, const uint64_t value) {
    if (!group.usable) {
      return;
    }
    if (value == 0U) {
      group.usable = false;
      return;
    }
    if (id == ProgramConstraint::MAX_NESTING_DEPTH) {
      group.maxNestingDepth = value;
      return;
    }
    if (id == ProgramConstraint::MAX_ITERATION_COUNT &&
        (feature == ControlFeature::CountedIteration ||
         feature == ControlFeature::ConditionalLoop)) {
      group.maxIterationCount = value;
      return;
    }
    if (id == ProgramConstraint::MAX_CASE_COUNT &&
        feature == ControlFeature::MultiwayBranching) {
      group.maxCaseCount = value;
      return;
    }
    group.usable = false;
  }

  std::array<CapabilityGroup, static_cast<size_t>(ControlFeature::Count)>
      groups{};
};

} // namespace

[[nodiscard]] static bool hasLinearCapture(Operation* operation) {
  bool hasCapture = false;
  visitUsedValuesDefinedAbove(operation->getRegions(), [&](OpOperand* operand) {
    hasCapture |= isLinearQubitType(operand->get().getType());
  });
  return hasCapture;
}

[[nodiscard]] static bool hasLinearBranchState(Operation* operation) {
  return llvm::any_of(operation->getResultTypes(), isLinearQubitType) ||
         hasLinearCapture(operation);
}

// Check the supported loop input form once before transforming control flow.
// Exactly one SSA use does not exclude captures in repeated regions.
static LogicalResult verifyLoopCarriedState(ModuleOp moduleOp) {
  const auto result = moduleOp.walk([](Operation* operation) {
    if (isa<scf::ForOp, scf::WhileOp>(operation) &&
        hasLinearCapture(operation)) {
      operation->emitOpError(
          "captures QCO linear values; pass them as iteration arguments");
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return result.wasInterrupted() ? failure() : success();
}

static LogicalResult foldStaticBranches(ModuleOp moduleOp) {
  mqt::simplifyClassicalControl(moduleOp);
  // Fold branches without transforming loops before the capture and unroll
  // safety checks below.
  RewritePatternSet patterns(moduleOp.getContext());
  populateFoldStaticIfPatterns(patterns);
  IndexSwitchOp::getCanonicalizationPatterns(patterns, moduleOp.getContext());
  scf::IfOp::getCanonicalizationPatterns(patterns, moduleOp.getContext());
  scf::IndexSwitchOp::getCanonicalizationPatterns(patterns,
                                                  moduleOp.getContext());
  return applyPatternsGreedily(
      moduleOp, std::move(patterns),
      GreedyRewriteConfig{}.setUseTopDownTraversal().setMaxIterations(
          GreedyRewriteConfig::kNoLimit));
}

[[nodiscard]] static bool isLegal(IfOp operation,
                                  const PayloadControlSupport& support) {
  return support.coversDepth(ControlFeature::ForwardBranching, operation);
}

[[nodiscard]] static bool isLegal(scf::IfOp operation,
                                  const PayloadControlSupport& support) {
  return !hasLinearBranchState(operation) &&
         support.coversDepth(ControlFeature::ForwardBranching, operation);
}

[[nodiscard]] static bool isLegal(scf::ForOp operation,
                                  const PayloadControlSupport& support) {
  return support.coversIteration(ControlFeature::CountedIteration, operation,
                                 getExactConstantTripCount(operation));
}

[[nodiscard]] static bool isLegal(scf::WhileOp operation,
                                  const PayloadControlSupport& support) {
  return support.coversIteration(ControlFeature::ConditionalLoop, operation,
                                 operation.getStaticTripCount());
}

[[nodiscard]] static bool isLegal(IndexSwitchOp operation,
                                  const PayloadControlSupport& support) {
  const uint64_t cases = operation.getNumCases();
  return cases > 0U && support.coversMultiwayBranching(operation, cases);
}

[[nodiscard]] static bool isLegal(scf::IndexSwitchOp operation,
                                  const PayloadControlSupport& support) {
  const uint64_t cases = operation.getNumCases();
  return !hasLinearBranchState(operation) && cases > 0U &&
         support.coversMultiwayBranching(operation, cases);
}

static void inlineDefaultRegion(Operation* operation, Block& block,
                                ValueRange blockArguments,
                                ConversionPatternRewriter& rewriter) {
  Operation* terminator = block.getTerminator();
  rewriter.inlineBlockBefore(&block, operation, blockArguments);
  SmallVector<Value> replacements(terminator->getOperands());
  rewriter.eraseOp(terminator);
  rewriter.replaceOp(operation, replacements);
}

namespace {

class LowerQCOIndexSwitch final : public OpConversionPattern<IndexSwitchOp> {
public:
  LowerQCOIndexSwitch(MLIRContext* context,
                      const PayloadControlSupport& supportIn)
      : OpConversionPattern(context), support(&supportIn) {}

  LogicalResult
  matchAndRewrite(IndexSwitchOp operation, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto cases = operation.getCaseRegions();
    Region* const defaultRegion = &operation.getDefaultRegion();

    if (cases.empty()) {
      inlineDefaultRegion(operation, defaultRegion->front(),
                          adaptor.getTargets(), rewriter);
      return success();
    }
    if (!support->canLowerSwitch(operation, cases.size())) {
      return rewriter.notifyMatchFailure(
          operation,
          "switch requires unsupported or excessive branching depth");
    }

    IfOp replacement;
    ValueRange targets = adaptor.getTargets();
    for (size_t index = 0; index < cases.size(); ++index) {
      auto constant = arith::ConstantIndexOp::create(
          rewriter, operation.getLoc(), operation.getCases()[index]);
      auto condition = arith::CmpIOp::create(
          rewriter, operation.getLoc(), arith::CmpIPredicate::eq,
          adaptor.getArg(), constant.getResult());
      auto ifOp = IfOp::create(rewriter, operation.getLoc(),
                               operation.getClassicalResults().getTypes(),
                               operation.getLinearResults().getTypes(),
                               condition, targets);
      if (index == 0) {
        replacement = ifOp;
      } else {
        YieldOp::create(rewriter, operation.getLoc(), ifOp.getResults());
      }
      rewriter.inlineRegionBefore(cases[index], ifOp.getThenRegion(),
                                  ifOp.getThenRegion().end());
      if (index + 1U == cases.size()) {
        rewriter.inlineRegionBefore(*defaultRegion, ifOp.getElseRegion(),
                                    ifOp.getElseRegion().end());
        break;
      }

      Block& elseBlock = ifOp.getElseRegion().emplaceBlock();
      elseBlock.addArguments(targets.getTypes(),
                             SmallVector(targets.size(), operation.getLoc()));
      rewriter.setInsertionPointToEnd(&elseBlock);
      targets = elseBlock.getArguments();
    }

    rewriter.replaceOp(operation, replacement.getResults());
    return success();
  }

private:
  const PayloadControlSupport* support;
};

class LowerSCFIndexSwitch final
    : public OpConversionPattern<scf::IndexSwitchOp> {
public:
  LowerSCFIndexSwitch(MLIRContext* context,
                      const PayloadControlSupport& supportIn)
      : OpConversionPattern(context), support(&supportIn) {}

  LogicalResult
  matchAndRewrite(scf::IndexSwitchOp operation, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    if (hasLinearBranchState(operation)) {
      return rewriter.notifyMatchFailure(
          operation, "SCF control flow cannot carry QCO linear values");
    }

    auto cases = operation.getCaseRegions();
    Region* const defaultRegion = &operation.getDefaultRegion();

    if (cases.empty()) {
      inlineDefaultRegion(operation, defaultRegion->front(), {}, rewriter);
      return success();
    }
    if (!support->canLowerSwitch(operation, cases.size())) {
      return rewriter.notifyMatchFailure(
          operation,
          "switch requires unsupported or excessive branching depth");
    }

    scf::IfOp replacement;
    for (size_t index = 0; index < cases.size(); ++index) {
      auto constant = arith::ConstantIndexOp::create(
          rewriter, operation.getLoc(), operation.getCases()[index]);
      auto condition = arith::CmpIOp::create(
          rewriter, operation.getLoc(), arith::CmpIPredicate::eq,
          adaptor.getArg(), constant.getResult());
      auto ifOp =
          scf::IfOp::create(rewriter, operation.getLoc(),
                            operation.getResultTypes(), condition, true);
      if (index == 0) {
        replacement = ifOp;
      } else {
        scf::YieldOp::create(rewriter, operation.getLoc(), ifOp.getResults());
      }
      rewriter.eraseBlock(&ifOp.getThenRegion().front());
      rewriter.eraseBlock(&ifOp.getElseRegion().front());
      rewriter.inlineRegionBefore(cases[index], ifOp.getThenRegion(),
                                  ifOp.getThenRegion().end());
      if (index + 1U == cases.size()) {
        rewriter.inlineRegionBefore(*defaultRegion, ifOp.getElseRegion(),
                                    ifOp.getElseRegion().end());
        break;
      }

      Block& elseBlock = ifOp.getElseRegion().emplaceBlock();
      rewriter.setInsertionPointToEnd(&elseBlock);
    }

    rewriter.replaceOp(operation, replacement.getResults());
    return success();
  }

private:
  const PayloadControlSupport* support;
};

struct UnrollLoopsForPayload final
    : impl::UnrollLoopsForPayloadBase<UnrollLoopsForPayload> {
  using UnrollLoopsForPayloadBase::UnrollLoopsForPayloadBase;

  void getDependentDialects(DialectRegistry& registry) const override {
    UnrollLoopsForPayloadBase::getDependentDialects(registry);
    mqt::createPromoteClassicalMemory()->getDependentDialects(registry);
  }

  static bool carriesQTensor(scf::ForOp loop) {
    return llvm::any_of(loop.getInitArgs(), [](Value value) {
      auto type = dyn_cast<RankedTensorType>(value.getType());
      return type && isa<QubitType>(type.getElementType());
    });
  }

  /// Specialize a nested sweep before expanding its enclosing repetition.
  static std::pair<scf::ForOp, llvm::APInt>
  selectTensorLoop(scf::ForOp loop, const llvm::APInt& tripCount) {
    if (tripCount.ugt(1)) {
      // Do not cross branches: an outer induction value may make them dead.
      for (auto nested : loop.getBody()->getOps<scf::ForOp>()) {
        if (const auto nestedCount = getExactConstantTripCount(nested);
            nestedCount && carriesQTensor(nested)) {
          return selectTensorLoop(nested, *nestedCount);
        }
      }
    }
    return {loop, tripCount};
  }

protected:
  void runOnOperation() override {
    const auto support = PayloadControlSupport::read(
        getOperation(), getAnalysis<TargetEnvironmentAnalysis>(), payloadSpec);
    if (!support) {
      signalPassFailure();
      return;
    }

    if (failed(foldStaticBranches(getOperation())) ||
        failed(verifyLoopCarriedState(getOperation()))) {
      signalPassFailure();
      return;
    }

    uint64_t clonedOperations = 0U;
    OpPassManager cleanup(ModuleOp::getOperationName());
    cleanup.addPass(mqt::createPromoteClassicalMemory());
    IRRewriter rewriter(&getContext());
    while (true) {
      if (failed(runPipeline(cleanup, getOperation())) ||
          failed(foldStaticBranches(getOperation()))) {
        signalPassFailure();
        return;
      }
      SmallVector<std::pair<scf::ForOp, llvm::APInt>> loops;
      SmallVector<scf::ForOp> repetitions;
      getOperation().walk<WalkOrder::PreOrder>([&](scf::ForOp loop) {
        const auto tripCount = getExactConstantTripCount(loop);
        const bool tensor = carriesQTensor(loop);
        if ((!tensor || support->indexedQubits) &&
            support->coversIteration(ControlFeature::CountedIteration, loop,
                                     tripCount)) {
          return WalkResult::advance();
        }
        if (!tripCount) {
          return WalkResult::skip();
        }
        auto selected = tensor && !support->indexedQubits
                            ? selectTensorLoop(loop, *tripCount)
                            : std::pair{loop, *tripCount};
        if (selected.first != loop) {
          for (auto parent = selected.first->getParentOfType<scf::ForOp>();;
               parent = parent->getParentOfType<scf::ForOp>()) {
            repetitions.push_back(parent);
            if (parent == loop) {
              break;
            }
          }
        }
        loops.push_back(selected);
        return WalkResult::skip();
      });
      if (loops.empty()) {
        return;
      }

      for (size_t index = 0; index < loops.size(); ++index) {
        auto [loop, tripCount] = loops[index];
        if (tripCount.ule(1)) {
          SmallVector<scf::ForOp> nestedLoops;
          if (tripCount == 1) {
            loop.getRegion().walk<WalkOrder::PreOrder>([&](scf::ForOp nested) {
              nestedLoops.push_back(nested);
              return WalkResult::skip();
            });
          }
          if (failed(loop.promoteIfSingleIteration(rewriter))) {
            loop.emitError(
                "failed to simplify a zero- or single-iteration loop");
            signalPassFailure();
            return;
          }
          // Promotion preserves child operations and can make their bounds
          // literal. Defer larger loops until static branches have folded.
          for (auto nested : nestedLoops) {
            const auto nestedTripCount = getExactConstantTripCount(nested);
            if (nestedTripCount && *nestedTripCount == 1 &&
                !support->coversIteration(ControlFeature::CountedIteration,
                                          nested, nestedTripCount)) {
              loops.emplace_back(nested, *nestedTripCount);
            }
          }
          continue;
        }
        if (llvm::hasSingleElement(loop.getBody()->getOperations())) {
          ValueRange yielded = loop.getYieldedValues();
          if (llvm::all_of(yielded, [&](Value value) {
                return loop.isDefinedOutsideOfLoop(value);
              })) {
            rewriter.replaceOp(loop, yielded);
            continue;
          }
        }

        if (failed(unrollStaticLoop(loop, tripCount, maxOperations,
                                    clonedOperations))) {
          signalPassFailure();
          return;
        }
      }

      if (!repetitions.empty()) {
        RewritePatternSet patterns(&getContext());
        getContext()
            .getLoadedDialect<qtensor::QTensorDialect>()
            ->getCanonicalizationPatterns(patterns);
        qtensor::ExtractOp::getCanonicalizationPatterns(patterns,
                                                        &getContext());
        qtensor::InsertOp::getCanonicalizationPatterns(patterns, &getContext());
        SmallVector<Operation*> operations;
        for (auto repetition : repetitions) {
          operations.push_back(repetition);
          for (Operation& operation : repetition.getBody()->getOperations()) {
            if (operation.getNumRegions() == 0) {
              operations.push_back(&operation);
            }
          }
        }
        // Canonicalize the enclosing loop chain without visiting unrelated
        // regions. General folding below may erase these saved roots.
        if (failed(applyOpPatternsGreedily(
                operations, std::move(patterns),
                GreedyRewriteConfig{}.setStrictness(
                    GreedyRewriteStrictness::ExistingAndNewOps)))) {
          signalPassFailure();
          return;
        }
      }
    }
  }
};

struct LegalizeControlFlow final
    : impl::LegalizeControlFlowBase<LegalizeControlFlow> {
  using LegalizeControlFlowBase::LegalizeControlFlowBase;

protected:
  void runOnOperation() override {
    const auto support = PayloadControlSupport::read(
        getOperation(), getAnalysis<TargetEnvironmentAnalysis>(), payloadSpec);
    if (!support || failed(verifyLoopCarriedState(getOperation()))) {
      signalPassFailure();
      return;
    }

    ConversionTarget target(getContext());
    target.addDynamicallyLegalOp<IfOp>(
        [&](IfOp operation) { return isLegal(operation, *support); });
    target.addDynamicallyLegalOp<scf::IfOp>(
        [&](scf::IfOp operation) { return isLegal(operation, *support); });
    target.addDynamicallyLegalOp<scf::ForOp>(
        [&](scf::ForOp operation) { return isLegal(operation, *support); });
    target.addDynamicallyLegalOp<scf::WhileOp>(
        [&](scf::WhileOp operation) { return isLegal(operation, *support); });
    target.addDynamicallyLegalOp<IndexSwitchOp>(
        [&](IndexSwitchOp operation) { return isLegal(operation, *support); });
    target.addDynamicallyLegalOp<scf::IndexSwitchOp>(
        [&](scf::IndexSwitchOp operation) {
          return isLegal(operation, *support);
        });
    target.markUnknownOpDynamicallyLegal([](Operation* operation) {
      if (isa<scf::ExecuteRegionOp>(operation)) {
        return true;
      }
      return !isa<BranchOpInterface, RegionBranchOpInterface>(operation);
    });

    RewritePatternSet patterns(&getContext());
    patterns.add<LowerQCOIndexSwitch, LowerSCFIndexSwitch>(&getContext(),
                                                           *support);
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace
} // namespace mlir::qco
