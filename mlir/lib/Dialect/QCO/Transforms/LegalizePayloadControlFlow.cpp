/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Compiler/TargetEnvironment.h"
#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QCO/QCOUtils.h"
#include "mlir/Dialect/QCO/Transforms/Passes.h"

#include <llvm/ADT/APInt.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SetVector.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/ADT/StringSwitch.h>
#include <llvm/Support/MathExtras.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/SCF/Utils/Utils.h>
#include <mlir/Dialect/Utils/StaticValueUtils.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Visitors.h>
#include <mlir/Interfaces/ControlFlowInterfaces.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Support/WalkResult.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <mlir/Transforms/RegionUtils.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <optional>
#include <utility>

namespace mlir::qco {

#define GEN_PASS_DEF_LEGALIZEPAYLOADCONTROLFLOW
#define GEN_PASS_DEF_UNROLLUNSUPPORTEDPAYLOADLOOPS
#include "mlir/Dialect/QCO/Transforms/Passes.h.inc"

namespace {

constexpr llvm::StringLiteral FORWARD_BRANCHING = "forward-branching";
constexpr llvm::StringLiteral COUNTED_ITERATION = "counted-iteration";
constexpr llvm::StringLiteral CONDITIONAL_LOOP = "conditional-loop";
constexpr llvm::StringLiteral MULTIWAY_BRANCHING = "multiway-branching";

constexpr llvm::StringLiteral MAX_NESTING_DEPTH =
    "max-control-flow-nesting-depth";
constexpr llvm::StringLiteral MAX_ITERATION_COUNT = "max-iteration-count";
constexpr llvm::StringLiteral MAX_CASE_COUNT = "max-case-count";

constexpr uint64_t MAX_UNROLLED_OPERATIONS = 65536U;

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
  read(ModuleOp moduleOp, const TargetEnvironmentAnalysis& analysis) {
    if (!analysis) {
      moduleOp.emitError()
          << "payload control-flow legalization requires a valid "
             "mqt.target_env: "
          << analysis.error();
      return std::nullopt;
    }

    PayloadControlSupport support;
    const auto& payload = analysis.environment().payloadSpecification();
    for (const ProgramCapability& capability : payload.capabilities()) {
      const auto feature =
          llvm::StringSwitch<std::optional<ControlFeature>>(capability.id)
              .Case(FORWARD_BRANCHING, ControlFeature::ForwardBranching)
              .Case(COUNTED_ITERATION, ControlFeature::CountedIteration)
              .Case(CONDITIONAL_LOOP, ControlFeature::ConditionalLoop)
              .Case(MULTIWAY_BRANCHING, ControlFeature::MultiwayBranching)
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

  [[nodiscard]] static uint64_t controlDepth(Operation* operation) {
    uint64_t depth = 1U;
    for (Operation* parent = operation->getParentOp(); parent != nullptr;
         parent = parent->getParentOp()) {
      if (isa<IfOp, IndexSwitchOp, scf::IfOp, scf::IndexSwitchOp, scf::ForOp,
              scf::WhileOp>(parent)) {
        ++depth;
      }
    }
    return depth;
  }

private:
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
    if (id == MAX_NESTING_DEPTH) {
      group.maxNestingDepth = value;
      return;
    }
    if (id == MAX_ITERATION_COUNT &&
        (feature == ControlFeature::CountedIteration ||
         feature == ControlFeature::ConditionalLoop)) {
      group.maxIterationCount = value;
      return;
    }
    if (id == MAX_CASE_COUNT && feature == ControlFeature::MultiwayBranching) {
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
  llvm::SetVector<Value> captures;
  getUsedValuesDefinedAbove(operation->getRegions(), captures);
  return llvm::any_of(
      captures, [](Value value) { return isLinearQubitType(value.getType()); });
}

[[nodiscard]] static bool hasLinearBranchState(Operation* operation) {
  return llvm::any_of(operation->getResultTypes(), isLinearQubitType) ||
         hasLinearCapture(operation);
}

/// Widen before subtracting so a large range cannot appear to have zero trips.
[[nodiscard]] static std::optional<llvm::APInt>
getExactConstantTripCount(scf::ForOp loop) {
  const auto constant = [](Value value) -> std::optional<llvm::APInt> {
    const auto result = getConstantAPIntValue(getAsOpFoldResult(value));
    return result ? std::optional(result->first) : std::nullopt;
  };

  const auto lowerBound = constant(loop.getLowerBound());
  const auto upperBound = constant(loop.getUpperBound());
  const auto step = constant(loop.getStep());
  if (!lowerBound || !upperBound || !step) {
    return std::nullopt;
  }

  const unsigned width = std::max({
                             lowerBound->getBitWidth(),
                             upperBound->getBitWidth(),
                             step->getBitWidth(),
                         }) +
                         1U;
  const bool isUnsigned = loop.getUnsignedCmp();
  const auto extend = [&](const llvm::APInt& value) {
    return isUnsigned ? value.zextOrTrunc(width) : value.sextOrTrunc(width);
  };
  const llvm::APInt lower = extend(*lowerBound);
  const llvm::APInt upper = extend(*upperBound);
  const llvm::APInt stride = extend(*step);
  const llvm::APInt one(width, 1U);

  if ((isUnsigned && (stride.isZero() || upper.ule(lower))) ||
      (!isUnsigned && (!stride.isStrictlyPositive() || upper.sle(lower)))) {
    return llvm::APInt(width, 0U);
  }
  const llvm::APInt difference = upper - lower;
  return ((difference - one).udiv(stride)) + one;
}

[[nodiscard]] static bool haveEqualTripCounts(const llvm::APInt& lhs,
                                              const llvm::APInt& rhs) {
  const unsigned width = std::max(lhs.getBitWidth(), rhs.getBitWidth());
  return lhs.zextOrTrunc(width) == rhs.zextOrTrunc(width);
}

static LogicalResult foldStaticBranches(ModuleOp moduleOp) {
  /// Do not load generic SCF loop patterns before the exact trip-count check.
  RewritePatternSet patterns(moduleOp.getContext());
  IfOp::getCanonicalizationPatterns(patterns, moduleOp.getContext());
  IndexSwitchOp::getCanonicalizationPatterns(patterns, moduleOp.getContext());
  scf::IfOp::getCanonicalizationPatterns(patterns, moduleOp.getContext());
  scf::IndexSwitchOp::getCanonicalizationPatterns(patterns,
                                                  moduleOp.getContext());
  return applyPatternsGreedily(
      moduleOp, std::move(patterns),
      GreedyRewriteConfig{}.setMaxIterations(GreedyRewriteConfig::kNoLimit));
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
  return !hasLinearCapture(operation) &&
         support.coversIteration(ControlFeature::CountedIteration, operation,
                                 getExactConstantTripCount(operation));
}

[[nodiscard]] static bool isLegal(scf::WhileOp operation,
                                  const PayloadControlSupport& support) {
  return !hasLinearCapture(operation) &&
         support.coversIteration(ControlFeature::ConditionalLoop, operation,
                                 operation.getStaticTripCount());
}

[[nodiscard]] static bool isLegal(IndexSwitchOp operation,
                                  const PayloadControlSupport& support) {
  const uint64_t cases = operation.getNumCases();
  return cases > 1U && support.coversMultiwayBranching(operation, cases);
}

[[nodiscard]] static bool isLegal(scf::IndexSwitchOp operation,
                                  const PayloadControlSupport& support) {
  const uint64_t cases = operation.getNumCases();
  return !hasLinearBranchState(operation) && cases > 1U &&
         support.coversMultiwayBranching(operation, cases);
}

[[nodiscard]] static bool canUseFullUnroll(scf::ForOp loop,
                                           const uint64_t iterations) {
  const auto constant = [&](Value value) -> std::optional<int64_t> {
    const auto result = getConstantAPIntValue(getAsOpFoldResult(value));
    if (!result || (loop.getUnsignedCmp() && result->first.isNegative())) {
      return std::nullopt;
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
  return llvm::MulOverflow(*step, static_cast<int64_t>(iterations),
                           scaledStep) == 0 &&
         llvm::AddOverflow(*lowerBound, scaledStep, unrolledUpperBound) == 0;
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
    if (!support->get(ControlFeature::ForwardBranching).usable) {
      return rewriter.notifyMatchFailure(
          operation, "selected payload cannot use forward branches");
    }

    const auto build = [&](auto&& self, const size_t index,
                           ValueRange targets) -> IfOp {
      auto constant = arith::ConstantIndexOp::create(
          rewriter, operation.getLoc(), operation.getCases()[index]);
      auto condition = arith::CmpIOp::create(
          rewriter, operation.getLoc(), arith::CmpIPredicate::eq,
          adaptor.getArg(), constant.getResult());
      auto ifOp = IfOp::create(rewriter, operation.getLoc(),
                               operation.getClassicalResults().getTypes(),
                               operation.getLinearResults().getTypes(),
                               condition, targets);
      rewriter.inlineRegionBefore(cases[index], ifOp.getThenRegion(),
                                  ifOp.getThenRegion().end());
      if (index + 1U == cases.size()) {
        rewriter.inlineRegionBefore(*defaultRegion, ifOp.getElseRegion(),
                                    ifOp.getElseRegion().end());
        return ifOp;
      }

      Block& elseBlock = ifOp.getElseRegion().emplaceBlock();
      elseBlock.addArguments(targets.getTypes(),
                             SmallVector(targets.size(), operation.getLoc()));
      const OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToEnd(&elseBlock);
      IfOp nested = self(self, index + 1U, elseBlock.getArguments());
      YieldOp::create(rewriter, operation.getLoc(), nested.getResults());
      return ifOp;
    };

    IfOp replacement = build(build, 0U, adaptor.getTargets());
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
    if (!support->get(ControlFeature::ForwardBranching).usable) {
      return rewriter.notifyMatchFailure(
          operation, "selected payload cannot use forward branches");
    }

    const auto build = [&](auto&& self, const size_t index) -> scf::IfOp {
      auto constant = arith::ConstantIndexOp::create(
          rewriter, operation.getLoc(), operation.getCases()[index]);
      auto condition = arith::CmpIOp::create(
          rewriter, operation.getLoc(), arith::CmpIPredicate::eq,
          adaptor.getArg(), constant.getResult());
      auto ifOp =
          scf::IfOp::create(rewriter, operation.getLoc(),
                            operation.getResultTypes(), condition, true);
      rewriter.eraseBlock(&ifOp.getThenRegion().front());
      rewriter.eraseBlock(&ifOp.getElseRegion().front());
      rewriter.inlineRegionBefore(cases[index], ifOp.getThenRegion(),
                                  ifOp.getThenRegion().end());
      if (index + 1U == cases.size()) {
        rewriter.inlineRegionBefore(*defaultRegion, ifOp.getElseRegion(),
                                    ifOp.getElseRegion().end());
        return ifOp;
      }

      Block& elseBlock = ifOp.getElseRegion().emplaceBlock();
      const OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToEnd(&elseBlock);
      scf::IfOp nested = self(self, index + 1U);
      scf::YieldOp::create(rewriter, operation.getLoc(), nested.getResults());
      return ifOp;
    };

    scf::IfOp replacement = build(build, 0U);
    rewriter.replaceOp(operation, replacement.getResults());
    return success();
  }

private:
  const PayloadControlSupport* support;
};

struct UnrollUnsupportedPayloadLoops final
    : impl::UnrollUnsupportedPayloadLoopsBase<UnrollUnsupportedPayloadLoops> {
  using UnrollUnsupportedPayloadLoopsBase::UnrollUnsupportedPayloadLoopsBase;

protected:
  void runOnOperation() override {
    const auto support = PayloadControlSupport::read(
        getOperation(), getAnalysis<TargetEnvironmentAnalysis>());
    if (!support) {
      signalPassFailure();
      return;
    }

    if (failed(foldStaticBranches(getOperation()))) {
      signalPassFailure();
      return;
    }

    uint64_t clonedOperations = 0U;
    IRRewriter rewriter(&getContext());
    while (true) {
      SmallVector<std::pair<scf::ForOp, llvm::APInt>> loops;
      const WalkResult result =
          getOperation().walk<WalkOrder::PreOrder>([&](scf::ForOp loop) {
            if (hasLinearCapture(loop)) {
              loop.emitError(
                  "SCF loop captures QCO linear values; pass them as "
                  "iteration arguments");
              return WalkResult::interrupt();
            }
            const auto tripCount = getExactConstantTripCount(loop);
            if (tripCount) {
              const auto mlirTripCount = loop.getStaticTripCount();
              if (!mlirTripCount ||
                  !haveEqualTripCounts(*tripCount, *mlirTripCount)) {
                loop.emitError(
                    "MLIR cannot safely normalize this static loop range");
                return WalkResult::interrupt();
              }
            }
            if (support->coversIteration(ControlFeature::CountedIteration, loop,
                                         tripCount)) {
              return WalkResult::advance();
            }
            if (!tripCount) {
              return WalkResult::skip();
            }
            loops.emplace_back(loop, *tripCount);
            return WalkResult::skip();
          });
      if (result.wasInterrupted()) {
        signalPassFailure();
        return;
      }
      if (loops.empty()) {
        return;
      }

      for (auto& [loop, tripCount] : loops) {
        if (tripCount.isZero()) {
          rewriter.replaceOp(loop, loop.getInitArgs());
          continue;
        }
        if (tripCount.isOne()) {
          if (failed(loop.promoteIfSingleIteration(rewriter))) {
            loop.emitError("failed to promote a single-iteration loop");
            signalPassFailure();
            return;
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
          loop.emitError("cannot fully unroll a terminator-only loop");
          signalPassFailure();
          return;
        }

        uint64_t bodyOperations = 0U;
        Operation* const terminator = loop.getBody()->getTerminator();
        loop.getRegion().walk([&](Operation* operation) {
          bodyOperations += operation != terminator;
        });
        const uint64_t remaining = MAX_UNROLLED_OPERATIONS - clonedOperations;
        const uint64_t maximumTripCount = (remaining / bodyOperations) + 1U;
        if (!tripCount.ule(maximumTripCount)) {
          loop.emitError() << "full legalization would clone more than "
                           << MAX_UNROLLED_OPERATIONS
                           << " loop-body operations";
          signalPassFailure();
          return;
        }
        const uint64_t iterations = tripCount.getZExtValue();
        clonedOperations += bodyOperations * (iterations - 1U);
        if (!canUseFullUnroll(loop, iterations)) {
          loop.emitError(
              "cannot safely apply MLIR full unrolling to these loop bounds");
          signalPassFailure();
          return;
        }
        if (failed(loopUnrollFull(loop))) {
          loop.emitError("failed to fully unroll a static counted loop");
          signalPassFailure();
          return;
        }
      }

      if (failed(foldStaticBranches(getOperation()))) {
        signalPassFailure();
        return;
      }
    }
  }
};

struct LegalizePayloadControlFlow final
    : impl::LegalizePayloadControlFlowBase<LegalizePayloadControlFlow> {
  using LegalizePayloadControlFlowBase::LegalizePayloadControlFlowBase;

protected:
  void runOnOperation() override {
    const auto support = PayloadControlSupport::read(
        getOperation(), getAnalysis<TargetEnvironmentAnalysis>());
    if (!support) {
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
      if (isa<scf::ExecuteRegionOp, CtrlOp, InvOp, PowOp>(operation)) {
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
