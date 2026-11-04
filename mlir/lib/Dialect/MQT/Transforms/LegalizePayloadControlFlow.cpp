/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/Dialect/MQT/Transforms/LegalizePayloadControlFlow.h"

#include "mqt/Compiler/TargetEnvironment.h"
#include "mqt/Dialect/CBit/IR/CBitOps.h"
#include "mqt/Dialect/MQT/IR/MQTAttributes.h"
#include "mqt/Dialect/MQT/IR/MQTDialect.h"
#include "mqt/Dialect/MQT/Transforms/Passes.h"
#include "mqt/Dialect/QC/IR/QCOps.h"
#include "mqt/Dialect/QCO/IR/QCOOps.h"

#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Support/WalkResult.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <utility>

namespace mlir::mqt {

#define GEN_PASS_DEF_LEGALIZECONTROLFLOW
#include "mqt/Dialect/MQT/Transforms/Passes.h.inc"

// Bound generated nesting to keep recursive IR processing manageable.
constexpr uint64_t MAX_SWITCH_EXPANSION_DEPTH = 256U;

std::optional<PayloadControlSupport>
PayloadControlSupport::read(ModuleOp moduleOp,
                            const TargetEnvironmentAnalysis& analysis,
                            StringRef specification) {
  std::optional<PayloadSpecification> explicitPayload;
  if (!specification.empty()) {
    auto attribute = dyn_cast_or_null<mqt::PayloadSpecAttr>(
        parseAttribute(specification, moduleOp.getContext()));
    auto payload = PayloadSpecification::create(attribute);
    if (!payload) {
      moduleOp.emitError() << "invalid payload-spec: "
                           << llvm::toString(payload.takeError());
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

  auto support = forPayload(
      explicitPayload ? *explicitPayload
                      : analysis.environment().payloadSpecification());
  support.indexedQubits_ =
      !explicitPayload && analysis.environment().supportsIndexedQubits();
  return support;
}

PayloadControlSupport
PayloadControlSupport::forPayload(const PayloadSpecification& payload) {
  PayloadControlSupport support;
  for (const ProgramCapability& capability : payload.capabilities()) {
    const auto feature =
        llvm::StringSwitch<std::optional<PayloadControlFeature>>(capability.id)
            .Case(ProgramCapability::FORWARD_BRANCHING,
                  PayloadControlFeature::ForwardBranching)
            .Case(ProgramCapability::COUNTED_ITERATION,
                  PayloadControlFeature::CountedIteration)
            .Case(ProgramCapability::CONDITIONAL_LOOP,
                  PayloadControlFeature::ConditionalLoop)
            .Case(ProgramCapability::MULTIWAY_BRANCHING,
                  PayloadControlFeature::MultiwayBranching)
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

const PayloadControlSupport::CapabilityGroup&
PayloadControlSupport::get(PayloadControlFeature feature) const {
  return groups_[static_cast<size_t>(feature)];
}

static bool isRegisterRead(Value value) {
  return value.getDefiningOp<cbit::ReadOp>() ||
         value.getDefiningOp<cbit::LoadOp>() ||
         value.getDefiningOp<qc::MeasureOp>() ||
         value.getDefiningOp<qco::MeasureOp>();
}

static bool isRegisterEquality(Value condition) {
  if (isRegisterRead(condition) && condition.getType().isInteger(1)) {
    return true;
  }
  if (auto compare = condition.getDefiningOp<arith::CmpIOp>()) {
    if (compare.getPredicate() != arith::CmpIPredicate::eq) {
      return false;
    }
    return (isRegisterRead(compare.getLhs()) &&
            matchPattern(compare.getRhs(), m_Constant())) ||
           (isRegisterRead(compare.getRhs()) &&
            matchPattern(compare.getLhs(), m_Constant()));
  }
  if (auto negate = condition.getDefiningOp<arith::XOrIOp>();
      negate && condition.getType().isInteger(1)) {
    return (isRegisterRead(negate.getLhs()) &&
            matchPattern(negate.getRhs(), m_One())) ||
           (isRegisterRead(negate.getRhs()) &&
            matchPattern(negate.getLhs(), m_One()));
  }
  return false;
}

bool PayloadControlSupport::coversBranch(Operation* operation,
                                         Value condition) const {
  return coversBranch(operation, condition, controlDepth(operation));
}

bool PayloadControlSupport::coversBranch(Operation* operation, Value condition,
                                         uint64_t depth) const {
  if (!coversDepth(PayloadControlFeature::ForwardBranching, depth)) {
    return false;
  }
  const auto& group = get(PayloadControlFeature::ForwardBranching);
  if (group.requiresRegisterEquality && !isRegisterEquality(condition)) {
    return false;
  }
  if (group.maxConditionWidth) {
    SmallVector<Value> pending{condition};
    llvm::DenseSet<Value> visited;
    while (!pending.empty()) {
      const auto value = pending.pop_back_val();
      if (!visited.insert(value).second) {
        continue;
      }
      if (auto integer = dyn_cast<IntegerType>(value.getType());
          integer && integer.getWidth() > *group.maxConditionWidth) {
        return false;
      }
      if (auto* producer = value.getDefiningOp();
          producer != nullptr && !isRegisterRead(value)) {
        llvm::append_range(pending, producer->getOperands());
      }
    }
  }
  if (group.unitaryOnlyBranches &&
      operation
          ->walk([](Operation* nested) {
            return isa<qc::MeasureOp, qc::ResetOp, qco::MeasureOp, qco::ResetOp,
                       cbit::StoreOp, cbit::WriteOp>(nested)
                       ? WalkResult::interrupt()
                       : WalkResult::advance();
          })
          .wasInterrupted()) {
    return false;
  }
  return true;
}

bool PayloadControlSupport::coversDepth(PayloadControlFeature feature,
                                        Operation* operation) const {
  return coversDepth(feature, controlDepth(operation));
}

bool PayloadControlSupport::coversDepth(PayloadControlFeature feature,
                                        uint64_t depth) const {
  const CapabilityGroup& group = get(feature);
  return group.usable &&
         (!group.maxNestingDepth || depth <= *group.maxNestingDepth);
}

bool PayloadControlSupport::coversIteration(
    PayloadControlFeature feature, Operation* operation,
    const std::optional<llvm::APInt>& tripCount) const {
  return coversIteration(feature, controlDepth(operation), tripCount);
}

bool PayloadControlSupport::coversIteration(
    PayloadControlFeature feature, uint64_t depth,
    const std::optional<llvm::APInt>& tripCount) const {
  if (!coversDepth(feature, depth)) {
    return false;
  }
  const CapabilityGroup& group = get(feature);
  if (!group.maxIterationCount) {
    return true;
  }
  return tripCount && tripCount->ule(*group.maxIterationCount);
}

bool PayloadControlSupport::coversMultiwayBranching(Operation* operation,
                                                    uint64_t caseCount) const {
  return coversMultiwayBranching(controlDepth(operation), caseCount);
}

bool PayloadControlSupport::coversMultiwayBranching(uint64_t depth,
                                                    uint64_t caseCount) const {
  if (!coversDepth(PayloadControlFeature::MultiwayBranching, depth)) {
    return false;
  }
  const CapabilityGroup& group = get(PayloadControlFeature::MultiwayBranching);
  return !group.maxCaseCount || caseCount <= *group.maxCaseCount;
}

bool PayloadControlSupport::canLowerSwitch(Operation* operation,
                                           uint64_t caseCount) const {
  const auto& group = get(PayloadControlFeature::ForwardBranching);
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

uint64_t PayloadControlSupport::controlDepth(Operation* operation) {
  uint64_t depth = 1U;
  for (Operation* parent = operation->getParentOp(); parent != nullptr;
       parent = parent->getParentOp()) {
    if (isStructuredControl(parent)) {
      ++depth;
    }
  }
  return depth;
}

bool PayloadControlSupport::isStructuredControl(Operation* operation) {
  return isa<qco::IfOp, qco::IndexSwitchOp, scf::IfOp, scf::IndexSwitchOp,
             scf::ForOp, scf::WhileOp>(operation);
}

PayloadControlSupport::CapabilityGroup&
PayloadControlSupport::get(PayloadControlFeature feature) {
  return groups_[static_cast<size_t>(feature)];
}

void PayloadControlSupport::applyConstraint(CapabilityGroup& group,
                                            PayloadControlFeature feature,
                                            StringRef id, uint64_t value) {
  if (!group.usable) {
    return;
  }
  if (value == 0U) {
    group.usable = false;
    return;
  }
  if (feature == PayloadControlFeature::ForwardBranching) {
    if (id == ProgramConstraint::MAX_CONDITION_WIDTH) {
      group.maxConditionWidth = value;
      return;
    }
    if (id == ProgramConstraint::REQUIRES_REGISTER_EQUALITY && value == 1U) {
      group.requiresRegisterEquality = true;
      return;
    }
    if (id == ProgramConstraint::UNITARY_ONLY_BRANCHES && value == 1U) {
      group.unitaryOnlyBranches = true;
      return;
    }
  }
  if (id == ProgramConstraint::MAX_NESTING_DEPTH) {
    group.maxNestingDepth = value;
    return;
  }
  if (id == ProgramConstraint::MAX_ITERATION_COUNT &&
      (feature == PayloadControlFeature::CountedIteration ||
       feature == PayloadControlFeature::ConditionalLoop)) {
    group.maxIterationCount = value;
    return;
  }
  if (id == ProgramConstraint::MAX_CASE_COUNT &&
      feature == PayloadControlFeature::MultiwayBranching) {
    group.maxCaseCount = value;
    return;
  }
  group.usable = false;
}

std::optional<llvm::APInt> getExactConstantTripCount(scf::ForOp loop) {
  if (!getConstantAPIntValue(getAsOpFoldResult(loop.getLowerBound())) ||
      !getConstantAPIntValue(getAsOpFoldResult(loop.getUpperBound())) ||
      !getConstantAPIntValue(getAsOpFoldResult(loop.getStep()))) {
    return std::nullopt;
  }
  return loop.getStaticTripCount();
}

[[nodiscard]] static bool isLegal(scf::IfOp operation,
                                  const PayloadControlSupport& support) {
  return support.coversBranch(operation, operation.getCondition());
}

[[nodiscard]] static bool isLegal(scf::ForOp operation,
                                  const PayloadControlSupport& support) {
  return support.coversIteration(PayloadControlFeature::CountedIteration,
                                 operation,
                                 getExactConstantTripCount(operation));
}

[[nodiscard]] static bool isLegal(scf::WhileOp operation,
                                  const PayloadControlSupport& support) {
  return support.coversIteration(PayloadControlFeature::ConditionalLoop,
                                 operation, operation.getStaticTripCount());
}

[[nodiscard]] static bool isLegal(scf::IndexSwitchOp operation,
                                  const PayloadControlSupport& support) {
  const uint64_t cases = operation.getNumCases();
  return cases > 0U && support.coversMultiwayBranching(operation, cases);
}

namespace {

class LowerSCFIndexSwitch final
    : public OpConversionPattern<scf::IndexSwitchOp> {
public:
  LowerSCFIndexSwitch(MLIRContext* context,
                      const PayloadControlSupport& supportIn)
      : OpConversionPattern(context), support_(&supportIn) {}

  LogicalResult
  matchAndRewrite(scf::IndexSwitchOp operation, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto cases = operation.getCaseRegions();
    Region* const defaultRegion = &operation.getDefaultRegion();

    if (cases.empty()) {
      Block& block = defaultRegion->front();
      Operation* terminator = block.getTerminator();
      rewriter.inlineBlockBefore(&block, operation);
      SmallVector<Value> replacements(terminator->getOperands());
      rewriter.eraseOp(terminator);
      rewriter.replaceOp(operation, replacements);
      return success();
    }
    if (!support_->canLowerSwitch(operation, cases.size())) {
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
  const PayloadControlSupport* support_;
};

} // namespace

void populatePayloadControlFlowPatternsAndLegality(
    ConversionTarget& target, RewritePatternSet& patterns,
    const PayloadControlSupport& support) {
  target.addDynamicallyLegalOp<scf::IfOp>(
      [&](scf::IfOp operation) { return isLegal(operation, support); });
  target.addDynamicallyLegalOp<scf::ForOp>(
      [&](scf::ForOp operation) { return isLegal(operation, support); });
  target.addDynamicallyLegalOp<scf::WhileOp>(
      [&](scf::WhileOp operation) { return isLegal(operation, support); });
  target.addDynamicallyLegalOp<scf::IndexSwitchOp>(
      [&](scf::IndexSwitchOp operation) {
        return isLegal(operation, support);
      });
  target.markUnknownOpDynamicallyLegal([](Operation* operation) {
    if (isa<scf::ExecuteRegionOp>(operation)) {
      return true;
    }
    return !isa<BranchOpInterface, RegionBranchOpInterface>(operation);
  });
  patterns.add<LowerSCFIndexSwitch>(patterns.getContext(), support);
}

LogicalResult legalizePayloadControlFlow(ModuleOp moduleOp,
                                         const PayloadSpecification& payload) {
  auto support = PayloadControlSupport::forPayload(payload);
  ConversionTarget target(*moduleOp.getContext());
  RewritePatternSet patterns(moduleOp.getContext());
  populatePayloadControlFlowPatternsAndLegality(target, patterns, support);
  return applyPartialConversion(moduleOp, target, std::move(patterns));
}

namespace {

struct LegalizeControlFlow final
    : impl::LegalizeControlFlowBase<LegalizeControlFlow> {
  using LegalizeControlFlowBase::LegalizeControlFlowBase;

protected:
  void runOnOperation() override {
    const auto support = PayloadControlSupport::read(
        getOperation(), getAnalysis<TargetEnvironmentAnalysis>(), payloadSpec);
    if (!support) {
      signalPassFailure();
      return;
    }
    ConversionTarget target(getContext());
    RewritePatternSet patterns(&getContext());
    populatePayloadControlFlowPatternsAndLegality(target, patterns, *support);
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::mqt
