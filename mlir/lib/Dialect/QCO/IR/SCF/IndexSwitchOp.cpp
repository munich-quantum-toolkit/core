/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "RegionBranchCompat.h"
#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QCO/QCOUtils.h"

#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/ErrorHandling.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Matchers.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Interfaces/ControlFlowInterfaces.h>
#include <mlir/Support/LLVM.h>

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <utility>

using namespace mlir;
using namespace mlir::qco;

void IndexSwitchOp::build(OpBuilder& odsBuilder, OperationState& odsState,
                          Value arg, Value target, ArrayRef<int64_t> cases,
                          ArrayRef<function_ref<Value(Value)>> caseBuilders,
                          function_ref<Value(Value)> defaultBuilder) {
  if (cases.size() != caseBuilders.size()) {
    llvm::reportFatalUsageError(
        "Each case must have a corresponding case body function");
  }

  const SmallVector<Type> linearResultTypes{target.getType()};
  build(odsBuilder, odsState, linearResultTypes, arg, cases, target,
        cases.size());

  const OpBuilder::InsertionGuard guard(odsBuilder);
  const auto buildRegion = [&](Region& region,
                               function_ref<Value(Value)> bodyBuilder) {
    auto& block = region.emplaceBlock();
    auto blockArgument = block.addArgument(target.getType(), odsState.location);
    odsBuilder.setInsertionPointToStart(&block);
    YieldOp::create(odsBuilder, odsState.location, bodyBuilder(blockArgument));
  };

  for (const auto [index, caseBuilder] : llvm::enumerate(caseBuilders)) {
    buildRegion(*odsState.regions[index + 1], caseBuilder);
  }
  buildRegion(*odsState.regions.front(), defaultBuilder);
}

// Adapted from
// https://github.com/llvm/llvm-project/blob/llvmorg-22.1.1/mlir/lib/Dialect/SCF/IR/SCF.cpp

void IndexSwitchOp::getSuccessorRegions(
    RegionBranchPoint point, SmallVectorImpl<RegionSuccessor>& regions) {
  if (!point.isParent()) {
    regions.push_back(
        detail::makeRegionSuccessor(getOperation(), getResults()));
    return;
  }

  for (Region* region : getRegions()) {
    regions.push_back(
        detail::makeRegionSuccessor(region, region->getArguments()));
  }
}

void IndexSwitchOp::getRegionInvocationBounds(
    ArrayRef<Attribute> operands, SmallVectorImpl<InvocationBounds>& bounds) {
  FoldAdaptor adaptor(operands, *this);

  // If the constant "arg" operand is not provided, we can't reason about the
  // invocation bounds and thus assume that all regions are invoked at most
  // once.

  auto arg = llvm::dyn_cast_or_null<IntegerAttr>(adaptor.getArg());
  if (!arg) {
    bounds.append(getNumRegions(), InvocationBounds(/*lb=*/0, /*ub=*/1));
    return;
  }

  // Otherwise, we can reason that all but the "live" case (can be the default
  // case) are invoked zero times.

  const auto nregions = getNumRegions();
  const auto* it = llvm::find(getCases(), arg.getInt());
  const auto liveIndex = it != getCases().end()
                             ? std::distance(getCases().begin(), it) + 1
                             : 0; // Default region.

  for (size_t i = 0; i < nregions; ++i) {
    bounds.emplace_back(/*lb=*/0, /*ub=*/std::cmp_equal(i, liveIndex) ? 1 : 0);
  }
}

void IndexSwitchOp::getEntrySuccessorRegions(
    ArrayRef<Attribute> operands, SmallVectorImpl<RegionSuccessor>& regions) {
  FoldAdaptor adaptor(operands, *this);

  // If a constant was not provided, all regions are possible successors.
  auto arg = dyn_cast_or_null<IntegerAttr>(adaptor.getArg());
  if (!arg) {
    for (Region* region : getRegions()) {
      regions.push_back(
          detail::makeRegionSuccessor(region, region->getArguments()));
    }
    return;
  }

  // Otherwise, try to find a case with a matching value. If not, the
  // default region is the only successor.

  const auto* it = llvm::find(getCases(), arg.getInt());
  if (it == getCases().end()) {
    regions.push_back(detail::makeRegionSuccessor(
        &getDefaultRegion(), getDefaultRegion().getArguments()));
    return;
  }

  const auto caseIndex = std::distance(getCases().begin(), it);
  auto& caseRegion = getCaseRegions()[caseIndex];
  regions.push_back(
      detail::makeRegionSuccessor(&caseRegion, caseRegion.getArguments()));
}

ValueRange IndexSwitchOp::getSuccessorInputs(RegionSuccessor successor) {
  if (detail::isOperationSuccessor(successor)) {
    return getResults();
  }
  return successor.getSuccessor()->getArguments();
}

OperandRange
IndexSwitchOp::getEntrySuccessorOperands(RegionSuccessor /*successor*/) {
  return getTargets();
}

namespace {
/** Inline the selected region when the switch argument is constant. */
struct RemoveStaticSelector final : OpRewritePattern<IndexSwitchOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(IndexSwitchOp op,
                                PatternRewriter& rewriter) const override {
    IntegerAttr selector;
    if (!matchPattern(op.getArg(), m_Constant(&selector))) {
      return failure();
    }

    Region* selectedRegion = &op.getDefaultRegion();
    const auto* const selectedCase =
        llvm::find(op.getCases(), selector.getInt());
    if (selectedCase != op.getCases().end()) {
      const auto caseIndex = static_cast<size_t>(
          std::distance(op.getCases().begin(), selectedCase));
      selectedRegion = &op.getCaseRegions()[caseIndex];
    }

    Block* selectedBlock = &selectedRegion->front();
    Operation* terminator = selectedBlock->getTerminator();
    rewriter.inlineBlockBefore(selectedBlock, op, op.getTargets());
    rewriter.replaceOp(op, terminator->getOperands());
    rewriter.eraseOp(terminator);
    return success();
  }
};
} // namespace

void IndexSwitchOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                                MLIRContext* context) {
  results.add<RemoveStaticSelector>(context);
}

LogicalResult IndexSwitchOp::verify() {
  if (getCases().size() != getNumCases()) {
    return emitOpError(
        "must have the same number of case values and case regions");
  }

  llvm::SmallDenseSet<int64_t, 4> uniqueCases;
  for (const int64_t caseValue : getCases()) {
    if (!uniqueCases.insert(caseValue).second) {
      return emitOpError("case values must be unique");
    }
  }

  auto targets = getTargets();
  const auto ntargets = targets.size();
  auto results = getLinearResults();
  const auto nresults = results.size();

  for (Region* region : getRegions()) {
    if (region->getNumArguments() != ntargets) {
      return emitOpError(
          "Region " + Twine(region->getRegionNumber()) +
          " must have the same number of arguments as the number of targets");
    }
    if (!llvm::equal(region->getArgumentTypes(), targets.getTypes())) {
      return emitOpError("region argument types must match the target types");
    }
  }

  for (Type type : getClassicalResults().getTypes()) {
    if (isLinearQubitType(type)) {
      return emitOpError("classical results must not use QCO linear types");
    }
  }

  SmallPtrSet<Value, 4> visited;
  for (auto target : targets) {
    if (!visited.insert(target).second) {
      return emitOpError("The operation requires unique values as targets.");
    }
  }

  if (nresults != ntargets) {
    return emitOpError(
        "The operation must consume and produce the same number of values.");
  }

  for (auto [resType, targetType] :
       llvm::zip_equal(results.getTypes(), targets.getTypes())) {
    if (resType != targetType) {
      return emitOpError(
          "The operation must consume and produce the same types.");
    }
  }

  return success();
}

OpResult IndexSwitchOp::getTiedResult(OpOperand* target) {
  if (target->getOwner() != getOperation() || target->getOperandNumber() == 0) {
    return {};
  }
  // Because the first operand is the index, subtract one.
  return getLinearResults()[target->getOperandNumber() - 1];
}

OpOperand* IndexSwitchOp::getTiedTarget(OpResult result) {
  if (result.getDefiningOp() != getOperation()) {
    return nullptr;
  }
  const auto numClassicalResults = getClassicalResults().size();
  if (result.getResultNumber() < numClassicalResults) {
    return nullptr;
  }
  return &getTargetsMutable()[result.getResultNumber() - numClassicalResults];
}

BlockArgument IndexSwitchOp::getTiedCaseBlockArgument(OpOperand* target,
                                                      size_t i) {
  if (target->getOwner() != getOperation() || target->getOperandNumber() == 0 ||
      i >= getNumCases()) {
    return {};
  }

  return getCaseBlock(i)->getArgument(target->getOperandNumber() - 1);
}

OpOperand* IndexSwitchOp::getTiedCaseYieldedValue(BlockArgument bbArg,
                                                  size_t i) {
  if (i >= getNumCases() || bbArg.getOwner() != getCaseBlock(i)) {
    return nullptr;
  }

  return &getCaseYield(i).getTargetsMutable()[getClassicalResults().size() +
                                              bbArg.getArgNumber()];
}

BlockArgument IndexSwitchOp::getTiedDefaultBlockArgument(OpOperand* target) {
  if (target->getOwner() != getOperation() || target->getOperandNumber() == 0) {
    return {};
  }

  return getDefaultBlock()->getArgument(target->getOperandNumber() - 1);
}

OpOperand* IndexSwitchOp::getTiedDefaultYieldedValue(BlockArgument bbArg) {
  if (bbArg.getOwner() != getDefaultBlock()) {
    return nullptr;
  }

  return &getDefaultYield().getTargetsMutable()[getClassicalResults().size() +
                                                bbArg.getArgNumber()];
}

IndexSwitchOp
IndexSwitchOp::replaceWithAdditionalTargets(RewriterBase& rewriter,
                                            ValueRange addons) {
  if (addons.empty()) {
    return *this;
  }

  auto targets = getTargets();
  const auto nregions = getNumRegions();

  SmallVector<Value> newTargets;
  newTargets.reserve(targets.size() + addons.size());
  newTargets.append(targets.begin(), targets.end());
  newTargets.append(addons.begin(), addons.end());
  const auto newTargetTypes = ValueRange(newTargets).getTypes();

  auto newSwitchOp =
      create(rewriter, getLoc(), getClassicalResults().getTypes(),
             newTargetTypes, getArg(), getCases(), newTargets, getNumCases());

  const auto rewriteRegion = [&](Region& oldRegion, Region& newRegion) {
    auto* oldBlock = &oldRegion.front();
    const auto numOldArgs = oldBlock->getNumArguments();
    auto* newBlock = rewriter.createBlock(
        &newRegion, {}, newTargetTypes,
        SmallVector<Location>(newTargets.size(), getLoc()));
    const auto oldArgs = newBlock->getArguments().take_front(numOldArgs);
    const auto addonArgs = newBlock->getArguments().drop_front(numOldArgs);

    rewriter.mergeBlocks(oldBlock, newBlock, oldArgs);

    auto yield = cast<YieldOp>(newBlock->getTerminator());
    SmallVector<Value> yieldedValues;
    yieldedValues.reserve(yield.getTargets().size() + addons.size());
    yieldedValues.append(yield.getTargets().begin(), yield.getTargets().end());
    yieldedValues.append(addonArgs.begin(), addonArgs.end());
    rewriter.replaceOpWithNewOp<YieldOp>(yield, yieldedValues);
  };

  for (size_t i = 0; i < nregions; ++i) {
    rewriteRegion(getRegion(i), newSwitchOp.getRegion(i));
  }

  rewriter.replaceOp(*this,
                     newSwitchOp.getResults().take_front(getNumResults()));

  return newSwitchOp;
}
