/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QCO/IR/QCOOps.h"

#include <llvm/ADT/STLExtras.h>
#include <llvm/Support/Casting.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Interfaces/ControlFlowInterfaces.h>
#include <mlir/Support/LLVM.h>

#include <cstddef>
#include <iterator>

using namespace mlir;
using namespace mlir::qco;

// Adapted from
// https://github.com/llvm/llvm-project/blob/llvmorg-22.1.1/mlir/lib/Dialect/SCF/IR/SCF.cpp

void IndexSwitchOp::getSuccessorRegions(
    RegionBranchPoint point, SmallVectorImpl<RegionSuccessor>& regions) {
  if (!point.isParent()) {
    regions.emplace_back(getOperation(), getResults());
    return;
  }

  llvm::append_range(regions, getRegions());
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
                             ? std::distance(getCases().begin(), it)
                             : 0; // Default region.

  for (size_t i = 0; i < nregions; ++i) {
    bounds.emplace_back(/*lb=*/0, /*ub=*/i == liveIndex ? 1 : 0);
  }
}

void IndexSwitchOp::getEntrySuccessorRegions(
    ArrayRef<Attribute> operands, SmallVectorImpl<RegionSuccessor>& regions) {
  FoldAdaptor adaptor(operands, *this);

  // If a constant was not provided, all regions are possible successors.
  auto arg = dyn_cast_or_null<IntegerAttr>(adaptor.getArg());
  if (!arg) {
    llvm::append_range(regions, getRegions());
    return;
  }

  // Otherwise, try to find a case with a matching value. If not, the
  // default region is the only successor.

  const auto* it = llvm::find(getCases(), arg.getInt());
  const auto liveIndex = it != getCases().end()
                             ? std::distance(getCases().begin(), it)
                             : 0; // Default region.

  regions.emplace_back(&getRegion(liveIndex));
}

LogicalResult IndexSwitchOp::verify() {
  const auto targets = getTargets();
  const auto ntargets = targets.size();
  const auto results = getResults();
  const auto nresults = results.size();

  for (Region* region : getRegions()) {
    if (region->getNumArguments() != ntargets) {
      return emitOpError(
          "Region " + Twine(region->getRegionNumber()) +
          " must have the same number of arguments as the number of targets");
    }
  }

  SmallPtrSet<Value, 4> visited;
  for (const auto target : targets) {
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
  if (target->getOwner() != getOperation()) {
    return {};
  }
  // Because the first operand is the index, subtract one.
  return getResults()[target->getOperandNumber() - 1];
}

OpOperand* IndexSwitchOp::getTiedTarget(OpResult result) {
  if (result.getDefiningOp() != getOperation()) {
    return nullptr;
  }
  return &getTargetsMutable()[result.getResultNumber()];
}

BlockArgument IndexSwitchOp::getTiedCaseBlockArgument(OpOperand* target,
                                                      size_t i) {
  if (target->getOwner() != getOperation() || i >= getNumCases()) {
    return {};
  }

  return getCaseBlock(i)->getArgument(target->getOperandNumber() - 1);
}

OpOperand* IndexSwitchOp::getTiedCaseYieldedValue(BlockArgument bbArg,
                                                  size_t i) {
  if (bbArg.getOwner()->getParentOp() != getOperation() || i >= getNumCases()) {
    return nullptr;
  }

  return &getCaseYield(i).getTargetsMutable()[bbArg.getArgNumber()];
}

BlockArgument IndexSwitchOp::getTiedDefaultBlockArgument(OpOperand* target) {
  if (target->getOwner() != getOperation()) {
    return {};
  }

  return getDefaultBlock()->getArgument(target->getOperandNumber() - 1);
}

OpOperand* IndexSwitchOp::getTiedDefaultYieldedValue(BlockArgument bbArg) {
  if (bbArg.getOwner()->getParentOp() != getOperation()) {
    return nullptr;
  }

  return &getDefaultYield().getTargetsMutable()[bbArg.getArgNumber()];
}

IndexSwitchOp
IndexSwitchOp::replaceWithAdditionalTargets(RewriterBase& rewriter,
                                            ValueRange addons) {
  if (addons.empty()) {
    return *this;
  }

  const auto targets = getTargets();
  const auto nregions = getNumRegions();

  SmallVector<Value> newTargets;
  newTargets.reserve(targets.size() + addons.size());
  newTargets.append(targets.begin(), targets.end());
  newTargets.append(addons.begin(), addons.end());
  const auto newTargetTypes = ValueRange(newTargets).getTypes();

  auto newSwitchOp = create(rewriter, getLoc(), newTargetTypes, getArg(),
                            getCases(), newTargets, getNumCases());

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
