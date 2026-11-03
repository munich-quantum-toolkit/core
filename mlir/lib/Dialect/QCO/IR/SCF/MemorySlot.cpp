/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/Dialect/QCO/IR/QCOOps.h"

#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/MemorySlotInterfaces.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::qco;

template <typename Op>
static Value
replacePromotedBranch(Op op, Op replacement, Value reachingDef,
                      const DenseMap<Block*, Value>& reachingAtBlockEnd,
                      IRRewriter& rewriter) {
  replacement->setDiscardableAttrs(op->getDiscardableAttrDictionary());
  const auto index = op.getClassicalResults().size();
  for (auto [region, newRegion] :
       llvm::zip_equal(op->getRegions(), replacement->getRegions())) {
    auto& block = region.front();
    auto value = reachingAtBlockEnd.lookup(&block);
    // Promotion adds a classical result before the positional quantum results.
    block.getTerminator()->insertOperands(index, value ? value : reachingDef);
    rewriter.inlineRegionBefore(region, newRegion, newRegion.end());
  }
  rewriter.replaceAllUsesWith(op.getClassicalResults(),
                              replacement.getClassicalResults().drop_back());
  rewriter.replaceAllUsesWith(op.getLinearResults(),
                              replacement.getLinearResults());
  rewriter.eraseOp(op);
  return replacement.getClassicalResults().back();
}

bool IfOp::isRegionPromotable(const MemorySlot& /*slot*/, Region* /*region*/,
                              bool /*hasValueStores*/) {
  return true;
}

void IfOp::setupPromotion(
    const MemorySlot& /*slot*/, Value reachingDef, bool /*hasValueStores*/,
    llvm::SmallMapVector<Region*, Value, 2>& regionsToProcess) {
  regionsToProcess.insert({&getThenRegion(), reachingDef});
  regionsToProcess.insert({&getElseRegion(), reachingDef});
}

Value IfOp::finalizePromotion(const MemorySlot& slot, Value reachingDef,
                              bool hasValueStores,
                              const DenseMap<Block*, Value>& reachingAtBlockEnd,
                              OpBuilder& builder) {
  if (!hasValueStores) {
    return reachingDef;
  }
  SmallVector<Type> resultTypes(getClassicalResults().getTypes());
  resultTypes.push_back(slot.elemType);
  IRRewriter rewriter(builder);
  rewriter.setInsertionPoint(*this);
  auto replacement =
      create(rewriter, getLoc(), resultTypes, getLinearResults().getTypes(),
             getCondition(), getQubits());
  return replacePromotedBranch(*this, replacement, reachingDef,
                               reachingAtBlockEnd, rewriter);
}

bool IndexSwitchOp::isRegionPromotable(const MemorySlot& /*slot*/,
                                       Region* /*region*/,
                                       bool /*hasValueStores*/) {
  return true;
}

void IndexSwitchOp::setupPromotion(
    const MemorySlot& /*slot*/, Value reachingDef, bool /*hasValueStores*/,
    llvm::SmallMapVector<Region*, Value, 2>& regionsToProcess) {
  for (Region& region : getOperation()->getRegions()) {
    regionsToProcess.insert({&region, reachingDef});
  }
}

Value IndexSwitchOp::finalizePromotion(
    const MemorySlot& slot, Value reachingDef, bool hasValueStores,
    const DenseMap<Block*, Value>& reachingAtBlockEnd, OpBuilder& builder) {
  if (!hasValueStores) {
    return reachingDef;
  }
  SmallVector<Type> resultTypes(getClassicalResults().getTypes());
  resultTypes.push_back(slot.elemType);
  IRRewriter rewriter(builder);
  rewriter.setInsertionPoint(*this);
  auto replacement =
      create(rewriter, getLoc(), resultTypes, getLinearResults().getTypes(),
             getArg(), getCases(), getTargets(), getNumCases());
  return replacePromotedBranch(*this, replacement, reachingDef,
                               reachingAtBlockEnd, rewriter);
}
