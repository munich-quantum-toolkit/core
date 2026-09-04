/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QC/Translation/LoopBuilder.h"

#include <llvm/ADT/STLExtras.h>
#include <mlir/Conversion/ControlFlowToSCF/ControlFlowToSCF.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/IR/Dominance.h>
#include <mlir/IR/Matchers.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/CFGToSCF.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <mlir/Transforms/RegionUtils.h>

#include <tuple>
#include <utility>

namespace mlir::qc {

LoopBuilder::LoopBuilder(OpBuilder& builder, Location location,
                         ValueRange initialState)
    : builder(builder), location(location) {
  regionOp =
      scf::ExecuteRegionOp::create(builder, location, initialState.getTypes());
  auto& region = regionOp.getRegion();
  auto* entry = builder.createBlock(&region);
  SmallVector<Location> locations(initialState.size(), location);
  header = builder.createBlock(&region, {}, initialState.getTypes(), locations);
  SmallVector<Type> decisionTypes{builder.getI1Type()};
  llvm::append_range(decisionTypes, initialState.getTypes());
  decision = builder.createBlock(
      &region, {}, decisionTypes,
      SmallVector<Location>(decisionTypes.size(), location));
  exit = builder.createBlock(&region, {}, initialState.getTypes(), locations);
  builder.setInsertionPointToEnd(entry);
  cf::BranchOp::create(builder, location, header, initialState);
  builder.setInsertionPointToEnd(header);
}

ValueRange LoopBuilder::arguments() { return header->getArguments(); }

void LoopBuilder::enterBody(Value condition, ValueRange state) {
  auto* current = builder.getInsertionBlock();
  auto* body = builder.createBlock(&regionOp.getRegion());
  builder.setInsertionPointToEnd(current);
  if (matchPattern(condition, m_One())) {
    cf::BranchOp::create(builder, location, body);
  } else {
    SmallVector<Value> exitValues{
        arith::ConstantIntOp::create(builder, location, 0, 1)};
    llvm::append_range(exitValues, state);
    cf::CondBranchOp::create(builder, location, condition, body, ValueRange{},
                             decision, exitValues);
  }
  builder.setInsertionPointToEnd(body);
}

void LoopBuilder::branch(bool continuing, ValueRange state) {
  SmallVector<Value> values{
      arith::ConstantIntOp::create(builder, location, continuing ? 1 : 0, 1)};
  llvm::append_range(values, state);
  cf::BranchOp::create(builder, location, decision, values);
}

FailureOr<SmallVector<Value>> LoopBuilder::finish() {
  builder.setInsertionPointToEnd(decision);
  cf::CondBranchOp::create(builder, location, decision->getArgument(0), header,
                           decision->getArguments().drop_front(), exit,
                           decision->getArguments().drop_front());
  builder.setInsertionPointToEnd(exit);
  scf::YieldOp::create(builder, location, exit->getArguments());
  IRRewriter rewriter(builder.getContext());
  std::ignore = eraseUnreachableBlocks(rewriter, regionOp->getRegions());
  DominanceInfo dominance;
  ControlFlowToSCFTransformation transformation;
  if (failed(
          transformCFGToSCF(regionOp.getRegion(), transformation, dominance))) {
    return regionOp.emitError("cannot structure loop with break");
  }
  RewritePatternSet patterns(builder.getContext());
  scf::WhileOp::getCanonicalizationPatterns(patterns, builder.getContext());
  scf::IfOp::getCanonicalizationPatterns(patterns, builder.getContext());
  arith::SelectOp::getCanonicalizationPatterns(patterns, builder.getContext());
  arith::TruncIOp::getCanonicalizationPatterns(patterns, builder.getContext());
  SmallVector<Operation*> operations;
  regionOp.getRegion().walk([&](Operation* op) { operations.push_back(op); });
  if (failed(applyOpPatternsGreedily(operations, std::move(patterns)))) {
    return regionOp.emitError("cannot canonicalize loop with break");
  }
  std::ignore = runRegionDCE(rewriter, regionOp->getRegions());
  auto& block = regionOp.getRegion().front();
  auto terminator = cast<scf::YieldOp>(block.getTerminator());
  SmallVector<Value> results(terminator.getResults());
  rewriter.inlineBlockBefore(&block, regionOp);
  rewriter.eraseOp(terminator);
  builder.setInsertionPointAfter(regionOp);
  rewriter.eraseOp(regionOp);
  return results;
}

} // namespace mlir::qc
