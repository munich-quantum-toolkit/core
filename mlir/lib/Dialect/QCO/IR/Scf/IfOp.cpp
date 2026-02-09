/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QCO/IR/QCODialect.h"

#include <cassert>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/STLFunctionalExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Casting.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Matchers.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Interfaces/ControlFlowInterfaces.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

using namespace mlir;
using namespace mlir::qco;

void IfOp::build(
    OpBuilder& odsBuilder, OperationState& odsState, Value condition,
    ValueRange qubits,
    llvm::function_ref<llvm::SmallVector<Value>(ValueRange)> thenBuilder,
    llvm::function_ref<llvm::SmallVector<Value>(ValueRange)> elseBuilder) {
  // Build the empty operation
  build(odsBuilder, odsState, qubits.getTypes(), condition, qubits);

  // Add the blocks to the regions
  auto& thenBlock = odsState.regions.front()->emplaceBlock();
  auto& elseBlock = odsState.regions.back()->emplaceBlock();

  // Add the block arguments and insert the yield operation
  thenBlock.addArguments(
      qubits.getTypes(),
      SmallVector<Location>(qubits.size(), odsState.location));
  odsBuilder.setInsertionPointToStart(&thenBlock);
  qco::YieldOp::create(odsBuilder, odsState.location,
                       thenBuilder(thenBlock.getArguments()));
  elseBlock.addArguments(
      qubits.getTypes(),
      SmallVector<Location>(qubits.size(), odsState.location));
  odsBuilder.setInsertionPointToStart(&elseBlock);
  qco::YieldOp::create(odsBuilder, odsState.location,
                       elseBuilder(elseBlock.getArguments()));
}

Block* IfOp::thenBlock() { return &getThenRegion().back(); }
YieldOp IfOp::thenYield() { return dyn_cast<YieldOp>(&thenBlock()->back()); }
Block* IfOp::elseBlock() { return &getElseRegion().back(); }
YieldOp IfOp::elseYield() { return dyn_cast<YieldOp>(&elseBlock()->back()); }

// Copied from
// https://github.com/llvm/llvm-project/blob/llvmorg-21.1.7/mlir/lib/Dialect/SCF/IR/SCF.cpp

void IfOp::getSuccessorRegions(RegionBranchPoint point,
                               SmallVectorImpl<RegionSuccessor>& regions) {
  // The `then` and the `else` region branch back to the parent operation.
  if (!point.isParent()) {
    regions.push_back(RegionSuccessor(getResults()));
    return;
  }

  regions.push_back(RegionSuccessor(&getThenRegion()));

  // Don't consider the else region if it is empty.
  Region* elseRegion = &this->getElseRegion();
  if (elseRegion->empty()) {
    regions.push_back(RegionSuccessor());
  } else {
    regions.push_back(RegionSuccessor(elseRegion));
  }
}

void IfOp::getEntrySuccessorRegions(ArrayRef<Attribute> operands,
                                    SmallVectorImpl<RegionSuccessor>& regions) {
  FoldAdaptor adaptor(operands, *this);
  auto boolAttr = dyn_cast_or_null<BoolAttr>(adaptor.getCondition());
  if (!boolAttr || boolAttr.getValue()) {
    regions.emplace_back(&getThenRegion());
  }

  // If the else region is empty, execution continues after the parent op.
  if (!boolAttr || !boolAttr.getValue()) {
    if (!getElseRegion().empty()) {
      regions.emplace_back(&getElseRegion());
    } else {
      regions.emplace_back(getResults());
    }
  }
}

void IfOp::getRegionInvocationBounds(
    ArrayRef<Attribute> operands,
    SmallVectorImpl<InvocationBounds>& invocationBounds) {
  if (auto cond = llvm::dyn_cast_or_null<BoolAttr>(operands[0])) {
    // If the condition is known, then one region is known to be executed once
    // and the other zero times.
    invocationBounds.emplace_back(0, cond.getValue() ? 1 : 0);
    invocationBounds.emplace_back(0, cond.getValue() ? 0 : 1);
  } else {
    // Non-constant condition. Each region may be executed 0 or 1 times.
    invocationBounds.assign(2, {0, 1});
  }
}

/// Replaces the given op with the contents of the given single-block region,
/// using the operands of the block terminator to replace operation results.
static void replaceOpWithRegion(PatternRewriter& rewriter, Operation* op,
                                Region& region, ValueRange blockArgs = {}) {
  assert(llvm::hasSingleElement(region) && "expected single-region block");
  Block* block = &region.front();
  Operation* terminator = block->getTerminator();
  const auto results = terminator->getOperands();
  rewriter.inlineBlockBefore(block, op, blockArgs);
  rewriter.replaceOp(op, results);
  rewriter.eraseOp(terminator);
}

struct RemoveStaticCondition : public OpRewritePattern<IfOp> {
  using OpRewritePattern<IfOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(IfOp op,
                                PatternRewriter& rewriter) const override {
    BoolAttr condition;
    if (!matchPattern(op.getCondition(), m_Constant(&condition))) {
      return failure();
    }

    if (condition.getValue()) {
      replaceOpWithRegion(rewriter, op, op.getThenRegion(), op.getQubits());
    } else if (!op.getElseRegion().empty()) {
      replaceOpWithRegion(rewriter, op, op.getElseRegion(), op.getQubits());
    } else {
      rewriter.eraseOp(op);
    }

    return success();
  }
};

void IfOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                       MLIRContext* context) {
  results.add<RemoveStaticCondition>(context);
}

LogicalResult IfOp::verify() {
  const auto& inputQubits = getQubits();
  const auto numInputQubits = inputQubits.size();
  const auto& outputQubits = getResults();
  const auto numOutputQubits = outputQubits.size();

  for (const auto& type : inputQubits.getTypes()) {
    if (llvm::isa<QubitType>(type)) {
      continue;
    }
    auto tensor = dyn_cast<TensorType>(type);
    if (tensor && llvm::isa<QubitType>(tensor.getElementType())) {
      continue;
    }
    return emitOpError("Inputs must be qubit type or tensor of qubit type!");
  }
  const auto numThenArgs = thenBlock()->getNumArguments();
  const auto numElseArgs = elseBlock()->getNumArguments();

  if (numThenArgs != numElseArgs) {
    return emitOpError(
        "Both regions must have the same number of qubits as arguments.");
  }
  if (numInputQubits != numOutputQubits) {
    return emitOpError("Operation must return the same number of qubits as the "
                       "number of input qubits.");
  }
  for (const auto& [inputQubitType, outputQubitType] :
       llvm::zip_equal(inputQubits.getTypes(), outputQubits.getTypes())) {
    if (inputQubitType != outputQubitType) {
      return emitOpError("Operation must return the same qubit types as its "
                         "input qubit types.");
    }
  }
  SmallPtrSet<Value, 4> uniqueQubitsIn;
  for (const auto& qubit : inputQubits) {
    if (!uniqueQubitsIn.insert(qubit).second) {
      return emitOpError("Input qubits must be unique.");
    }
  }

  return success();
}
