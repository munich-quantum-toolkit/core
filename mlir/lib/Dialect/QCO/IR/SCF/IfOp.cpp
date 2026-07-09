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
#include <llvm/ADT/STLFunctionalExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
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

#include <cassert>

using namespace mlir;
using namespace mlir::qco;

void IfOp::build(OpBuilder& odsBuilder, OperationState& odsState,
                 Value condition, ValueRange qubits,
                 function_ref<SmallVector<Value>(ValueRange)> thenBuilder,
                 function_ref<SmallVector<Value>(ValueRange)> elseBuilder) {
  // Build the empty operation
  build(odsBuilder, odsState, qubits.getTypes(), condition, qubits);

  // Add the blocks to the regions
  auto& thenBlock = odsState.regions.front()->emplaceBlock();
  auto& elseBlock = odsState.regions.back()->emplaceBlock();

  const OpBuilder::InsertionGuard guard(odsBuilder);
  // Add the block arguments and insert the yield operation
  thenBlock.addArguments(qubits.getTypes(),
                         SmallVector(qubits.size(), odsState.location));
  odsBuilder.setInsertionPointToStart(&thenBlock);
  YieldOp::create(odsBuilder, odsState.location,
                  thenBuilder(thenBlock.getArguments()));
  elseBlock.addArguments(qubits.getTypes(),
                         SmallVector(qubits.size(), odsState.location));
  odsBuilder.setInsertionPointToStart(&elseBlock);
  if (elseBuilder) {
    YieldOp::create(odsBuilder, odsState.location,
                    elseBuilder(elseBlock.getArguments()));
  } else {
    YieldOp::create(odsBuilder, odsState.location, elseBlock.getArguments());
  }
}

// Adjusted from
// https://github.com/llvm/llvm-project/blob/llvmorg-22.1.1/mlir/lib/Dialect/SCF/IR/SCF.cpp

void IfOp::getSuccessorRegions(RegionBranchPoint point,
                               SmallVectorImpl<RegionSuccessor>& regions) {
  // The `then` and the `else` region branch back to the parent operation or
  // one of the recursive parent operations (early exit case).
  if (!point.isParent()) {
    regions.push_back(RegionSuccessor(getOperation(), getResults()));
    return;
  }

  regions.push_back(RegionSuccessor(&getThenRegion()));

  // If the else region is empty, execution continues after the parent op.
  Region* elseRegion = &getElseRegion();
  if (elseRegion->empty()) {
    regions.push_back(
        RegionSuccessor(getOperation(), getOperation()->getResults()));
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
      regions.emplace_back(getOperation(), getResults());
    }
  }
}
// NOLINTNEXTLINE(readability-convert-member-functions-to-static)
void IfOp::getRegionInvocationBounds(
    ArrayRef<Attribute> operands,
    SmallVectorImpl<InvocationBounds>& invocationBounds) {
  if (auto cond = dyn_cast_or_null<BoolAttr>(operands[0])) {
    // If the condition is known, then one region is known to be executed once
    // and the other zero times.
    invocationBounds.emplace_back(0, cond.getValue() ? 1 : 0);
    invocationBounds.emplace_back(0, cond.getValue() ? 0 : 1);
  } else {
    // Non-constant condition. Each region may be executed 0 or 1 times.
    invocationBounds.assign(2, {0, 1});
  }
}

/**
 * @brief Replace operation with the contents of a region
 *
 * @details
 * Replaces the given op with the contents of the given single-block region,
 * using the operands of the block terminator to replace operation results.
 *
 * @param rewriter The used rewriter
 * @param op The operation that is replcaed
 * @param region The region with the replacement content
 * @param blockArgs The block arguments of the region
 *
 */
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

namespace {

/**
 * @brief Remove static conditions
 *
 * @details
 * Removes a qco.if operation with a static condition and replace it with the
 * contents of the selected branch.
 *
 */
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
    } else {
      replaceOpWithRegion(rewriter, op, op.getElseRegion(), op.getQubits());
    }

    return success();
  }
};

/**
 * @brief Propagate the condition into the branches
 *
 * @details
 * Allow the true region of an if to assume the condition is true
 * and vice versa. For example:
 *
 *   qco.if %cmp args(%arg0 = %q0) -> (!qco.qubit) {
 *      print(true)
 *      ...
 *   } else args(%arg = %q0) {
 *      print(false)
 *      ...
 *   }
 *
 */
struct ConditionPropagation : public OpRewritePattern<IfOp> {
  using OpRewritePattern<IfOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(IfOp op,
                                PatternRewriter& rewriter) const override {
    // Early exit if the condition is constant since replacing a constant
    // in the body with another constant isn't a simplification.
    if (matchPattern(op.getCondition(), m_Constant())) {
      return failure();
    }

    bool changed = false;
    Type i1Ty = rewriter.getI1Type();

    // These variables serve to prevent creating duplicate constants
    // and hold constant true or false values.
    Value constantTrue = nullptr;
    Value constantFalse = nullptr;

    for (auto& use : llvm::make_early_inc_range(op.getCondition().getUses())) {
      if (op.getThenRegion().isAncestor(use.getOwner()->getParentRegion())) {
        changed = true;

        if (!constantTrue) {
          constantTrue = arith::ConstantOp::create(
              rewriter, op.getLoc(), i1Ty, rewriter.getIntegerAttr(i1Ty, 1));
        }

        rewriter.modifyOpInPlace(use.getOwner(),
                                 [&]() { use.set(constantTrue); });
      } else if (op.getElseRegion().isAncestor(
                     use.getOwner()->getParentRegion())) {
        changed = true;

        if (!constantFalse) {
          constantFalse = arith::ConstantOp::create(
              rewriter, op.getLoc(), i1Ty, rewriter.getIntegerAttr(i1Ty, 0));
        }

        rewriter.modifyOpInPlace(use.getOwner(),
                                 [&]() { use.set(constantFalse); });
      }
    }

    return success(changed);
  }
};
} // namespace

void IfOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                       MLIRContext* context) {
  results.add<RemoveStaticCondition, ConditionPropagation>(context);
  populateRegionBranchOpInterfaceCanonicalizationPatterns(
      results, IfOp::getOperationName());
}

LogicalResult IfOp::verify() {
  const auto& inputQubits = getQubits();
  const auto numInputQubits = inputQubits.size();
  const auto& outputQubits = getResults();
  const auto numOutputQubits = outputQubits.size();

  const auto numThenArgs = thenBlock()->getNumArguments();
  const auto numElseArgs = elseBlock()->getNumArguments();

  if (numThenArgs != numElseArgs) {
    return emitOpError(
        "Both regions must have the same number of qubits as arguments.");
  }
  if (numThenArgs != numInputQubits) {
    return emitOpError("Both regions must have the same number of qubits as "
                       "arguments as the number of input qubits");
  }
  if (numInputQubits != numOutputQubits) {
    return emitOpError("Operation must return the same number of qubits as the "
                       "number of input qubits.");
  }
  for (auto [inputQubitType, outputQubitType] :
       llvm::zip_equal(inputQubits.getTypes(), outputQubits.getTypes())) {
    if (inputQubitType != outputQubitType) {
      return emitOpError("Operation must return the same qubit types as its "
                         "input qubit types.");
    }
  }
  SmallPtrSet<Value, 4> uniqueQubitsIn;
  for (auto qubit : inputQubits) {
    if (!uniqueQubitsIn.insert(qubit).second) {
      return emitOpError("Input qubits must be unique.");
    }
  }

  return success();
}

OpResult IfOp::getTiedResult(OpOperand* qubit) {
  if (qubit->getOwner() != getOperation()) {
    return {};
  }
  // Because the first operand is the if-condition, subtract one.
  return getResults()[qubit->getOperandNumber() - 1];
}

OpOperand* IfOp::getTiedQubit(OpResult result) {
  if (result.getDefiningOp() != getOperation()) {
    return nullptr;
  }
  return &getQubitsMutable()[result.getResultNumber()];
}

BlockArgument IfOp::getTiedThenBlockArgument(OpOperand* qubit) {
  if (qubit->getOwner() != getOperation()) {
    return {};
  }
  // Because the first operand is the if-condition, subtract one.
  return thenBlock()->getArguments()[qubit->getOperandNumber() - 1];
}

BlockArgument IfOp::getTiedElseBlockArgument(OpOperand* qubit) {
  if (qubit->getOwner() != getOperation()) {
    return {};
  }
  // Because the first operand is the if-condition, subtract one.
  return elseBlock()->getArguments()[qubit->getOperandNumber() - 1];
}

OpOperand* IfOp::getTiedThenYieldedValue(BlockArgument bbArg) {
  if (bbArg.getDefiningOp() != getOperation()) {
    return nullptr;
  }
  return &thenYield().getTargetsMutable()[bbArg.getArgNumber()];
}

OpOperand* IfOp::getTiedElseYieldedValue(BlockArgument bbArg) {
  if (bbArg.getDefiningOp() != getOperation()) {
    return nullptr;
  }
  return &elseYield().getTargetsMutable()[bbArg.getArgNumber()];
}

IfOp IfOp::replaceWithAdditionalQubits(RewriterBase& rewriter,
                                       ValueRange addons) {
  SmallVector<Value> inits;
  inits.reserve(getQubits().size() + addons.size());
  inits.append(getQubits().begin(), getQubits().end());
  inits.append(addons.begin(), addons.end());

  SmallVector<Type> types;
  types.reserve(getQubits().size() + addons.size());
  types.append(getQubits().getTypes().begin(), getQubits().getTypes().end());
  types.append(addons.getTypes().begin(), addons.getTypes().end());

  SmallVector<Location> locs(getQubits().size() + addons.size(), getLoc());

  auto newIfOp = rewriter.create<IfOp>(getLoc(), getCondition(), inits);

  const auto processRegion = [&](Region& oldRegion, Region& newRegion) {
    Block* oldBlock = &oldRegion.front();
    Block* newBlock = rewriter.createBlock(&newRegion, {}, types, locs);

    // Merge the old block into the new block,
    // keeping only the original arguments.
    rewriter.mergeBlocks(
        oldBlock, newBlock,
        newBlock->getArguments().take_front(oldBlock->getNumArguments()));

    // Update the yield operation to include additional qubits.
    auto yield = cast<YieldOp>(newBlock->getTerminator());

    const auto args = newBlock->getArguments().take_back(addons.size());

    SmallVector<Value> newResults;
    newResults.reserve(inits.size());
    newResults.append(yield.getTargets().begin(), yield.getTargets().end());
    newResults.append(args.begin(), args.end());

    rewriter.replaceOpWithNewOp<YieldOp>(yield, newResults);
  };

  // Process both regions
  processRegion(getThenRegion(), newIfOp.getThenRegion());
  processRegion(getElseRegion(), newIfOp.getElseRegion());

  return newIfOp;
}
