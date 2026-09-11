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
#include "mqt/Dialect/QTensor/IR/QTensorDialect.h"
#include "mqt/Dialect/QTensor/IR/QTensorOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>

using namespace mlir;
using namespace mlir::qtensor;

namespace {

struct QTensorAccess {
  ExtractOp extract;
  InsertOp insert;
};

struct BranchQTensorAccesses {
  DenseMap<int64_t, QTensorAccess> accesses;
  SmallVector<Operation*> qTensorOperations;
  size_t yieldedOperand = 0;
};

} // namespace

/// Analyze a QTensor's complete lifetime in one branch.
///
/// Supported branches extract distinct constant-index qubits, perform
/// QTensor-independent computation, reinsert one qubit at every extracted
/// index, and yield the resulting QTensor. Dynamic indices, repeated accesses,
/// and partial updates do not match.
static std::optional<BranchQTensorAccesses>
analyzeQTensorBranch(Block* block, size_t qTensorArgumentIndex,
                     std::optional<size_t> qTensorYieldIndex = std::nullopt) {
  BranchQTensorAccesses result;
  Value currentQTensor = block->getArgument(qTensorArgumentIndex);
  bool reachedInsertPhase = false;

  while (true) {
    assert(currentQTensor.hasOneUse() && "expected linear semantics");
    Operation* user = *currentQTensor.getUsers().begin();
    if (user->getBlock() != block) {
      return std::nullopt;
    }

    if (auto extract = dyn_cast<ExtractOp>(user)) {
      auto index = getConstantIntValue(extract.getIndex());
      if (reachedInsertPhase || !index ||
          !result.accesses
               .try_emplace(*index, QTensorAccess{.extract = extract})
               .second) {
        return std::nullopt;
      }
      result.qTensorOperations.push_back(user);
      currentQTensor = extract.getOutTensor();
      continue;
    }

    if (auto insert = dyn_cast<InsertOp>(user)) {
      reachedInsertPhase = true;
      auto index = getConstantIntValue(insert.getIndex());
      if (!index) {
        return std::nullopt;
      }
      auto access = result.accesses.find(*index);
      if (access == result.accesses.end() || access->second.insert) {
        return std::nullopt;
      }
      access->second.insert = insert;
      result.qTensorOperations.push_back(user);
      currentQTensor = insert.getResult();
      continue;
    }

    if (!isa<qco::YieldOp, scf::YieldOp, scf::ConditionOp>(user) ||
        user != block->getTerminator() ||
        (qTensorYieldIndex && currentQTensor.use_begin()->getOperandNumber() !=
                                  *qTensorYieldIndex) ||
        llvm::any_of(result.accesses, [](const auto& access) {
          return !access.second.insert;
        })) {
      return std::nullopt;
    }
    result.yieldedOperand = currentQTensor.use_begin()->getOperandNumber();
    return result;
  }
}

/// Move a branch while replacing QTensor accesses with scalar qubits.
static void moveScalarizedQTensorBranch(Value originalQTensor, Block* oldBlock,
                                        Block* newBlock,
                                        size_t qTensorArgumentIndex,
                                        BranchQTensorAccesses& accesses,
                                        ArrayRef<int64_t> indices,
                                        PatternRewriter& rewriter) {
  Operation* oldYield = oldBlock->getTerminator();
  auto scalarArguments = newBlock->getArguments().take_back(indices.size());
  auto carriedArguments = newBlock->getArguments().drop_back(indices.size());

  SmallVector<Value> argumentReplacements;
  argumentReplacements.reserve(oldBlock->getNumArguments());
  size_t carriedIndex = 0;
  for (size_t oldIndex : llvm::seq(oldBlock->getNumArguments())) {
    argumentReplacements.push_back(oldIndex == qTensorArgumentIndex
                                       ? originalQTensor
                                       : carriedArguments[carriedIndex++]);
  }
  assert(carriedIndex == carriedArguments.size());
  rewriter.mergeBlocks(oldBlock, newBlock, argumentReplacements);

  SmallVector<Value> scalarYields;
  scalarYields.reserve(indices.size());
  for (auto [indexPosition, index] : llvm::enumerate(indices)) {
    auto access = accesses.accesses.find(index);
    if (access == accesses.accesses.end()) {
      scalarYields.push_back(scalarArguments[indexPosition]);
    } else {
      rewriter.replaceAllUsesWith(access->second.extract.getResult(),
                                  scalarArguments[indexPosition]);
      scalarYields.push_back(access->second.insert.getScalar());
    }
  }

  auto oldTargets = oldYield->getOperands();
  SmallVector<Value> newYieldValues;
  newYieldValues.reserve(oldTargets.size() - 1 + scalarYields.size());
  for (auto [oldIndex, value] : llvm::enumerate(oldTargets)) {
    if (oldIndex != accesses.yieldedOperand) {
      newYieldValues.push_back(value);
    }
  }
  llvm::append_range(newYieldValues, scalarYields);

  rewriter.modifyOpInPlace(oldYield,
                           [&] { oldYield->setOperands(newYieldValues); });

  for (Operation* operation : llvm::reverse(accesses.qTensorOperations)) {
    rewriter.eraseOp(operation);
  }
}

/// Extract the selected tensor slots before structured control flow.
static std::pair<Value, SmallVector<Value>>
extractQTensorScalars(Value tensor, ArrayRef<int64_t> indices, Location loc,
                      PatternRewriter& rewriter) {
  SmallVector<Value> scalars;
  scalars.reserve(indices.size());
  for (int64_t index : indices) {
    auto indexValue = arith::ConstantIndexOp::create(rewriter, loc, index);
    auto extract = ExtractOp::create(rewriter, loc, tensor, indexValue);
    scalars.push_back(extract.getResult());
    tensor = extract.getOutTensor();
  }
  return {tensor, std::move(scalars)};
}

/// Restore the selected slots after structured control flow.
static Value insertQTensorScalars(Value tensor, ValueRange scalars,
                                  ArrayRef<int64_t> indices, Location loc,
                                  PatternRewriter& rewriter) {
  for (auto [scalar, index] : llvm::zip_equal(scalars, indices)) {
    auto indexValue = arith::ConstantIndexOp::create(rewriter, loc, index);
    tensor =
        InsertOp::create(rewriter, loc, scalar, tensor, indexValue).getResult();
  }
  return tensor;
}

namespace {

/// Replace constant-index QTensor updates in an if with scalar threading.
///
/// A QTensor carried through an if hides its qubits from target mapping. This
/// pattern extracts the union of constant indices accessed by either branch,
/// threads those qubits through both branches, and reinserts the results.
/// Untouched elements remain in the QTensor outside the if.
struct ScalarizeQTensorInputs final : OpRewritePattern<qco::IfOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(qco::IfOp op,
                                PatternRewriter& rewriter) const override {
    size_t classicalResultCount = op.getClassicalResults().size();
    auto oldQubits = op.getQubits();

    for (auto [qTensorIndex, qTensor] : llvm::enumerate(oldQubits)) {
      auto qTensorType = dyn_cast<RankedTensorType>(qTensor.getType());
      if (!qTensorType || !qTensorType.hasStaticShape()) {
        continue;
      }

      auto thenAccesses = analyzeQTensorBranch(
          op.thenBlock(), qTensorIndex, classicalResultCount + qTensorIndex);
      auto elseAccesses = analyzeQTensorBranch(
          op.elseBlock(), qTensorIndex, classicalResultCount + qTensorIndex);
      if (!thenAccesses || !elseAccesses) {
        continue;
      }

      SmallVector<int64_t> accessedIndices(thenAccesses->accesses.keys());
      llvm::append_range(accessedIndices, elseAccesses->accesses.keys());
      llvm::sort(accessedIndices);
      accessedIndices.erase(llvm::unique(accessedIndices),
                            accessedIndices.end());
      ArrayRef<int64_t> indices(accessedIndices);

      rewriter.setInsertionPoint(op);
      auto [qTensorWithoutScalars, scalarInputs] =
          extractQTensorScalars(qTensor, indices, op.getLoc(), rewriter);

      SmallVector<Value> newQubits(oldQubits);
      newQubits.erase(newQubits.begin() + qTensorIndex);
      llvm::append_range(newQubits, scalarInputs);

      auto newIf = qco::IfOp::create(
          rewriter, op.getLoc(), op.getClassicalResults().getTypes(),
          ValueRange(newQubits).getTypes(), op.getCondition(), newQubits);
      newIf->setDiscardableAttrs(op->getDiscardableAttrDictionary());

      SmallVector<Location> locations(newQubits.size(), op.getLoc());
      Block* oldThenBlock = op.thenBlock();
      Block* oldElseBlock = op.elseBlock();
      Block* newThenBlock =
          rewriter.createBlock(&newIf.getThenRegion(), {},
                               ValueRange(newQubits).getTypes(), locations);
      Block* newElseBlock =
          rewriter.createBlock(&newIf.getElseRegion(), {},
                               ValueRange(newQubits).getTypes(), locations);
      moveScalarizedQTensorBranch(qTensor, oldThenBlock, newThenBlock,
                                  qTensorIndex, *thenAccesses, indices,
                                  rewriter);
      moveScalarizedQTensorBranch(qTensor, oldElseBlock, newElseBlock,
                                  qTensorIndex, *elseAccesses, indices,
                                  rewriter);

      rewriter.setInsertionPointAfter(newIf);
      Value updatedQTensor = insertQTensorScalars(
          qTensorWithoutScalars,
          newIf.getLinearResults().take_back(indices.size()), indices,
          op.getLoc(), rewriter);

      SmallVector<Value> replacements(
          newIf.getLinearResults().drop_back(indices.size()));
      replacements.insert(replacements.begin() + qTensorIndex, updatedQTensor);
      replacements.insert(replacements.begin(),
                          newIf.getClassicalResults().begin(),
                          newIf.getClassicalResults().end());
      rewriter.replaceOp(op, replacements);
      return success();
    }
    return failure();
  }
};
/// Keep constant-index QTensor accesses outside a while loop and carry scalars.
/// The before region may reorder results, but the after region must return
/// each tensor to its original iteration argument.
struct ScalarizeWhileQTensorInputs final : OpRewritePattern<scf::WhileOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::WhileOp op,
                                PatternRewriter& rewriter) const override {
    for (auto [inputIndex, qTensor] : llvm::enumerate(op.getInits())) {
      auto type = dyn_cast<RankedTensorType>(qTensor.getType());
      if (!type || !type.hasStaticShape() ||
          !isa<qco::QubitType>(type.getElementType())) {
        continue;
      }
      auto beforeAccesses =
          analyzeQTensorBranch(op.getBeforeBody(), inputIndex);
      if (!beforeAccesses) {
        continue;
      }
      /// The condition is operand zero of scf.condition.
      const auto resultIndex = beforeAccesses->yieldedOperand - 1;
      auto afterAccesses =
          analyzeQTensorBranch(op.getAfterBody(), resultIndex, inputIndex);
      if (!afterAccesses) {
        continue;
      }
      SmallVector<int64_t> indices(beforeAccesses->accesses.keys());
      llvm::append_range(indices, afterAccesses->accesses.keys());
      llvm::sort(indices);
      indices.erase(llvm::unique(indices), indices.end());

      rewriter.setInsertionPoint(op);
      auto [remainingTensor, scalarInputs] =
          extractQTensorScalars(qTensor, indices, op.getLoc(), rewriter);

      SmallVector<Value> newInputs(op.getInits());
      newInputs.erase(newInputs.begin() + inputIndex);
      llvm::append_range(newInputs, scalarInputs);
      SmallVector<Type> newTypes(op.getResultTypes());
      newTypes.erase(newTypes.begin() + resultIndex);
      llvm::append_range(newTypes, ValueRange(scalarInputs).getTypes());
      auto newWhile =
          scf::WhileOp::create(rewriter, op.getLoc(), newTypes, newInputs);
      newWhile->setDiscardableAttrs(op->getDiscardableAttrDictionary());
      Block* newBefore = rewriter.createBlock(
          &newWhile.getBefore(), {}, ValueRange(newInputs).getTypes(),
          SmallVector<Location>(newInputs.size(), op.getLoc()));
      Block* newAfter = rewriter.createBlock(
          &newWhile.getAfter(), {}, newTypes,
          SmallVector<Location>(newTypes.size(), op.getLoc()));
      moveScalarizedQTensorBranch(qTensor, op.getBeforeBody(), newBefore,
                                  inputIndex, *beforeAccesses, indices,
                                  rewriter);
      moveScalarizedQTensorBranch(qTensor, op.getAfterBody(), newAfter,
                                  resultIndex, *afterAccesses, indices,
                                  rewriter);

      rewriter.setInsertionPointAfter(newWhile);
      remainingTensor = insertQTensorScalars(
          remainingTensor, newWhile.getResults().take_back(indices.size()),
          indices, op.getLoc(), rewriter);
      SmallVector<Value> replacements(
          newWhile.getResults().drop_back(indices.size()));
      replacements.insert(replacements.begin() + resultIndex, remainingTensor);
      rewriter.replaceOp(op, replacements);
      return success();
    }
    return failure();
  }
};
/// Carry constant slots through a for loop without expanding its iterations.
struct ScalarizeForQTensorInputs final : OpRewritePattern<scf::ForOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp op,
                                PatternRewriter& rewriter) const override {
    for (auto [inputIndex, tensor] : llvm::enumerate(op.getInitArgs())) {
      auto type = dyn_cast<RankedTensorType>(tensor.getType());
      if (!type || !type.hasStaticShape() ||
          !isa<qco::QubitType>(type.getElementType())) {
        continue;
      }
      auto accesses =
          analyzeQTensorBranch(op.getBody(), inputIndex + 1, inputIndex);
      if (!accesses) {
        continue;
      }
      SmallVector<int64_t> indices(accesses->accesses.keys());
      llvm::sort(indices);
      auto [remainingTensor, scalars] =
          extractQTensorScalars(tensor, indices, op.getLoc(), rewriter);
      SmallVector<Value> inputs(op.getInitArgs());
      inputs.erase(inputs.begin() + inputIndex);
      llvm::append_range(inputs, scalars);
      auto loop = scf::ForOp::create(rewriter, op.getLoc(), op.getLowerBound(),
                                     op.getUpperBound(), op.getStep(), inputs);
      loop->setDiscardableAttrs(op->getDiscardableAttrDictionary());
      if (!loop.getBody()->empty()) {
        rewriter.eraseOp(loop.getBody()->getTerminator());
      }
      moveScalarizedQTensorBranch(tensor, op.getBody(), loop.getBody(),
                                  inputIndex + 1, *accesses, indices, rewriter);
      rewriter.setInsertionPointAfter(loop);
      remainingTensor = insertQTensorScalars(
          remainingTensor, loop.getResults().take_back(indices.size()), indices,
          op.getLoc(), rewriter);
      SmallVector<Value> replacements(
          loop.getResults().drop_back(indices.size()));
      replacements.insert(replacements.begin() + inputIndex, remainingTensor);
      rewriter.replaceOp(op, replacements);
      return success();
    }
    return failure();
  }
};
} // namespace

void QTensorDialect::getCanonicalizationPatterns(
    RewritePatternSet& results) const {
  results.add<ScalarizeQTensorInputs, ScalarizeWhileQTensorInputs,
              ScalarizeForQTensorInputs>(getContext());
}
