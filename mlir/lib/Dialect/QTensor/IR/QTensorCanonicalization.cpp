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

/// Recognize discarded qubits through same-block tensor insertions.
///
/// ponytail: shared tails cost O(swaps * inserts); cache if this dominates.
static bool isDiscardedAfterInsertions(Value value, Block* block) {
  while (true) {
    auto* user = *value.user_begin();
    if (user->getBlock() != block) {
      return false;
    }
    if (isa<qco::SinkOp, DeallocOp>(user)) {
      return true;
    }
    auto insert = dyn_cast<InsertOp>(user);
    if (!insert) {
      return false;
    }
    value = insert.getResult();
  }
}

namespace {

struct ElideTerminalSwap final : OpRewritePattern<qco::SWAPOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(qco::SWAPOp op,
                                PatternRewriter& rewriter) const override {
    SmallVector<qco::MeasureOp, 2> measurements;
    for (auto qubit : op.getResults()) {
      auto measure = dyn_cast<qco::MeasureOp>(*qubit.user_begin());
      if (!measure || measure->getBlock() != op->getBlock() ||
          !isDiscardedAfterInsertions(measure.getQubitOut(), op->getBlock())) {
        return failure();
      }
      measurements.push_back(measure);
    }

    /// Measure at the SWAP so both bits dominate their uses. Keep quantum
    /// outputs in their original tensor slots and exchange only the bits.
    auto first = qco::MeasureOp::create(rewriter, measurements[0].getLoc(),
                                        op.getQubit0In());
    auto second = qco::MeasureOp::create(rewriter, measurements[1].getLoc(),
                                         op.getQubit1In());
    rewriter.replaceOp(measurements[0],
                       {first.getQubitOut(), second.getResult()});
    rewriter.replaceOp(measurements[1],
                       {second.getQubitOut(), first.getResult()});
    rewriter.eraseOp(op);
    return success();
  }
};

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
    /// SCF folding may temporarily leave a loop-carried argument unused.
    if (!currentQTensor.hasOneUse()) {
      return std::nullopt;
    }
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

struct TensorSlots {
  Value original;
  Value remaining;
  SmallVector<Value> inputs;
  SmallVector<int64_t> indices;
  SmallVector<BranchQTensorAccesses, 2> branches;
};

using SlotLayout = SmallVector<TensorSlots*>;

} // namespace

static bool isStaticQTensor(Value value) {
  auto type = dyn_cast<RankedTensorType>(value.getType());
  return type && type.hasStaticShape() &&
         isa<qco::QubitType>(type.getElementType());
}

static void prepareSlots(TensorSlots& slots, Location loc,
                         PatternRewriter& rewriter) {
  for (auto& branch : slots.branches) {
    llvm::append_range(slots.indices, branch.accesses.keys());
  }
  llvm::sort(slots.indices);
  slots.indices.erase(llvm::unique(slots.indices), slots.indices.end());
  auto [remaining, inputs] =
      extractQTensorScalars(slots.original, slots.indices, loc, rewriter);
  slots.remaining = remaining;
  slots.inputs = std::move(inputs);
}

static SmallVector<Value> expandInputs(ValueRange values,
                                       ArrayRef<TensorSlots*> layout) {
  SmallVector<Value> result;
  for (auto [value, slots] : llvm::zip_equal(values, layout)) {
    if (slots != nullptr) {
      llvm::append_range(result, slots->inputs);
    } else {
      result.push_back(value);
    }
  }
  return result;
}

static SmallVector<Type> expandTypes(TypeRange types,
                                     ArrayRef<TensorSlots*> layout) {
  SmallVector<Type> result;
  for (auto [type, slots] : llvm::zip_equal(types, layout)) {
    if (slots != nullptr) {
      llvm::append_range(result, ValueRange(slots->inputs).getTypes());
    } else {
      result.push_back(type);
    }
  }
  return result;
}

/// Rebuild a branch once, replacing every eligible tensor in its signature.
static void moveScalarizedBranch(Block* oldBlock, Block* newBlock,
                                 ArrayRef<TensorSlots*> arguments,
                                 ArrayRef<TensorSlots*> yields, size_t branch,
                                 PatternRewriter& rewriter) {
  auto* oldYield = oldBlock->getTerminator();
  SmallVector<Value> replacements;
  DenseMap<TensorSlots*, ValueRange> scalarArguments;
  size_t nextArgument = 0;
  for (auto* slots : arguments) {
    if (slots == nullptr) {
      replacements.push_back(newBlock->getArgument(nextArgument++));
      continue;
    }
    auto scalars =
        newBlock->getArguments().slice(nextArgument, slots->indices.size());
    scalarArguments.try_emplace(slots, scalars);
    nextArgument += slots->indices.size();
    replacements.push_back(slots->original);
    for (auto [index, scalar] : llvm::zip_equal(slots->indices, scalars)) {
      auto found = slots->branches[branch].accesses.find(index);
      if (found != slots->branches[branch].accesses.end()) {
        rewriter.replaceAllUsesWith(found->second.extract.getResult(), scalar);
      }
    }
  }
  rewriter.mergeBlocks(oldBlock, newBlock, replacements);
  SmallVector<Value> values;
  for (auto [value, slots] : llvm::zip_equal(oldYield->getOperands(), yields)) {
    if (slots == nullptr) {
      values.push_back(value);
      continue;
    }
    for (auto [index, scalar] :
         llvm::zip_equal(slots->indices, scalarArguments.at(slots))) {
      auto found = slots->branches[branch].accesses.find(index);
      values.push_back(found == slots->branches[branch].accesses.end()
                           ? scalar
                           : found->second.insert.getScalar());
    }
  }
  rewriter.modifyOpInPlace(oldYield, [&] { oldYield->setOperands(values); });
  for (auto* slots : arguments) {
    if (slots != nullptr) {
      for (Operation* operation :
           llvm::reverse(slots->branches[branch].qTensorOperations)) {
        rewriter.eraseOp(operation);
      }
    }
  }
}

static SmallVector<Value> restoreResults(ValueRange results,
                                         ArrayRef<TensorSlots*> layout,
                                         Location loc,
                                         PatternRewriter& rewriter) {
  SmallVector<Value> replacements;
  size_t nextResult = 0;
  for (auto* slots : layout) {
    if (slots == nullptr) {
      replacements.push_back(results[nextResult++]);
      continue;
    }
    replacements.push_back(insertQTensorScalars(
        slots->remaining, results.slice(nextResult, slots->indices.size()),
        slots->indices, loc, rewriter));
    nextResult += slots->indices.size();
  }
  return replacements;
}

namespace {

/// Expose constant tensor slots to mapping without expanding control flow.
struct ScalarizeQTensorInputs final : OpRewritePattern<qco::IfOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(qco::IfOp op,
                                PatternRewriter& rewriter) const override {
    const size_t classical = op.getClassicalResults().size();
    SmallVector<TensorSlots, 1> slots;
    slots.reserve(op.getQubits().size());
    SlotLayout arguments(op.getQubits().size(), nullptr);
    SlotLayout results(op.getNumResults(), nullptr);
    for (auto [index, tensor] : llvm::enumerate(op.getQubits())) {
      if (!isStaticQTensor(tensor)) {
        continue;
      }
      auto thenAccesses =
          analyzeQTensorBranch(op.thenBlock(), index, classical + index);
      auto elseAccesses =
          analyzeQTensorBranch(op.elseBlock(), index, classical + index);
      if (!thenAccesses || !elseAccesses) {
        continue;
      }
      slots.push_back({
          .original = tensor,
          .branches = {std::move(*thenAccesses), std::move(*elseAccesses)},
      });
      arguments[index] = &slots.back();
      results[classical + index] = &slots.back();
    }
    if (slots.empty()) {
      return failure();
    }
    for (auto& tensor : slots) {
      prepareSlots(tensor, op.getLoc(), rewriter);
    }
    auto inputs = expandInputs(op.getQubits(), arguments);
    auto newIf = qco::IfOp::create(
        rewriter, op.getLoc(), op.getClassicalResults().getTypes(),
        ValueRange(inputs).getTypes(), op.getCondition(), inputs);
    newIf->setDiscardableAttrs(op->getDiscardableAttrDictionary());
    for (size_t branch = 0; branch < 2; ++branch) {
      auto* block = rewriter.createBlock(
          &newIf->getRegion(branch), {}, ValueRange(inputs).getTypes(),
          SmallVector<Location>(inputs.size(), op.getLoc()));
      moveScalarizedBranch(&op->getRegion(branch).front(), block, arguments,
                           results, branch, rewriter);
    }
    rewriter.setInsertionPointAfter(newIf);
    rewriter.replaceOp(
        op, restoreResults(newIf.getResults(), results, op.getLoc(), rewriter));
    return success();
  }
};

/// Before/after signatures may differ and reorder tensor results.
struct ScalarizeWhileQTensorInputs final : OpRewritePattern<scf::WhileOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::WhileOp op,
                                PatternRewriter& rewriter) const override {
    SmallVector<TensorSlots, 1> slots;
    slots.reserve(op.getInits().size());
    SlotLayout inputs(op.getInits().size(), nullptr);
    SlotLayout outputs(op.getNumResults(), nullptr);
    for (auto [index, tensor] : llvm::enumerate(op.getInits())) {
      if (!isStaticQTensor(tensor)) {
        continue;
      }
      auto before = analyzeQTensorBranch(op.getBeforeBody(), index);
      if (!before) {
        continue;
      }
      const auto resultIndex = before->yieldedOperand - 1;
      auto after = analyzeQTensorBranch(op.getAfterBody(), resultIndex, index);
      if (!after) {
        continue;
      }
      slots.push_back({
          .original = tensor,
          .branches = {std::move(*before), std::move(*after)},
      });
      inputs[index] = &slots.back();
      outputs[resultIndex] = &slots.back();
    }
    if (slots.empty()) {
      return failure();
    }
    for (auto& tensor : slots) {
      prepareSlots(tensor, op.getLoc(), rewriter);
    }
    auto newInputs = expandInputs(op.getInits(), inputs);
    auto newTypes = expandTypes(op.getResultTypes(), outputs);
    auto loop =
        scf::WhileOp::create(rewriter, op.getLoc(), newTypes, newInputs);
    loop->setDiscardableAttrs(op->getDiscardableAttrDictionary());
    auto* before = rewriter.createBlock(
        &loop.getBefore(), {}, ValueRange(newInputs).getTypes(),
        SmallVector<Location>(newInputs.size(), op.getLoc()));
    auto* after = rewriter.createBlock(
        &loop.getAfter(), {}, newTypes,
        SmallVector<Location>(newTypes.size(), op.getLoc()));
    SlotLayout conditionYields{nullptr};
    llvm::append_range(conditionYields, outputs);
    moveScalarizedBranch(op.getBeforeBody(), before, inputs, conditionYields, 0,
                         rewriter);
    moveScalarizedBranch(op.getAfterBody(), after, outputs, inputs, 1,
                         rewriter);
    rewriter.setInsertionPointAfter(loop);
    rewriter.replaceOp(
        op, restoreResults(loop.getResults(), outputs, op.getLoc(), rewriter));
    return success();
  }
};

struct ScalarizeForQTensorInputs final : OpRewritePattern<scf::ForOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp op,
                                PatternRewriter& rewriter) const override {
    SmallVector<TensorSlots, 1> slots;
    slots.reserve(op.getInitArgs().size());
    SlotLayout inputs(op.getInitArgs().size(), nullptr);
    for (auto [index, tensor] : llvm::enumerate(op.getInitArgs())) {
      if (!isStaticQTensor(tensor)) {
        continue;
      }
      auto accesses = analyzeQTensorBranch(op.getBody(), index + 1, index);
      if (!accesses) {
        continue;
      }
      slots.push_back({.original = tensor, .branches = {std::move(*accesses)}});
      inputs[index] = &slots.back();
    }
    if (slots.empty()) {
      return failure();
    }
    for (auto& tensor : slots) {
      prepareSlots(tensor, op.getLoc(), rewriter);
    }
    auto newInputs = expandInputs(op.getInitArgs(), inputs);
    auto loop = scf::ForOp::create(rewriter, op.getLoc(), op.getLowerBound(),
                                   op.getUpperBound(), op.getStep(), newInputs);
    loop->setDiscardableAttrs(op->getDiscardableAttrDictionary());
    if (!loop.getBody()->empty()) {
      rewriter.eraseOp(loop.getBody()->getTerminator());
    }
    SlotLayout arguments{nullptr};
    llvm::append_range(arguments, inputs);
    moveScalarizedBranch(op.getBody(), loop.getBody(), arguments, inputs, 0,
                         rewriter);
    rewriter.setInsertionPointAfter(loop);
    rewriter.replaceOp(
        op, restoreResults(loop.getResults(), inputs, op.getLoc(), rewriter));
    return success();
  }
};
} // namespace

void QTensorDialect::getCanonicalizationPatterns(
    RewritePatternSet& results) const {
  results.add<ScalarizeQTensorInputs, ScalarizeWhileQTensorInputs,
              ScalarizeForQTensorInputs, ElideTerminalSwap>(getContext());
}
