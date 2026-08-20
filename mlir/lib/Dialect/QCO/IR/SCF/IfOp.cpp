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
#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QCO/QCOUtils.h"
#include "mlir/Dialect/QTensor/IR/QTensorOps.h"

#include <llvm/ADT/BitVector.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/STLFunctionalExtras.h>
#include <llvm/ADT/Sequence.h>
#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Utils/StaticValueUtils.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/IRMapping.h>
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
#include <cstddef>
#include <cstdint>
#include <optional>

using namespace mlir;
using namespace mlir::qco;

void IfOp::build(OpBuilder& odsBuilder, OperationState& odsState,
                 Value condition, ValueRange qubits,
                 function_ref<SmallVector<Value>(ValueRange)> thenBuilder,
                 function_ref<SmallVector<Value>(ValueRange)> elseBuilder) {
  // Build the empty operation
  build(odsBuilder, odsState, TypeRange{}, qubits.getTypes(), condition,
        qubits);

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

void IfOp::build(OpBuilder& odsBuilder, OperationState& odsState,
                 Value condition, Value input,
                 function_ref<Value(Value)> thenBuilder,
                 function_ref<Value(Value)> elseBuilder) {
  const SmallVector<Type> linearResultTypes{input.getType()};
  build(odsBuilder, odsState, TypeRange{}, linearResultTypes, condition, input);

  auto& thenBlock = odsState.regions.front()->emplaceBlock();
  auto& elseBlock = odsState.regions.back()->emplaceBlock();

  const OpBuilder::InsertionGuard guard(odsBuilder);
  const auto location = odsState.location;
  auto thenArgument = thenBlock.addArgument(input.getType(), location);
  odsBuilder.setInsertionPointToStart(&thenBlock);
  YieldOp::create(odsBuilder, location, thenBuilder(thenArgument));

  auto elseArgument = elseBlock.addArgument(input.getType(), location);
  odsBuilder.setInsertionPointToStart(&elseBlock);
  YieldOp::create(odsBuilder, location,
                  elseBuilder ? elseBuilder(elseArgument) : elseArgument);
}

// Adjusted from
// https://github.com/llvm/llvm-project/blob/llvmorg-22.1.1/mlir/lib/Dialect/SCF/IR/SCF.cpp

void IfOp::getSuccessorRegions(RegionBranchPoint point,
                               SmallVectorImpl<RegionSuccessor>& regions) {
  // The `then` and the `else` region branch back to the parent operation or
  // one of the recursive parent operations (early exit case).
  if (!point.isParent()) {
    regions.push_back(
        detail::makeRegionSuccessor(getOperation(), getResults()));
    return;
  }

  regions.push_back(detail::makeRegionSuccessor(
      &getThenRegion(), getThenRegion().getArguments()));

  // If the else region is empty, execution continues after the parent op.
  Region* elseRegion = &getElseRegion();
  if (elseRegion->empty()) {
    regions.push_back(detail::makeRegionSuccessor(
        getOperation(), getOperation()->getResults()));
  } else {
    regions.push_back(
        detail::makeRegionSuccessor(elseRegion, elseRegion->getArguments()));
  }
}

void IfOp::getEntrySuccessorRegions(ArrayRef<Attribute> operands,
                                    SmallVectorImpl<RegionSuccessor>& regions) {
  FoldAdaptor adaptor(operands, *this);
  auto boolAttr = dyn_cast_or_null<BoolAttr>(adaptor.getCondition());
  if (!boolAttr || boolAttr.getValue()) {
    regions.push_back(detail::makeRegionSuccessor(
        &getThenRegion(), getThenRegion().getArguments()));
  }

  // If the else region is empty, execution continues after the parent op.
  if (!boolAttr || !boolAttr.getValue()) {
    if (!getElseRegion().empty()) {
      regions.push_back(detail::makeRegionSuccessor(
          &getElseRegion(), getElseRegion().getArguments()));
    } else {
      regions.push_back(
          detail::makeRegionSuccessor(getOperation(), getResults()));
    }
  }
}

ValueRange IfOp::getSuccessorInputs(RegionSuccessor successor) {
  if (detail::isOperationSuccessor(successor)) {
    return getResults();
  }
  return successor.getSuccessor()->getArguments();
}

OperandRange IfOp::getEntrySuccessorOperands(RegionSuccessor /*successor*/) {
  return getQubits();
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

/**
 * @brief Forward redundant classical results
 *
 * @details
 * Replaces a classical result with a value yielded by both branches or with an
 * earlier classical result whose pair of yielded values is identical. A
 * separate pattern removes the result and its yield operands once they become
 * unused. Linear results are intentionally excluded because their explicit
 * branch threading is part of QCO's quantum dataflow.
 */
struct ForwardClassicalResults : public OpRewritePattern<IfOp> {
  using OpRewritePattern<IfOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(IfOp op,
                                PatternRewriter& rewriter) const override {
    const auto classicalResults = op.getClassicalResults();
    if (classicalResults.empty()) {
      return failure();
    }

    const auto thenValues =
        op.thenYield().getTargets().take_front(classicalResults.size());
    const auto elseValues =
        op.elseYield().getTargets().take_front(classicalResults.size());

    bool changed = false;
    for (const auto [index, result] : llvm::enumerate(classicalResults)) {
      Value replacement;
      if (thenValues[index] == elseValues[index]) {
        replacement = thenValues[index];
      } else {
        for (const auto candidate : llvm::seq(index)) {
          if (thenValues[candidate] == thenValues[index] &&
              elseValues[candidate] == elseValues[index]) {
            replacement = classicalResults[candidate];
            break;
          }
        }
      }

      if (replacement && !result.use_empty()) {
        rewriter.replaceAllUsesWith(result, replacement);
        changed = true;
      }
    }
    return success(changed);
  }
};

/**
 * @brief Remove unused classical results
 *
 * @details
 * Removes unused classical results and the corresponding operands from both
 * branch terminators. The result segment property is updated on the replacement
 * operation. The linear result suffix and all quantum dataflow remain intact.
 */
struct RemoveUnusedClassicalResults : public OpRewritePattern<IfOp> {
  using OpRewritePattern<IfOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(IfOp op,
                                PatternRewriter& rewriter) const override {
    llvm::BitVector resultsToErase(op.getNumResults());
    for (const OpResult result : op.getClassicalResults()) {
      if (result.use_empty()) {
        resultsToErase.set(result.getResultNumber());
      }
    }
    if (resultsToErase.none()) {
      return failure();
    }

    const auto numLinearResults = op.getLinearResults().size();
    const auto numClassicalResults =
        op.getClassicalResults().size() - resultsToErase.count();

    llvm::BitVector yieldOperandsToErase(op.thenYield().getNumOperands());
    for (const auto result : op.getClassicalResults()) {
      if (resultsToErase.test(result.getResultNumber())) {
        yieldOperandsToErase.set(result.getResultNumber());
      }
    }
    rewriter.modifyOpInPlace(op.thenYield(), [&]() {
      op.thenYield()->eraseOperands(yieldOperandsToErase);
    });
    rewriter.modifyOpInPlace(op.elseYield(), [&]() {
      op.elseYield()->eraseOperands(yieldOperandsToErase);
    });

    auto replacement = cast<IfOp>(rewriter.eraseOpResults(op, resultsToErase));
    rewriter.modifyOpInPlace(replacement, [&]() {
      replacement.getProperties().setResultSegmentSizes(
          ArrayRef<int32_t>({static_cast<int32_t>(numClassicalResults),
                             static_cast<int32_t>(numLinearResults)}));
    });
    return success();
  }
};

} // namespace

namespace mlir::qco {

namespace {

struct TensorAccess {
  int64_t index;
  Value inputQubit;
  Value outputQubit;
};

struct BranchTensorAccesses {
  SmallVector<TensorAccess> accesses;
  llvm::SmallPtrSet<Operation*, 8> tensorOperations;
};

} // namespace

/**
 * @brief Analyze a tensor's complete life chain in one branch.
 *
 * @details The supported shape extracts a set of distinct constant indices,
 * performs arbitrary tensor-independent computation, reinserts exactly one
 * qubit at every extracted index, and yields the resulting tensor. Keeping the
 * analysis deliberately narrow makes the rewrite below safe to use as a
 * canonicalization: dynamic indices, repeated accesses, and partial tensor
 * updates are simply left unchanged.
 */
static std::optional<BranchTensorAccesses>
analyzeTensorBranch(Block* block, const size_t tensorArgumentIndex,
                    const size_t tensorYieldIndex) {
  BranchTensorAccesses result;
  Value currentTensor = block->getArgument(tensorArgumentIndex);
  bool reachedInsertPhase = false;

  while (true) {
    if (!currentTensor.hasOneUse()) {
      return std::nullopt;
    }
    Operation* user = *currentTensor.getUsers().begin();
    if (user->getBlock() != block) {
      return std::nullopt;
    }

    if (auto extract = dyn_cast<qtensor::ExtractOp>(user)) {
      const auto index = getConstantIntValue(extract.getIndex());
      if (reachedInsertPhase || !index ||
          llvm::any_of(result.accesses, [&](const TensorAccess& access) {
            return access.index == *index;
          })) {
        return std::nullopt;
      }
      result.accesses.push_back(
          TensorAccess{.index = *index, .inputQubit = extract.getResult()});
      result.tensorOperations.insert(user);
      currentTensor = extract.getOutTensor();
      continue;
    }

    if (auto insert = dyn_cast<qtensor::InsertOp>(user)) {
      reachedInsertPhase = true;
      const auto index = getConstantIntValue(insert.getIndex());
      if (!index) {
        return std::nullopt;
      }
      auto* access =
          llvm::find_if(result.accesses, [&](const TensorAccess& it) {
            return it.index == *index;
          });
      if (access == result.accesses.end() || access->outputQubit) {
        return std::nullopt;
      }
      access->outputQubit = insert.getScalar();
      result.tensorOperations.insert(user);
      currentTensor = insert.getResult();
      continue;
    }

    auto yield = dyn_cast<YieldOp>(user);
    if (!yield || user != block->getTerminator() ||
        tensorYieldIndex >= yield.getTargets().size() ||
        yield.getTargets()[tensorYieldIndex] != currentTensor ||
        llvm::any_of(result.accesses, [](const TensorAccess& access) {
          return !access.outputQubit;
        })) {
      return std::nullopt;
    }
    return result;
  }
}

/** Return the access for `index`, if the branch touches that tensor element. */
static const TensorAccess*
findTensorAccess(const BranchTensorAccesses& accesses, const int64_t index) {
  const auto* const access =
      llvm::find_if(accesses.accesses, [&](const TensorAccess& candidate) {
        return candidate.index == index;
      });
  return access == accesses.accesses.end() ? nullptr : &*access;
}

/**
 * @brief Clone one branch while replacing tensor accesses with scalar qubits.
 */
static void cloneScalarizedBranch(IfOp oldIf, Block* oldBlock, Block* newBlock,
                                  const size_t tensorArgumentIndex,
                                  const BranchTensorAccesses& accesses,
                                  const ArrayRef<int64_t> indices,
                                  PatternRewriter& rewriter) {
  IRMapping mapping;
  size_t newArgumentIndex = 0;
  for (const auto [oldIndex, oldArgument] :
       llvm::enumerate(oldBlock->getArguments())) {
    if (oldIndex == tensorArgumentIndex) {
      continue;
    }
    mapping.map(oldArgument, newBlock->getArgument(newArgumentIndex++));
  }

  const auto scalarArguments =
      newBlock->getArguments().take_back(indices.size());
  for (const auto [indexPosition, index] : llvm::enumerate(indices)) {
    if (const auto* access = findTensorAccess(accesses, index)) {
      mapping.map(access->inputQubit, scalarArguments[indexPosition]);
    }
  }

  rewriter.setInsertionPointToEnd(newBlock);
  for (Operation& operation : oldBlock->without_terminator()) {
    if (!accesses.tensorOperations.contains(&operation)) {
      rewriter.clone(operation, mapping);
    }
  }

  auto oldYield = cast<YieldOp>(oldBlock->getTerminator());
  SmallVector<Value> newYieldValues;
  newYieldValues.reserve(oldYield.getNumOperands() - 1 + indices.size());
  llvm::append_range(
      newYieldValues,
      llvm::map_range(
          oldYield.getTargets().take_front(oldIf.getClassicalResults().size()),
          [&](Value value) { return mapping.lookupOrDefault(value); }));

  const auto oldLinearYields =
      oldYield.getTargets().drop_front(oldIf.getClassicalResults().size());
  for (const auto [oldIndex, value] : llvm::enumerate(oldLinearYields)) {
    if (oldIndex != tensorArgumentIndex) {
      newYieldValues.push_back(mapping.lookupOrDefault(value));
    }
  }

  for (const auto [indexPosition, index] : llvm::enumerate(indices)) {
    const auto* access = findTensorAccess(accesses, index);
    newYieldValues.push_back(access != nullptr
                                 ? mapping.lookupOrDefault(access->outputQubit)
                                 : scalarArguments[indexPosition]);
  }
  YieldOp::create(rewriter, oldYield.getLoc(), newYieldValues);
}

} // namespace mlir::qco

namespace {

/**
 * @brief Replace constant-index tensor updates in an if with scalar threading.
 *
 * @details A tensor carried through an if prevents target mapping from seeing
 * the accessed qubits as ordinary scalar wires. This pattern extracts the
 * union of the constant indices accessed by either branch before the if,
 * threads only those elements as scalar qubits through both branches, and
 * inserts the results afterwards. A tensor untouched in both branches is
 * forwarded around the if without extracting any element. Keeping untouched
 * elements out of the scalar operand list is important for large registers:
 * target mapping establishes the required sequencing for the remaining
 * physical wires when it expands the composite operation.
 */
struct ScalarizeTensorInputs final : OpRewritePattern<IfOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(IfOp op,
                                PatternRewriter& rewriter) const override {
    const auto classicalResultCount = op.getClassicalResults().size();
    const auto oldQubits = op.getQubits();

    for (const auto [tensorIndex, tensor] : llvm::enumerate(oldQubits)) {
      const auto tensorType = dyn_cast<RankedTensorType>(tensor.getType());
      if (!tensorType || !tensorType.hasStaticShape() ||
          tensorType.getRank() != 1) {
        continue;
      }

      auto thenAccesses = analyzeTensorBranch(
          op.thenBlock(), tensorIndex, classicalResultCount + tensorIndex);
      auto elseAccesses = analyzeTensorBranch(
          op.elseBlock(), tensorIndex, classicalResultCount + tensorIndex);
      if (!thenAccesses || !elseAccesses) {
        continue;
      }

      SmallVector<int64_t> accessedIndices;
      for (const auto& access : thenAccesses->accesses) {
        accessedIndices.push_back(access.index);
      }
      for (const auto& access : elseAccesses->accesses) {
        if (!llvm::is_contained(accessedIndices, access.index)) {
          accessedIndices.push_back(access.index);
        }
      }
      if (llvm::any_of(accessedIndices, [&](const int64_t index) {
            return index < 0 || index >= tensorType.getDimSize(0);
          })) {
        continue;
      }

      llvm::sort(accessedIndices);
      const ArrayRef<int64_t> indices(accessedIndices);

      rewriter.setInsertionPoint(op);
      SmallVector<Value> indexValues;
      SmallVector<Value> scalarInputs;
      indexValues.reserve(indices.size());
      scalarInputs.reserve(indices.size());
      Value tensorWithoutScalars = tensor;
      for (const int64_t index : indices) {
        auto indexValue =
            arith::ConstantIndexOp::create(rewriter, op.getLoc(), index);
        auto extract = qtensor::ExtractOp::create(rewriter, op.getLoc(),
                                                  tensorWithoutScalars,
                                                  indexValue.getResult());
        indexValues.push_back(indexValue.getResult());
        scalarInputs.push_back(extract.getResult());
        tensorWithoutScalars = extract.getOutTensor();
      }

      SmallVector<Value> newQubits;
      newQubits.reserve(oldQubits.size() - 1 + scalarInputs.size());
      for (const auto [oldIndex, qubit] : llvm::enumerate(oldQubits)) {
        if (oldIndex != tensorIndex) {
          newQubits.push_back(qubit);
        }
      }
      llvm::append_range(newQubits, scalarInputs);

      auto newIf = IfOp::create(
          rewriter, op.getLoc(), op.getClassicalResults().getTypes(),
          ValueRange(newQubits).getTypes(), op.getCondition(), newQubits);
      newIf->setDiscardableAttrs(op->getDiscardableAttrDictionary());
      const SmallVector locations(newQubits.size(), op.getLoc());
      auto* newThenBlock =
          rewriter.createBlock(&newIf.getThenRegion(), {},
                               ValueRange(newQubits).getTypes(), locations);
      auto* newElseBlock =
          rewriter.createBlock(&newIf.getElseRegion(), {},
                               ValueRange(newQubits).getTypes(), locations);
      cloneScalarizedBranch(op, op.thenBlock(), newThenBlock, tensorIndex,
                            *thenAccesses, indices, rewriter);
      cloneScalarizedBranch(op, op.elseBlock(), newElseBlock, tensorIndex,
                            *elseAccesses, indices, rewriter);

      rewriter.setInsertionPointAfter(newIf);
      Value updatedTensor = tensorWithoutScalars;
      const auto scalarResults =
          newIf.getLinearResults().take_back(indices.size());
      for (const auto [scalar, indexValue] :
           llvm::zip_equal(scalarResults, indexValues)) {
        updatedTensor = qtensor::InsertOp::create(rewriter, op.getLoc(), scalar,
                                                  updatedTensor, indexValue)
                            .getResult();
      }

      SmallVector<Value> replacements;
      replacements.reserve(op.getNumResults());
      llvm::append_range(replacements, newIf.getClassicalResults());
      size_t newLinearIndex = 0;
      for (const auto oldIndex : llvm::seq(oldQubits.size())) {
        replacements.push_back(
            oldIndex == tensorIndex
                ? updatedTensor
                : newIf.getLinearResults()[newLinearIndex++]);
      }
      rewriter.replaceOp(op, replacements);
      return success();
    }
    return failure();
  }
};
} // namespace

bool mlir::qco::hasOnlyScalarizableTensorInputs(IfOp op) {
  const size_t classicalResultCount = op.getClassicalResults().size();
  for (const auto [tensorIndex, value] : llvm::enumerate(op.getQubits())) {
    if (isa<QubitType>(value.getType())) {
      continue;
    }

    const auto tensorType = dyn_cast<RankedTensorType>(value.getType());
    if (!tensorType || tensorType.getRank() != 1 ||
        !tensorType.hasStaticShape() ||
        !isa<QubitType>(tensorType.getElementType())) {
      return false;
    }

    const auto thenAccesses = analyzeTensorBranch(
        op.thenBlock(), tensorIndex, classicalResultCount + tensorIndex);
    const auto elseAccesses = analyzeTensorBranch(
        op.elseBlock(), tensorIndex, classicalResultCount + tensorIndex);
    if (!thenAccesses || !elseAccesses) {
      return false;
    }

    const auto isOutOfBounds = [&](const TensorAccess& access) {
      return access.index < 0 || access.index >= tensorType.getDimSize(0);
    };
    if (llvm::any_of(thenAccesses->accesses, isOutOfBounds) ||
        llvm::any_of(elseAccesses->accesses, isOutOfBounds)) {
      return false;
    }
  }
  return true;
}

void IfOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                       MLIRContext* context) {
  results
      .add<RemoveStaticCondition, ConditionPropagation, ForwardClassicalResults,
           RemoveUnusedClassicalResults, ScalarizeTensorInputs>(context);
}

LogicalResult IfOp::verify() {
  const auto& inputQubits = getQubits();
  const auto numInputQubits = inputQubits.size();
  const auto& outputQubits = getLinearResults();
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
  for (Type type : getClassicalResults().getTypes()) {
    if (isLinearQubitType(type)) {
      return emitOpError("classical results must not use QCO linear types");
    }
  }
  for (auto [inputQubitType, outputQubitType] :
       llvm::zip_equal(inputQubits.getTypes(), outputQubits.getTypes())) {
    if (inputQubitType != outputQubitType) {
      return emitOpError("Operation must return the same qubit types as its "
                         "input qubit types.");
    }
  }
  for (const auto [inputType, thenType, elseType] :
       llvm::zip_equal(inputQubits.getTypes(), thenBlock()->getArgumentTypes(),
                       elseBlock()->getArgumentTypes())) {
    if (inputType != thenType || inputType != elseType) {
      return emitOpError(
          "branch argument types must match the input qubit types");
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
  if (qubit->getOwner() != getOperation() || qubit->getOperandNumber() == 0) {
    return {};
  }
  // Because the first operand is the if-condition, subtract one.
  return getLinearResults()[qubit->getOperandNumber() - 1];
}

OpOperand* IfOp::getTiedQubit(OpResult result) {
  if (result.getDefiningOp() != getOperation()) {
    return nullptr;
  }
  const auto numClassicalResults = getClassicalResults().size();
  if (result.getResultNumber() < numClassicalResults) {
    return nullptr;
  }
  return &getQubitsMutable()[result.getResultNumber() - numClassicalResults];
}

BlockArgument IfOp::getTiedThenBlockArgument(OpOperand* qubit) {
  if (qubit->getOwner() != getOperation() || qubit->getOperandNumber() == 0) {
    return {};
  }
  // Because the first operand is the if-condition, subtract one.
  return thenBlock()->getArguments()[qubit->getOperandNumber() - 1];
}

BlockArgument IfOp::getTiedElseBlockArgument(OpOperand* qubit) {
  if (qubit->getOwner() != getOperation() || qubit->getOperandNumber() == 0) {
    return {};
  }
  // Because the first operand is the if-condition, subtract one.
  return elseBlock()->getArguments()[qubit->getOperandNumber() - 1];
}

OpOperand* IfOp::getTiedThenYieldedValue(BlockArgument bbArg) {
  if (bbArg.getOwner() != thenBlock()) {
    return nullptr;
  }
  return &thenYield().getTargetsMutable()[getClassicalResults().size() +
                                          bbArg.getArgNumber()];
}

OpOperand* IfOp::getTiedElseYieldedValue(BlockArgument bbArg) {
  if (bbArg.getOwner() != elseBlock()) {
    return nullptr;
  }
  return &elseYield().getTargetsMutable()[getClassicalResults().size() +
                                          bbArg.getArgNumber()];
}

IfOp IfOp::replaceWithAdditionalQubits(RewriterBase& rewriter,
                                       ValueRange addons) {
  if (addons.empty()) {
    return *this;
  }

  const auto qubits = getQubits();

  SmallVector<Value> newQubits;
  newQubits.reserve(qubits.size() + addons.size());
  newQubits.append(qubits.begin(), qubits.end());
  newQubits.append(addons.begin(), addons.end());
  const auto allQubitTypes = ValueRange(newQubits).getTypes();

  auto newIfOp =
      create(rewriter, getLoc(), getClassicalResults().getTypes(),
             ValueRange(newQubits).getTypes(), getCondition(), newQubits);

  const auto rewriteRegion = [&](Region& oldRegion, Region& newRegion) {
    auto* oldBlock = &oldRegion.front();
    const auto numOldArgs = oldBlock->getNumArguments();
    auto* newBlock =
        rewriter.createBlock(&newRegion, {}, allQubitTypes,
                             SmallVector<Location>(newQubits.size(), getLoc()));
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

  rewriteRegion(getThenRegion(), newIfOp.getThenRegion());
  rewriteRegion(getElseRegion(), newIfOp.getElseRegion());

  rewriter.replaceOp(*this, newIfOp.getResults().take_front(getNumResults()));

  return newIfOp;
}
