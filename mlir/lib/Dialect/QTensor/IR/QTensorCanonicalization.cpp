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
#include "mlir/Dialect/QTensor/IR/QTensorDialect.h"
#include "mlir/Dialect/QTensor/IR/QTensorOps.h"

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/Sequence.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Utils/StaticValueUtils.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <optional>

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
                     size_t qTensorYieldIndex) {
  BranchQTensorAccesses result;
  Value currentQTensor = block->getArgument(qTensorArgumentIndex);
  bool reachedInsertPhase = false;

  while (true) {
    assert(currentQTensor.hasOneUse() && "expected linear typing");
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

    auto yield = dyn_cast<qco::YieldOp>(user);
    if (!yield || user != block->getTerminator() ||
        qTensorYieldIndex >= yield.getTargets().size() ||
        yield.getTargets()[qTensorYieldIndex] != currentQTensor ||
        llvm::any_of(result.accesses, [](const auto& access) {
          return !access.second.insert;
        })) {
      return std::nullopt;
    }
    return result;
  }
}

/// Move a branch while replacing QTensor accesses with scalar qubits.
static void moveScalarizedQTensorBranch(qco::IfOp oldIf, Block* oldBlock,
                                        Block* newBlock,
                                        size_t qTensorArgumentIndex,
                                        BranchQTensorAccesses& accesses,
                                        ArrayRef<int64_t> indices,
                                        PatternRewriter& rewriter) {
  auto oldYield = cast<qco::YieldOp>(oldBlock->getTerminator());
  auto scalarArguments = newBlock->getArguments().take_back(indices.size());
  auto carriedArguments = newBlock->getArguments().drop_back(indices.size());

  SmallVector<Value> argumentReplacements;
  argumentReplacements.reserve(oldBlock->getNumArguments());
  size_t carriedIndex = 0;
  for (size_t oldIndex : llvm::seq(oldBlock->getNumArguments())) {
    argumentReplacements.push_back(oldIndex == qTensorArgumentIndex
                                       ? oldIf.getQubits()[qTensorArgumentIndex]
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

  auto oldTargets = oldYield.getTargets();
  size_t classicalResultCount = oldIf.getClassicalResults().size();
  SmallVector<Value> newYieldValues;
  newYieldValues.reserve(oldTargets.size() - 1 + scalarYields.size());
  llvm::append_range(newYieldValues,
                     oldTargets.take_front(classicalResultCount));
  for (auto [oldIndex, value] :
       llvm::enumerate(oldTargets.drop_front(classicalResultCount))) {
    if (oldIndex != qTensorArgumentIndex) {
      newYieldValues.push_back(value);
    }
  }
  llvm::append_range(newYieldValues, scalarYields);

  rewriter.setInsertionPoint(oldYield);
  rewriter.replaceOpWithNewOp<qco::YieldOp>(oldYield, newYieldValues);

  for (Operation* operation : llvm::reverse(accesses.qTensorOperations)) {
    rewriter.eraseOp(operation);
  }
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
      SmallVector<Value> indexValues;
      SmallVector<Value> scalarInputs;
      indexValues.reserve(indices.size());
      scalarInputs.reserve(indices.size());
      Value qTensorWithoutScalars = qTensor;
      for (int64_t index : indices) {
        auto indexValue =
            arith::ConstantIndexOp::create(rewriter, op.getLoc(), index);
        auto extract =
            ExtractOp::create(rewriter, op.getLoc(), qTensorWithoutScalars,
                              indexValue.getResult());
        indexValues.push_back(indexValue.getResult());
        scalarInputs.push_back(extract.getResult());
        qTensorWithoutScalars = extract.getOutTensor();
      }

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
      moveScalarizedQTensorBranch(op, oldThenBlock, newThenBlock, qTensorIndex,
                                  *thenAccesses, indices, rewriter);
      moveScalarizedQTensorBranch(op, oldElseBlock, newElseBlock, qTensorIndex,
                                  *elseAccesses, indices, rewriter);

      rewriter.setInsertionPointAfter(newIf);
      Value updatedQTensor = qTensorWithoutScalars;
      auto scalarResults = newIf.getLinearResults().take_back(indices.size());
      for (auto [scalar, indexValue] :
           llvm::zip_equal(scalarResults, indexValues)) {
        updatedQTensor = InsertOp::create(rewriter, op.getLoc(), scalar,
                                          updatedQTensor, indexValue)
                             .getResult();
      }

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
} // namespace

void QTensorDialect::getCanonicalizationPatterns(
    RewritePatternSet& results) const {
  results.add<ScalarizeQTensorInputs>(getContext());
}
