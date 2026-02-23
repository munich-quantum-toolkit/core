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

#include <cstddef>
#include <llvm/ADT/STLExtras.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/ErrorHandling.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

using namespace mlir;
using namespace mlir::qco;

namespace {

/**
 * @brief Merge subsequent barriers on the same qubits into a single barrier.
 */
struct MergeSubsequentBarrier final : OpRewritePattern<BarrierOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(BarrierOp op,
                                PatternRewriter& rewriter) const override {
    const auto& qubitsIn = op.getQubitsIn();

    auto anythingToMerge = false;
    DenseMap<size_t, Value> newQubitsOutMap;

    SmallVector<Value> newQubitsIn;
    SmallVector<size_t> indicesToFill;

    for (size_t i = 0; i < qubitsIn.size(); ++i) {
      if (llvm::isa<BarrierOp>(
              *op.getOutputForInput(qubitsIn[i]).getUsers().begin())) {
        anythingToMerge = true;
        newQubitsOutMap[i] = qubitsIn[i];
      } else {
        newQubitsIn.push_back(qubitsIn[i]);
        indicesToFill.push_back(i);
      }
    }

    if (!anythingToMerge) {
      return failure();
    }

    auto newBarrier = rewriter.create<BarrierOp>(op.getLoc(), newQubitsIn);

    for (size_t i = 0; i < indicesToFill.size(); ++i) {
      newQubitsOutMap[indicesToFill[i]] = newBarrier.getQubitsOut()[i];
    }

    SmallVector<Value> newQubitsOut;
    newQubitsOut.reserve(op.getQubitsIn().size());
    for (size_t i = 0; i < op.getQubitsIn().size(); ++i) {
      newQubitsOut.push_back(newQubitsOutMap[i]);
    }

    rewriter.replaceOp(op, newQubitsOut);
    return success();
  }
};

} // namespace

Value BarrierOp::getInputTarget(const size_t i) {
  if (i < getNumTargets()) {
    return getQubitsIn()[i];
  }
  llvm::reportFatalUsageError("Invalid qubit index");
}

Value BarrierOp::getOutputTarget(const size_t i) {
  if (i < getNumTargets()) {
    return getQubitsOut()[i];
  }
  llvm::reportFatalUsageError("Invalid qubit index");
}

Value BarrierOp::getInputForOutput(Value output) {
  for (size_t i = 0; i < getNumTargets(); ++i) {
    if (output == getQubitsOut()[i]) {
      return getQubitsIn()[i];
    }
  }
  llvm::reportFatalUsageError("Given qubit is not an output of the operation");
}

Value BarrierOp::getOutputForInput(Value input) {
  for (size_t i = 0; i < getNumTargets(); ++i) {
    if (input == getQubitsIn()[i]) {
      return getQubitsOut()[i];
    }
  }
  llvm::reportFatalUsageError("Given qubit is not an input of the operation");
}

void BarrierOp::build(OpBuilder& odsBuilder, OperationState& odsState,
                      ValueRange qubits) {
  SmallVector<Type> resultTypes;
  resultTypes.reserve(qubits.size());
  for (auto qubit : qubits) {
    resultTypes.push_back(qubit.getType());
  }
  build(odsBuilder, odsState, resultTypes, qubits);
}

void BarrierOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                            MLIRContext* context) {
  results.add<MergeSubsequentBarrier>(context);
}
