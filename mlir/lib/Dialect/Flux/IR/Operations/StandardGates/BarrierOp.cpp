/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/Flux/FluxUtils.h"
#include "mlir/Dialect/Flux/IR/FluxDialect.h"

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
using namespace mlir::flux;

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

size_t BarrierOp::getNumQubits() { return getNumTargets(); }

size_t BarrierOp::getNumTargets() { return getQubitsIn().size(); }

size_t BarrierOp::getNumControls() { return 0; }

size_t BarrierOp::getNumPosControls() { return 0; }

size_t BarrierOp::getNumNegControls() { return 0; }

Value BarrierOp::getInputQubit(const size_t i) { return getInputTarget(i); }

Value BarrierOp::getOutputQubit(const size_t i) { return getOutputTarget(i); }

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

Value BarrierOp::getInputPosControl(const size_t /*i*/) {
  llvm::reportFatalUsageError("BarrierOp cannot be controlled");
}

Value BarrierOp::getOutputPosControl(const size_t /*i*/) {
  llvm::reportFatalUsageError("BarrierOp cannot be controlled");
}

Value BarrierOp::getInputNegControl(const size_t /*i*/) {
  llvm::reportFatalUsageError("BarrierOp cannot be controlled");
}

Value BarrierOp::getOutputNegControl(const size_t /*i*/) {
  llvm::reportFatalUsageError("BarrierOp cannot be controlled");
}

Value BarrierOp::getInputForOutput(const Value output) {
  for (size_t i = 0; i < getNumTargets(); ++i) {
    if (output == getQubitsOut()[i]) {
      return getQubitsIn()[i];
    }
  }
  llvm::reportFatalUsageError("Given qubit is not an output of the operation");
}

Value BarrierOp::getOutputForInput(const Value input) {
  for (size_t i = 0; i < getNumTargets(); ++i) {
    if (input == getQubitsIn()[i]) {
      return getQubitsOut()[i];
    }
  }
  llvm::reportFatalUsageError("Given qubit is not an input of the operation");
}

size_t BarrierOp::getNumParams() { return 0; }

Value BarrierOp::getParameter(const size_t /*i*/) {
  llvm::reportFatalUsageError("BarrierOp has no parameters");
}

void BarrierOp::build(OpBuilder& odsBuilder, OperationState& odsState,
                      const ValueRange qubits) {
  SmallVector<Type> resultTypes;
  resultTypes.reserve(qubits.size());
  for (const Value qubit : qubits) {
    resultTypes.push_back(qubit.getType());
  }
  build(odsBuilder, odsState, resultTypes, qubits);
}

void BarrierOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                            MLIRContext* context) {
  results.add<MergeSubsequentBarrier>(context);
}
