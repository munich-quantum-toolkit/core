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
#include "mlir/Dialect/QCO/QCOUtils.h"
#include "mlir/Dialect/QCO/Utils/Matrix.h"

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/ErrorHandling.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>

#include <cstddef>
#include <cstdint>
#include <optional>

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
    auto outputs = op.getQubitsOut();
    if (outputs.empty()) {
      return failure();
    }

    BarrierOp nextBarrier;
    for (Value output : outputs) {
      auto user = dyn_cast<BarrierOp>(*output.getUsers().begin());
      if (!user || user->getBlock() != op->getBlock() ||
          (nextBarrier && user != nextBarrier)) {
        return failure();
      }
      nextBarrier = user;
    }

    if (nextBarrier.getNumTargets() != outputs.size()) {
      return failure();
    }

    rewriter.replaceOp(op, op.getQubitsIn());
    return success();
  }
};

} // namespace

LogicalResult BarrierOp::verify() {
  if (getQubitsIn().size() != getQubitsOut().size()) {
    return emitOpError(
        "number of input qubits must match the number of output qubits");
  }
  return success();
}

Value BarrierOp::getInputForOutput(Value output) {
  if (auto result = dyn_cast<OpResult>(output);
      result && result.getOwner() == getOperation()) {
    return getQubitsIn()[result.getResultNumber()];
  }
  llvm::reportFatalUsageError("Given qubit is not an output of the operation");
}

Value BarrierOp::getOutputForInput(Value input) {
  for (auto [in, out] : llvm::zip_equal(getQubitsIn(), getQubitsOut())) {
    if (in == input) {
      return out;
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

bool BarrierOp::hasCompileTimeKnownUnitaryMatrix() {
  return isModifierMatrixSizeSupported(getNumTargets());
}

std::optional<DynamicMatrix> BarrierOp::getUnitaryMatrix() {
  if (!hasCompileTimeKnownUnitaryMatrix()) {
    return std::nullopt;
  }
  return DynamicMatrix::identity(static_cast<int64_t>(1ULL << getNumTargets()));
}
