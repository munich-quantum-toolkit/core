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
#include "mqt/Dialect/QCO/Utils/Matrix.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"

#include <cstddef>
#include <cstdint>

using namespace mlir;
using namespace mlir::qco;

namespace {

/// Merge subsequent barriers on the same qubits into a single barrier.
struct MergeSubsequentBarrier final : OpRewritePattern<BarrierOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(BarrierOp op,
                                PatternRewriter& rewriter) const override {
    auto qubitsIn = op.getQubitsIn();

    auto anythingToMerge = false;
    SmallVector<Value> newQubitsOut(qubitsIn);

    SmallVector<Value> newQubitsIn;
    SmallVector<size_t> indicesToFill;

    for (size_t i = 0; i < qubitsIn.size(); ++i) {
      if (auto output = op.getQubitsOut()[i];
          isa<BarrierOp>(*output.user_begin())) {
        anythingToMerge = true;
      } else {
        newQubitsIn.push_back(qubitsIn[i]);
        indicesToFill.push_back(i);
      }
    }

    if (!anythingToMerge) {
      return failure();
    }

    auto newBarrier = BarrierOp::create(rewriter, op.getLoc(), newQubitsIn);

    for (size_t i = 0; i < indicesToFill.size(); ++i) {
      newQubitsOut[indicesToFill[i]] = newBarrier.getQubitsOut()[i];
    }

    rewriter.replaceOp(op, newQubitsOut);
    return success();
  }
};

struct FoldMaskedBarrier final : OpRewritePattern<MaskedBarrierOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(MaskedBarrierOp op,
                                PatternRewriter& rewriter) const override {
    SmallVector<Value> selected;
    SmallVector<Value> masks;
    SmallVector<size_t> positions;
    bool allConstant = true;
    for (auto [index, mask] : llvm::enumerate(op.getMasks())) {
      APInt value;
      const bool constant = matchPattern(mask, m_ConstantInt(&value));
      allConstant &= constant;
      if (!constant || !value.isZero()) {
        positions.push_back(index);
        selected.push_back(op.getQubitsIn()[index]);
        masks.push_back(mask);
      }
    }
    if (!allConstant && selected.size() == op.getQubitsIn().size()) {
      return failure();
    }
    SmallVector<Value> results(op.getQubitsIn());
    if (!selected.empty()) {
      auto outputs = allConstant
                         ? BarrierOp::create(rewriter, op.getLoc(), selected)
                               .getQubitsOut()
                         : MaskedBarrierOp::create(
                               rewriter, op.getLoc(),
                               SmallVector<Type>(selected.size(),
                                                 selected.front().getType()),
                               selected, masks)
                               .getQubitsOut();
      for (auto [position, result] : llvm::zip_equal(positions, outputs)) {
        results[position] = result;
      }
    }
    rewriter.replaceOp(op, results);
    return success();
  }
};

} // namespace

LogicalResult BarrierOp::verify() {
  if (getQubitsIn().size() != getQubitsOut().size()) {
    return emitOpError("requires one output qubit for each input qubit");
  }
  return success();
}

LogicalResult MaskedBarrierOp::verify() {
  if (getQubitsIn().size() != getMasks().size() ||
      getQubitsIn().size() != getQubitsOut().size()) {
    return emitOpError("requires one mask and one output for each input qubit");
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

void MaskedBarrierOp::getCanonicalizationPatterns(RewritePatternSet& patterns,
                                                  MLIRContext* context) {
  patterns.add<FoldMaskedBarrier>(context);
}

DynamicMatrix BarrierOp::getUnitaryMatrix() {
  const auto numQubits = getQubitsIn().size();
  return DynamicMatrix::identity(
      static_cast<int64_t>(uint64_t{1} << numQubits));
}
