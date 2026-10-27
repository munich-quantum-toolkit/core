/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/Dialect/MQT/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/STLExtras.h"

namespace mlir::mqt {

#define GEN_PASS_DEF_FOLDCLASSICALVALUES
#include "mqt/Dialect/MQT/Transforms/Passes.h.inc"

namespace {
struct FoldClassicalValues final
    : impl::FoldClassicalValuesBase<FoldClassicalValues> {
  using FoldClassicalValuesBase::FoldClassicalValuesBase;

protected:
  void runOnOperation() override {
    SmallVector<Operation*> operations;
    getOperation()->walk([&](Operation* op) {
      const auto isNumeric = [](Type type) {
        if (auto shaped = dyn_cast<ShapedType>(type)) {
          type = shaped.getElementType();
        }
        return type.isIntOrIndexOrFloat();
      };
      if (op->getNumRegions() != 0 ||
          !llvm::all_of(op->getOperandTypes(), isNumeric) ||
          !llvm::all_of(op->getResultTypes(), isNumeric)) {
        return;
      }
      if (isa_and_nonnull<arith::ArithDialect, math::MathDialect>(
              op->getDialect()) ||
          (isa<tensor::ExtractOp>(op) && op->getResult(0).getType().isF64())) {
        operations.push_back(op);
      }
    });
    // Strictness prevents quantum users from entering the folding worklist.
    const auto config = GreedyRewriteConfig{}
                            .setStrictness(GreedyRewriteStrictness::ExistingOps)
                            .enableConstantCSE(false);
    const FrozenRewritePatternSet patterns(&getContext());
    if (failed(applyOpPatternsGreedily(operations, patterns, config))) {
      getOperation()->emitError("classical value folding did not converge");
      signalPassFailure();
    }
  }
};
} // namespace
} // namespace mlir::mqt
