/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/Dialect/QCO/IR/QCODialect.h"
#include "mqt/Dialect/QCO/IR/QCOOps.h"
#include "mqt/Dialect/QTensor/IR/QTensorOps.h"

#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

#include <cassert>
#include <cstdint>

using namespace mlir;
using namespace mlir::qtensor;

void AllocOp::build(OpBuilder& builder, OperationState& result, Value size) {
  auto sizeValue = getConstantIntValue(size);
  if (sizeValue) {
    assert(*sizeValue > 0 && "qtensor.alloc size must be positive");
  }

  auto resultType =
      RankedTensorType::get({sizeValue ? *sizeValue : ShapedType::kDynamic},
                            qco::QubitType::get(builder.getContext()));
  build(builder, result, resultType, size);
}

LogicalResult AllocOp::verify() {
  auto resultType = getResult().getType();
  auto sizeValue = getConstantIntValue(getSize());
  auto resultSize = resultType.getShape()[0];

  if (sizeValue && *sizeValue <= 0) {
    return emitOpError("Constant size operand must be positive");
  }
  if (!resultType.isDynamicDim(0)) {
    if (!sizeValue) {
      return emitOpError("Static result type requires constant size operand");
    }
    if (resultSize != *sizeValue) {
      return emitOpError("Constant size operand (")
             << *sizeValue << ") does not match static result size ("
             << resultSize << ")";
    }
  }

  return success();
}

namespace {
/// Discover fresh slots once per allocation, including unsuccessful searches.
struct RemoveFreshSlotResets final : OpRewritePattern<AllocOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(AllocOp alloc,
                                PatternRewriter& rewriter) const override {
    llvm::SmallDenseSet<int64_t> accessed;
    SmallVector<qco::ResetOp> resets;
    auto tensor = alloc.getResult();
    while (true) {
      auto* user = *tensor.user_begin();
      if (user->getBlock() != alloc->getBlock()) {
        break;
      }
      if (auto extract = dyn_cast<ExtractOp>(user)) {
        const auto index = getConstantIntValue(extract.getIndex());
        if (!index) {
          break;
        }
        if (accessed.insert(*index).second) {
          if (auto reset =
                  dyn_cast<qco::ResetOp>(*extract.getResult().user_begin());
              reset && reset->getBlock() == alloc->getBlock()) {
            resets.push_back(reset);
          }
        }
        tensor = extract.getOutTensor();
        continue;
      }
      if (auto insert = dyn_cast<InsertOp>(user)) {
        const auto index = getConstantIntValue(insert.getIndex());
        if (!index) {
          break;
        }
        accessed.insert(*index);
        tensor = insert.getResult();
        continue;
      }
      break;
    }
    if (resets.empty()) {
      return failure();
    }
    for (auto reset : resets) {
      rewriter.replaceOp(reset, reset.getQubitIn());
    }
    /// Replace the pattern root and preserve allocation attributes.
    auto* replacement = rewriter.clone(*alloc);
    rewriter.replaceOp(alloc, replacement->getResults());
    return success();
  }
};
} // namespace

void AllocOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                          MLIRContext* context) {
  results.add<RemoveFreshSlotResets>(context);
}
