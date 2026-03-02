/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QTensor/IR/QTensorOps.h"

#include "mlir/Dialect/QTensor/IR/QTensorDialect.h" // IWYU pragma: associated

#include <llvm/ADT/STLExtras.h>
#include <llvm/Support/Casting.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h> // for affine::AffineDialect
#include <mlir/Dialect/Arith/IR/Arith.h>      // for arith::ArithDialect
#include <mlir/Dialect/Arith/Utils/Utils.h>
#include <mlir/Dialect/Complex/IR/Complex.h> // for complex::ComplexDialect
#include <mlir/Dialect/Linalg/IR/RelayoutOpInterface.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Dialect/Utils/StaticValueUtils.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/TypeUtilities.h>
#include <mlir/Interfaces/DestinationStyleOpInterface.h>
#include <mlir/Interfaces/ViewLikeInterface.h>
#include <mlir/Support/LLVM.h>

// The following headers are needed for some template instantiations.
// IWYU pragma: begin_keep
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/DialectImplementation.h>
// IWYU pragma: end_keep

using namespace mlir;
using namespace mlir::qtensor;

/// Materialize a single constant operation from a given attribute value with
/// the desired resultant type.
Operation* QTensorDialect::materializeConstant(OpBuilder& builder,
                                               Attribute value, Type type,
                                               Location loc) {
  if (auto op = arith::ConstantOp::materialize(builder, value, type, loc)) {
    return op;
  }
  if (complex::ConstantOp::isBuildableWith(value, type)) {
    return builder.create<complex::ConstantOp>(loc, type,
                                               llvm::cast<ArrayAttr>(value));
  }
  return nullptr;
}

//===----------------------------------------------------------------------===//
// Common Canonicalizers and Folders.
//===----------------------------------------------------------------------===//

namespace {

bool foldTensorCastPrecondition(DestinationStyleOpInterface op) {
  // 1. InsertSliceOp has its own logic about folding tensor.cast ops.
  // 2. Exclude DPS ops that are also LoopLike from this interface as they
  // might need special handling of attached regions.
  if (isa<InsertSliceOp>(op.getOperation()) ||
      isa<LoopLikeOpInterface>(op.getOperation())) {
    return false;
  }

  return mlir::tensor::hasFoldableTensorCastOperand(op);
}
} // namespace

struct FoldTensorCastProducerOp
    : public OpInterfaceRewritePattern<DestinationStyleOpInterface> {
  using OpInterfaceRewritePattern<
      DestinationStyleOpInterface>::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(DestinationStyleOpInterface op,
                                PatternRewriter& rewriter) const override {

    // Reject PackOp/UnpackOp (i.e. RelayoutOps) - there are dedicated patterns
    // for that instead.
    if (!foldTensorCastPrecondition(op) ||
        isa<linalg::RelayoutOpInterface>(*op)) {
      return failure();
    }

    SmallVector<Type> newResultTypes(op->getResultTypes());
    SmallVector<Value> newOperands =
        mlir::tensor::getUpdatedOperandsAfterCastOpFolding(op, newResultTypes);

    // Clone op
    auto newOp = clone(rewriter, op, newResultTypes, newOperands);

    SmallVector<Value, 4> replacements;
    replacements.reserve(newOp->getNumResults());
    for (auto [oldResult, newResult] :
         llvm::zip(op->getResults(), newOp->getResults())) {
      if (newResult.getType() != oldResult.getType()) {
        replacements.push_back(rewriter.create<tensor::CastOp>(
            op->getLoc(), oldResult.getType(), newResult));
      } else {
        replacements.push_back(newResult);
      }
    }
    rewriter.replaceOp(op, replacements);

    return success();
  }
};

void QTensorDialect::getCanonicalizationPatterns(
    RewritePatternSet& results) const {
  results.add<FoldTensorCastProducerOp>(getContext());
}

//===----------------------------------------------------------------------===//
// Dialect
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/QTensor/IR/QTensorOpsDialect.cpp.inc"

void QTensorDialect::initialize() {
  // NOLINTNEXTLINE(clang-analyzer-core.StackAddressEscape)
  addTypes<
#define GET_TYPEDEF_LIST
#include "mlir/Dialect/QTensor/IR/QTensorOpsTypes.cpp.inc"

      >();

  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/QTensor/IR/QTensorOps.cpp.inc"

      >();
}

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/QTensor/IR/QTensorOpsTypes.cpp.inc"

//===----------------------------------------------------------------------===//
// Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/QTensor/IR/QTensorOps.cpp.inc"
