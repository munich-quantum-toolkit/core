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
#include <llvm/ADT/SmallBitVector.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/ErrorHandling.h>
#include <mlir/Dialect/Linalg/IR/RelayoutOpInterface.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Dialect/Utils/StaticValueUtils.h>
#include <mlir/Dialect/Utils/StructuredOpsUtils.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/TypeUtilities.h>
#include <mlir/Interfaces/DestinationStyleOpInterface.h>
#include <mlir/Interfaces/LoopLikeInterface.h>
#include <mlir/Interfaces/ViewLikeInterface.h>
#include <mlir/Support/LLVM.h>

#include <cassert>
#include <cstddef>
#include <cstdint>

// The following headers are needed for some template instantiations.
// IWYU pragma: begin_keep
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/DialectImplementation.h>
// IWYU pragma: end_keep

using namespace mlir;
using namespace mlir::qtensor;

namespace mlir::qtensor {

llvm::SmallBitVector getDroppedDims(ArrayRef<int64_t> reducedShape,
                                    ArrayRef<OpFoldResult> mixedSizes) {
  llvm::SmallBitVector droppedDims(mixedSizes.size());
  int64_t shapePos = static_cast<int64_t>(reducedShape.size()) - 1;

  for (const auto& size : enumerate(llvm::reverse(mixedSizes))) {
    size_t idx = mixedSizes.size() - size.index() - 1;
    // Rank-reduced dims must have a static unit dimension.
    bool isStaticUnitSize =
        isa<Attribute>(size.value()) &&
        llvm::cast<IntegerAttr>(cast<Attribute>(size.value())).getInt() == 1;

    if (shapePos < 0) {
      // There are no more dims in the reduced shape. All remaining sizes must
      // be rank-reduced dims.
      assert(isStaticUnitSize && "expected unit dim");
      droppedDims.set(idx);
      continue;
    }

    // Dim is preserved if the size is not a static 1.
    if (!isStaticUnitSize) {
      --shapePos;
      continue;
    }

    // Dim is preserved if the reduced shape dim is also 1.
    if (reducedShape[shapePos] == 1) {
      --shapePos;
      continue;
    }

    // Otherwise: Dim is dropped.
    droppedDims.set(idx);
  }

  assert(shapePos < 0 && "dimension mismatch");
  return droppedDims;
}

LogicalResult produceSliceErrorMsg(SliceVerificationResult result,
                                   Operation* op,
                                   RankedTensorType expectedType) {
  switch (result) {
  case SliceVerificationResult::Success:
    return success();
  case SliceVerificationResult::RankTooLarge:
    return op->emitError("expected rank to be smaller or equal to ")
           << "the other rank. ";
  case SliceVerificationResult::SizeMismatch:
    return op->emitError("expected type to be ")
           << expectedType << " or a rank-reduced version. (size mismatch) ";
  case SliceVerificationResult::ElemTypeMismatch:
    return op->emitError("expected element type to be ")
           << expectedType.getElementType();
  default:
    llvm_unreachable("unexpected extract_slice op verification result");
  }
}

LogicalResult
foldIdentityOffsetSizeAndStrideOpInterface(OffsetSizeAndStrideOpInterface op,
                                           ShapedType shapedType) {
  OpBuilder b(op.getContext());
  for (OpFoldResult opFold : op.getMixedOffsets()) {
    if (getConstantIntValue(opFold) != static_cast<int64_t>(0)) {
      return failure();
    }
  }
  // Rank-reducing noops only need to inspect the leading dimensions:
  // llvm::zip is appropriate.
  auto shape = shapedType.getShape();
  for (auto it : llvm::zip(op.getMixedSizes(), shape)) {
    if (getConstantIntValue(std::get<0>(it)) != std::get<1>(it)) {
      return failure();
    }
  }
  for (OpFoldResult opFold : op.getMixedStrides()) {
    if (getConstantIntValue(opFold) != static_cast<int64_t>(1)) {
      return failure();
    }
  }
  return success();
}
} // namespace mlir::qtensor

//===----------------------------------------------------------------------===//
// Common Canonicalizers and Folders.
//===----------------------------------------------------------------------===//

static bool foldTensorCastPrecondition(DestinationStyleOpInterface op) {
  // 1. InsertSliceOp has its own logic about folding tensor.cast ops.
  // 2. Exclude DPS ops that are also LoopLike from this interface as they
  // might need special handling of attached regions.
  if (isa<InsertSliceOp>(op.getOperation()) ||
      isa<LoopLikeOpInterface>(op.getOperation())) {
    return false;
  }

  return mlir::tensor::hasFoldableTensorCastOperand(op);
}

namespace {
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
} // namespace

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
