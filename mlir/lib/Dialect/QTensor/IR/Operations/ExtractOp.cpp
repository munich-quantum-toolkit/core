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

#include <llvm/ADT/STLExtras.h>
#include <llvm/Support/Casting.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Dialect/Utils/StaticValueUtils.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/TypeUtilities.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

using namespace mlir;
using namespace mlir::qtensor;

//===----------------------------------------------------------------------===//
// ExtractOp
//===----------------------------------------------------------------------===//

void ExtractOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult(), "q_extracted");
}

LogicalResult ExtractOp::verify() {
  // Verify the # indices match if we have a ranked type.
  auto tensorType = llvm::cast<RankedTensorType>(getTensor().getType());
  if (tensorType.getRank() != static_cast<int64_t>(getIndices().size())) {
    return emitOpError("incorrect number of indices for extract_element");
  }
  return success();
}
struct ExtractFromTensorCast : public OpRewritePattern<ExtractOp> {
  using OpRewritePattern<ExtractOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ExtractOp extract,
                                PatternRewriter& rewriter) const final {
    auto tensorCast = extract.getTensor().getDefiningOp<tensor::CastOp>();
    if (!tensorCast) {
      return failure();
    }
    if (!llvm::isa<RankedTensorType>(tensorCast.getSource().getType())) {
      return failure();
    }
    rewriter.replaceOpWithNewOp<ExtractOp>(extract, tensorCast.getSource(),
                                           extract.getIndices());
    return success();
  }
};

/// If we have an ExtractOp consuming an InsertOp with the same
/// indices, we can return the InsertOp's scalar directly.
// TODO: This only checks the immediate producer; extend to go up the
// insert/extract chain if the slices are disjoint.
static Value foldExtractAfterInsert(ExtractOp extractOp) {
  auto insertOp = extractOp.getTensor().getDefiningOp<tensor::InsertOp>();

  auto isSame = [](Value a, Value b) {
    return getAsOpFoldResult(a) == getAsOpFoldResult(b);
  };
  if (insertOp && insertOp.getScalar().getType() == extractOp.getType(0) &&
      llvm::equal(insertOp.getIndices(), extractOp.getIndices(), isSame)) {
    return insertOp.getScalar();
  }
  return {};
}

LogicalResult ExtractOp::fold(FoldAdaptor adaptor,
                              SmallVectorImpl<OpFoldResult>& results) {
  // Collect the constant indices into the tensor.
  SmallVector<uint64_t, 8> indices;
  for (Attribute indice : adaptor.getIndices()) {
    if (!indice || !llvm::isa<IntegerAttr>(indice)) {
      return failure();
    }
    indices.push_back(llvm::cast<IntegerAttr>(indice).getInt());
  }

  if (Value result = foldExtractAfterInsert(*this)) {
    results.push_back(result);
    results.push_back(getTensor());
    return success();
  }

  return failure();
}

void ExtractOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                            MLIRContext* context) {
  results.add<ExtractFromTensorCast>(context);
}
