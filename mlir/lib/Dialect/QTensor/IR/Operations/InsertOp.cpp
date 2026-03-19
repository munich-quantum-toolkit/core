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

#include <mlir/Dialect/Utils/StaticValueUtils.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

using namespace mlir;
using namespace mlir::qtensor;

LogicalResult InsertOp::verify() {
  auto dstDim = getDest().getType().getDimSize(0);
  auto index = getConstantIntValue(getIndex());

  if (index) {
    if (*index < 0) {
      return emitOpError("Index must be non-negative");
    }
    if (!ShapedType::isDynamic(dstDim) && *index >= dstDim) {
      return emitOpError("Index exceeds tensor dimension");
    }
  }

  return success();
}

/**
 * @brief If an InsertOp consumes an ExtractOp with the same index,
 * return the tensor from the extractOp directly.
 */
static Value foldInsertAfterExtract(InsertOp insertOp) {
  auto extractOp = insertOp.getScalar().getDefiningOp<ExtractOp>();

  if (!extractOp) {
    return nullptr;
  }
  if (insertOp.getDest() != extractOp.getOutTensor()) {
    return nullptr;
  }

  auto insertIndex = insertOp.getIndex();
  auto extractIndex = extractOp.getIndex();

  if (getAsOpFoldResult(insertIndex) != getAsOpFoldResult(extractIndex)) {
    return nullptr;
  }

  return extractOp.getTensor();
}

OpFoldResult InsertOp::fold(FoldAdaptor /*adaptor*/) {
  if (auto result = foldInsertAfterExtract(*this)) {
    return result;
  }

  return {};
}
