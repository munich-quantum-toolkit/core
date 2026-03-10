/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/Interfaces/DestinationStyleOpInterface.h>
#include <mlir/Interfaces/ViewLikeInterface.h>

#define DIALECT_NAME_QTensor "qtensor"

namespace mlir::qtensor {

/// Compute the dropped dimensions of a rank-reducing tensor.extract_slice op or
/// rank-extending tensor.insert_slice op.
llvm::SmallBitVector getDroppedDims(ArrayRef<int64_t> reducedShape,
                                    ArrayRef<OpFoldResult> mixedSizes);

LogicalResult produceSliceErrorMsg(SliceVerificationResult result,
                                   Operation* op,
                                   RankedTensorType expectedType);

LogicalResult
foldIdentityOffsetSizeAndStrideOpInterface(OffsetSizeAndStrideOpInterface op,
                                           ShapedType shapedType);
} // namespace mlir::qtensor

//===----------------------------------------------------------------------===//
// QTensor Dialect
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/QTensor/IR/QTensorOpsDialect.h.inc" // IWYU pragma: export

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/QTensor/IR/QTensorOpsTypes.h.inc" // IWYU pragma: export

//===----------------------------------------------------------------------===//
// QTensor Dialect Helpers
//===----------------------------------------------------------------------===//
