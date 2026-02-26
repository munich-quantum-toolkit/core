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

#include "mlir/Dialect/QTensor/IR/QTensorOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

#define DIALECT_NAME_QTensor "qtensor"

//===----------------------------------------------------------------------===//
// QTensor Dialect Helpers
//===----------------------------------------------------------------------===//

namespace mlir {

/// Return the list of Range (i.e. offset, size, stride). Each Range
/// entry contains either the dynamic value or a ConstantIndexOp constructed
/// with `b` at location `loc`.
SmallVector<Range, 8> getOrCreateRanges(OffsetSizeAndStrideOpInterface op,
                                        OpBuilder& b, Location loc);

} // namespace mlir

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

namespace mlir::qtensor {} // namespace mlir::qtensor
