/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QTensor/IR/QTensorOps.h"

#include <cstdint>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypes.h>

using namespace mlir;
using namespace mlir::qtensor;

void AllocOp::build(OpBuilder& builder, OperationState& result, int64_t size) {
  auto resultType =
      RankedTensorType::get({size}, qco::QubitType::get(builder.getContext()));
  build(builder, result, resultType,
        IntegerAttr::get(builder.getIntegerType(64), size));
}
