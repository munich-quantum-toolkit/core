/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/Dialect/MQT/Utils/Parameters.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/STLExtras.h"

#include <cstdint>

namespace mlir::mqt {

Value constantFromScalar(OpBuilder& builder, Location loc, const double value) {
  return arith::ConstantOp::create(builder, loc,
                                   builder.getF64FloatAttr(value));
}

Value constantFromScalar(OpBuilder& builder, Location loc,
                         const int64_t value) {
  return arith::ConstantOp::create(builder, loc, builder.getIndexAttr(value));
}

Value constantFromScalar(OpBuilder& builder, Location loc, const bool value) {
  return arith::ConstantOp::create(builder, loc, builder.getBoolAttr(value));
}

LogicalResult verifyFiniteConstantParameters(Operation* operation,
                                             ValueRange parameters) {
  for (auto [index, parameter] : llvm::enumerate(parameters)) {
    Attribute constant;
    if (matchPattern(parameter, m_Constant(&constant))) {
      if (auto floating = dyn_cast<FloatAttr>(constant);
          floating && !floating.getValue().isFinite()) {
        return operation->emitOpError()
               << "constant parameter expression at index " << index
               << " must be finite";
      }
    }
  }
  return success();
}

} // namespace mlir::mqt
