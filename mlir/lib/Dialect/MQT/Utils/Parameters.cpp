/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/MQT/Utils/Parameters.h"

#include "mlir/Dialect/MQT/Utils/ConstantFolding.h"

#include <llvm/ADT/STLExtras.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

#include <cstdint>
#include <optional>

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
  DenseMap<Value, std::optional<Attribute>> constantCache;
  DenseSet<Value> visited;
  for (const auto [index, parameter] : llvm::enumerate(parameters)) {
    SmallVector<Value> worklist{parameter};
    while (!worklist.empty()) {
      auto value = worklist.pop_back_val();
      if (!visited.insert(value).second) {
        continue;
      }
      if (const auto constant = valueToConstantAttr(value, constantCache)) {
        if (auto floating = dyn_cast<FloatAttr>(*constant);
            floating && !floating.getValue().isFinite()) {
          return operation->emitOpError()
                 << "constant parameter expression at index " << index
                 << " must be finite";
        }
      }
      Operation* definingOp = value.getDefiningOp();
      if (definingOp == nullptr || definingOp->getNumRegions() != 0 ||
          !isPure(definingOp)) {
        continue;
      }
      llvm::append_range(worklist, definingOp->getOperands());
    }
  }
  return success();
}

} // namespace mlir::mqt
