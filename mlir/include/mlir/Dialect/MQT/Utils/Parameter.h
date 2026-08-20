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

#include "mlir/Support/MQT/ConstantFolding.h"

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

#include <cstdint>
#include <optional>
#include <variant>

namespace mlir::mqt {

inline Value constantFromScalar(OpBuilder& builder, Location loc,
                                const double value) {
  return arith::ConstantOp::create(builder, loc,
                                   builder.getF64FloatAttr(value));
}

inline Value constantFromScalar(OpBuilder& builder, Location loc,
                                const int64_t value) {
  return arith::ConstantOp::create(builder, loc, builder.getIndexAttr(value));
}

inline Value constantFromScalar(OpBuilder& builder, Location loc,
                                const bool value) {
  return arith::ConstantOp::create(builder, loc, builder.getBoolAttr(value));
}

/// Convert a scalar or existing SSA value to an SSA value.
template <typename T>
[[nodiscard]] inline Value
variantToValue(OpBuilder& builder, Location loc,
               const std::variant<T, Value>& parameter) {
  if (const auto* value = std::get_if<Value>(&parameter)) {
    return *value;
  }
  return constantFromScalar(builder, loc, std::get<T>(parameter));
}

/// Verify that each statically known floating-point parameter is finite.
[[nodiscard]] inline LogicalResult
verifyFiniteConstantParameters(Operation* operation,
                               const ValueRange parameters) {
  DenseMap<Value, std::optional<Attribute>> constantCache;
  DenseSet<Value> visited;
  for (const auto [index, parameter] : llvm::enumerate(parameters)) {
    SmallVector<Value> worklist{parameter};
    while (!worklist.empty()) {
      const Value value = worklist.pop_back_val();
      if (!visited.insert(value).second) {
        continue;
      }
      if (const auto constant = valueToConstantAttr(value, constantCache)) {
        if (const auto floating = dyn_cast<FloatAttr>(*constant);
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
