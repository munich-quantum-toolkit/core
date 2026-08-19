/*
 * Copyright (c) 2026 Chair for Design Automation, TUM
 * Copyright (c) 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

#include <llvm/ADT/DenseMap.h>
#include <mlir/IR/Value.h>

namespace mlir {
class Operation;
class Region;
class RewritePatternSet;
class TypeConverter;

namespace cbit {

/** Tracks the current tensor value for each CBit register during conversion. */
class CBitToTensorState {
public:
  /// Resolves a tensor alias to its source CBit register.
  [[nodiscard]] Value resolveRegister(Value regOrAlias) const;

  /// Returns the source register if @p value denotes a CBit register.
  [[nodiscard]] Value findRegister(Value value) const;

  /// Returns the current tensor value for @p reg at @p anchor.
  [[nodiscard]] Value getCurrentRegister(Value reg, Operation* anchor) const;

  /// Records @p tensor as the current value of @p reg at @p anchor.
  void setCurrentRegister(Value reg, Value tensor, Operation* anchor);

  /// Records @p tensor as the current value of @p reg in @p region.
  void setCurrentRegister(Value reg, Value tensor, Region* region);

  /// Records @p tensor as an alias for @p reg.
  void addRegisterAlias(Value tensor, Value reg);

  /// Returns the register represented by @p tensor, if any.
  [[nodiscard]] Value getRegisterForAlias(Value tensor) const;

  /// Returns the current register values recorded for @p region.
  [[nodiscard]] DenseMap<Value, Value>* getRegionRegisters(Region* region);

  /// Records the source register operands of CBit loads and stores under root.
  void recordRegisterUses(Operation* root);

  /// Returns the source register recorded for @p operation, if any.
  [[nodiscard]] Value getRecordedRegister(Operation* operation) const;

private:
  DenseMap<Region*, DenseMap<Value, Value>> registerTensors;
  DenseMap<Value, Value> registerAliases;
  DenseMap<Operation*, Value> operationRegisters;
};

/// Adds the `!cbit.reg<N>` to `tensor<Nxi1>` type conversion.
void addCBitToTensorTypeConversion(TypeConverter& typeConverter);

/// Adds CBit allocation, load, and store conversion patterns.
void populateCBitToTensorConversionPatterns(TypeConverter& typeConverter,
                                            RewritePatternSet& patterns,
                                            CBitToTensorState& state);

} // namespace cbit
} // namespace mlir
