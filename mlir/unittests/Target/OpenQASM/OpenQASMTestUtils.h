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

#include "mqt/Dialect/CBit/IR/CBitAttributes.h"
#include "mqt/Dialect/CBit/IR/CBitOps.h"

#include "gtest/gtest.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"

#include <cstddef>
#include <optional>

namespace mlir::openqasm::test {

inline constexpr llvm::StringLiteral BROADCAST_PROGRAM = R"qasm(
OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
h q;
bit[2] c = measure q;
)qasm";

inline SmallVector<std::optional<Value>> returnedBitValues(ModuleOp moduleOp) {
  func::ReturnOp result;
  moduleOp.walk([&](func::ReturnOp operation) { result = operation; });
  if (!result) {
    ADD_FAILURE() // NOLINT(readability-implicit-bool-conversion)
        << "translated module has no return operation";
    return {};
  }

  SmallVector<std::optional<Value>> values;
  for (auto operand : result.getOperands()) {
    const auto registerType = dyn_cast<cbit::RegisterType>(operand.getType());
    auto allocation = operand.getDefiningOp<cbit::AllocOp>();
    if (!registerType || !allocation) {
      ADD_FAILURE() << "classical bit output is not a direct CBit allocation";
      return {};
    }
    SmallVector<std::optional<Value>> registerValues(
        static_cast<size_t>(registerType.getWidth()));
    moduleOp.walk([&](cbit::StoreOp store) {
      if (store.getReg() != operand) {
        return;
      }
      APInt index;
      if (!matchPattern(store.getIndex(), m_ConstantInt(&index)) ||
          index.getActiveBits() > 63) {
        return;
      }
      const auto position = static_cast<size_t>(index.getZExtValue());
      if (position < registerValues.size()) {
        registerValues[position] = store.getValue();
      }
    });
    if (allocation.getInitialization() == cbit::Initialization::Undefined &&
        llvm::any_of(registerValues,
                     [](const auto& value) { return !value.has_value(); })) {
      ADD_FAILURE() << "undefined classical bit output is not fully stored";
      return {};
    }
    llvm::append_range(values, registerValues);
  }
  return values;
}

} // namespace mlir::openqasm::test
