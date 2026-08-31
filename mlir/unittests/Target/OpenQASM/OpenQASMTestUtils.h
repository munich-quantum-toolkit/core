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

#include "mlir/Dialect/CBit/IR/CBitAttributes.h"
#include "mlir/Dialect/CBit/IR/CBitDialect.h"
#include "mlir/Dialect/CBit/IR/CBitOps.h"
#include "mlir/Dialect/QC/Translation/TranslateQASM3ToQC.h"

#include <gtest/gtest.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Matchers.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/Passes.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace mlir::oq3::test {

inline constexpr llvm::StringLiteral BROADCAST_PROGRAM = R"qasm(
OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
h q;
bit[2] c = measure q;
)qasm";

inline std::optional<APInt> evaluateConstantInteger(Value value) {
  APInt constant;
  if (matchPattern(value, m_ConstantInt(&constant))) {
    return constant;
  }
  auto* operation = value.getDefiningOp();
  if (operation == nullptr || operation->getNumOperands() == 0) {
    return std::nullopt;
  }
  const auto operand = [&](const unsigned index) {
    return evaluateConstantInteger(operation->getOperand(index));
  };
  const auto integerType = dyn_cast<IntegerType>(value.getType());
  if (!integerType && !value.getType().isIndex()) {
    return std::nullopt;
  }
  const auto width = integerType ? integerType.getWidth() : 64U;
  if (isa<arith::TruncIOp, arith::IndexCastOp>(operation)) {
    auto input = operand(0);
    if (!input) {
      return std::nullopt;
    }
    if (input->getBitWidth() == width) {
      return input;
    }
    return input->getBitWidth() > width ? std::optional(input->trunc(width))
                                        : std::optional(input->sext(width));
  }
  if (isa<arith::ExtUIOp>(operation)) {
    const auto input = operand(0);
    return input ? std::optional(input->zext(width)) : std::nullopt;
  }
  auto lhs = operand(0);
  if (!lhs) {
    return std::nullopt;
  }
  if (isa<math::CtPopOp>(operation)) {
    return APInt(width, lhs->popcount());
  }
  if (isa<arith::SelectOp>(operation)) {
    return evaluateConstantInteger(
        operation->getOperand(lhs->isZero() ? 2 : 1));
  }
  const auto rhs = operand(1);
  if (!rhs) {
    return std::nullopt;
  }
  if (isa<arith::AddIOp>(operation)) {
    return *lhs + *rhs;
  }
  if (isa<arith::RemSIOp>(operation)) {
    return lhs->srem(*rhs);
  }
  if (isa<arith::ShLIOp>(operation)) {
    return lhs->shl(rhs->getLimitedValue());
  }
  if (isa<arith::ShRUIOp>(operation)) {
    return lhs->lshr(rhs->getLimitedValue());
  }
  if (isa<arith::OrIOp>(operation)) {
    return *lhs | *rhs;
  }
  if (isa<LLVM::FshlOp, LLVM::FshrOp>(operation)) {
    const auto shift = evaluateConstantInteger(operation->getOperand(2));
    if (!shift) {
      return std::nullopt;
    }
    const auto amount = shift->urem(APInt(shift->getBitWidth(), width));
    const auto distance = amount.getLimitedValue();
    if (distance == 0) {
      return lhs;
    }
    if (isa<LLVM::FshlOp>(operation)) {
      return lhs->shl(distance) | rhs->lshr(width - distance);
    }
    return lhs->lshr(distance) | rhs->shl(width - distance);
  }
  return std::nullopt;
}

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
      const auto index = evaluateConstantInteger(store.getIndex());
      if (!index || index->getActiveBits() > 63) {
        return;
      }
      const auto position = static_cast<size_t>(index->getZExtValue());
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

inline std::vector<bool> canonicalizedBitOutputs(const StringRef source) {
  MLIRContext context;
  auto moduleOp = qc::translateQASM3ToQC(source, &context);
  if (!moduleOp) {
    ADD_FAILURE() // NOLINT(readability-implicit-bool-conversion)
        << "translation failed";
    return {};
  }
  if (failed(verify(*moduleOp))) {
    ADD_FAILURE() // NOLINT(readability-implicit-bool-conversion)
        << "translation produced an invalid module";
    return {};
  }
  PassManager canonicalizer(&context);
  canonicalizer.addPass(createCanonicalizerPass());
  if (failed(canonicalizer.run(*moduleOp))) {
    ADD_FAILURE() // NOLINT(readability-implicit-bool-conversion)
        << "canonicalization failed";
    return {};
  }

  const auto returned = returnedBitValues(*moduleOp);
  std::vector<bool> outputs;
  outputs.reserve(returned.size());
  for (const auto operand : returned) {
    if (!operand) {
      outputs.push_back(false);
      continue;
    }
    const auto value = evaluateConstantInteger(*operand);
    if (!value) {
      std::string description;
      llvm::raw_string_ostream stream(description);
      operand->print(stream);
      ADD_FAILURE() << "canonicalized output is not constant: " << description;
      return {};
    }
    outputs.push_back(!value->isZero());
  }
  return outputs;
}

inline std::vector<bool> rotateBits(const std::array<bool, 5>& bits,
                                    const int64_t distance, const bool left) {
  constexpr int64_t width = 5;
  auto normalized = distance % width;
  if (normalized < 0) {
    normalized += width;
  }
  std::vector<bool> result(width);
  for (int64_t bit = 0; bit < width; ++bit) {
    const auto source =
        left ? (bit + width - normalized) % width : (bit + normalized) % width;
    result[bit] = bits[static_cast<size_t>(source)];
  }
  return result;
}

} // namespace mlir::oq3::test
