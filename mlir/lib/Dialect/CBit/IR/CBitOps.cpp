/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/CBit/IR/CBitOps.h"

#include "mlir/Dialect/CBit/IR/CBitAttributes.h" // IWYU pragma: associated
#include "mlir/Dialect/CBit/IR/CBitDialect.h"

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/TypeSwitch.h> // IWYU pragma: keep
#include <llvm/Support/ErrorHandling.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Utils/StaticValueUtils.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/DialectImplementation.h> // IWYU pragma: keep
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

#include <cstdint>
#include <optional>
#include <variant>

using namespace mlir;
using namespace mlir::cbit;

#include "mlir/Dialect/CBit/IR/CBitOpsDialect.cpp.inc"
#include "mlir/Dialect/CBit/IR/CBitOpsEnums.cpp.inc"

void CBitDialect::initialize() {
  // NOLINTNEXTLINE(clang-analyzer-core.StackAddressEscape)
  addAttributes<
#define GET_ATTRDEF_LIST
#include "mlir/Dialect/CBit/IR/CBitOpsAttributes.cpp.inc"

      >();

  // NOLINTNEXTLINE(clang-analyzer-core.StackAddressEscape)
  addTypes<
#define GET_TYPEDEF_LIST
#include "mlir/Dialect/CBit/IR/CBitOpsTypes.cpp.inc"

      >();

  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/CBit/IR/CBitOps.cpp.inc"

      >();
}

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/CBit/IR/CBitOpsAttributes.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/CBit/IR/CBitOpsTypes.cpp.inc"

LogicalResult
RegisterType::verify(const function_ref<InFlightDiagnostic()> emitError,
                     const int64_t width) {
  if (width <= 0) {
    return emitError() << "register width must be positive";
  }
  return success();
}

void mlir::cbit::validateStaticRegisterIndex(
    Value reg, const std::variant<int64_t, Value>& index) {
  const auto type = dyn_cast<RegisterType>(reg.getType());
  if (!type) {
    llvm::reportFatalUsageError("Expected a CBit register");
  }

  const auto* constant = std::get_if<int64_t>(&index);
  if (constant == nullptr) {
    return;
  }
  if (*constant < 0) {
    llvm::reportFatalUsageError("Register index must be non-negative");
  }
  if (*constant >= type.getWidth()) {
    llvm::reportFatalUsageError("Register index is out of bounds");
  }
}

static LogicalResult verifyIndex(Operation* operation, Value registerValue,
                                 Value indexValue) {
  const auto index = getConstantIntValue(indexValue);
  if (!index) {
    return success();
  }

  if (*index < 0) {
    return operation->emitOpError("index must be non-negative");
  }

  const auto width = cast<RegisterType>(registerValue.getType()).getWidth();
  if (*index >= width) {
    return operation->emitOpError("index exceeds register width");
  }
  return success();
}

namespace {
struct KnownLoadValue {
  Value value;
  bool isZeroInitialization = false;
};
} // namespace

static std::optional<KnownLoadValue> findKnownLoadValue(LoadOp load) {
  const auto loadIndex = getConstantIntValue(load.getIndex());
  for (auto* candidate = load->getPrevNode(); candidate != nullptr;
       candidate = candidate->getPrevNode()) {
    if (auto store = dyn_cast<StoreOp>(candidate);
        store && store.getReg() == load.getReg()) {
      if (store.getIndex() == load.getIndex()) {
        return KnownLoadValue{.value = store.getValue()};
      }
      const auto storeIndex = getConstantIntValue(store.getIndex());
      if (loadIndex && storeIndex && *loadIndex != *storeIndex) {
        continue;
      }
      return std::nullopt;
    }

    if (auto alloc = dyn_cast<AllocOp>(candidate);
        alloc && alloc.getResult() == load.getReg()) {
      if (alloc.getInitialization() == Initialization::Zero) {
        return KnownLoadValue{.isZeroInitialization = true};
      }
      return std::nullopt;
    }

    if (isa<LoadOp>(candidate)) {
      continue;
    }
    if (candidate->getNumRegions() != 0 ||
        llvm::is_contained(candidate->getOperands(), load.getReg())) {
      return std::nullopt;
    }
  }
  return std::nullopt;
}

namespace {
struct ForwardKnownLoad final : OpRewritePattern<LoadOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(LoadOp load,
                                PatternRewriter& rewriter) const override {
    const auto known = findKnownLoadValue(load);
    if (!known) {
      return failure();
    }
    if (known->value) {
      rewriter.replaceOp(load, known->value);
      return success();
    }
    rewriter.replaceOpWithNewOp<arith::ConstantIntOp>(load, false, 1);
    return success();
  }
};
} // namespace

LogicalResult LoadOp::verify() {
  return verifyIndex(getOperation(), getReg(), getIndex());
}

void LoadOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                         MLIRContext* context) {
  results.add<ForwardKnownLoad>(context);
}

LogicalResult StoreOp::verify() {
  return verifyIndex(getOperation(), getReg(), getIndex());
}

#define GET_OP_CLASSES
#include "mlir/Dialect/CBit/IR/CBitOps.cpp.inc"
