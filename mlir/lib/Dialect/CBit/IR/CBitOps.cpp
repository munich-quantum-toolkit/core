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
#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/TypeSwitch.h> // IWYU pragma: keep
#include <llvm/Support/ErrorHandling.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Utils/StaticValueUtils.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/DialectImplementation.h> // IWYU pragma: keep
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

#include <cstdint>
#include <optional>
#include <utility>
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

static arith::CmpIPredicate
swapPredicate(const arith::CmpIPredicate predicate) {
  switch (predicate) {
  case arith::CmpIPredicate::eq:
  case arith::CmpIPredicate::ne:
    return predicate;
  case arith::CmpIPredicate::slt:
    return arith::CmpIPredicate::sgt;
  case arith::CmpIPredicate::sle:
    return arith::CmpIPredicate::sge;
  case arith::CmpIPredicate::sgt:
    return arith::CmpIPredicate::slt;
  case arith::CmpIPredicate::sge:
    return arith::CmpIPredicate::sle;
  case arith::CmpIPredicate::ult:
    return arith::CmpIPredicate::ugt;
  case arith::CmpIPredicate::ule:
    return arith::CmpIPredicate::uge;
  case arith::CmpIPredicate::ugt:
    return arith::CmpIPredicate::ult;
  case arith::CmpIPredicate::uge:
    return arith::CmpIPredicate::ule;
  }
  llvm_unreachable("unknown integer comparison predicate");
}

static std::optional<llvm::APInt> integerConstant(Value value) {
  auto constant = value.getDefiningOp<arith::ConstantOp>();
  auto attribute =
      constant ? dyn_cast<IntegerAttr>(constant.getValue()) : IntegerAttr{};
  if (!attribute) {
    return std::nullopt;
  }
  return attribute.getValue();
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

struct FoldUntouchedZeroComparison final : OpRewritePattern<CompareOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(CompareOp compare,
                                PatternRewriter& rewriter) const override {
    auto alloc = compare.getReg().getDefiningOp<AllocOp>();
    if (!alloc || alloc.getInitialization() != Initialization::Zero ||
        alloc->getBlock() != compare->getBlock() ||
        !alloc->isBeforeInBlock(compare)) {
      return failure();
    }
    for (auto* user : compare.getReg().getUsers()) {
      if (!isa<LoadOp, ReadOp, CompareOp>(user)) {
        auto* ancestor = compare->getBlock()->findAncestorOpInBlock(*user);
        if (ancestor != nullptr && ancestor->isBeforeInBlock(compare)) {
          return failure();
        }
      }
    }
    const auto result = arith::applyCmpPredicate(
        compare.getPredicate(), llvm::APInt(compare.getRhs().getBitWidth(), 0),
        compare.getRhs());
    rewriter.replaceOpWithNewOp<arith::ConstantIntOp>(compare, result, 1);
    return success();
  }
};

struct CanonicalizeReadComparison final : OpRewritePattern<arith::CmpIOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::CmpIOp compare,
                                PatternRewriter& rewriter) const override {
    Value candidate = compare.getLhs();
    auto expected = integerConstant(compare.getRhs());
    auto predicate = compare.getPredicate();
    if (!expected) {
      expected = integerConstant(compare.getLhs());
      candidate = compare.getRhs();
      predicate = swapPredicate(predicate);
    }
    if (!expected) {
      return failure();
    }

    while (auto extension = candidate.getDefiningOp<arith::ExtUIOp>()) {
      if (predicate != arith::CmpIPredicate::eq &&
          predicate != arith::CmpIPredicate::ne &&
          getUnsignedPredicate(predicate) != predicate) {
        return failure();
      }
      const auto width =
          cast<IntegerType>(extension.getIn().getType()).getWidth();
      if (expected->getActiveBits() > width) {
        return failure();
      }
      candidate = extension.getIn();
      *expected = expected->trunc(width);
    }

    auto read = candidate.getDefiningOp<ReadOp>();
    if (!read) {
      auto bias = candidate.getDefiningOp<arith::XOrIOp>();
      if (!bias) {
        return failure();
      }
      read = bias.getLhs().getDefiningOp<ReadOp>();
      auto mask = integerConstant(bias.getRhs());
      if (!read) {
        read = bias.getRhs().getDefiningOp<ReadOp>();
        mask = integerConstant(bias.getLhs());
      }
      const auto width = expected->getBitWidth();
      if (!read || !mask || *mask != llvm::APInt::getSignMask(width)) {
        return failure();
      }
      switch (predicate) {
      case arith::CmpIPredicate::eq:
      case arith::CmpIPredicate::ne:
        break;
      case arith::CmpIPredicate::ult:
        predicate = arith::CmpIPredicate::slt;
        break;
      case arith::CmpIPredicate::ule:
        predicate = arith::CmpIPredicate::sle;
        break;
      case arith::CmpIPredicate::ugt:
        predicate = arith::CmpIPredicate::sgt;
        break;
      case arith::CmpIPredicate::uge:
        predicate = arith::CmpIPredicate::sge;
        break;
      default:
        return failure();
      }
      expected->flipBit(width - 1U);
    }

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointAfter(read);
    auto rhs = rewriter.getIntegerAttr(read.getResult().getType(), *expected);
    auto replacement =
        CompareOp::create(rewriter, compare.getLoc(), rewriter.getI1Type(),
                          predicate, read.getReg(), rhs);
    rewriter.replaceOp(compare, replacement.getResult());
    return success();
  }
};

} // namespace

LogicalResult LoadOp::verify() {
  return verifyIndex(getOperation(), getReg(), getIndex());
}

LogicalResult ReadOp::verify() {
  if (std::cmp_not_equal(getResult().getType().getWidth(),
                         getReg().getType().getWidth())) {
    return emitOpError("result width must match register width");
  }
  return success();
}

void ReadOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                         MLIRContext* context) {
  results.add<CanonicalizeReadComparison>(context);
}

LogicalResult WriteOp::verify() {
  if (std::cmp_not_equal(getValue().getType().getWidth(),
                         getReg().getType().getWidth())) {
    return emitOpError("value width must match register width");
  }
  return success();
}

LogicalResult CompareOp::verify() {
  if (std::cmp_not_equal(getRhs().getBitWidth(),
                         getReg().getType().getWidth())) {
    return emitOpError("expected integer width must match register width");
  }
  return success();
}

Value mlir::cbit::buildRead(OpBuilder& builder, const Location location,
                            const unsigned width,
                            const llvm::function_ref<Value(int64_t)> loadBit) {
  assert(width > 0);
  if (width == 1) {
    return loadBit(0);
  }

  const auto type = builder.getIntegerType(width);
  Value result = arith::ExtUIOp::create(builder, location, type, loadBit(0));
  for (unsigned index = 1; index < width; ++index) {
    Value bit = arith::ExtUIOp::create(builder, location, type, loadBit(index));
    auto shift = arith::ConstantIntOp::create(builder, location, type, index);
    bit = arith::ShLIOp::create(builder, location, bit, shift);
    result = arith::OrIOp::create(builder, location, result, bit);
  }
  return result;
}

void mlir::cbit::buildWrite(
    OpBuilder& builder, const Location location, Value value,
    const unsigned width,
    const llvm::function_ref<void(int64_t, Value)> storeBit) {
  assert(width > 0);
  const auto type = builder.getIntegerType(width);
  for (unsigned index = 0; index < width; ++index) {
    Value selected = value;
    if (index != 0) {
      auto shift = arith::ConstantIntOp::create(builder, location, type, index);
      selected = arith::ShRUIOp::create(builder, location, value, shift);
    }
    if (width != 1) {
      selected = arith::TruncIOp::create(builder, location, builder.getI1Type(),
                                         selected);
    }
    storeBit(index, selected);
  }
}

arith::CmpIPredicate
mlir::cbit::getUnsignedPredicate(const arith::CmpIPredicate predicate) {
  switch (predicate) {
  case arith::CmpIPredicate::slt:
    return arith::CmpIPredicate::ult;
  case arith::CmpIPredicate::sle:
    return arith::CmpIPredicate::ule;
  case arith::CmpIPredicate::sgt:
    return arith::CmpIPredicate::ugt;
  case arith::CmpIPredicate::sge:
    return arith::CmpIPredicate::uge;
  default:
    return predicate;
  }
}

bool mlir::cbit::isRegisterBitVector(Value value) {
  SmallVector<Value, 8> worklist{value};
  llvm::SmallPtrSet<Operation*, 16> visited;
  while (!worklist.empty()) {
    auto* operation = worklist.pop_back_val().getDefiningOp();
    if (operation == nullptr || !visited.insert(operation).second) {
      continue;
    }
    if (isa<ReadOp>(operation)) {
      return true;
    }
    const auto name = operation->getName().getStringRef();
    const bool rotation =
        (name == "llvm.intr.fshl" || name == "llvm.intr.fshr") &&
        operation->getNumOperands() == 3 &&
        operation->getOperand(0) == operation->getOperand(1);
    if (isa<arith::ShLIOp, arith::ShRUIOp>(operation) || rotation) {
      worklist.push_back(operation->getOperand(0));
    } else if (isa<arith::AndIOp, arith::OrIOp, arith::XOrIOp>(operation)) {
      llvm::append_range(worklist, operation->getOperands());
    }
  }
  return false;
}

Value mlir::cbit::buildComparison(
    OpBuilder& builder, const Location location,
    const arith::CmpIPredicate predicate, const llvm::APInt& rhs,
    const llvm::function_ref<Value(int64_t)> loadBit) {
  const auto encodedPredicate = getUnsignedPredicate(predicate);
  auto encodedRhs = rhs;
  const bool biasSignBit = encodedPredicate != predicate;
  if (biasSignBit) {
    encodedRhs.flipBit(encodedRhs.getBitWidth() - 1U);
  }

  auto one = arith::ConstantIntOp::create(builder, location, 1, 1);
  Value equal = one;
  Value less;
  if (encodedPredicate != arith::CmpIPredicate::eq &&
      encodedPredicate != arith::CmpIPredicate::ne) {
    less = arith::ConstantIntOp::create(builder, location, 0, 1);
  }
  for (int64_t index = static_cast<int64_t>(encodedRhs.getBitWidth()) - 1;
       index >= 0; --index) {
    auto bit = loadBit(index);
    if (biasSignBit &&
        index == static_cast<int64_t>(encodedRhs.getBitWidth()) - 1) {
      bit = arith::XOrIOp::create(builder, location, bit, one);
    }
    Value matches = bit;
    if (!encodedRhs[static_cast<unsigned>(index)]) {
      matches = arith::XOrIOp::create(builder, location, bit, one);
    } else if (less) {
      auto lower = arith::XOrIOp::create(builder, location, bit, one);
      auto firstDifference =
          arith::AndIOp::create(builder, location, equal, lower);
      less = arith::OrIOp::create(builder, location, less, firstDifference);
    }
    equal = arith::AndIOp::create(builder, location, equal, matches);
  }
  switch (encodedPredicate) {
  case arith::CmpIPredicate::eq:
    return equal;
  case arith::CmpIPredicate::ne:
    return arith::XOrIOp::create(builder, location, equal, one);
  case arith::CmpIPredicate::ult:
    return less;
  case arith::CmpIPredicate::ule:
    return arith::OrIOp::create(builder, location, less, equal);
  case arith::CmpIPredicate::ugt: {
    auto lessOrEqual = arith::OrIOp::create(builder, location, less, equal);
    return arith::XOrIOp::create(builder, location, lessOrEqual, one);
  }
  case arith::CmpIPredicate::uge:
    return arith::XOrIOp::create(builder, location, less, one);
  default:
    llvm_unreachable("signed CBit predicate must be encoded as unsigned");
  }
}

void LoadOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                         MLIRContext* context) {
  results.add<ForwardKnownLoad>(context);
}

void CompareOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                            MLIRContext* context) {
  results.add<FoldUntouchedZeroComparison>(context);
}

LogicalResult StoreOp::verify() {
  return verifyIndex(getOperation(), getReg(), getIndex());
}

#define GET_OP_CLASSES
#include "mlir/Dialect/CBit/IR/CBitOps.cpp.inc"
