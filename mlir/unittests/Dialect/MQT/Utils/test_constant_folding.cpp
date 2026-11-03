/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/Dialect/MQT/Utils/ConstantFolding.h"
#include "mqt/Dialect/MQT/Utils/Parameters.h"

#include "gtest/gtest.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/DenseMap.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>

using namespace mlir;

namespace {

class ConstantFoldingTest : public ::testing::Test {
protected:
  MLIRContext context;
  OwningOpRef<ModuleOp> module;
  std::unique_ptr<ImplicitLocOpBuilder> builder;

  void SetUp() override {
    context.loadDialect<arith::ArithDialect, func::FuncDialect,
                        index::IndexDialect>();

    auto loc = FileLineColLoc::get(&context, "<utils-test-builder>", 1, 1);
    module = ModuleOp::create(loc);
    builder = std::make_unique<ImplicitLocOpBuilder>(loc, &context);
    builder->setInsertionPointToStart(module->getBody());
  }
};

} // namespace

TEST_F(ConstantFoldingTest, valueToDoubleLiteralConversions) {
  for (const auto& [attribute, expected] : {
           std::pair<TypedAttr, double>{builder->getF64FloatAttr(1.234), 1.234},
           {builder->getI32IntegerAttr(42), 42.0},
           {builder->getSI32IntegerAttr(-123), -123.0},
           {
               builder->getIntegerAttr(builder->getIntegerType(64, false),
                                       APInt::getMaxValue(64)),
               static_cast<double>(std::numeric_limits<uint64_t>::max()),
           },
           {
               builder->getIntegerAttr(builder->getIntegerType(128, true),
                                       APInt::getAllOnes(128)),
               -1.0,
           },
           {
               builder->getIntegerAttr(builder->getIntegerType(128, false),
                                       APInt::getOneBitSet(128, 127)),
               std::ldexp(1.0, 127),
           },
       }) {
    SCOPED_TRACE(expected);
    auto op = arith::ConstantOp::create(*builder, attribute);
    const auto value = mlir::mqt::valueToDouble(op.getResult());
    ASSERT_TRUE(value);
    EXPECT_DOUBLE_EQ(*value, expected);
  }
  auto op = index::ConstantOp::create(*builder, 42);
  EXPECT_EQ(mlir::mqt::valueToDouble(op.getResult()), std::optional{42.0});
}

TEST_F(ConstantFoldingTest, valueToDoubleWrongType) {
  EXPECT_FALSE(mlir::mqt::attributeToDouble({}));
  auto op = arith::ConstantOp::create(*builder, builder->getStringAttr("test"));
  EXPECT_FALSE(mlir::mqt::valueToDouble(op.getResult()).has_value());
}

TEST_F(ConstantFoldingTest, valueToDoubleNonStaticValue) {
  auto lhs = arith::ConstantOp::create(*builder, builder->getF64FloatAttr(9.5));
  auto rhs =
      arith::ConstantOp::create(*builder, builder->getF64FloatAttr(21.5));
  auto op = arith::AddFOp::create(*builder, lhs, rhs);
  EXPECT_FALSE(mlir::mqt::valueToDouble(op.getResult()).has_value());
}

TEST_F(ConstantFoldingTest, valueToConstantAttrPreservesLiteralAttributes) {
  auto floatAttr = builder->getF64FloatAttr(-0.0);
  auto floatOp = arith::ConstantOp::create(*builder, floatAttr);
  EXPECT_EQ(mlir::mqt::valueToConstantAttr(floatOp.getResult()), floatAttr);

  auto indexOp = index::ConstantOp::create(*builder, 42);
  EXPECT_EQ(mlir::mqt::valueToConstantAttr(indexOp.getResult()),
            builder->getIndexAttr(42));
}

TEST_F(ConstantFoldingTest, valueToConstantDoubleNestedFold) {
  auto lhs = arith::ConstantOp::create(*builder, builder->getF64FloatAttr(5.0));
  auto num = arith::ConstantOp::create(*builder, builder->getF64FloatAttr(1.0));
  auto den = arith::ConstantOp::create(*builder, builder->getF64FloatAttr(2.0));
  auto quot = arith::DivFOp::create(*builder, num, den);
  auto op = arith::SubFOp::create(*builder, lhs, quot);
  const auto stdValue = mlir::mqt::valueToConstantDouble(op.getResult());
  ASSERT_TRUE(stdValue.has_value());
  EXPECT_DOUBLE_EQ(*stdValue, 4.5);
}

TEST_F(ConstantFoldingTest, valueToConstantDoubleNegF) {
  auto operand =
      arith::ConstantOp::create(*builder, builder->getF64FloatAttr(2.25));
  auto op = arith::NegFOp::create(*builder, operand);
  const auto stdValue = mlir::mqt::valueToConstantDouble(op.getResult());
  ASSERT_TRUE(stdValue.has_value());
  EXPECT_DOUBLE_EQ(*stdValue, -2.25);
}

TEST_F(ConstantFoldingTest, valueToConstantDoubleUIToFP) {
  constexpr uint64_t expectedValue = 7;
  auto intConst = arith::ConstantOp::create(
      *builder, builder->getIntegerAttr(builder->getIntegerType(64, false),
                                        expectedValue));
  auto op = arith::UIToFPOp::create(*builder, builder->getF64Type(),
                                    intConst.getResult());
  const auto stdValue = mlir::mqt::valueToConstantDouble(op.getResult());
  ASSERT_TRUE(stdValue.has_value());
  EXPECT_DOUBLE_EQ(*stdValue, static_cast<double>(expectedValue));
}

TEST_F(ConstantFoldingTest, valueToConstantDoubleSIToFP) {
  constexpr int64_t expectedValue = -7;
  auto intConst = arith::ConstantOp::create(
      *builder,
      builder->getIntegerAttr(builder->getIntegerType(64, /*isSigned=*/true),
                              expectedValue));
  auto op = arith::SIToFPOp::create(*builder, builder->getF64Type(),
                                    intConst.getResult());
  const auto stdValue = mlir::mqt::valueToConstantDouble(op.getResult());
  ASSERT_TRUE(stdValue.has_value());
  EXPECT_DOUBLE_EQ(*stdValue, static_cast<double>(expectedValue));
}

TEST_F(ConstantFoldingTest, valueToConstantAttrDynamicOperand) {
  auto func =
      func::FuncOp::create(*builder, "dyn",
                           FunctionType::get(&context, {builder->getF64Type()},
                                             {builder->getF64Type()}));
  auto* entry = func.addEntryBlock();
  OpBuilder::InsertionGuard guard(*builder);
  builder->setInsertionPointToStart(entry);
  auto lhs = arith::ConstantOp::create(*builder, builder->getF64FloatAttr(5.0));
  auto dynamic = entry->getArgument(0);
  EXPECT_FALSE(mlir::mqt::valueToConstantDouble(dynamic));
  for (const bool dynamicFirst : {false, true}) {
    Value left = dynamicFirst ? dynamic : lhs.getResult();
    Value right = dynamicFirst ? lhs.getResult() : dynamic;
    auto op = arith::SubFOp::create(*builder, left, right);
    llvm::DenseMap<Value, Attribute> cache;
    EXPECT_FALSE(mlir::mqt::valueToConstantAttr(op.getResult(), cache));
    EXPECT_TRUE(cache.contains(op.getResult()));
    EXPECT_TRUE(cache.contains(dynamic));
    EXPECT_EQ(cache.contains(lhs), !dynamicFirst);
    EXPECT_FALSE(mlir::mqt::valueToConstantAttr(op.getResult(), cache));
    EXPECT_EQ(mlir::mqt::valueToConstantAttr(lhs, cache), lhs.getValue());
  }
}

TEST_F(ConstantFoldingTest, valueToConstantAttrFoldFailure) {
  // Pure op whose fold fails (shift >= bitwidth).
  auto lhs = arith::ConstantOp::create(*builder, builder->getI32IntegerAttr(1));
  auto rhs =
      arith::ConstantOp::create(*builder, builder->getI32IntegerAttr(32));
  auto op = arith::ShLIOp::create(*builder, lhs, rhs);
  EXPECT_FALSE(mlir::mqt::valueToConstantAttr(op.getResult()));
}

TEST_F(ConstantFoldingTest, valueToConstantAttrMultiResultFold) {
  // addui_extended folds to two results; valueToConstantAttr requires exactly
  // one OpFoldResult.
  auto lhs = arith::ConstantOp::create(*builder, builder->getI32IntegerAttr(1));
  auto rhs = arith::ConstantOp::create(*builder, builder->getI32IntegerAttr(2));
  auto op = arith::AddUIExtendedOp::create(*builder, lhs, rhs);
  EXPECT_FALSE(mlir::mqt::valueToConstantAttr(op.getSum()));
}

TEST_F(ConstantFoldingTest, valueToConstantAttrIdentityFold) {
  // select(true, x, y) folds to an existing SSA value, not an Attribute.
  constexpr double expectedValue = 3.25;
  auto cond = arith::ConstantOp::create(*builder, builder->getBoolAttr(true));
  auto x = arith::ConstantOp::create(*builder,
                                     builder->getF64FloatAttr(expectedValue));
  auto y = arith::ConstantOp::create(*builder, builder->getF64FloatAttr(9.0));
  auto op = arith::SelectOp::create(*builder, cond, x, y);
  auto attr = mlir::mqt::valueToConstantAttr(op.getResult());
  ASSERT_TRUE(attr);
  EXPECT_DOUBLE_EQ(*mlir::mqt::attributeToDouble(attr), expectedValue);
}

TEST_F(ConstantFoldingTest, VerifyFiniteParametersChecksDirectConstants) {
  auto finite =
      arith::ConstantOp::create(*builder, builder->getF64FloatAttr(1.0));
  auto nan = arith::ConstantOp::create(
      *builder,
      builder->getF64FloatAttr(std::numeric_limits<double>::quiet_NaN()));
  std::string diagnostic;
  ScopedDiagnosticHandler handler(&context, [&](Diagnostic& emitted) {
    diagnostic = emitted.str();
    return success();
  });
  EXPECT_TRUE(succeeded(mlir::mqt::verifyFiniteConstantParameters(
      module->getOperation(), {finite.getResult()})));
  EXPECT_TRUE(diagnostic.empty());
  EXPECT_TRUE(failed(mlir::mqt::verifyFiniteConstantParameters(
      module->getOperation(), {finite.getResult(), nan.getResult()})));
  EXPECT_EQ(diagnostic, "'builtin.module' op constant parameter expression at "
                        "index 1 must be finite");
}

TEST_F(ConstantFoldingTest, valueToConstantAttrDeepSharedExpressions) {
  auto func =
      func::FuncOp::create(*builder, "dyn",
                           FunctionType::get(&context, {builder->getF64Type()},
                                             {builder->getF64Type()}));
  auto* entry = func.addEntryBlock();
  OpBuilder::InsertionGuard guard(*builder);
  builder->setInsertionPointToStart(entry);
  Value one =
      arith::ConstantOp::create(*builder, builder->getF64FloatAttr(1.0));
  for (const bool dynamic : {false, true}) {
    SCOPED_TRACE(dynamic);
    Value value = dynamic ? entry->getArgument(0) : one;
    // Shared operands require memoization; the linear tail exceeds the stack.
    for (int i = 0; i < 40; ++i) {
      value = arith::AddFOp::create(*builder, value, value);
    }
    for (int i = 0; i < 64000; ++i) {
      value = arith::AddFOp::create(*builder, value, one);
    }
    llvm::DenseMap<Value, Attribute> cache;
    auto folded = mlir::mqt::valueToConstantAttr(value, cache);
    EXPECT_TRUE(cache.contains(value));
    EXPECT_EQ(mlir::mqt::valueToConstantAttr(value, cache), folded);
    if (dynamic) {
      EXPECT_FALSE(folded);
    } else {
      ASSERT_TRUE(folded);
      EXPECT_EQ(mlir::mqt::attributeToDouble(folded),
                std::ldexp(1.0, 40) + 64000.0);
    }
  }
}
