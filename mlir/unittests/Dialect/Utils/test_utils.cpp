/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/Utils/Utils.h"

#include <gtest/gtest.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>

using namespace mlir;

namespace {

class UtilsTest : public ::testing::Test {
protected:
  MLIRContext context;
  OwningOpRef<ModuleOp> module;
  std::unique_ptr<ImplicitLocOpBuilder> builder;

  void SetUp() override {
    context.loadDialect<arith::ArithDialect, func::FuncDialect>();

    auto loc = FileLineColLoc::get(&context, "<utils-test-builder>", 1, 1);
    module = ModuleOp::create(loc);
    builder = std::make_unique<ImplicitLocOpBuilder>(loc, &context);
    builder->setInsertionPointToStart(module->getBody());
  }
};

} // namespace

TEST_F(UtilsTest, valueToDouble) {
  constexpr double expectedValue = 1.234;
  auto op = arith::ConstantOp::create(*builder,
                                      builder->getF64FloatAttr(expectedValue));
  const auto stdValue = utils::valueToDouble(op.getResult());
  ASSERT_TRUE(stdValue.has_value());
  EXPECT_DOUBLE_EQ(*stdValue, expectedValue);
}

TEST_F(UtilsTest, valueToDoubleCastFromInteger) {
  auto op = arith::ConstantOp::create(*builder, builder->getI32IntegerAttr(42));
  const auto stdValue = utils::valueToDouble(op.getResult());
  ASSERT_TRUE(stdValue.has_value());
  EXPECT_DOUBLE_EQ(*stdValue, 42.0);
}

TEST_F(UtilsTest, valueToDoubleCastFromNegativeInteger) {
  auto op =
      arith::ConstantOp::create(*builder, builder->getSI32IntegerAttr(-123));
  const auto stdValue = utils::valueToDouble(op.getResult());
  ASSERT_TRUE(stdValue.has_value());
  EXPECT_DOUBLE_EQ(*stdValue, -123.0);
}

TEST_F(UtilsTest, valueToDoubleCastFromMaxUnsignedInteger) {
  constexpr auto bitCount = 64;
  auto op = arith::ConstantOp::create(
      *builder,
      builder->getIntegerAttr(builder->getIntegerType(bitCount, false),
                              APInt::getMaxValue(bitCount)));
  const auto stdValue = utils::valueToDouble(op.getResult());
  ASSERT_TRUE(stdValue.has_value());
  EXPECT_DOUBLE_EQ(*stdValue,
                   static_cast<double>(std::numeric_limits<uint64_t>::max()));
}

TEST_F(UtilsTest, valueToDoubleWrongType) {
  auto op = arith::ConstantOp::create(*builder, builder->getStringAttr("test"));
  EXPECT_FALSE(utils::valueToDouble(op.getResult()).has_value());
}

TEST_F(UtilsTest, valueToDoubleNonStaticValue) {
  auto lhs = arith::ConstantOp::create(*builder, builder->getF64FloatAttr(9.5));
  auto rhs =
      arith::ConstantOp::create(*builder, builder->getF64FloatAttr(21.5));
  auto op = arith::AddFOp::create(*builder, lhs, rhs);
  EXPECT_FALSE(utils::valueToDouble(op.getResult()).has_value());
}

TEST_F(UtilsTest, attributeToDoubleSignedI128) {
  constexpr unsigned bitWidth = 128;
  const auto attr = builder->getIntegerAttr(
      builder->getIntegerType(bitWidth, /*isSigned=*/true),
      APInt::getAllOnes(bitWidth));
  const auto asDouble = utils::attributeToDouble(attr);
  ASSERT_TRUE(asDouble.has_value());
  EXPECT_DOUBLE_EQ(*asDouble, -1.0);
}

TEST_F(UtilsTest, attributeToDoubleUnsignedI128) {
  constexpr unsigned bitWidth = 128;
  APInt bits(bitWidth, 0);
  bits.setBit(127);
  const auto attr = builder->getIntegerAttr(
      builder->getIntegerType(bitWidth, /*isSigned=*/false), bits);
  const auto asDouble = utils::attributeToDouble(attr);
  ASSERT_TRUE(asDouble.has_value());
  EXPECT_DOUBLE_EQ(*asDouble, std::ldexp(1.0, 127));
}

TEST_F(UtilsTest, valueToConstantDoubleNestedFold) {
  auto lhs = arith::ConstantOp::create(*builder, builder->getF64FloatAttr(5.0));
  auto num = arith::ConstantOp::create(*builder, builder->getF64FloatAttr(1.0));
  auto den = arith::ConstantOp::create(*builder, builder->getF64FloatAttr(2.0));
  auto quot = arith::DivFOp::create(*builder, num, den);
  auto op = arith::SubFOp::create(*builder, lhs, quot);
  const auto stdValue = utils::valueToConstantDouble(op.getResult());
  ASSERT_TRUE(stdValue.has_value());
  EXPECT_DOUBLE_EQ(*stdValue, 4.5);
}

TEST_F(UtilsTest, valueToConstantDoubleNegF) {
  auto operand =
      arith::ConstantOp::create(*builder, builder->getF64FloatAttr(2.25));
  auto op = arith::NegFOp::create(*builder, operand);
  const auto stdValue = utils::valueToConstantDouble(op.getResult());
  ASSERT_TRUE(stdValue.has_value());
  EXPECT_DOUBLE_EQ(*stdValue, -2.25);
}

TEST_F(UtilsTest, valueToConstantDoubleUIToFP) {
  constexpr uint64_t expectedValue = 7;
  auto intConst = arith::ConstantOp::create(
      *builder, builder->getIntegerAttr(builder->getIntegerType(64, false),
                                        expectedValue));
  auto op = arith::UIToFPOp::create(*builder, builder->getF64Type(),
                                    intConst.getResult());
  const auto stdValue = utils::valueToConstantDouble(op.getResult());
  ASSERT_TRUE(stdValue.has_value());
  EXPECT_DOUBLE_EQ(*stdValue, static_cast<double>(expectedValue));
}

TEST_F(UtilsTest, valueToConstantDoubleSIToFP) {
  constexpr int64_t expectedValue = -7;
  auto intConst = arith::ConstantOp::create(
      *builder,
      builder->getIntegerAttr(builder->getIntegerType(64, /*isSigned=*/true),
                              expectedValue));
  auto op = arith::SIToFPOp::create(*builder, builder->getF64Type(),
                                    intConst.getResult());
  const auto stdValue = utils::valueToConstantDouble(op.getResult());
  ASSERT_TRUE(stdValue.has_value());
  EXPECT_DOUBLE_EQ(*stdValue, static_cast<double>(expectedValue));
}

TEST_F(UtilsTest, valueToConstantDoubleDynamicOperand) {
  auto func =
      func::FuncOp::create(*builder, "dyn",
                           FunctionType::get(&context, {builder->getF64Type()},
                                             {builder->getF64Type()}));
  auto* entry = func.addEntryBlock();
  OpBuilder::InsertionGuard guard(*builder);
  builder->setInsertionPointToStart(entry);
  auto lhs = arith::ConstantOp::create(*builder, builder->getF64FloatAttr(5.0));
  auto op = arith::SubFOp::create(*builder, lhs, entry->getArgument(0));
  EXPECT_FALSE(utils::valueToConstantDouble(op.getResult()).has_value());
}

TEST_F(UtilsTest, valueToConstantAttrFoldFailure) {
  // Pure op whose fold fails (shift >= bitwidth).
  auto lhs = arith::ConstantOp::create(*builder, builder->getI32IntegerAttr(1));
  auto rhs =
      arith::ConstantOp::create(*builder, builder->getI32IntegerAttr(32));
  auto op = arith::ShLIOp::create(*builder, lhs, rhs);
  EXPECT_FALSE(utils::valueToConstantAttr(op.getResult()).has_value());
}

TEST_F(UtilsTest, valueToConstantAttrMultiResultFold) {
  // addui_extended folds to two results; valueToConstantAttr requires exactly
  // one OpFoldResult.
  auto lhs = arith::ConstantOp::create(*builder, builder->getI32IntegerAttr(1));
  auto rhs = arith::ConstantOp::create(*builder, builder->getI32IntegerAttr(2));
  auto op = arith::AddUIExtendedOp::create(*builder, lhs, rhs);
  EXPECT_FALSE(utils::valueToConstantAttr(op.getSum()).has_value());
}

TEST_F(UtilsTest, valueToConstantAttrIdentityFold) {
  // select(true, x, y) folds to an existing SSA value, not an Attribute.
  constexpr double expectedValue = 3.25;
  auto cond = arith::ConstantOp::create(*builder, builder->getBoolAttr(true));
  auto x = arith::ConstantOp::create(*builder,
                                     builder->getF64FloatAttr(expectedValue));
  auto y = arith::ConstantOp::create(*builder, builder->getF64FloatAttr(9.0));
  auto op = arith::SelectOp::create(*builder, cond, x, y);
  const auto attr = utils::valueToConstantAttr(op.getResult());
  ASSERT_TRUE(attr.has_value());
  EXPECT_DOUBLE_EQ(*utils::attributeToDouble(*attr), expectedValue);
}

TEST_F(UtilsTest, valueToConstantDoubleSharedOperandsSuccess) {
  // Repeated doubling reuses the same SSA value as both operands. Without
  // memoization this is exponential in `depth`.
  constexpr int depth = 40;
  Value v = arith::ConstantOp::create(*builder, builder->getF64FloatAttr(1.0));
  for (int i = 0; i < depth; ++i) {
    v = arith::AddFOp::create(*builder, v, v);
  }
  const auto stdValue = utils::valueToConstantDouble(v);
  ASSERT_TRUE(stdValue.has_value());
  EXPECT_DOUBLE_EQ(*stdValue, static_cast<double>(1ULL << depth));
}

TEST_F(UtilsTest, valueToConstantDoubleSharedOperandsFailure) {
  // Same doubling shape rooted at a dynamic value; failures must be cached
  // too or evaluation is exponential.
  constexpr int depth = 40;
  auto func =
      func::FuncOp::create(*builder, "dyn",
                           FunctionType::get(&context, {builder->getF64Type()},
                                             {builder->getF64Type()}));
  auto* entry = func.addEntryBlock();
  OpBuilder::InsertionGuard guard(*builder);
  builder->setInsertionPointToStart(entry);

  const Value arg = entry->getArgument(0);
  SmallVector<Value> nodes = {arg};
  Value v = arg;
  for (int i = 0; i < depth; ++i) {
    v = arith::AddFOp::create(*builder, v, v);
    nodes.push_back(v);
  }

  llvm::DenseMap<Value, std::optional<Attribute>> cache;
  EXPECT_FALSE(utils::valueToConstantAttr(v, cache).has_value());
  ASSERT_EQ(cache.size(), nodes.size());
  for (const Value node : nodes) {
    const auto it = cache.find(node);
    ASSERT_NE(it, cache.end());
    EXPECT_FALSE(it->second.has_value());
  }
}
