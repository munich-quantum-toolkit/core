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
#include <llvm/ADT/SmallVector.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>

#include <cstdint>
#include <limits>
#include <memory>

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

  [[nodiscard]] arith::AddFOp createAddition(const double a, const double b) {
    auto firstOperand =
        arith::ConstantOp::create(*builder, builder->getF64FloatAttr(a));
    auto secondOperand =
        arith::ConstantOp::create(*builder, builder->getF64FloatAttr(b));
    return arith::AddFOp::create(*builder, firstOperand, secondOperand);
  }
};

} // namespace

TEST_F(UtilsTest, valueToDouble) {
  constexpr double expectedValue = 1.234;
  auto op = arith::ConstantOp::create(*builder,
                                      builder->getF64FloatAttr(expectedValue));
  ASSERT_TRUE(op);

  const auto stdValue = utils::valueToDouble(op.getResult());
  ASSERT_TRUE(stdValue.has_value());
  EXPECT_DOUBLE_EQ(stdValue.value(), expectedValue);
}

TEST_F(UtilsTest, valueToDoubleCastFromInteger) {
  constexpr int expectedValue = 42;
  auto op = arith::ConstantOp::create(
      *builder, builder->getI32IntegerAttr(expectedValue));
  ASSERT_TRUE(op);

  const auto stdValue = utils::valueToDouble(op.getResult());
  ASSERT_TRUE(stdValue.has_value());
  EXPECT_DOUBLE_EQ(stdValue.value(), expectedValue);
}

TEST_F(UtilsTest, valueToDoubleCastFromNegativeInteger) {
  constexpr int expectedValue = -123;
  auto op = arith::ConstantOp::create(
      *builder, builder->getSI32IntegerAttr(expectedValue));
  ASSERT_TRUE(op);

  const auto stdValue = utils::valueToDouble(op.getResult());
  ASSERT_TRUE(stdValue.has_value());
  EXPECT_DOUBLE_EQ(stdValue.value(), expectedValue);
}

TEST_F(UtilsTest, valueToDoubleCastFromMaxUnsignedInteger) {
  constexpr auto expectedValue = std::numeric_limits<uint64_t>::max();
  constexpr auto bitCount = 64;
  auto op = arith::ConstantOp::create(
      *builder,
      builder->getIntegerAttr(builder->getIntegerType(bitCount, false),
                              APInt::getMaxValue(bitCount)));
  ASSERT_TRUE(op);

  const auto stdValue = utils::valueToDouble(op.getResult());
  ASSERT_TRUE(stdValue.has_value());
  // cast to double will lose precision, but difference to maximum value of
  // int64_t is large enough that the check still makes sense
  EXPECT_DOUBLE_EQ(stdValue.value(), static_cast<double>(expectedValue));
}

TEST_F(UtilsTest, valueToDoubleWrongType) {
  auto op = arith::ConstantOp::create(*builder, builder->getStringAttr("test"));
  ASSERT_TRUE(op);

  const auto stdValue = utils::valueToDouble(op.getResult());
  EXPECT_FALSE(stdValue.has_value());
}

TEST_F(UtilsTest, valueToDoubleNonStaticValue) {
  auto op = createAddition(9.5, 21.5);
  ASSERT_TRUE(op);

  const auto stdValue = utils::valueToDouble(op.getResult());
  EXPECT_FALSE(stdValue.has_value());
}

TEST_F(UtilsTest, valueToDoubleFoldedConstant) {
  auto op = createAddition(1.5, 2.0);
  ASSERT_TRUE(op);

  SmallVector<Value> tmp;
  SmallVector<Operation*> newConstants;
  ASSERT_TRUE(builder->tryFold(op, tmp, &newConstants).succeeded());
  ASSERT_EQ(newConstants.size(), 1);
  auto cst = dyn_cast<arith::ConstantOp>(newConstants[0]);
  ASSERT_TRUE(cst);
  const auto stdValue = utils::valueToDouble(cst.getResult());
  ASSERT_TRUE(stdValue.has_value());
  EXPECT_DOUBLE_EQ(stdValue.value(), 3.5);
}

TEST_F(UtilsTest, valueToConstantDoubleAddF) {
  auto op = createAddition(1.25, 2.5);
  ASSERT_TRUE(op);

  const auto stdValue = utils::valueToConstantDouble(op.getResult());
  ASSERT_TRUE(stdValue.has_value());
  EXPECT_DOUBLE_EQ(*stdValue, 3.75);
}

TEST_F(UtilsTest, valueToConstantDoubleSubF) {
  auto lhs = arith::ConstantOp::create(*builder, builder->getF64FloatAttr(5.0));
  auto rhs = arith::ConstantOp::create(*builder, builder->getF64FloatAttr(1.5));
  auto op = arith::SubFOp::create(*builder, lhs, rhs);

  const auto stdValue = utils::valueToConstantDouble(op.getResult());
  ASSERT_TRUE(stdValue.has_value());
  EXPECT_DOUBLE_EQ(*stdValue, 3.5);
}

TEST_F(UtilsTest, valueToConstantDoubleDivF) {
  auto lhs = arith::ConstantOp::create(*builder, builder->getF64FloatAttr(5.0));
  auto num = arith::ConstantOp::create(*builder, builder->getF64FloatAttr(1.0));
  auto den = arith::ConstantOp::create(*builder, builder->getF64FloatAttr(2.0));
  auto quot = arith::DivFOp::create(*builder, num, den);
  auto op = arith::SubFOp::create(*builder, lhs, quot);

  const auto stdValue = utils::valueToConstantDouble(op.getResult());
  ASSERT_TRUE(stdValue.has_value());
  EXPECT_DOUBLE_EQ(*stdValue, 4.5);
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
