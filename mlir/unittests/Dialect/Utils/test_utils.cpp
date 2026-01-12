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

#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <llvm/ADT/APInt.h>
#include <llvm/ADT/SmallVector.h>
#include <memory>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>

using namespace mlir;

class UtilsTest : public ::testing::Test {
protected:
  MLIRContext context;
  std::unique_ptr<OpBuilder> builder;

  void SetUp() override {
    context.loadDialect<arith::ArithDialect>();

    builder = std::make_unique<OpBuilder>(&context);
  }

  arith::AddFOp createAddition(double a, double b) {
    auto firstOperand = builder->create<arith::ConstantOp>(
        builder->getUnknownLoc(), builder->getF64FloatAttr(a));
    auto secondOperand = builder->create<arith::ConstantOp>(
        builder->getUnknownLoc(), builder->getF64FloatAttr(b));
    return builder->create<arith::AddFOp>(builder->getUnknownLoc(),
                                          firstOperand, secondOperand);
  }
};

TEST_F(UtilsTest, valueToDouble) {
  constexpr double expectedValue = 1.234;
  auto op = builder->create<arith::ConstantOp>(
      builder->getUnknownLoc(), builder->getF64FloatAttr(expectedValue));
  ASSERT_TRUE(op);

  auto value = op.getResult();
  auto stdValue = utils::valueToDouble(value);
  ASSERT_TRUE(stdValue.has_value());
  EXPECT_DOUBLE_EQ(stdValue.value(), expectedValue);
}

TEST_F(UtilsTest, valueToDoubleCastFromInteger) {
  constexpr int expectedValue = 42;
  auto op = builder->create<arith::ConstantOp>(
      builder->getUnknownLoc(), builder->getI32IntegerAttr(expectedValue));
  ASSERT_TRUE(op);

  auto value = op.getResult();
  auto stdValue = utils::valueToDouble(value);
  ASSERT_TRUE(stdValue.has_value());
  EXPECT_DOUBLE_EQ(stdValue.value(), expectedValue);
}

TEST_F(UtilsTest, valueToDoubleCastFromNegativeInteger) {
  constexpr int expectedValue = -123;
  auto op = builder->create<arith::ConstantOp>(
      builder->getUnknownLoc(), builder->getSI32IntegerAttr(expectedValue));
  ASSERT_TRUE(op);

  auto value = op.getResult();
  auto stdValue = utils::valueToDouble(value);
  ASSERT_TRUE(stdValue.has_value());
  EXPECT_DOUBLE_EQ(stdValue.value(), expectedValue);
}

TEST_F(UtilsTest, valueToDoubleCastFromMaxUnsignedInteger) {
  constexpr auto expectedValue = std::numeric_limits<uint64_t>::max();
  constexpr auto bitCount = 64;
  auto op = builder->create<arith::ConstantOp>(
      builder->getUnknownLoc(),
      builder->getIntegerAttr(builder->getIntegerType(bitCount, false),
                              llvm::APInt::getMaxValue(bitCount)));
  ASSERT_TRUE(op);

  auto value = op.getResult();
  auto stdValue = utils::valueToDouble(value);
  ASSERT_TRUE(stdValue.has_value());
  // cast to double will lose precision, but difference to maximum value of
  // int64_t is large enough that the check still makes sense
  EXPECT_DOUBLE_EQ(stdValue.value(), static_cast<double>(expectedValue));
}

TEST_F(UtilsTest, valueToDoubleWrongType) {
  auto op = builder->create<arith::ConstantOp>(builder->getUnknownLoc(),
                                               builder->getStringAttr("test"));
  ASSERT_TRUE(op);

  auto value = op.getResult();
  auto stdValue = utils::valueToDouble(value);
  EXPECT_FALSE(stdValue.has_value());
}

TEST_F(UtilsTest, valueToDoubleNonStaticValue) {
  auto op = createAddition(9.5, 21.5);
  ASSERT_TRUE(op);

  auto value = op.getResult();
  auto stdValue = utils::valueToDouble(value);
  EXPECT_FALSE(stdValue.has_value());
}

TEST_F(UtilsTest, valueToDoubleFoldedConstant) {
  auto op = createAddition(1.5, 2.0);
  ASSERT_TRUE(op);

  llvm::SmallVector<Value> tmp;
  llvm::SmallVector<Operation*> newConstants;
  ASSERT_TRUE(builder->tryFold(op, tmp, &newConstants).succeeded());
  ASSERT_EQ(newConstants.size(), 1);
  auto value = newConstants[0]->getResult(0);
  auto stdValue = utils::valueToDouble(value);
  ASSERT_TRUE(stdValue.has_value());
  EXPECT_DOUBLE_EQ(stdValue.value(), 3.5);
}
