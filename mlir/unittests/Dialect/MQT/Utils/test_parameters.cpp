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
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Support/LogicalResult.h"

#include <cmath>
#include <limits>
#include <numbers>
#include <optional>
#include <variant>

using namespace mlir;

namespace {

class ParametersTest : public testing::Test {
protected:
  MLIRContext context_;
  OpBuilder builder_{&context_};
  OwningOpRef<ModuleOp> moduleOp_;
  func::FuncOp function_;

  void SetUp() override {
    context_.loadDialect<arith::ArithDialect, math::MathDialect,
                         func::FuncDialect>();
    auto loc = builder_.getUnknownLoc();
    moduleOp_ = ModuleOp::create(loc);
    builder_.setInsertionPointToStart(moduleOp_->getBody());
    auto f64 = builder_.getF64Type();
    function_ = func::FuncOp::create(builder_, loc, "parameters",
                                     builder_.getFunctionType({f64, f64}, {}));
    auto* entry = function_.addEntryBlock();
    builder_.setInsertionPointToEnd(entry);
    func::ReturnOp::create(builder_, loc);
    builder_.setInsertionPointToStart(entry);
  }

  void TearDown() override { EXPECT_TRUE(succeeded(verify(*moduleOp_))); }
};

} // namespace

TEST_F(ParametersTest, KnownParametersStayInHostArithmetic) {
  auto loc = builder_.getUnknownLoc();
  const auto size = function_.getBody().front().getOperations().size();
  EXPECT_EQ(mqt::parameterToConstantDouble(
                mqt::addParameters(builder_, loc, 1.25, -0.5)),
            std::optional{0.75});
  EXPECT_EQ(mqt::parameterToConstantDouble(
                mqt::scaleParameter(builder_, loc, 1.25, -2.)),
            std::optional{-2.5});
  EXPECT_EQ(function_.getBody().front().getOperations().size(), size);

  auto a = mqt::constantFromScalar(builder_, loc, 1.25);
  auto b = mqt::constantFromScalar(builder_, loc, -0.5);
  auto sum = arith::AddFOp::create(builder_, loc, a, b).getResult();
  const auto expressionSize =
      function_.getBody().front().getOperations().size();
  auto result = mqt::addParameters(builder_, loc, sum, 2.);
  ASSERT_TRUE(std::holds_alternative<double>(result));
  EXPECT_DOUBLE_EQ(std::get<double>(result), 2.75);
  EXPECT_EQ(mqt::parameterToConstantDouble(
                mqt::scaleParameter(builder_, loc, sum, -2.)),
            std::optional{-1.5});
  EXPECT_EQ(function_.getBody().front().getOperations().size(), expressionSize);
}

TEST_F(ParametersTest, MixedArithmeticReusesKnownConstantExpressions) {
  auto loc = builder_.getUnknownLoc();
  auto input = function_.getArgument(0);
  auto a = mqt::constantFromScalar(builder_, loc, 1.25);
  auto b = mqt::constantFromScalar(builder_, loc, -0.5);
  auto constantExpression =
      arith::AddFOp::create(builder_, loc, a, b).getResult();
  EXPECT_EQ(std::get<double>(mqt::foldParameter(constantExpression)), 0.75);
  EXPECT_EQ(std::get<Value>(mqt::foldParameter(input)), input);
  EXPECT_EQ(mqt::FloatExpression(builder_, loc, mqt::FloatParameter{input})
                .getValue(),
            input);
  for (bool constantFirst : {false, true}) {
    auto result = std::get<Value>(
        constantFirst
            ? mqt::addParameters(builder_, loc, constantExpression, input)
            : mqt::addParameters(builder_, loc, input, constantExpression));
    auto addition = result.getDefiningOp<arith::AddFOp>();
    ASSERT_TRUE(addition);
    EXPECT_TRUE(addition.getLhs() == input || addition.getRhs() == input);
    EXPECT_EQ(mqt::valueToDouble(addition.getLhs() == input
                                     ? addition.getRhs()
                                     : addition.getLhs()),
              std::optional{0.75});
  }
}

TEST_F(ParametersTest, RuntimeParametersRetainIdentityAndDominance) {
  auto loc = builder_.getUnknownLoc();
  auto lhs = function_.getArgument(0);
  auto rhs = function_.getArgument(1);
  EXPECT_EQ(mqt::variantToValue(builder_, loc, mqt::FloatParameter{lhs}), lhs);
  EXPECT_FALSE(mqt::parameterToConstantDouble(lhs));
  auto sum = std::get<Value>(mqt::addParameters(builder_, loc, lhs, rhs));
  auto addition = sum.getDefiningOp<arith::AddFOp>();
  ASSERT_TRUE(addition);
  EXPECT_EQ(addition.getLhs(), lhs);
  EXPECT_EQ(addition.getRhs(), rhs);
  auto scaled = std::get<Value>(mqt::scaleParameter(builder_, loc, sum, -0.5));
  auto product = scaled.getDefiningOp<arith::MulFOp>();
  ASSERT_TRUE(product);
  EXPECT_EQ(product.getLhs(), sum);
  EXPECT_EQ(mqt::valueToConstantDouble(product.getRhs()), std::optional{-0.5});
}

TEST_F(ParametersTest, MaterializationUsesTheOwningDialectBuilder) {
  auto loc = builder_.getUnknownLoc();
  auto input = function_.getArgument(0);
  bool materialized = false;
  const auto materialize = [&](double value) {
    materialized = true;
    return mqt::constantFromScalar(builder_, loc, value);
  };
  EXPECT_EQ(mqt::variantToValue(mqt::FloatParameter{input}, materialize),
            input);
  EXPECT_FALSE(materialized);
  auto constant = mqt::variantToValue(mqt::FloatParameter{0.25}, materialize);
  EXPECT_TRUE(materialized);
  EXPECT_EQ(mqt::valueToConstantDouble(constant), std::optional{0.25});
}

TEST_F(ParametersTest, SSAArithmeticUsesOperationFolders) {
  auto loc = builder_.getUnknownLoc();
  const auto scalar = [&](double value) {
    return mqt::FloatExpression::constant(builder_, loc, value);
  };
  const auto check = [](mqt::FloatExpression expression, double expected) {
    const auto value = mqt::valueToDouble(expression.getValue());
    ASSERT_TRUE(value);
    EXPECT_DOUBLE_EQ(*value, expected);
  };
  const auto a = scalar(4.);
  const auto b = scalar(2.);
  check(a + b, 6.);
  check(a - b, 2.);
  check(a * b, 8.);
  check(a / b, 2.);
  check(a.pow(b), 16.);
  check(-b, -2.);
  check(a.sqrt(), 2.);
  check((-b).abs(), 2.);
  check(scalar(-1.25).floor(), -2.);
  check(b.sin(), std::sin(2.));
  check(b.cos(), std::cos(2.));
  check(b.tan(), std::tan(2.));
  check(b.atan(), std::atan(2.));
  check(b.atan2(a), std::atan2(2., 4.));
  auto nan = scalar(std::numeric_limits<double>::quiet_NaN());
  check(mqt::FloatExpression::select(a.oge(b), a, b), 4.);
  check(mqt::FloatExpression::select(b.olt(a), b, a), 2.);
  check(mqt::FloatExpression::select(nan.oge(b), a, b), 2.);
  check(mqt::FloatExpression::select(nan.olt(b), a, b), 2.);
  auto condition = mqt::constantFromScalar(builder_, loc, true);
  EXPECT_EQ(mqt::FloatExpression::select(condition, a, b).getValue(),
            a.getValue());
}

TEST_F(ParametersTest,
       ScalarArithmeticPreservesSignedZeroAndNonfiniteOperands) {
  auto loc = builder_.getUnknownLoc();
  auto zeroSum = mqt::parameterToConstantDouble(
      mqt::addParameters(builder_, loc, -0., 0.));
  ASSERT_TRUE(zeroSum);
  EXPECT_FALSE(std::signbit(*zeroSum));
  auto negativeZero = mqt::parameterToConstantDouble(
      mqt::scaleParameter(builder_, loc, 0., -1.));
  ASSERT_TRUE(negativeZero);
  EXPECT_TRUE(std::signbit(*negativeZero));
  const auto y = mqt::FloatExpression::constant(builder_, loc, -0.);
  const auto x = mqt::FloatExpression::constant(builder_, loc, -1.);
  EXPECT_EQ(mqt::valueToConstantDouble(y.atan2(x).getValue()),
            std::optional{-std::numbers::pi});

  auto input = function_.getArgument(0);
  auto sum = std::get<Value>(mqt::addParameters(builder_, loc, input, 0.));
  auto product = std::get<Value>(mqt::scaleParameter(builder_, loc, input, 0.));
  builder_.setInsertionPointToStart(&function_.getBody().front());
  auto negativeInput = mqt::constantFromScalar(builder_, loc, -0.);
  input.replaceAllUsesWith(negativeInput);
  auto runtimeSum = mqt::valueToConstantDouble(sum);
  auto runtimeProduct = mqt::valueToConstantDouble(product);
  ASSERT_TRUE(runtimeSum);
  ASSERT_TRUE(runtimeProduct);
  EXPECT_FALSE(std::signbit(*runtimeSum));
  EXPECT_TRUE(std::signbit(*runtimeProduct));
  auto infinity = mqt::constantFromScalar(
      builder_, loc, std::numeric_limits<double>::infinity());
  negativeInput.replaceAllUsesWith(infinity);
  auto added = mqt::valueToConstantDouble(sum);
  auto multiplied = mqt::valueToConstantDouble(product);
  ASSERT_TRUE(added);
  ASSERT_TRUE(multiplied);
  EXPECT_TRUE(std::isinf(*added));
  EXPECT_TRUE(std::isnan(*multiplied));
}
