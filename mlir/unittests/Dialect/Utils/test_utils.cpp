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
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Parser/Parser.h>

using namespace mlir;

class UtilsTest : public ::testing::Test {
protected:
  MLIRContext context;

  void SetUp() override {
    context.loadDialect<func::FuncDialect>();
    context.loadDialect<arith::ArithDialect>();
  }
};

TEST_F(UtilsTest, valueToDouble) {
  auto moduleOp = parseSourceString<ModuleOp>(
      "func.func @test() { arith.constant 1.234 : f64\n return }", &context);
  ASSERT_TRUE(moduleOp);

  for (auto&& funcOp : moduleOp->getOps<func::FuncOp>()) {
    for (auto&& constantOp : funcOp.getOps<arith::ConstantOp>()) {
      auto value = constantOp.getResult();
      auto stdValue = utils::valueToDouble(value);
      ASSERT_TRUE(stdValue.has_value());
      EXPECT_DOUBLE_EQ(stdValue.value(), 1.234);
      return;
    }
    FAIL() << "No arith::ConstantOp found in function!";
  }
  FAIL() << "No func::FuncOp found in module!";
}

TEST_F(UtilsTest, valueToDoubleWrongType) {
  auto moduleOp = parseSourceString<ModuleOp>(
      "func.func @test() { arith.constant 42 : i32\n return }", &context);
  ASSERT_TRUE(moduleOp);

  for (auto&& funcOp : moduleOp->getOps<func::FuncOp>()) {
    for (auto&& constantOp : funcOp.getOps<arith::ConstantOp>()) {
      auto value = constantOp.getResult();
      auto stdValue = utils::valueToDouble(value);
      EXPECT_FALSE(stdValue.has_value());
      return;
    }
    FAIL() << "No arith::ConstantOp found in function!";
  }
  FAIL() << "No func::FuncOp found in module!";
}

TEST_F(UtilsTest, valueToDoubleNonStaticValue) {
  auto moduleOp = parseSourceString<ModuleOp>("func.func @test() {\n"
                                              "%0 = arith.constant 1.1 : f64\n"
                                              "%1 = arith.constant 2.2 : f64\n"
                                              "arith.addf %0, %1 : f64\n"
                                              "return }",
                                              &context);
  ASSERT_TRUE(moduleOp);

  for (auto&& funcOp : moduleOp->getOps<func::FuncOp>()) {
    for (auto&& addOp : funcOp.getOps<arith::AddFOp>()) {
      auto value = addOp.getResult();
      auto stdValue = utils::valueToDouble(value);
      EXPECT_FALSE(stdValue.has_value());
      return;
    }
    FAIL() << "No arith::AddFOp found in function!";
  }
  FAIL() << "No func::FuncOp found in module!";
}
