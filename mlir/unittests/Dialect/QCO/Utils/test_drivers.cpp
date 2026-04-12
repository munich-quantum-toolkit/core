/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QCO/Builder/QCOProgramBuilder.h"
#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QCO/Utils/Drivers.h"

#include <gtest/gtest.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Operation.h>
#include <mlir/Support/WalkResult.h>

#include <memory>

using namespace mlir;

namespace {
class DriversTest : public testing::Test {
protected:
  void SetUp() override {
    DialectRegistry registry;
    registry.insert<qco::QCODialect, arith::ArithDialect, func::FuncDialect>();

    context = std::make_unique<MLIRContext>();
    context->appendDialectRegistry(registry);
    context->loadAllAvailableDialects();
  }

  std::unique_ptr<MLIRContext> context;
};
} // namespace

TEST_F(DriversTest, FullWalkStatic) {
  qco::QCOProgramBuilder builder(context.get());
  builder.initialize();
  const auto q00 = builder.staticQubit(5);
  const auto q10 = builder.staticQubit(6);
  const auto q20 = builder.staticQubit(7);
  const auto q30 = builder.staticQubit(8);

  const auto q01 = builder.h(q00);
  const auto [q02, q11] = builder.cx(q01, q10);
  const auto [q21, q31] = builder.cx(q20, q30);

  const auto [q03, c0] = builder.measure(q02);
  const auto [q12, c1] = builder.measure(q11);
  const auto [q22, c2] = builder.measure(q21);
  const auto [q32, c3] = builder.measure(q31);

  builder.sink(q03);
  builder.sink(q12);
  builder.sink(q22);
  builder.sink(q32);

  auto module = builder.finalize();
  auto func = *(module->getOps<mlir::func::FuncOp>().begin());

  Value ex0 = nullptr;
  Value ex1 = nullptr;
  Value ex2 = nullptr;
  Value ex3 = nullptr;

  qco::walkUnit(func.getBody(), [&](Operation* op, const qco::Qubits& qubits) {
    if (op == q03.getDefiningOp()) {
      ex0 = qubits.getQubit(5);
      ex1 = qubits.getQubit(6);
      ex2 = qubits.getQubit(7);
      ex3 = qubits.getQubit(8);
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });

  ASSERT_EQ(ex0, q02);
  ASSERT_EQ(ex1, q11);
  ASSERT_EQ(ex2, q21);
  ASSERT_EQ(ex3, q31);
}

TEST_F(DriversTest, FullWalkDynamic) {
  qco::QCOProgramBuilder builder(context.get());
  builder.initialize();
  const auto q00 = builder.allocQubit();
  const auto q10 = builder.allocQubit();
  const auto q20 = builder.allocQubit();
  const auto q30 = builder.allocQubit();

  const auto q01 = builder.h(q00);
  const auto [q02, q11] = builder.cx(q01, q10);
  const auto [q21, q31] = builder.cx(q20, q30);

  const auto [q03, c0] = builder.measure(q02);
  const auto [q12, c1] = builder.measure(q11);
  const auto [q22, c2] = builder.measure(q21);
  const auto [q32, c3] = builder.measure(q31);

  builder.sink(q03);
  builder.sink(q12);
  builder.sink(q22);
  builder.sink(q32);

  auto module = builder.finalize();
  auto func = *(module->getOps<mlir::func::FuncOp>().begin());

  Value ex0 = nullptr;
  Value ex1 = nullptr;
  Value ex2 = nullptr;
  Value ex3 = nullptr;

  qco::walkUnit(func.getBody(), [&](Operation* op, const qco::Qubits& qubits) {
    if (op == q03.getDefiningOp()) {
      ex0 = qubits.getQubit(0);
      ex1 = qubits.getQubit(1);
      ex2 = qubits.getQubit(2);
      ex3 = qubits.getQubit(3);
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });

  ASSERT_EQ(ex0, q02);
  ASSERT_EQ(ex1, q11);
  ASSERT_EQ(ex2, q21);
  ASSERT_EQ(ex3, q31);
}
