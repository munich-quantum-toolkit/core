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
#include "mlir/Dialect/QCO/IR/QCOInterfaces.h"
#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QCO/Transforms/Decomposition/GateKind.h"
#include "mlir/Dialect/QCO/Transforms/Decomposition/Helpers.h"

#include <gtest/gtest.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/WalkResult.h>

using namespace mlir;
using namespace mlir::qco;

class DecompositionGetGateKindTest : public ::testing::Test {
protected:
  MLIRContext context;
  QCOProgramBuilder builder{&context};

  void SetUp() override {
    context.loadDialect<QCODialect>();
    context.loadDialect<func::FuncDialect>();
    context.loadDialect<arith::ArithDialect>();
    context.loadDialect<scf::SCFDialect>();
    builder.initialize();
  }
};

TEST_F(DecompositionGetGateKindTest, MapsBareSingleQubitOps) {
  Value q = builder.staticQubit(0);
  q = builder.rx(0.25, q);
  auto mod = builder.finalize();
  ASSERT_TRUE(mod);
  RXOp rx;
  mod->walk([&](RXOp op) {
    rx = op;
    return WalkResult::interrupt();
  });
  ASSERT_TRUE(rx);
  EXPECT_EQ(helpers::getGateKind(cast<UnitaryOpInterface>(rx.getOperation())),
            decomposition::GateKind::RX);
}

TEST_F(DecompositionGetGateKindTest, MapsCtrlBodyNotWrapper) {
  Value c = builder.staticQubit(0);
  Value t = builder.staticQubit(1);
  auto [cOut, tOut] =
      builder.ctrl(ValueRange{c}, ValueRange{t},
                   [&](ValueRange targets) -> SmallVector<Value> {
                     return {builder.z(targets[0])};
                   });
  (void)cOut;
  (void)tOut;
  auto mod = builder.finalize();
  ASSERT_TRUE(mod);
  CtrlOp ctrl;
  mod->walk([&](CtrlOp op) {
    ctrl = op;
    return WalkResult::interrupt();
  });
  ASSERT_TRUE(ctrl);
  EXPECT_EQ(helpers::getGateKind(cast<UnitaryOpInterface>(ctrl.getOperation())),
            decomposition::GateKind::Z);
}
