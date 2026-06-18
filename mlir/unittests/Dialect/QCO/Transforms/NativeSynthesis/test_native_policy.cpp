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
#include "mlir/Dialect/QCO/Transforms/NativeSynthesis/NativeSpec.h"
#include "mlir/Dialect/QCO/Transforms/NativeSynthesis/Policy.h"
#include "mlir/Dialect/QCO/Transforms/NativeSynthesis/Types.h"

#include <gtest/gtest.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Casting.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/WalkResult.h>

using namespace mlir;
using namespace mlir::qco;
using namespace mlir::qco::decomposition;
using namespace mlir::qco::native_synth;

TEST(NativePolicyTest, UsesCxAndCzFromResolvedSpec) {
  const auto cxOnly = resolveNativeGatesSpec("u,cx");
  ASSERT_TRUE(cxOnly);
  EXPECT_TRUE(usesCxEntangler(*cxOnly));
  EXPECT_FALSE(usesCzEntangler(*cxOnly));

  const auto both = resolveNativeGatesSpec("u,cx,cz");
  ASSERT_TRUE(both);
  EXPECT_TRUE(usesCxEntangler(*both));
  EXPECT_TRUE(usesCzEntangler(*both));
}

// NOLINTNEXTLINE(misc-use-internal-linkage)
class NativePolicyAllowsOpTest : public ::testing::Test {
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

TEST_F(NativePolicyAllowsOpTest, AllowsSingleQubitOpRespectsMenu) {
  const auto spec = resolveNativeGatesSpec("x,sx,rz,cx");
  ASSERT_TRUE(spec);
  Value q = builder.staticQubit(0);
  q = builder.x(q);
  auto mod = builder.finalize();
  ASSERT_TRUE(mod);
  XOp xop;
  mod->walk([&](XOp op) {
    xop = op;
    return WalkResult::interrupt();
  });
  ASSERT_TRUE(xop);
  EXPECT_TRUE(allowsSingleQubitOp(
      llvm::cast<UnitaryOpInterface>(xop.getOperation()), *spec));
}

TEST_F(NativePolicyAllowsOpTest, RejectsSingleQubitOpNotInMenu) {
  const auto spec = resolveNativeGatesSpec("u,cx");
  ASSERT_TRUE(spec);
  Value q = builder.staticQubit(0);
  q = builder.x(q);
  auto mod = builder.finalize();
  ASSERT_TRUE(mod);
  XOp xop;
  mod->walk([&](XOp op) {
    xop = op;
    return WalkResult::interrupt();
  });
  ASSERT_TRUE(xop);
  EXPECT_FALSE(allowsSingleQubitOp(
      llvm::cast<UnitaryOpInterface>(xop.getOperation()), *spec));
}

TEST_F(NativePolicyAllowsOpTest, CanDirectlyDecomposeToU3OnRxInCircuit) {
  Value q = builder.staticQubit(0);
  q = builder.rx(0.1, q);
  auto mod = builder.finalize();
  ASSERT_TRUE(mod);
  RXOp rx;
  mod->walk([&](RXOp op) {
    rx = op;
    return WalkResult::interrupt();
  });
  ASSERT_TRUE(rx);
  EXPECT_TRUE(canDirectlyDecomposeToU3(rx.getOperation()));
}

TEST_F(NativePolicyAllowsOpTest, CannotDirectlyDecomposeHToU3) {
  Value q = builder.staticQubit(0);
  q = builder.h(q);
  auto mod = builder.finalize();
  ASSERT_TRUE(mod);
  HOp hop;
  mod->walk([&](HOp op) {
    hop = op;
    return WalkResult::interrupt();
  });
  ASSERT_TRUE(hop);
  EXPECT_FALSE(canDirectlyDecomposeToU3(hop.getOperation()));
}
