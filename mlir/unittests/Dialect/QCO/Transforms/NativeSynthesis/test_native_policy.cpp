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
#include "mlir/Dialect/QCO/Transforms/Decomposition/GateSequence.h"
#include "mlir/Dialect/QCO/Transforms/NativeSynthesis/NativeSpec.h"
#include "mlir/Dialect/QCO/Transforms/NativeSynthesis/Policy.h"

#include <gtest/gtest.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Support/WalkResult.h>

using namespace mlir;
using namespace mlir::qco;
using namespace mlir::qco::decomposition;
using namespace mlir::qco::native_synth;

TEST(NativePolicyTest, ComputeGateSequenceMetricsDepth) {
  QubitGateSequence seq;
  seq.gates.push_back(
      {.type = GateKind::RZ, .parameter = {0.1}, .qubitId = {0}});
  seq.gates.push_back(
      {.type = GateKind::RZ, .parameter = {0.2}, .qubitId = {0}});
  seq.gates.push_back(
      {.type = GateKind::RZZ, .parameter = {0.3}, .qubitId = {0, 1}});
  const CandidateMetrics m = computeGateSequenceMetrics(seq);
  EXPECT_EQ(m.numOneQ, 2U);
  EXPECT_EQ(m.numTwoQ, 1U);
  EXPECT_EQ(m.depth, 3U);
}

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
  EXPECT_TRUE(
      allowsSingleQubitOp(cast<UnitaryOpInterface>(xop.getOperation()), *spec));
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
