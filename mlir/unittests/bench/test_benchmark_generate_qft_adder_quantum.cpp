/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "TestUtils.h"
#include "bench/QFTAdderQuantum.hpp"
#include "mlir/Dialect/CBit/IR/CBitOps.h"
#include "mlir/Dialect/QC/IR/QCOps.h"
#include "mlir/bench/Generate.h"

#include <gtest/gtest.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>

#include <cmath>
#include <cstdint>
#include <numbers>

namespace mqt::bench {

using namespace mlir;

static void expectConstantIndex(Value value, int64_t expected) {
  auto constant = value.getDefiningOp<arith::ConstantIndexOp>();
  ASSERT_TRUE(constant);
  EXPECT_EQ(constant.value(), expected);
}

static void expectConstantFloat(Value value, double expected) {
  auto constant = value.getDefiningOp<arith::ConstantOp>();
  ASSERT_TRUE(constant);
  auto attribute = dyn_cast<FloatAttr>(constant.getValue());
  ASSERT_TRUE(attribute);
  EXPECT_DOUBLE_EQ(attribute.getValueAsDouble(), expected);
}

TEST(GenerateProgramTest, EmitsQuantumQFTAdderCircuit) {
  constexpr int64_t qubits = 3;
  auto program = generate(QFTAdderQuantum{{.qubits = qubits}});
  ASSERT_TRUE(program);
  auto moduleOp = program->module();

  EXPECT_EQ(test::countOps<memref::AllocOp>(moduleOp), 2U);
  EXPECT_EQ(test::countOps<cbit::AllocOp>(moduleOp), 1U);
  EXPECT_EQ(test::countOps<qc::HOp>(moduleOp), 3U);
  EXPECT_EQ(test::countOps<qc::XOp>(moduleOp), 1U);
  EXPECT_EQ(test::countOps<qc::CtrlOp>(moduleOp), 3U);
  EXPECT_EQ(test::countOps<qc::POp>(moduleOp), 3U);
  EXPECT_EQ(test::countOps<qc::MeasureOp>(moduleOp), 2U);
  EXPECT_EQ(test::countOps<qc::SWAPOp>(moduleOp), 0U);

  // Unlike the QFT phases, the addition phase connects the two registers.
  qc::CtrlOp addition;
  moduleOp.walk([&](qc::CtrlOp op) {
    auto control = op.getControl(0).getDefiningOp<memref::LoadOp>();
    auto target = op.getTarget(0).getDefiningOp<memref::LoadOp>();
    if (control && target && control.getMemref() != target.getMemref()) {
      EXPECT_FALSE(addition);
      addition = op;
    }
  });
  ASSERT_TRUE(addition);

  auto sourceLoad = addition.getControl(0).getDefiningOp<memref::LoadOp>();
  auto targetLoad = addition.getTarget(0).getDefiningOp<memref::LoadOp>();
  ASSERT_TRUE(sourceLoad);
  ASSERT_TRUE(targetLoad);

  auto inner = addition->getParentOfType<scf::ForOp>();
  ASSERT_TRUE(inner);
  auto outer = inner->getParentOfType<scf::ForOp>();
  ASSERT_TRUE(outer);

  auto target = targetLoad.getIndices().front();
  auto targetIndex = target.getDefiningOp<arith::SubIOp>();
  ASSERT_TRUE(targetIndex);
  expectConstantIndex(targetIndex.getLhs(), qubits - 1);
  EXPECT_EQ(targetIndex.getRhs(), outer.getInductionVar());

  auto sourceIndex =
      sourceLoad.getIndices().front().getDefiningOp<arith::SubIOp>();
  ASSERT_TRUE(sourceIndex);
  EXPECT_EQ(sourceIndex.getLhs(), target);
  EXPECT_EQ(sourceIndex.getRhs(), inner.getInductionVar());

  auto upper = inner.getUpperBound().getDefiningOp<arith::SubIOp>();
  ASSERT_TRUE(upper);
  expectConstantIndex(upper.getLhs(), qubits);
  EXPECT_EQ(upper.getRhs(), outer.getInductionVar());
  expectConstantIndex(inner.getLowerBound(), 0);
  expectConstantIndex(inner.getStep(), 1);

  ASSERT_EQ(inner.getInitArgs().size(), 1U);
  expectConstantFloat(inner.getInitArgs().front(), std::numbers::pi);
  qc::POp phase;
  addition.walk([&](qc::POp op) { phase = op; });
  ASSERT_TRUE(phase);
  EXPECT_EQ(phase.getTheta(), inner.getRegionIterArg(0));

  auto yield = dyn_cast<scf::YieldOp>(inner.getBody()->getTerminator());
  ASSERT_TRUE(yield);
  ASSERT_EQ(yield.getNumOperands(), 1U);
  auto nextAngle = yield.getOperand(0).getDefiningOp<arith::MulFOp>();
  ASSERT_TRUE(nextAngle);
  EXPECT_EQ(nextAngle.getLhs(), inner.getRegionIterArg(0));
  expectConstantFloat(nextAngle.getRhs(), 0.5);
}

TEST(GenerateProgramTest, KeepsLargestQuantumQFTAdderFiniteAndStructured) {
  auto program =
      generate(QFTAdderQuantum{{.qubits = QFTAdderQuantumOptions::MAX_QUBITS}});
  ASSERT_TRUE(program);
  auto moduleOp = program->module();

  EXPECT_LT(test::countOperations(moduleOp), 200U);
  moduleOp.walk([&](arith::ConstantOp op) {
    if (auto value = dyn_cast<FloatAttr>(op.getValue())) {
      EXPECT_TRUE(std::isfinite(value.getValueAsDouble()));
    }
  });
}

} // namespace mqt::bench
