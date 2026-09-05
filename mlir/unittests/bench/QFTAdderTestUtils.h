/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

#include "mlir/Dialect/QC/IR/QCOps.h"

#include <gtest/gtest.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>

#include <cstddef>
#include <cstdint>
#include <numbers>

namespace mqt::bench::test {

using namespace mlir;

inline void expectConstantIndex(Value value, const int64_t expected) {
  auto constant = value.getDefiningOp<arith::ConstantIndexOp>();
  ASSERT_TRUE(constant);
  EXPECT_EQ(constant.value(), expected);
}

inline void expectConstantFloat(Value value, const double expected) {
  auto constant = value.getDefiningOp<arith::ConstantOp>();
  ASSERT_TRUE(constant);
  auto attribute = dyn_cast<FloatAttr>(constant.getValue());
  ASSERT_TRUE(attribute);
  EXPECT_DOUBLE_EQ(attribute.getValueAsDouble(), expected);
}

inline void expectStaticLoop(scf::ForOp loop, const int64_t lower,
                             const int64_t upper) {
  expectConstantIndex(loop.getLowerBound(), lower);
  expectConstantIndex(loop.getUpperBound(), upper);
  expectConstantIndex(loop.getStep(), 1);
}

[[nodiscard]] inline SmallVector<scf::ForOp> topLevelLoops(ModuleOp moduleOp) {
  SmallVector<scf::ForOp> loops;
  moduleOp.walk([&](scf::ForOp loop) {
    if (!loop->getParentOfType<scf::ForOp>()) {
      loops.push_back(loop);
    }
  });
  return loops;
}

[[nodiscard]] inline SmallVector<scf::ForOp> nestedLoops(scf::ForOp outer) {
  SmallVector<scf::ForOp> loops;
  outer.walk([&](scf::ForOp loop) {
    if (loop != outer) {
      loops.push_back(loop);
    }
  });
  return loops;
}

inline void expectAngleRecurrence(scf::ForOp loop, Value initialAngle,
                                  const double factor) {
  ASSERT_EQ(loop.getInitArgs().size(), 1U);
  EXPECT_EQ(loop.getInitArgs().front(), initialAngle);
  auto angle = loop.getRegionIterArg(0);

  auto yield = dyn_cast<scf::YieldOp>(loop.getBody()->getTerminator());
  ASSERT_TRUE(yield);
  ASSERT_EQ(yield.getNumOperands(), 1U);
  auto scale = yield.getOperand(0).getDefiningOp<arith::MulFOp>();
  ASSERT_TRUE(scale);
  EXPECT_EQ(scale.getLhs(), angle);
  expectConstantFloat(scale.getRhs(), factor);
}

struct ControlledPhase {
  qc::CtrlOp control;
  qc::POp phase;
};

[[nodiscard]] inline ControlledPhase controlledPhase(scf::ForOp loop) {
  ControlledPhase result;
  size_t controls = 0;
  size_t phases = 0;
  loop.walk([&](qc::CtrlOp op) {
    result.control = op;
    ++controls;
  });
  loop.walk([&](qc::POp op) {
    result.phase = op;
    ++phases;
  });
  EXPECT_EQ(controls, 1U);
  EXPECT_EQ(phases, 1U);
  if (result.control) {
    EXPECT_EQ(result.control.getNumControls(), 1U);
    EXPECT_EQ(result.control.getNumTargets(), 1U);
  }
  return result;
}

inline void expectForwardQFT(scf::ForOp forward, Value qubitRegister,
                             const int64_t qubits) {
  qc::HOp forwardH;
  forward.walk([&](qc::HOp op) { forwardH = op; });
  ASSERT_TRUE(forwardH);
  auto targetLoad = forwardH.getQubit(0).getDefiningOp<memref::LoadOp>();
  ASSERT_TRUE(targetLoad);
  EXPECT_EQ(targetLoad.getMemref(), qubitRegister);
  auto target = targetLoad.getIndices().front();
  auto targetExpression = target.getDefiningOp<arith::SubIOp>();
  ASSERT_TRUE(targetExpression);
  expectConstantIndex(targetExpression.getLhs(), qubits - 1);
  EXPECT_EQ(targetExpression.getRhs(), forward.getInductionVar());

  auto innerLoops = nestedLoops(forward);
  ASSERT_EQ(innerLoops.size(), 1U);
  auto inner = innerLoops.front();
  EXPECT_TRUE(forwardH->isBeforeInBlock(inner));
  expectConstantIndex(inner.getLowerBound(), 0);
  EXPECT_EQ(inner.getUpperBound(), target);
  expectConstantIndex(inner.getStep(), 1);
  ASSERT_EQ(inner.getInitArgs().size(), 1U);
  expectConstantFloat(inner.getInitArgs().front(), std::numbers::pi / 2.);
  expectAngleRecurrence(inner, inner.getInitArgs().front(), 0.5);

  auto controlled = controlledPhase(inner);
  ASSERT_TRUE(controlled.control);
  ASSERT_TRUE(controlled.phase);
  EXPECT_EQ(controlled.phase.getTheta(), inner.getRegionIterArg(0));
  auto controlLoad =
      controlled.control.getControl(0).getDefiningOp<memref::LoadOp>();
  auto phaseTargetLoad =
      controlled.control.getTarget(0).getDefiningOp<memref::LoadOp>();
  ASSERT_TRUE(controlLoad);
  ASSERT_TRUE(phaseTargetLoad);
  EXPECT_EQ(controlLoad.getMemref(), qubitRegister);
  EXPECT_EQ(phaseTargetLoad.getMemref(), qubitRegister);
  EXPECT_EQ(phaseTargetLoad.getIndices().front(), target);
  auto controlExpression =
      controlLoad.getIndices().front().getDefiningOp<arith::SubIOp>();
  ASSERT_TRUE(controlExpression);
  EXPECT_EQ(controlExpression.getRhs(), inner.getInductionVar());
  auto previous = controlExpression.getLhs().getDefiningOp<arith::SubIOp>();
  ASSERT_TRUE(previous);
  expectConstantIndex(previous.getLhs(), qubits - 2);
  EXPECT_EQ(previous.getRhs(), forward.getInductionVar());
}

inline void expectInverseQFT(scf::ForOp inverse, Value qubitRegister) {
  ASSERT_EQ(inverse.getInitArgs().size(), 1U);
  expectConstantFloat(inverse.getInitArgs().front(), -std::numbers::pi);
  auto innerLoops = nestedLoops(inverse);
  ASSERT_EQ(innerLoops.size(), 1U);
  auto inner = innerLoops.front();
  expectConstantIndex(inner.getLowerBound(), 0);
  EXPECT_EQ(inner.getUpperBound(), inverse.getInductionVar());
  expectConstantIndex(inner.getStep(), 1);
  expectAngleRecurrence(inner, inverse.getRegionIterArg(0), 2.);

  auto controlled = controlledPhase(inner);
  ASSERT_TRUE(controlled.control);
  ASSERT_TRUE(controlled.phase);
  EXPECT_EQ(controlled.phase.getTheta(), inner.getRegionIterArg(0));
  auto controlLoad =
      controlled.control.getControl(0).getDefiningOp<memref::LoadOp>();
  auto targetLoad =
      controlled.control.getTarget(0).getDefiningOp<memref::LoadOp>();
  ASSERT_TRUE(controlLoad);
  ASSERT_TRUE(targetLoad);
  EXPECT_EQ(controlLoad.getMemref(), qubitRegister);
  EXPECT_EQ(controlLoad.getIndices().front(), inner.getInductionVar());
  EXPECT_EQ(targetLoad.getMemref(), qubitRegister);
  EXPECT_EQ(targetLoad.getIndices().front(), inverse.getInductionVar());

  qc::HOp inverseH;
  inverse.walk([&](qc::HOp op) { inverseH = op; });
  ASSERT_TRUE(inverseH);
  EXPECT_TRUE(inner->isBeforeInBlock(inverseH));
  auto hLoad = inverseH.getQubit(0).getDefiningOp<memref::LoadOp>();
  ASSERT_TRUE(hLoad);
  EXPECT_EQ(hLoad.getMemref(), qubitRegister);
  EXPECT_EQ(hLoad.getIndices().front(), inverse.getInductionVar());

  auto yield = dyn_cast<scf::YieldOp>(inverse.getBody()->getTerminator());
  ASSERT_TRUE(yield);
  ASSERT_EQ(yield.getNumOperands(), 1U);
  auto nextAngle = yield.getOperand(0).getDefiningOp<arith::MulFOp>();
  ASSERT_TRUE(nextAngle);
  EXPECT_EQ(nextAngle.getLhs(), inverse.getRegionIterArg(0));
  expectConstantFloat(nextAngle.getRhs(), 0.5);
}

} // namespace mqt::bench::test
