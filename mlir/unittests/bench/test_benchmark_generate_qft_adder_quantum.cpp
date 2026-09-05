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
#include <llvm/ADT/SmallVector.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numbers>
#include <utility>

namespace mqt::bench {

using namespace mlir;

namespace {

struct ControlledPhase {
  qc::CtrlOp control;
  qc::POp phase;
};

} // namespace

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

static void expectStaticLoop(scf::ForOp loop, int64_t lower, int64_t upper) {
  expectConstantIndex(loop.getLowerBound(), lower);
  expectConstantIndex(loop.getUpperBound(), upper);
  expectConstantIndex(loop.getStep(), 1);
}

[[nodiscard]] static SmallVector<scf::ForOp> topLevelLoops(ModuleOp moduleOp) {
  SmallVector<scf::ForOp> loops;
  moduleOp.walk([&](scf::ForOp loop) {
    if (!loop->getParentOfType<scf::ForOp>()) {
      loops.push_back(loop);
    }
  });
  return loops;
}

[[nodiscard]] static SmallVector<scf::ForOp> nestedLoops(scf::ForOp outer) {
  SmallVector<scf::ForOp> loops;
  outer.walk([&](scf::ForOp loop) {
    if (loop != outer) {
      loops.push_back(loop);
    }
  });
  return loops;
}

static void expectAngleRecurrence(scf::ForOp loop, Value initialAngle,
                                  double factor) {
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

[[nodiscard]] static ControlledPhase controlledPhase(scf::ForOp loop) {
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

TEST(GenerateProgramTest, EmitsExactQuantumQFTAdderSchedule) {
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
  EXPECT_EQ(test::countOps<qc::ResetOp>(moduleOp), 0U);
  EXPECT_EQ(test::countOps<scf::ForOp>(moduleOp), 9U);
  size_t unitaries = 0;
  moduleOp.walk([&](qc::UnitaryOpInterface /*unused*/) { ++unitaries; });
  EXPECT_EQ(unitaries, 10U);

  cbit::AllocOp resultAllocation;
  moduleOp.walk([&](cbit::AllocOp op) { resultAllocation = op; });
  ASSERT_TRUE(resultAllocation);
  EXPECT_EQ(resultAllocation.getResult().getType().getWidth(), 2 * qubits);

  auto loops = topLevelLoops(moduleOp);
  ASSERT_EQ(loops.size(), 6U);
  for (auto loop : loops) {
    expectStaticLoop(loop, 0, qubits);
  }

  qc::HOp prepareAddend;
  loops[0].walk([&](qc::HOp op) { prepareAddend = op; });
  ASSERT_TRUE(prepareAddend);
  auto addendLoad = prepareAddend.getQubit(0).getDefiningOp<memref::LoadOp>();
  ASSERT_TRUE(addendLoad);
  EXPECT_EQ(addendLoad.getIndices().front(), loops[0].getInductionVar());
  auto addend = addendLoad.getMemref();

  qc::XOp prepareOne;
  moduleOp.walk([&](qc::XOp op) { prepareOne = op; });
  ASSERT_TRUE(prepareOne);
  auto sumLoad = prepareOne.getQubit(0).getDefiningOp<memref::LoadOp>();
  ASSERT_TRUE(sumLoad);
  expectConstantIndex(sumLoad.getIndices().front(), 0);
  auto sum = sumLoad.getMemref();
  EXPECT_NE(addend, sum);
  EXPECT_TRUE(loops[0]->isBeforeInBlock(prepareOne));
  EXPECT_TRUE(prepareOne->isBeforeInBlock(loops[1]));

  // The positive, no-swap QFT visits t = n - 1 ... 0. For each t, it
  // applies H(y[t]) before CP(pi / 2^(t-c), y[c], y[t]) for c = t - 1 ... 0.
  auto forward = loops[1];
  qc::HOp forwardH;
  forward.walk([&](qc::HOp op) { forwardH = op; });
  ASSERT_TRUE(forwardH);
  auto forwardTargetLoad = forwardH.getQubit(0).getDefiningOp<memref::LoadOp>();
  ASSERT_TRUE(forwardTargetLoad);
  EXPECT_EQ(forwardTargetLoad.getMemref(), sum);
  auto forwardTarget = forwardTargetLoad.getIndices().front();
  auto forwardTargetExpression = forwardTarget.getDefiningOp<arith::SubIOp>();
  ASSERT_TRUE(forwardTargetExpression);
  expectConstantIndex(forwardTargetExpression.getLhs(), qubits - 1);
  EXPECT_EQ(forwardTargetExpression.getRhs(), forward.getInductionVar());

  auto forwardInnerLoops = nestedLoops(forward);
  ASSERT_EQ(forwardInnerLoops.size(), 1U);
  auto forwardInner = forwardInnerLoops.front();
  EXPECT_TRUE(forwardH->isBeforeInBlock(forwardInner));
  expectConstantIndex(forwardInner.getLowerBound(), 0);
  EXPECT_EQ(forwardInner.getUpperBound(), forwardTarget);
  expectConstantIndex(forwardInner.getStep(), 1);
  ASSERT_EQ(forwardInner.getInitArgs().size(), 1U);
  expectConstantFloat(forwardInner.getInitArgs().front(),
                      std::numbers::pi / 2.);
  expectAngleRecurrence(forwardInner, forwardInner.getInitArgs().front(), 0.5);
  auto forwardPhase = controlledPhase(forwardInner);
  ASSERT_TRUE(forwardPhase.control);
  ASSERT_TRUE(forwardPhase.phase);
  EXPECT_EQ(forwardPhase.phase.getTheta(), forwardInner.getRegionIterArg(0));
  auto forwardControl =
      forwardPhase.control.getControl(0).getDefiningOp<memref::LoadOp>();
  auto forwardPhaseTarget =
      forwardPhase.control.getTarget(0).getDefiningOp<memref::LoadOp>();
  ASSERT_TRUE(forwardControl);
  ASSERT_TRUE(forwardPhaseTarget);
  EXPECT_EQ(forwardControl.getMemref(), sum);
  EXPECT_EQ(forwardPhaseTarget.getMemref(), sum);
  EXPECT_EQ(forwardPhaseTarget.getIndices().front(), forwardTarget);
  auto forwardControlExpression =
      forwardControl.getIndices().front().getDefiningOp<arith::SubIOp>();
  ASSERT_TRUE(forwardControlExpression);
  EXPECT_EQ(forwardControlExpression.getRhs(), forwardInner.getInductionVar());
  auto previous =
      forwardControlExpression.getLhs().getDefiningOp<arith::SubIOp>();
  ASSERT_TRUE(previous);
  expectConstantIndex(previous.getLhs(), qubits - 2);
  EXPECT_EQ(previous.getRhs(), forward.getInductionVar());

  // Draper's addition visits t = n - 1 ... 0 and c = t ... 0. The first
  // gate at each target is CP(pi), including the matching source bit.
  auto addition = loops[2];
  EXPECT_TRUE(forward->isBeforeInBlock(addition));
  auto additionInnerLoops = nestedLoops(addition);
  ASSERT_EQ(additionInnerLoops.size(), 1U);
  auto additionInner = additionInnerLoops.front();
  expectConstantIndex(additionInner.getLowerBound(), 0);
  auto additionUpper =
      additionInner.getUpperBound().getDefiningOp<arith::SubIOp>();
  ASSERT_TRUE(additionUpper);
  expectConstantIndex(additionUpper.getLhs(), qubits);
  EXPECT_EQ(additionUpper.getRhs(), addition.getInductionVar());
  expectConstantIndex(additionInner.getStep(), 1);
  ASSERT_EQ(additionInner.getInitArgs().size(), 1U);
  expectConstantFloat(additionInner.getInitArgs().front(), std::numbers::pi);
  expectAngleRecurrence(additionInner, additionInner.getInitArgs().front(),
                        0.5);
  auto additionPhase = controlledPhase(additionInner);
  ASSERT_TRUE(additionPhase.control);
  ASSERT_TRUE(additionPhase.phase);
  EXPECT_EQ(additionPhase.phase.getTheta(), additionInner.getRegionIterArg(0));
  auto sourceControl =
      additionPhase.control.getControl(0).getDefiningOp<memref::LoadOp>();
  auto sumTarget =
      additionPhase.control.getTarget(0).getDefiningOp<memref::LoadOp>();
  ASSERT_TRUE(sourceControl);
  ASSERT_TRUE(sumTarget);
  auto additionTarget = sumTarget.getIndices().front();
  auto descendingAdditionTarget = additionTarget.getDefiningOp<arith::SubIOp>();
  ASSERT_TRUE(descendingAdditionTarget);
  expectConstantIndex(descendingAdditionTarget.getLhs(), qubits - 1);
  EXPECT_EQ(descendingAdditionTarget.getRhs(), addition.getInductionVar());
  EXPECT_EQ(sourceControl.getMemref(), addend);
  EXPECT_EQ(sumTarget.getMemref(), sum);
  EXPECT_EQ(sumTarget.getIndices().front(), additionTarget);
  auto sourceIndex =
      sourceControl.getIndices().front().getDefiningOp<arith::SubIOp>();
  ASSERT_TRUE(sourceIndex);
  EXPECT_EQ(sourceIndex.getLhs(), additionTarget);
  EXPECT_EQ(sourceIndex.getRhs(), additionInner.getInductionVar());

  // The inverse reverses every QFT gate: c = 0 ... t - 1, then H(y[t]),
  // while its first inner-loop angle progresses as -pi / 2^t.
  auto inverse = loops[3];
  EXPECT_TRUE(addition->isBeforeInBlock(inverse));
  ASSERT_EQ(inverse.getInitArgs().size(), 1U);
  expectConstantFloat(inverse.getInitArgs().front(), -std::numbers::pi);
  auto inverseInnerLoops = nestedLoops(inverse);
  ASSERT_EQ(inverseInnerLoops.size(), 1U);
  auto inverseInner = inverseInnerLoops.front();
  expectConstantIndex(inverseInner.getLowerBound(), 0);
  EXPECT_EQ(inverseInner.getUpperBound(), inverse.getInductionVar());
  expectConstantIndex(inverseInner.getStep(), 1);
  expectAngleRecurrence(inverseInner, inverse.getRegionIterArg(0), 2.);
  auto inversePhase = controlledPhase(inverseInner);
  ASSERT_TRUE(inversePhase.control);
  ASSERT_TRUE(inversePhase.phase);
  EXPECT_EQ(inversePhase.phase.getTheta(), inverseInner.getRegionIterArg(0));
  auto inverseControl =
      inversePhase.control.getControl(0).getDefiningOp<memref::LoadOp>();
  auto inverseTarget =
      inversePhase.control.getTarget(0).getDefiningOp<memref::LoadOp>();
  ASSERT_TRUE(inverseControl);
  ASSERT_TRUE(inverseTarget);
  EXPECT_EQ(inverseControl.getMemref(), sum);
  EXPECT_EQ(inverseControl.getIndices().front(),
            inverseInner.getInductionVar());
  EXPECT_EQ(inverseTarget.getMemref(), sum);
  EXPECT_EQ(inverseTarget.getIndices().front(), inverse.getInductionVar());
  qc::HOp inverseH;
  inverse.walk([&](qc::HOp op) { inverseH = op; });
  ASSERT_TRUE(inverseH);
  EXPECT_TRUE(inverseInner->isBeforeInBlock(inverseH));
  auto inverseHLoad = inverseH.getQubit(0).getDefiningOp<memref::LoadOp>();
  ASSERT_TRUE(inverseHLoad);
  EXPECT_EQ(inverseHLoad.getMemref(), sum);
  EXPECT_EQ(inverseHLoad.getIndices().front(), inverse.getInductionVar());
  auto inverseYield =
      dyn_cast<scf::YieldOp>(inverse.getBody()->getTerminator());
  ASSERT_TRUE(inverseYield);
  ASSERT_EQ(inverseYield.getNumOperands(), 1U);
  auto nextInverseAngle =
      inverseYield.getOperand(0).getDefiningOp<arith::MulFOp>();
  ASSERT_TRUE(nextInverseAngle);
  EXPECT_EQ(nextInverseAngle.getLhs(), inverse.getRegionIterArg(0));
  expectConstantFloat(nextInverseAngle.getRhs(), 0.5);

  // Register bit zero is least significant. Measuring sum at i and addend at
  // n + i makes the displayed big-endian result `addend || sum`.
  auto sumMeasurementLoop = loops[4];
  auto addendMeasurementLoop = loops[5];
  EXPECT_TRUE(inverse->isBeforeInBlock(sumMeasurementLoop));
  EXPECT_TRUE(sumMeasurementLoop->isBeforeInBlock(addendMeasurementLoop));
  qc::MeasureOp sumMeasurement;
  sumMeasurementLoop.walk([&](qc::MeasureOp op) { sumMeasurement = op; });
  qc::MeasureOp addendMeasurement;
  addendMeasurementLoop.walk([&](qc::MeasureOp op) { addendMeasurement = op; });
  ASSERT_TRUE(sumMeasurement);
  ASSERT_TRUE(addendMeasurement);
  auto measuredSum = sumMeasurement.getQubit().getDefiningOp<memref::LoadOp>();
  auto measuredAddend =
      addendMeasurement.getQubit().getDefiningOp<memref::LoadOp>();
  ASSERT_TRUE(measuredSum);
  ASSERT_TRUE(measuredAddend);
  EXPECT_EQ(measuredSum.getMemref(), sum);
  EXPECT_EQ(measuredSum.getIndices().front(),
            sumMeasurementLoop.getInductionVar());
  EXPECT_EQ(measuredAddend.getMemref(), addend);
  EXPECT_EQ(measuredAddend.getIndices().front(),
            addendMeasurementLoop.getInductionVar());

  auto sumStore =
      dyn_cast<cbit::StoreOp>(*sumMeasurement.getResult().user_begin());
  auto addendStore =
      dyn_cast<cbit::StoreOp>(*addendMeasurement.getResult().user_begin());
  ASSERT_TRUE(sumStore);
  ASSERT_TRUE(addendStore);
  EXPECT_EQ(sumStore.getReg(), resultAllocation.getResult());
  EXPECT_EQ(sumStore.getIndex(), sumMeasurementLoop.getInductionVar());
  EXPECT_EQ(addendStore.getReg(), resultAllocation.getResult());
  auto displayedAddendIndex =
      addendStore.getIndex().getDefiningOp<arith::AddIOp>();
  ASSERT_TRUE(displayedAddendIndex);
  if (displayedAddendIndex.getLhs() ==
      addendMeasurementLoop.getInductionVar()) {
    expectConstantIndex(displayedAddendIndex.getRhs(), qubits);
  } else {
    expectConstantIndex(displayedAddendIndex.getLhs(), qubits);
    EXPECT_EQ(displayedAddendIndex.getRhs(),
              addendMeasurementLoop.getInductionVar());
  }
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
  test::expectJeffRoundTrip(std::move(*program));
}

} // namespace mqt::bench
