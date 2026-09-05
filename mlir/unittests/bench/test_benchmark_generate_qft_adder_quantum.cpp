/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "QFTAdderTestUtils.h"
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
using test::controlledPhase;
using test::expectAngleRecurrence;
using test::expectConstantFloat;
using test::expectConstantIndex;
using test::expectStaticLoop;
using test::nestedLoops;
using test::topLevelLoops;

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

  auto forward = loops[1];
  test::expectForwardQFT(forward, sum, qubits);

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

  auto inverse = loops[3];
  EXPECT_TRUE(addition->isBeforeInBlock(inverse));
  test::expectInverseQFT(inverse, sum);

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
