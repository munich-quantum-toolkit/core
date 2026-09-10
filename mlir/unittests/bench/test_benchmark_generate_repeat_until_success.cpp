/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "bench/RepeatUntilSuccess.hpp"
#include "mqt/Dialect/QC/IR/QCOps.h"
#include "mqt/bench/Generate.h"

#include "TestUtils.h"

#include "gtest/gtest.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/SmallVector.h"

#include <cstddef>

namespace mqt::bench {

using namespace mlir;

TEST(GenerateProgramTest, EmitsRepeatUntilSuccessAlgorithm) {
  auto program = generate(RepeatUntilSuccess{{.dataQubits = 5}});
  ASSERT_TRUE(program);
  auto moduleOp = program->module();
  SmallVector<scf::WhileOp> loops;
  moduleOp.walk([&](scf::WhileOp loop) { loops.push_back(loop); });
  ASSERT_EQ(loops.size(), 1U);
  auto loop = loops.front();
  SmallVector<Operation*> attempt;
  for (Operation& operation : loop.getBefore().front().without_terminator()) {
    attempt.push_back(&operation);
  }
  ASSERT_EQ(attempt.size(), 8U);
  auto firstH = dyn_cast<qc::HOp>(attempt[0]);
  auto firstT = dyn_cast<qc::TOp>(attempt[1]);
  auto middleH = dyn_cast<qc::HOp>(attempt[3]);
  auto secondT = dyn_cast<qc::TOp>(attempt[5]);
  auto finalH = dyn_cast<qc::HOp>(attempt[6]);
  auto measurement = dyn_cast<qc::MeasureOp>(attempt[7]);
  ASSERT_TRUE(firstH && firstT && middleH && secondT && finalH && measurement);
  auto ancilla = firstH.getQubit(0);
  EXPECT_EQ(firstT.getQubit(0), ancilla);
  EXPECT_EQ(middleH.getQubit(0), ancilla);
  EXPECT_EQ(secondT.getQubit(0), ancilla);
  EXPECT_EQ(finalH.getQubit(0), ancilla);
  EXPECT_EQ(measurement.getQubit(), ancilla);
  EXPECT_EQ(loop.getConditionOp().getCondition(), measurement.getResult());

  Value data;
  for (auto index : {2, 4}) {
    auto sweep = dyn_cast<scf::ForOp>(attempt[index]);
    ASSERT_TRUE(sweep);
    EXPECT_EQ(getConstantIntValue(sweep.getLowerBound()), 0);
    EXPECT_EQ(getConstantIntValue(sweep.getUpperBound()), 5);
    EXPECT_EQ(sweep.getConstantStep(), 1);
    SmallVector<qc::CtrlOp> controls;
    sweep.walk([&](qc::CtrlOp op) { controls.push_back(op); });
    ASSERT_EQ(controls.size(), 1U);
    auto control = controls.front();
    ASSERT_EQ(control.getNumControls(), 1U);
    ASSERT_EQ(control.getNumTargets(), 1U);
    EXPECT_EQ(control.getControl(0), ancilla);
    ASSERT_EQ(control.getNumBodyUnitaries(), 1U);
    EXPECT_TRUE(isa<qc::XOp>(control.getBodyUnitary(0).getOperation()));
    auto load = control.getTarget(0).getDefiningOp<memref::LoadOp>();
    ASSERT_TRUE(load);
    ASSERT_EQ(load.getIndices().size(), 1U);
    EXPECT_EQ(load.getIndices().front(), sweep.getInductionVar());
    if (data) {
      EXPECT_EQ(load.getMemRef(), data);
    }
    data = load.getMemRef();
  }

  SmallVector<Operation*> retry;
  for (Operation& operation : loop.getAfter().front().without_terminator()) {
    retry.push_back(&operation);
  }
  ASSERT_EQ(retry.size(), 1U);
  auto cleanup = dyn_cast<qc::XOp>(retry.front());
  ASSERT_TRUE(cleanup);
  EXPECT_EQ(cleanup.getQubit(0), ancilla);
}

TEST(GenerateProgramTest, SamplesRepeatUntilSuccessAgainstReference) {
  /// At 16,384 shots the binomial tail bound for a 0.01 error is below 7e-12.
  /// An all-zero sampler has error 0.0286 and must fail this check.
  for (const size_t width : {1U, 2U, 5U, 32U}) {
    SCOPED_TRACE(width);
    test::expectSamplingMatchesReference(
        RepeatUntilSuccess{{.dataQubits = width}}, 0.01);
  }
}

TEST(GenerateProgramTest, KeepsRepeatUntilSuccessGenerationCompact) {
  auto small = generate(RepeatUntilSuccess{{.dataQubits = 5}});
  auto large = generate(RepeatUntilSuccess{
      {.dataQubits = RepeatUntilSuccessOptions::MAX_DATA_QUBITS}});
  ASSERT_TRUE(small);
  ASSERT_TRUE(large);
  EXPECT_EQ(test::countOperations(small->module()),
            test::countOperations(large->module()));
}

} // namespace mqt::bench
