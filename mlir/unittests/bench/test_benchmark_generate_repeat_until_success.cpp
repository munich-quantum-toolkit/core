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
#include "bench/RepeatUntilSuccess.hpp"
#include "mlir/Dialect/CBit/IR/CBitOps.h"
#include "mlir/Dialect/QC/IR/QCOps.h"
#include "mlir/bench/Generate.h"

#include <gtest/gtest.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>

#include <utility>

namespace mqt::bench {

using namespace mlir;

static void expectCNOT(qc::CtrlOp control, Value expectedControl,
                       Value expectedTarget) {
  ASSERT_TRUE(control);
  ASSERT_EQ(control.getNumControls(), 1U);
  ASSERT_EQ(control.getNumTargets(), 1U);
  EXPECT_EQ(control.getControl(0), expectedControl);
  EXPECT_EQ(control.getTarget(0), expectedTarget);
  ASSERT_EQ(control.getNumBodyUnitaries(), 1U);
  EXPECT_TRUE(isa<qc::XOp>(control.getBodyUnitary(0).getOperation()));
}

TEST(GenerateProgramTest, EmitsExactRepeatUntilSuccessSchedule) {
  auto program = generate(RepeatUntilSuccess{});
  ASSERT_TRUE(program);
  auto moduleOp = program->module();

  EXPECT_EQ(test::countOps<qc::AllocOp>(moduleOp), 2U);
  EXPECT_EQ(test::countOps<cbit::AllocOp>(moduleOp), 1U);
  EXPECT_EQ(test::countOps<scf::WhileOp>(moduleOp), 1U);
  EXPECT_EQ(test::countOps<scf::IfOp>(moduleOp), 0U);
  EXPECT_EQ(test::countOps<qc::HOp>(moduleOp), 4U);
  EXPECT_EQ(test::countOps<qc::TOp>(moduleOp), 2U);
  EXPECT_EQ(test::countOps<qc::CtrlOp>(moduleOp), 2U);
  EXPECT_EQ(test::countOps<qc::XOp>(moduleOp), 3U);
  EXPECT_EQ(test::countOps<qc::SdgOp>(moduleOp), 1U);
  EXPECT_EQ(test::countOps<qc::MeasureOp>(moduleOp), 2U);
  EXPECT_EQ(test::countOps<qc::ResetOp>(moduleOp), 0U);

  SmallVector<scf::WhileOp> loops;
  moduleOp.walk([&](scf::WhileOp loop) { loops.push_back(loop); });
  ASSERT_EQ(loops.size(), 1U);
  auto loop = loops.front();
  EXPECT_TRUE(loop.getInits().empty());
  EXPECT_TRUE(loop.getResults().empty());

  SmallVector<Operation*> attempt;
  for (Operation& operation : loop.getBefore().front().without_terminator()) {
    attempt.push_back(&operation);
  }
  ASSERT_EQ(attempt.size(), 8U);

  auto firstH = dyn_cast<qc::HOp>(attempt[0]);
  auto firstT = dyn_cast<qc::TOp>(attempt[1]);
  auto firstCNOT = dyn_cast<qc::CtrlOp>(attempt[2]);
  auto middleH = dyn_cast<qc::HOp>(attempt[3]);
  auto secondCNOT = dyn_cast<qc::CtrlOp>(attempt[4]);
  auto secondT = dyn_cast<qc::TOp>(attempt[5]);
  auto finalH = dyn_cast<qc::HOp>(attempt[6]);
  auto ancillaMeasurement = dyn_cast<qc::MeasureOp>(attempt[7]);
  ASSERT_TRUE(firstH);
  ASSERT_TRUE(firstT);
  ASSERT_TRUE(firstCNOT);
  ASSERT_TRUE(middleH);
  ASSERT_TRUE(secondCNOT);
  ASSERT_TRUE(secondT);
  ASSERT_TRUE(finalH);
  ASSERT_TRUE(ancillaMeasurement);

  auto ancilla = firstH.getQubit(0);
  auto data = firstCNOT.getTarget(0);
  EXPECT_NE(ancilla, data);
  EXPECT_EQ(firstT.getQubit(0), ancilla);
  expectCNOT(firstCNOT, ancilla, data);
  EXPECT_EQ(middleH.getQubit(0), ancilla);
  expectCNOT(secondCNOT, ancilla, data);
  EXPECT_EQ(secondT.getQubit(0), ancilla);
  EXPECT_EQ(finalH.getQubit(0), ancilla);
  EXPECT_EQ(ancillaMeasurement.getQubit(), ancilla);
  EXPECT_EQ(loop.getConditionOp().getCondition(),
            ancillaMeasurement.getResult());
  EXPECT_TRUE(loop.getConditionOp().getArgs().empty());

  SmallVector<Operation*> retry;
  for (Operation& operation : loop.getAfter().front().without_terminator()) {
    retry.push_back(&operation);
  }
  ASSERT_EQ(retry.size(), 1U);
  auto reprepareAncilla = dyn_cast<qc::XOp>(retry.front());
  ASSERT_TRUE(reprepareAncilla);
  EXPECT_EQ(reprepareAncilla.getQubit(0), ancilla);

  qc::SdgOp readoutPhase;
  qc::HOp readoutH;
  qc::MeasureOp readoutMeasurement;
  moduleOp.walk([&](qc::SdgOp op) { readoutPhase = op; });
  moduleOp.walk([&](qc::HOp op) {
    if (!op->getParentOfType<scf::WhileOp>()) {
      readoutH = op;
    }
  });
  moduleOp.walk([&](qc::MeasureOp op) {
    if (!op->getParentOfType<scf::WhileOp>()) {
      readoutMeasurement = op;
    }
  });
  ASSERT_TRUE(readoutPhase);
  ASSERT_TRUE(readoutH);
  ASSERT_TRUE(readoutMeasurement);
  EXPECT_EQ(readoutPhase.getQubit(0), data);
  EXPECT_EQ(readoutH.getQubit(0), data);
  EXPECT_EQ(readoutMeasurement.getQubit(), data);
  EXPECT_TRUE(loop->isBeforeInBlock(readoutPhase));
  EXPECT_TRUE(readoutPhase->isBeforeInBlock(readoutH));
  EXPECT_TRUE(readoutH->isBeforeInBlock(readoutMeasurement));

  SmallVector<cbit::StoreOp> stores;
  moduleOp.walk([&](cbit::StoreOp store) { stores.push_back(store); });
  ASSERT_EQ(stores.size(), 1U);
  auto store = stores.front();
  EXPECT_EQ(store.getValue(), readoutMeasurement.getResult());
  auto result = store.getReg().getDefiningOp<cbit::AllocOp>();
  ASSERT_TRUE(result);
  EXPECT_EQ(result.getResult().getType().getWidth(), 1);
  auto index = store.getIndex().getDefiningOp<arith::ConstantIndexOp>();
  ASSERT_TRUE(index);
  EXPECT_EQ(index.value(), 0);
}

TEST(GenerateProgramTest, SerializesRepeatUntilSuccessControlFlow) {
  auto program = generate(RepeatUntilSuccess{});
  ASSERT_TRUE(program);
  test::expectJeffRoundTrip(std::move(*program));
}

} // namespace mqt::bench
