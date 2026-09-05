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
#include "bench/QFTAdderClassical.hpp"
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
#include <string>
#include <utility>
#include <vector>

namespace mqt::bench {

using namespace mlir;

namespace {

struct ClassicalAdderCase {
  std::string addend;
  std::vector<double> angles;
};

class ClassicalQFTAdderStructureTest
    : public testing::TestWithParam<ClassicalAdderCase> {};

} // namespace

TEST_P(ClassicalQFTAdderStructureTest, EmitsExactClassicalQFTAdderSchedule) {
  const auto& testCase = GetParam();
  const auto qubits = static_cast<int64_t>(testCase.addend.size() + 1U);
  auto program = generate(QFTAdderClassical{{.addend = testCase.addend}});
  ASSERT_TRUE(program);
  auto moduleOp = program->module();

  EXPECT_EQ(test::countOps<memref::AllocOp>(moduleOp), 1U);
  EXPECT_EQ(test::countOps<cbit::AllocOp>(moduleOp), 1U);
  EXPECT_EQ(test::countOps<qc::HOp>(moduleOp), 2U);
  EXPECT_EQ(test::countOps<qc::XOp>(moduleOp), 1U);
  EXPECT_EQ(test::countOps<qc::CtrlOp>(moduleOp), 2U);
  EXPECT_EQ(test::countOps<qc::POp>(moduleOp), testCase.angles.size() + 2U);
  EXPECT_EQ(test::countOps<qc::MeasureOp>(moduleOp), 1U);
  EXPECT_EQ(test::countOps<qc::ResetOp>(moduleOp), 0U);
  EXPECT_EQ(test::countOps<qc::SWAPOp>(moduleOp), 0U);
  EXPECT_EQ(test::countOps<scf::ForOp>(moduleOp), 5U);

  cbit::AllocOp resultAllocation;
  moduleOp.walk([&](cbit::AllocOp op) { resultAllocation = op; });
  ASSERT_TRUE(resultAllocation);
  EXPECT_EQ(resultAllocation.getResult().getType().getWidth(), qubits);

  auto loops = test::topLevelLoops(moduleOp);
  ASSERT_EQ(loops.size(), 3U);
  for (auto loop : loops) {
    test::expectStaticLoop(loop, 0, qubits);
  }

  qc::XOp prepareOne;
  moduleOp.walk([&](qc::XOp op) { prepareOne = op; });
  ASSERT_TRUE(prepareOne);
  auto oneLoad = prepareOne.getQubit(0).getDefiningOp<memref::LoadOp>();
  ASSERT_TRUE(oneLoad);
  test::expectConstantIndex(oneLoad.getIndices().front(), 0);
  auto sum = oneLoad.getMemref();

  auto forward = loops[0];
  EXPECT_TRUE(prepareOne->isBeforeInBlock(forward));
  test::expectForwardQFT(forward, sum, qubits);

  // Beauregard's classical-input optimization combines all known rotations
  // into one unconditional phase per wire, including the overflow wire.
  SmallVector<qc::POp> additionPhases;
  moduleOp.walk([&](qc::POp op) {
    if (!op->getParentOfType<qc::CtrlOp>()) {
      additionPhases.push_back(op);
    }
  });
  ASSERT_EQ(additionPhases.size(), testCase.angles.size());
  for (size_t target = 0; target < additionPhases.size(); ++target) {
    auto phase = additionPhases[target];
    EXPECT_FALSE(phase->getParentOfType<qc::CtrlOp>());
    test::expectConstantFloat(phase.getTheta(), testCase.angles[target]);
    auto targetLoad = phase.getQubit(0).getDefiningOp<memref::LoadOp>();
    ASSERT_TRUE(targetLoad);
    EXPECT_EQ(targetLoad.getMemref(), sum);
    test::expectConstantIndex(targetLoad.getIndices().front(),
                              static_cast<int64_t>(target));
    EXPECT_TRUE(forward->isBeforeInBlock(phase));
    if (target != 0U) {
      EXPECT_TRUE(additionPhases[target - 1]->isBeforeInBlock(phase));
    }
  }

  auto inverse = loops[1];
  EXPECT_TRUE(additionPhases.back()->isBeforeInBlock(inverse));
  test::expectInverseQFT(inverse, sum);

  // Register bit zero is least significant. Equal source and destination
  // indices therefore produce the declared big-endian result register.
  auto measurementLoop = loops[2];
  EXPECT_TRUE(inverse->isBeforeInBlock(measurementLoop));
  qc::MeasureOp measurement;
  measurementLoop.walk([&](qc::MeasureOp op) { measurement = op; });
  ASSERT_TRUE(measurement);
  auto measured = measurement.getQubit().getDefiningOp<memref::LoadOp>();
  ASSERT_TRUE(measured);
  EXPECT_EQ(measured.getMemref(), sum);
  EXPECT_EQ(measured.getIndices().front(), measurementLoop.getInductionVar());
  auto store = dyn_cast<cbit::StoreOp>(*measurement.getResult().user_begin());
  ASSERT_TRUE(store);
  EXPECT_EQ(store.getReg(), resultAllocation.getResult());
  EXPECT_EQ(store.getIndex(), measurementLoop.getInductionVar());
}

TEST(GenerateProgramTest, KeepsLargestClassicalQFTAdderFiniteAndSerializable) {
  auto addend = std::string(QFTAdderClassicalOptions::MAX_ADDEND_BITS, '1');
  auto program = generate(QFTAdderClassical{{.addend = std::move(addend)}});
  ASSERT_TRUE(program);
  auto moduleOp = program->module();

  EXPECT_EQ(test::countOps<qc::POp>(moduleOp),
            QFTAdderClassicalOptions::MAX_ADDEND_BITS + 3U);
  const auto operations = test::countOperations(moduleOp);
  EXPECT_GT(operations, QFTAdderClassicalOptions::MAX_ADDEND_BITS);
  EXPECT_LT(operations, 5U * QFTAdderClassicalOptions::MAX_ADDEND_BITS);
  moduleOp.walk([&](arith::ConstantOp op) {
    if (const auto value = dyn_cast<FloatAttr>(op.getValue())) {
      EXPECT_TRUE(std::isfinite(value.getValueAsDouble()));
    }
  });
  test::expectJeffRoundTrip(std::move(*program));
}

INSTANTIATE_TEST_SUITE_P(
    ExactPhases, ClassicalQFTAdderStructureTest,
    testing::Values(ClassicalAdderCase{"0", {0., 0.}},
                    ClassicalAdderCase{
                        "1", {std::numbers::pi, std::numbers::pi / 2.}},
                    ClassicalAdderCase{"101",
                                       {
                                           std::numbers::pi,
                                           std::numbers::pi / 2.,
                                           5. * std::numbers::pi / 4.,
                                           5. * std::numbers::pi / 8.,
                                       }},
                    ClassicalAdderCase{"111",
                                       {
                                           std::numbers::pi,
                                           3. * std::numbers::pi / 2.,
                                           7. * std::numbers::pi / 4.,
                                           7. * std::numbers::pi / 8.,
                                       }},
                    ClassicalAdderCase{"110",
                                       {
                                           0.,
                                           std::numbers::pi,
                                           3. * std::numbers::pi / 2.,
                                           3. * std::numbers::pi / 4.,
                                       }},
                    ClassicalAdderCase{"001",
                                       {
                                           std::numbers::pi,
                                           std::numbers::pi / 2.,
                                           std::numbers::pi / 4.,
                                           std::numbers::pi / 8.,
                                       }}));

} // namespace mqt::bench
