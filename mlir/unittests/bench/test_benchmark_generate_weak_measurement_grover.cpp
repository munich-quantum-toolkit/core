/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "bench/WeakMeasurementGrover.hpp"
#include "mqt/Dialect/QC/IR/QCOps.h"
#include "mqt/bench/Generate.h"

#include "TestUtils.h"

#include "gtest/gtest.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/SmallVector.h"

#include <numbers>

namespace mqt::bench {

using namespace mlir;

TEST(GenerateProgramTest, EmitsWeakMeasurementAfterEachGroverIteration) {
  auto program = generate(WeakMeasurementGrover{
      {.markedBitstring = "01", .measurementStrength = 0.5}});
  ASSERT_TRUE(program);
  auto moduleOp = program->module();

  SmallVector<scf::WhileOp> loops;
  moduleOp.walk([&](scf::WhileOp loop) { loops.push_back(loop); });
  ASSERT_EQ(loops.size(), 1U);
  auto loop = loops.front();

  SmallVector<qc::CtrlOp> controls;
  loop.walk([&](qc::CtrlOp control) { controls.push_back(control); });
  ASSERT_EQ(controls.size(), 6U);
  for (auto control : controls) {
    ASSERT_EQ(control.getNumTargets(), 1U);
    ASSERT_EQ(control.getNumBodyUnitaries(), 1U);
  }
  EXPECT_EQ(controls[0].getNumControls(), 1U); // Grover oracle
  EXPECT_EQ(controls[1].getNumControls(), 1U); // Grover diffusion
  EXPECT_EQ(controls[2].getNumControls(), 2U); // Compute chi
  EXPECT_EQ(controls[5].getNumControls(), 2U); // Uncompute chi
  for (const auto index : {0U, 1U, 2U, 3U, 5U}) {
    EXPECT_TRUE(isa<qc::ZOp>(controls[index].getBodyUnitary(0).getOperation()));
  }
  auto rotation =
      dyn_cast<qc::RYOp>(controls[4].getBodyUnitary(0).getOperation());
  ASSERT_TRUE(rotation);

  auto work = controls[2].getTarget(0);
  EXPECT_EQ(controls[3].getControl(0), work);
  EXPECT_EQ(controls[4].getControl(0), work);
  EXPECT_EQ(controls[5].getTarget(0), work);
  EXPECT_EQ(controls[3]->getNextNode(), controls[4].getOperation());

  SmallVector<qc::MeasureOp> loopMeasurements;
  loop.walk([&](qc::MeasureOp measurement) {
    loopMeasurements.push_back(measurement);
  });
  ASSERT_EQ(loopMeasurements.size(), 1U);
  auto detection = loopMeasurements.front();
  EXPECT_EQ(controls[3].getTarget(0), detection.getQubit());
  EXPECT_TRUE(controls[5]->isBeforeInBlock(detection.getOperation()));
  auto keepSearching =
      loop.getConditionOp().getCondition().getDefiningOp<arith::XOrIOp>();
  ASSERT_TRUE(keepSearching);
  EXPECT_EQ(keepSearching.getLhs(), detection.getResult());
  auto one = keepSearching.getRhs().getDefiningOp<arith::ConstantIntOp>();
  ASSERT_TRUE(one);
  EXPECT_TRUE(one.value());

  auto angle = rotation.getTheta().getDefiningOp<arith::ConstantOp>();
  ASSERT_TRUE(angle);
  auto angleValue = dyn_cast<FloatAttr>(angle.getValue());
  ASSERT_TRUE(angleValue);
  EXPECT_NEAR(angleValue.getValueAsDouble(), std::numbers::pi / 2., 1e-15);
}

TEST(GenerateProgramTest, SamplesWeakMeasurementGroverAgainstReference) {
  auto program =
      test::generateQCO(WeakMeasurementGrover{{.markedBitstring = "110"}});
  ASSERT_TRUE(program);
  auto counts =
      qco::sample(mlir::mqt::getEntryPoint(program->module()), 64, 17);
  ASSERT_TRUE(succeeded(counts));
  EXPECT_EQ(*counts, (Counts{{"110", 64}}));
}

} // namespace mqt::bench
