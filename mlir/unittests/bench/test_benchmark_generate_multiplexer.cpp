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
#include "bench/Multiplexer.hpp"
#include "mlir/Dialect/QC/IR/QCOps.h"
#include "mlir/bench/Generate.h"

#include <gtest/gtest.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Support/LLVM.h>

#include <cstdint>
#include <numbers>
#include <utility>

namespace mqt::bench {

using namespace mlir;

TEST(GenerateProgramTest, EmitsUniformLinearQuantumMultiplexer) {
  auto program = generate(Multiplexer{{.qubits = 3}});
  ASSERT_TRUE(program);
  auto moduleOp = program->module();

  EXPECT_EQ(test::countOps<qc::HOp>(moduleOp), 1U);
  EXPECT_EQ(test::countOps<qc::CtrlOp>(moduleOp), 1U);
  EXPECT_EQ(test::countOps<qc::RYOp>(moduleOp), 1U);
  EXPECT_EQ(test::countOps<qc::XOp>(moduleOp), 0U);

  qc::CtrlOp controlledRotation;
  moduleOp.walk([&](qc::CtrlOp op) { controlledRotation = op; });
  ASSERT_TRUE(controlledRotation);
  EXPECT_EQ(controlledRotation.getNumControls(), 1U);

  auto rotationLoop = controlledRotation->getParentOfType<scf::ForOp>();
  ASSERT_TRUE(rotationLoop);
  ASSERT_EQ(rotationLoop.getInitArgs().size(), 1U);
  auto upper =
      rotationLoop.getUpperBound().getDefiningOp<arith::ConstantIndexOp>();
  ASSERT_TRUE(upper);
  EXPECT_EQ(upper.value(), 2);

  auto controlLoad =
      controlledRotation.getControl(0).getDefiningOp<memref::LoadOp>();
  ASSERT_TRUE(controlLoad);
  auto controlIndex =
      controlLoad.getIndices().front().getDefiningOp<arith::SubIOp>();
  ASSERT_TRUE(controlIndex);
  EXPECT_EQ(controlIndex.getRhs(), rotationLoop.getInductionVar());
  auto lastControl =
      controlIndex.getLhs().getDefiningOp<arith::ConstantIndexOp>();
  ASSERT_TRUE(lastControl);
  EXPECT_EQ(lastControl.value(), 1);

  auto firstAngle =
      rotationLoop.getInitArgs().front().getDefiningOp<arith::ConstantOp>();
  ASSERT_TRUE(firstAngle);
  auto firstAngleValue = dyn_cast<FloatAttr>(firstAngle.getValue());
  ASSERT_TRUE(firstAngleValue);
  EXPECT_DOUBLE_EQ(firstAngleValue.getValueAsDouble(), std::numbers::pi / 2.);

  arith::MulFOp scaleAngle;
  rotationLoop.walk([&](arith::MulFOp op) { scaleAngle = op; });
  ASSERT_TRUE(scaleAngle);
  EXPECT_EQ(scaleAngle.getLhs(), rotationLoop.getRegionIterArg(0));
  auto scale = scaleAngle.getRhs().getDefiningOp<arith::ConstantOp>();
  ASSERT_TRUE(scale);
  auto scaleValue = dyn_cast<FloatAttr>(scale.getValue());
  ASSERT_TRUE(scaleValue);
  EXPECT_DOUBLE_EQ(scaleValue.getValueAsDouble(), 0.5);
}

TEST(GenerateProgramTest, SerializesTheLargestQuantumMultiplexer) {
  auto program =
      generate(Multiplexer{{.qubits = MultiplexerOptions::MAX_QUBITS}});
  ASSERT_TRUE(program);
  scf::ForOp stateLoop;
  program->module().walk([&](scf::ForOp loop) {
    if (!loop.getInitArgs().empty()) {
      stateLoop = loop;
    }
  });
  ASSERT_TRUE(stateLoop);
  auto upper =
      stateLoop.getUpperBound().getDefiningOp<arith::ConstantIndexOp>();
  ASSERT_TRUE(upper);
  EXPECT_EQ(upper.value(),
            static_cast<int64_t>(MultiplexerOptions::MAX_QUBITS - 1));

  EXPECT_LT(test::countOperations(program->module()), 150U);

  test::expectJeffRoundTrip(std::move(*program));
}

} // namespace mqt::bench
