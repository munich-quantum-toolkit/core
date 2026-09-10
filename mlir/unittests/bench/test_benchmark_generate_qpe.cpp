/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "bench/QPE.hpp"
#include "mqt/Dialect/QC/IR/QCOps.h"
#include "mqt/bench/Generate.h"

#include "TestUtils.h"
#include "gtest/gtest.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/STLExtras.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numbers>

namespace mqt::bench {

using namespace mlir;

TEST(GenerateProgramTest, KeepsStandardQPEPowerAndResultOrderAligned) {
  const QPE benchmark({.precision = 2, .phase = Phase(1, 4)});
  EXPECT_DOUBLE_EQ(benchmark.probability("01"), 1.);

  auto program = generate(benchmark);
  ASSERT_TRUE(program);
  auto moduleOp = program->module();
  auto table = test::angleTable(moduleOp);
  ASSERT_TRUE(table);
  const auto angles = llvm::to_vector(table.getValues<double>());
  ASSERT_EQ(angles.size(), 2U);
  EXPECT_NEAR(angles[0], std::numbers::pi / 2., 1e-15);
  EXPECT_NEAR(angles[1], std::numbers::pi, 1e-15);

  tensor::ExtractOp extract;
  moduleOp.walk([&](tensor::ExtractOp op) { extract = op; });
  ASSERT_TRUE(extract);
  auto powerLoop = extract->getParentOfType<scf::ForOp>();
  ASSERT_TRUE(powerLoop);
  EXPECT_EQ(extract.getIndices().front(), powerLoop.getInductionVar());

  qc::CtrlOp controlledPower;
  powerLoop.walk([&](qc::CtrlOp op) { controlledPower = op; });
  ASSERT_TRUE(controlledPower);
  auto controlLoad =
      controlledPower.getControl(0).getDefiningOp<memref::LoadOp>();
  ASSERT_TRUE(controlLoad);
  auto controlIndex =
      controlLoad.getIndices().front().getDefiningOp<arith::SubIOp>();
  ASSERT_TRUE(controlIndex);
  EXPECT_EQ(controlIndex.getRhs(), powerLoop.getInductionVar());

  EXPECT_EQ(test::countOps<qc::SWAPOp>(moduleOp), 0U);
}

TEST(GenerateProgramTest, KeepsLargeQPEFiniteAndStructured) {
  constexpr size_t precision = 1025;
  for (const auto method : {QPEMethod::Standard, QPEMethod::Iterative}) {
    SCOPED_TRACE(static_cast<int>(method));
    const QPE benchmark({
        .precision = precision,
        .phase = Phase(std::numeric_limits<uint64_t>::max() - 1,
                       std::numeric_limits<uint64_t>::max()),
        .method = method,
    });
    auto program = generate(benchmark);
    ASSERT_TRUE(program);
    auto moduleOp = program->module();

    auto table = test::angleTable(moduleOp);
    ASSERT_TRUE(table);
    EXPECT_EQ(table.getNumElements(), precision);
    for (const auto angle : table.getValues<double>()) {
      EXPECT_TRUE(std::isfinite(angle));
    }

    EXPECT_LT(test::countOperations(moduleOp), 150U);
  }
}

TEST(GenerateProgramTest, DoublesQPEPhaseModuloOneWithoutOverflow) {
  const QPE benchmark({
      .precision = 4,
      .phase = Phase(uint64_t{1} << 63U, std::numeric_limits<uint64_t>::max()),
  });
  auto program = generate(benchmark);
  ASSERT_TRUE(program);
  const auto table = test::angleTable(program->module());
  ASSERT_TRUE(table);
  const auto angles = llvm::to_vector(table.getValues<double>());
  ASSERT_EQ(angles.size(), 4U);

  const auto denominator =
      static_cast<long double>(std::numeric_limits<uint64_t>::max());
  const auto turn = 2.L * std::numbers::pi_v<long double> / denominator;
  EXPECT_DOUBLE_EQ(angles[0], static_cast<double>((uint64_t{1} << 63U) * turn));
  EXPECT_DOUBLE_EQ(angles[1], static_cast<double>(turn));
  EXPECT_DOUBLE_EQ(angles[2], static_cast<double>(2.L * turn));
  EXPECT_DOUBLE_EQ(angles[3], static_cast<double>(4.L * turn));
}

TEST(GenerateProgramTest, SamplesQPEAgainstReference) {
  for (const auto method : {QPEMethod::Standard, QPEMethod::Iterative}) {
    for (const auto phase : {Phase(3, 8), Phase(1, 3)}) {
      SCOPED_TRACE(static_cast<int>(method));
      test::expectSamplingMatchesReference(
          QPE{{.precision = 3, .phase = phase, .method = method}});
    }
  }
}

} // namespace mqt::bench
