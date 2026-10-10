/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "bench/BV.hpp"
#include "mqt/Dialect/QC/IR/QCOps.h"
#include "mqt/bench/Generate.h"

#include "TestUtils.h"
#include "support/TestSupport.hpp"

#include "gtest/gtest.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"

namespace mqt::bench {

using namespace mlir;

TEST(GenerateProgramTest, EmitsStructuredBVWithMethodSpecificResources) {
  const auto staticBenchmark =
      ::mqt::test::value(BV::create({.hiddenBitstring = "101"}));
  const auto dynamicBenchmark = ::mqt::test::value(
      BV::create({.hiddenBitstring = "101", .method = BVMethod::Dynamic}));
  auto staticProgram = generate(staticBenchmark);
  auto dynamicProgram = generate(dynamicBenchmark);
  ASSERT_TRUE(succeeded(staticProgram));
  ASSERT_TRUE(succeeded(dynamicProgram));

  EXPECT_EQ(test::countOps<qc::AllocOp>(staticProgram->module()), 1U);
  EXPECT_EQ(test::countOps<memref::AllocOp>(staticProgram->module()), 1U);
  EXPECT_EQ(test::countOps<qc::AllocOp>(dynamicProgram->module()), 2U);
  EXPECT_EQ(test::countOps<memref::AllocOp>(dynamicProgram->module()), 0U);
}

} // namespace mqt::bench
