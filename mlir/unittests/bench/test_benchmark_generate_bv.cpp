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
#include "gtest/gtest.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"

namespace mqt::bench {

using namespace mlir;

TEST(GenerateProgramTest, EmitsStructuredBVWithMethodSpecificResources) {
  const BV staticBenchmark{{.hiddenBitstring = "101"}};
  const BV dynamicBenchmark{
      {.hiddenBitstring = "101", .method = BVMethod::Dynamic}};
  auto staticProgram = generate(staticBenchmark);
  auto dynamicProgram = generate(dynamicBenchmark);
  ASSERT_TRUE(staticProgram);
  ASSERT_TRUE(dynamicProgram);

  EXPECT_EQ(test::countOps<qc::AllocOp>(staticProgram->module()), 1U);
  EXPECT_EQ(test::countOps<memref::AllocOp>(staticProgram->module()), 1U);
  EXPECT_EQ(test::countOps<qc::AllocOp>(dynamicProgram->module()), 2U);
  EXPECT_EQ(test::countOps<memref::AllocOp>(dynamicProgram->module()), 0U);
}

} // namespace mqt::bench
