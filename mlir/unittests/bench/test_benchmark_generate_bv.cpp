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

#include "llvm/Support/Error.h"

namespace mqt::bench {

using namespace mlir;

TEST(GenerateProgramTest, SamplesAsymmetricBVInputsAgainstReference) {
  for (const auto method : {BVMethod::Static, BVMethod::Dynamic}) {
    SCOPED_TRACE(static_cast<int>(method));
    test::expectSamplingMatchesReference(llvm::cantFail(
        BV::create({.hiddenBitstring = "001", .method = method})));
  }
}

TEST(GenerateProgramTest, EmitsStructuredBVWithMethodSpecificResources) {
  const auto staticBenchmark =
      llvm::cantFail(BV::create({.hiddenBitstring = "101"}));
  const auto dynamicBenchmark = llvm::cantFail(
      BV::create({.hiddenBitstring = "101", .method = BVMethod::Dynamic}));
  auto staticProgram = generate(staticBenchmark);
  auto dynamicProgram = generate(dynamicBenchmark);
  ASSERT_TRUE(mlir::succeeded(staticProgram));
  ASSERT_TRUE(mlir::succeeded(dynamicProgram));

  EXPECT_EQ(test::countOps<qc::AllocOp>(staticProgram->module()), 1U);
  EXPECT_EQ(test::countOps<memref::AllocOp>(staticProgram->module()), 1U);
  EXPECT_EQ(test::countOps<qc::AllocOp>(dynamicProgram->module()), 2U);
  EXPECT_EQ(test::countOps<memref::AllocOp>(dynamicProgram->module()), 0U);
}

} // namespace mqt::bench
