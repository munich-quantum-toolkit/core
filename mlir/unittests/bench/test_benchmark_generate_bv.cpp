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
#include "bench/BV.hpp"
#include "mlir/Dialect/CBit/IR/CBitOps.h"
#include "mlir/Dialect/QC/IR/QCOps.h"
#include "mlir/bench/Generate.h"

#include <gtest/gtest.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Support/LLVM.h>

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
  EXPECT_EQ(test::countOps<tensor::ExtractOp>(staticProgram->module()), 1U);
  EXPECT_EQ(test::countOps<tensor::ExtractOp>(dynamicProgram->module()), 1U);

  const auto checkIndexing = [](ModuleOp moduleOp) {
    tensor::ExtractOp secret;
    moduleOp.walk([&](tensor::ExtractOp op) { secret = op; });
    ASSERT_TRUE(secret);
    auto loop = secret->getParentOfType<scf::ForOp>();
    ASSERT_TRUE(loop);
    EXPECT_EQ(secret.getIndices().front(), loop.getInductionVar());

    qc::MeasureOp measure;
    moduleOp.walk([&](qc::MeasureOp op) { measure = op; });
    ASSERT_TRUE(measure);
    auto measurementLoop = measure->getParentOfType<scf::ForOp>();
    ASSERT_TRUE(measurementLoop);
    auto store = dyn_cast<cbit::StoreOp>(*measure.getResult().user_begin());
    ASSERT_TRUE(store);
    EXPECT_EQ(store.getIndex(), measurementLoop.getInductionVar());
  };
  checkIndexing(staticProgram->module());
  checkIndexing(dynamicProgram->module());
}

} // namespace mqt::bench
