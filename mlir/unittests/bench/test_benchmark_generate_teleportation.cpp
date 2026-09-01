/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "bench/Teleportation.hpp"
#include "mlir/Dialect/CBit/IR/CBitOps.h"
#include "mlir/Dialect/QC/IR/QCOps.h"
#include "mlir/bench/Generate.h"

#include <gtest/gtest.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>

#include <array>
#include <cstddef>

namespace mqt::bench {

using namespace mlir;

TEST(GenerateProgramTest, KeepsTeleportationFeedForwardAndResultOrder) {
  auto program = generate(Teleportation{});
  ASSERT_TRUE(program);
  auto moduleOp = program->module();

  std::array<qc::MeasureOp, 3> measurements;
  Value result;
  size_t stores = 0;
  moduleOp.walk([&](cbit::StoreOp store) {
    auto index = store.getIndex().getDefiningOp<arith::ConstantIndexOp>();
    ASSERT_TRUE(index);
    const auto resultIndex = static_cast<size_t>(index.value());
    ASSERT_LT(resultIndex, measurements.size());

    auto measure = store.getValue().getDefiningOp<qc::MeasureOp>();
    ASSERT_TRUE(measure);
    EXPECT_FALSE(measurements[resultIndex]);
    measurements[resultIndex] = measure;
    if (!result) {
      result = store.getReg();
    } else {
      EXPECT_EQ(store.getReg(), result);
    }
    ++stores;
  });
  ASSERT_EQ(stores, measurements.size());

  SmallVector<scf::IfOp> corrections;
  moduleOp.walk([&](scf::IfOp op) { corrections.push_back(op); });
  ASSERT_EQ(corrections.size(), 2U);
  EXPECT_EQ(corrections[0].getCondition(), measurements[1].getResult());
  EXPECT_EQ(corrections[1].getCondition(), measurements[0].getResult());

  qc::XOp xCorrection;
  qc::ZOp zCorrection;
  corrections[0].walk([&](qc::XOp op) { xCorrection = op; });
  corrections[1].walk([&](qc::ZOp op) { zCorrection = op; });
  ASSERT_TRUE(xCorrection);
  ASSERT_TRUE(zCorrection);
  EXPECT_EQ(xCorrection.getQubit(0), measurements[2].getQubit());
  EXPECT_EQ(zCorrection.getQubit(0), measurements[2].getQubit());
  EXPECT_TRUE(corrections[0]->isBeforeInBlock(corrections[1]));
  EXPECT_TRUE(corrections[1]->isBeforeInBlock(measurements[2]));
}

} // namespace mqt::bench
