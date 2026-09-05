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
#include "bench/Grover.hpp"
#include "mlir/Dialect/QC/IR/QCOps.h"
#include "mlir/bench/Generate.h"

#include <gtest/gtest.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Support/LLVM.h>

#include <cstddef>

namespace mqt::bench {

using namespace mlir;

TEST(GenerateProgramTest, EmitsDirectGroverOracleWithBigEndianMarkedState) {
  const Grover benchmark({.markedBitstring = "01", .iterations = 2});
  auto program = generate(benchmark);
  ASSERT_TRUE(program);
  auto moduleOp = program->module();

  EXPECT_EQ(test::countOps<qc::AllocOp>(moduleOp), 0U);
  EXPECT_EQ(test::countOps<qc::CtrlOp>(moduleOp), 2U);

  scf::ForOp iterations;
  moduleOp.walk([&](qc::CtrlOp op) {
    if (!iterations) {
      iterations = op->getParentOfType<scf::ForOp>();
    }
  });
  ASSERT_TRUE(iterations);

  size_t markedStateFlips = 0;
  iterations.walk([&](qc::XOp op) {
    if (op->getParentOp() != iterations.getOperation()) {
      return;
    }
    ++markedStateFlips;
    auto load = op.getQubit(0).getDefiningOp<memref::LoadOp>();
    ASSERT_TRUE(load);
    auto index =
        load.getIndices().front().getDefiningOp<arith::ConstantIndexOp>();
    ASSERT_TRUE(index);
    EXPECT_EQ(index.value(), 1);
  });
  EXPECT_EQ(markedStateFlips, 2U);
}

} // namespace mqt::bench
