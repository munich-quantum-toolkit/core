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
#include "bench/QFTAdderClassical.hpp"
#include "mlir/Dialect/QC/IR/QCOps.h"
#include "mlir/bench/Generate.h"

#include <gtest/gtest.h>
#include <llvm/ADT/STLExtras.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>

#include <cmath>
#include <cstdint>
#include <numbers>
#include <string>
#include <utility>

namespace mqt::bench {

using namespace mlir;

static void expectPhaseLoopConstantIndex(Value value, int64_t expected) {
  auto constant = value.getDefiningOp<arith::ConstantIndexOp>();
  ASSERT_TRUE(constant);
  EXPECT_EQ(constant.value(), expected);
}

TEST(GenerateProgramTest, UsesConfiguredClassicalQFTAdderPhases) {
  auto program = generate(QFTAdderClassical{{.addend = "101"}});
  ASSERT_TRUE(program);
  auto moduleOp = program->module();

  auto table = test::angleTable(moduleOp);
  ASSERT_TRUE(table);
  const auto angles = llvm::to_vector(table.getValues<double>());
  ASSERT_EQ(angles.size(), 4U);
  EXPECT_DOUBLE_EQ(angles[0], std::numbers::pi);
  EXPECT_DOUBLE_EQ(angles[1], std::numbers::pi / 2.);
  EXPECT_DOUBLE_EQ(angles[2], 5. * std::numbers::pi / 4.);
  EXPECT_DOUBLE_EQ(angles[3], 5. * std::numbers::pi / 8.);

  tensor::ExtractOp extract;
  moduleOp.walk([&](tensor::ExtractOp op) {
    EXPECT_FALSE(extract);
    extract = op;
  });
  ASSERT_TRUE(extract);
  auto loop = extract->getParentOfType<scf::ForOp>();
  ASSERT_TRUE(loop);
  expectPhaseLoopConstantIndex(loop.getLowerBound(), 0);
  expectPhaseLoopConstantIndex(loop.getUpperBound(), 4);
  expectPhaseLoopConstantIndex(loop.getStep(), 1);
  EXPECT_EQ(extract.getIndices().front(), loop.getInductionVar());

  qc::POp phase;
  moduleOp.walk([&](qc::POp op) {
    if (!op->getParentOfType<qc::CtrlOp>()) {
      EXPECT_FALSE(phase);
      phase = op;
    }
  });
  ASSERT_TRUE(phase);
  EXPECT_EQ(phase->getParentOfType<scf::ForOp>(), loop);
  EXPECT_EQ(phase.getTheta(), extract.getResult());
  auto target = phase.getQubit(0).getDefiningOp<memref::LoadOp>();
  ASSERT_TRUE(target);
  EXPECT_EQ(target.getIndices().front(), loop.getInductionVar());
}

TEST(GenerateProgramTest, KeepsLargestClassicalQFTAdderFiniteAndStructured) {
  auto addend = std::string(QFTAdderClassicalOptions::MAX_ADDEND_BITS, '1');
  auto program = generate(QFTAdderClassical{{.addend = std::move(addend)}});
  ASSERT_TRUE(program);
  auto moduleOp = program->module();

  auto table = test::angleTable(moduleOp);
  ASSERT_TRUE(table);
  EXPECT_EQ(table.getNumElements(),
            QFTAdderClassicalOptions::MAX_ADDEND_BITS + 1U);
  for (const auto angle : table.getValues<double>()) {
    EXPECT_TRUE(std::isfinite(angle));
  }

  EXPECT_EQ(test::countOps<tensor::ExtractOp>(moduleOp), 1U);
  EXPECT_LT(test::countOperations(moduleOp), 100U);
}

} // namespace mqt::bench
