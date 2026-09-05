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
#include "bench/QFT.hpp"
#include "mlir/Dialect/CBit/IR/CBitOps.h"
#include "mlir/Dialect/QC/IR/QCOps.h"
#include "mlir/bench/Generate.h"

#include <gtest/gtest.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Support/LLVM.h>

#include <cmath>

namespace mqt::bench {

using namespace mlir;

TEST(GenerateProgramTest, EmitsStandardQFTWithoutSwaps) {
  const QFT benchmark{{.qubits = 4, .periodExponent = 2}};
  auto program = generate(benchmark);
  ASSERT_TRUE(program);
  auto moduleOp = program->module();
  EXPECT_EQ(test::countOps<qc::SWAPOp>(moduleOp), 0U);

  qc::MeasureOp measure;
  moduleOp.walk([&](qc::MeasureOp op) { measure = op; });
  ASSERT_TRUE(measure);
  auto loop = measure->getParentOfType<scf::ForOp>();
  ASSERT_TRUE(loop);
  auto store = dyn_cast<cbit::StoreOp>(*measure.getResult().user_begin());
  ASSERT_TRUE(store);
  auto index = store.getIndex().getDefiningOp<arith::SubIOp>();
  ASSERT_TRUE(index);
  EXPECT_EQ(index.getRhs(), loop.getInductionVar());
}

TEST(GenerateProgramTest, KeepsLargeQFTStructured) {
  for (const auto method : {QFTMethod::Standard, QFTMethod::Semiclassical}) {
    SCOPED_TRACE(static_cast<int>(method));
    auto program =
        generate(QFT{{.qubits = 1025, .periodExponent = 10, .method = method}});
    ASSERT_TRUE(program);
    EXPECT_LT(test::countOperations(program->module()), 100U);
    program->module().walk([&](arith::ConstantOp op) {
      if (const auto value = dyn_cast<FloatAttr>(op.getValue())) {
        EXPECT_TRUE(std::isfinite(value.getValueAsDouble()));
      }
    });
  }
}

} // namespace mqt::bench
