/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "bench/QFT.hpp"
#include "mqt/Dialect/QC/IR/QCOps.h"
#include "mqt/bench/Generate.h"

#include "TestUtils.h"

#include "gtest/gtest.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"

#include <cmath>

namespace mqt::bench {

using namespace mlir;

TEST(GenerateProgramTest, EmitsStandardQFTWithoutSwaps) {
  const QFT benchmark{{.qubits = 4, .periodExponent = 2}};
  auto program = generate(benchmark);
  ASSERT_TRUE(program);
  auto moduleOp = program->module();
  EXPECT_EQ(test::countOps<qc::SWAPOp>(moduleOp), 0U);
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

TEST(GenerateProgramTest, SamplesQFTAgainstReference) {
  for (const auto method : {QFTMethod::Standard, QFTMethod::Semiclassical}) {
    SCOPED_TRACE(static_cast<int>(method));
    test::expectSamplingMatchesReference(
        QFT{{.qubits = 3, .periodExponent = 1, .method = method}});
  }
}

} // namespace mqt::bench
