/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "bench/RepeatUntilSuccess.hpp"
#include "mqt/bench/Generate.h"

#include "TestUtils.h"

#include "gtest/gtest.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/LogicalResult.h"

#include <cstddef>

namespace mqt::bench {

TEST(GenerateProgramTest, SamplesRepeatUntilSuccessAgainstReference) {
  // At 16,384 shots the binomial tail bound for a 0.01 error is below 7e-12.
  // An all-zero sampler has error 0.0286 and must fail this check.
  for (const size_t width : {1U, 2U, 5U, 32U}) {
    SCOPED_TRACE(width);
    test::expectSamplingMatchesReference(
        llvm::cantFail(RepeatUntilSuccess::create({.dataQubits = width})),
        0.01);
  }
}

TEST(GenerateProgramTest, KeepsRepeatUntilSuccessGenerationCompact) {
  auto small =
      generate(llvm::cantFail(RepeatUntilSuccess::create({.dataQubits = 5})));
  auto large = generate(llvm::cantFail(RepeatUntilSuccess::create(
      {.dataQubits = RepeatUntilSuccessOptions::MAX_DATA_QUBITS})));
  ASSERT_TRUE(mlir::succeeded(small));
  ASSERT_TRUE(mlir::succeeded(large));
  EXPECT_EQ(test::countOperations(small->module()),
            test::countOperations(large->module()));
}

} // namespace mqt::bench
