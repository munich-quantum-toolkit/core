/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "bench/QAE.hpp"
#include "mqt/bench/Generate.h"

#include "TestUtils.h"

#include "gtest/gtest.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/LogicalResult.h"

namespace mqt::bench {
TEST(GenerateProgramTest, SamplesQAEPhaseAndAmplifiedData) {
  const QAEProblem problem{
      .goodStates = {"001", "100", "110"},
      .amplitude = 0.25,
  };
  for (const QAEOptions& options : {
           QAEOptions{.problem = problem, .precision = 3},
           QAEOptions{.problem = problem, .iterations = 0},
           QAEOptions{.problem = problem, .iterations = 1},
       }) {
    test::expectSamplingMatchesReference(llvm::cantFail(QAE::create(options)));
  }
}

TEST(GenerateProgramTest, PreservesQAEGroverPowersInJeff) {
  auto benchmark = llvm::cantFail(QAE::create({
      .problem = {.goodStates = {"0"}, .amplitude = 0.6},
      .precision = QAEOptions::MAX_PRECISION,
  }));
  auto program = generate(benchmark);
  ASSERT_TRUE(mlir::succeeded(program));
  EXPECT_LT(program->str().size(), 8192U);
  test::expectJeffRoundTrip(std::move(*program));
}
} // namespace mqt::bench
