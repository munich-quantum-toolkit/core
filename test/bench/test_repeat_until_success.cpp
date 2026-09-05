/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "bench/Evaluation.hpp"
#include "bench/RepeatUntilSuccess.hpp"

#include <gtest/gtest.h>

#include <numbers>
#include <stdexcept>

namespace {

using mqt::bench::Output;
using mqt::bench::RepeatUntilSuccess;

TEST(RepeatUntilSuccess, HasTheFixedOutput) {
  const RepeatUntilSuccess benchmark;
  EXPECT_EQ(benchmark.output(), (Output{"result", 1}));
}

TEST(RepeatUntilSuccess, HasThePhaseSensitiveReference) {
  const RepeatUntilSuccess benchmark;
  constexpr auto bias = std::numbers::sqrt2 / 3.;
  EXPECT_DOUBLE_EQ(benchmark.probability("0"), 0.5 + bias);
  EXPECT_DOUBLE_EQ(benchmark.probability("1"), 0.5 - bias);
  EXPECT_THROW(static_cast<void>(benchmark.probability("")),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(benchmark.probability("00")),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(benchmark.probability("x")),
               std::invalid_argument);
}

TEST(RepeatUntilSuccess, EvaluatesTheReferenceWithoutASuccessOutcome) {
  const RepeatUntilSuccess benchmark;
  constexpr auto bias = std::numbers::sqrt2 / 3.;
  const auto allZero = benchmark.evaluate({{"0", 10}});
  EXPECT_NEAR(allZero.totalVariationDistance, 0.5 - bias, 1e-15);
  EXPECT_NEAR(allZero.squaredHellingerFidelity, 0.5 + bias, 1e-15);
  EXPECT_FALSE(allZero.successProbability);
}

} // namespace
