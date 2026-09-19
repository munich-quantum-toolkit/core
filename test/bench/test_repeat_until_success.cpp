/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "bench/Error.hpp"
#include "bench/Evaluation.hpp"
#include "bench/RepeatUntilSuccess.hpp"
#include "bench/TestUtils.hpp"

#include "gtest/gtest.h"

#include <numbers>

namespace test = mqt::bench::test;

namespace {

using mqt::bench::Output;
using mqt::bench::RepeatUntilSuccess;
using mqt::bench::RepeatUntilSuccessOptions;

TEST(RepeatUntilSuccess, HasTheOutput) {
  const auto benchmark = test::value(RepeatUntilSuccess::create());
  EXPECT_EQ(benchmark.output(), (Output{"result", 1}));
}

TEST(RepeatUntilSuccess, ValidatesDataWidth) {
  EXPECT_EQ(test::value(RepeatUntilSuccess::create()).options().dataQubits, 1U);
  EXPECT_EQ(test::value(RepeatUntilSuccess::create({.dataQubits = 5}))
                .options()
                .dataQubits,
            5U);
  EXPECT_NO_THROW(test::value(RepeatUntilSuccess::create(
      {.dataQubits = RepeatUntilSuccessOptions::MAX_DATA_QUBITS})));
  EXPECT_EQ(test::errorKind(RepeatUntilSuccess::create({.dataQubits = 0})),
            mqt::bench::Error::Kind::InvalidArgument);
  EXPECT_EQ(
      test::errorKind(RepeatUntilSuccess::create(
          {.dataQubits = RepeatUntilSuccessOptions::MAX_DATA_QUBITS + 1})),
      mqt::bench::Error::Kind::InvalidArgument);
}

TEST(RepeatUntilSuccess, HasThePhaseSensitiveReference) {
  const auto benchmark = test::value(RepeatUntilSuccess::create());
  constexpr auto bias = std::numbers::sqrt2 / 3.;
  EXPECT_DOUBLE_EQ(test::value(benchmark.probability("0")), 0.5 + bias);
  EXPECT_DOUBLE_EQ(test::value(benchmark.probability("1")), 0.5 - bias);
  EXPECT_EQ(test::errorKind(benchmark.probability("")),
            mqt::bench::Error::Kind::InvalidArgument);
  EXPECT_EQ(test::errorKind(benchmark.probability("00")),
            mqt::bench::Error::Kind::InvalidArgument);
  EXPECT_EQ(test::errorKind(benchmark.probability("x")),
            mqt::bench::Error::Kind::InvalidArgument);
}

TEST(RepeatUntilSuccess, EvaluatesTheReferenceWithoutASuccessOutcome) {
  const auto benchmark = test::value(RepeatUntilSuccess::create());
  constexpr auto bias = std::numbers::sqrt2 / 3.;
  const auto allZero = test::value(benchmark.evaluate({{"0", 10}}));
  EXPECT_NEAR(allZero.totalVariationDistance, 0.5 - bias, 1e-15);
  EXPECT_NEAR(allZero.squaredHellingerFidelity, 0.5 + bias, 1e-15);
  EXPECT_FALSE(allZero.successProbability);
}

} // namespace
