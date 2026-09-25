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

#include "support/Diagnostics.hpp"
#include "support/TestSupport.hpp"

#include "gtest/gtest.h"

#include <numbers>

namespace {

using mqt::bench::Output;
using mqt::bench::RepeatUntilSuccess;
using mqt::bench::RepeatUntilSuccessOptions;

TEST(RepeatUntilSuccess, HasTheOutput) {
  const auto benchmark = ::mqt::test::value(RepeatUntilSuccess::create());
  EXPECT_EQ(benchmark.output(), (Output{"result", 1}));
}

TEST(RepeatUntilSuccess, ValidatesDataWidth) {
  EXPECT_EQ(
      ::mqt::test::value(RepeatUntilSuccess::create()).options().dataQubits,
      1U);
  EXPECT_EQ(::mqt::test::value(RepeatUntilSuccess::create({.dataQubits = 5}))
                .options()
                .dataQubits,
            5U);
  EXPECT_NO_THROW(::mqt::test::value(RepeatUntilSuccess::create(
      {.dataQubits = RepeatUntilSuccessOptions::MAX_DATA_QUBITS})));
  EXPECT_EQ(::mqt::test::errorKind(
                [&] { return RepeatUntilSuccess::create({.dataQubits = 0}); }),
            ::mqt::ErrorCategory::InvalidArgument);
  EXPECT_EQ(
      ::mqt::test::errorKind([&] {
        return RepeatUntilSuccess::create(
            {.dataQubits = RepeatUntilSuccessOptions::MAX_DATA_QUBITS + 1});
      }),
      ::mqt::ErrorCategory::InvalidArgument);
}

TEST(RepeatUntilSuccess, HasThePhaseSensitiveReference) {
  const auto benchmark = ::mqt::test::value(RepeatUntilSuccess::create());
  constexpr auto bias = std::numbers::sqrt2 / 3.;
  EXPECT_DOUBLE_EQ(::mqt::test::value(benchmark.probability("0")), 0.5 + bias);
  EXPECT_DOUBLE_EQ(::mqt::test::value(benchmark.probability("1")), 0.5 - bias);
  EXPECT_EQ(::mqt::test::errorKind([&] { return benchmark.probability(""); }),
            ::mqt::ErrorCategory::InvalidArgument);
  EXPECT_EQ(::mqt::test::errorKind([&] { return benchmark.probability("00"); }),
            ::mqt::ErrorCategory::InvalidArgument);
  EXPECT_EQ(::mqt::test::errorKind([&] { return benchmark.probability("x"); }),
            ::mqt::ErrorCategory::InvalidArgument);
}

TEST(RepeatUntilSuccess, EvaluatesTheReferenceWithoutASuccessOutcome) {
  const auto benchmark = ::mqt::test::value(RepeatUntilSuccess::create());
  constexpr auto bias = std::numbers::sqrt2 / 3.;
  const auto allZero = ::mqt::test::value(benchmark.evaluate({{"0", 10}}));
  EXPECT_NEAR(allZero.totalVariationDistance, 0.5 - bias, 1e-15);
  EXPECT_NEAR(allZero.squaredHellingerFidelity, 0.5 + bias, 1e-15);
  EXPECT_FALSE(allZero.successProbability);
}

} // namespace
