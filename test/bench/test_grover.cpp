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
#include "bench/Grover.hpp"
#include "bench/TestUtils.hpp"

#include "gtest/gtest.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>

namespace test = mqt::bench::test;

namespace {

using mqt::bench::Grover;
using mqt::bench::Output;

TEST(Grover, ResolvesTheDefaultIterationCountOnce) {
  const auto grover = test::value(Grover::create({.markedBitstring = "10"}));
  ASSERT_TRUE(grover.options().iterations.has_value());
  EXPECT_EQ(*grover.options().iterations, 1);
  EXPECT_EQ(grover.qubits(), 2);
  EXPECT_EQ(grover.output(), (Output{"result", 2}));
  EXPECT_DOUBLE_EQ(test::value(grover.probability("10")), 1.);
  EXPECT_NEAR(test::value(grover.probability("00")), 0., 1e-31);
}

TEST(Grover, ResolvesLargeDefaultIterationCountsWithoutProbabilityRounding) {
  const auto grover =
      test::value(Grover::create({.markedBitstring = std::string(62, '0')}));
  EXPECT_EQ(*grover.options().iterations, 1'686'629'713);
}

TEST(Grover, AcceptsAnExplicitZeroIterationCount) {
  const auto grover =
      test::value(Grover::create({.markedBitstring = "01", .iterations = 0}));
  EXPECT_EQ(*grover.options().iterations, 0);
  EXPECT_DOUBLE_EQ(test::value(grover.probability("01")), 0.25);
  EXPECT_DOUBLE_EQ(test::value(grover.probability("11")), 0.25);
}

TEST(Grover, RejectsUnsupportedOptions) {
  EXPECT_EQ(test::errorKind(Grover::create({.markedBitstring = "0"})),
            mqt::bench::Error::Kind::InvalidArgument);
  EXPECT_EQ(test::errorKind(Grover::create({.markedBitstring = "0x"})),
            mqt::bench::Error::Kind::InvalidArgument);
  EXPECT_EQ(test::errorKind(
                Grover::create({.markedBitstring = std::string(63, '0')})),
            mqt::bench::Error::Kind::InvalidArgument);
  EXPECT_EQ(
      test::errorKind(Grover::create({
          .markedBitstring = "00",
          .iterations =
              static_cast<size_t>(std::numeric_limits<int32_t>::max()) + 1,
      })),
      mqt::bench::Error::Kind::InvalidArgument);
}

TEST(Grover, EvaluatesTheMarkedOutcomeAsSuccess) {
  const auto grover =
      test::value(Grover::create({.markedBitstring = "01", .iterations = 0}));
  const auto evaluation = test::value(
      grover.evaluate({{"00", 25}, {"01", 25}, {"10", 25}, {"11", 25}}));
  EXPECT_NEAR(evaluation.totalVariationDistance, 0., 1e-15);
  EXPECT_DOUBLE_EQ(evaluation.squaredHellingerFidelity, 1.);
  ASSERT_TRUE(evaluation.successProbability.has_value());
  EXPECT_DOUBLE_EQ(*evaluation.successProbability, 0.25);
}

} // namespace
