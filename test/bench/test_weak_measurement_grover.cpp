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
#include "bench/WeakMeasurementGrover.hpp"

#include "gtest/gtest.h"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

using mqt::bench::Output;
using mqt::bench::WeakMeasurementGrover;

TEST(WeakMeasurementGrover, ResolvesTheDefaultMeasurementStrength) {
  const WeakMeasurementGrover benchmark{{.markedBitstring = "101"}};
  ASSERT_TRUE(benchmark.options().measurementStrength);
  EXPECT_DOUBLE_EQ(*benchmark.options().measurementStrength,
                   std::exp2(-3. / 2.));
  EXPECT_EQ(benchmark.qubits(), 3);
  EXPECT_EQ(benchmark.output(), (Output{"result", 3}));
  EXPECT_DOUBLE_EQ(benchmark.probability("101"), 1.);
  EXPECT_DOUBLE_EQ(benchmark.probability("001"), 0.);
}

TEST(WeakMeasurementGrover, AcceptsStrengthsInTheProvenRegime) {
  const WeakMeasurementGrover boundary{
      {.markedBitstring = "10", .measurementStrength = 0.5}};
  EXPECT_DOUBLE_EQ(*boundary.options().measurementStrength, 0.5);

  const WeakMeasurementGrover weaker{
      {.markedBitstring = "0000", .measurementStrength = 0.125}};
  EXPECT_DOUBLE_EQ(*weaker.options().measurementStrength, 0.125);
}

TEST(WeakMeasurementGrover, RejectsUnsupportedOptions) {
  for (const auto& marked :
       {std::string{"0"}, std::string{"0x"}, std::string(63, '0')}) {
    EXPECT_THROW(
        static_cast<void>(WeakMeasurementGrover{{.markedBitstring = marked}}),
        std::invalid_argument);
  }

  for (const auto strength : {
           0.,
           -0.1,
           0.51,
           std::numeric_limits<double>::infinity(),
           std::numeric_limits<double>::quiet_NaN(),
       }) {
    EXPECT_THROW(
        static_cast<void>(WeakMeasurementGrover{
            {.markedBitstring = "00", .measurementStrength = strength}}),
        std::invalid_argument);
  }
  EXPECT_THROW(static_cast<void>(WeakMeasurementGrover{
                   {.markedBitstring = "0000", .measurementStrength = 0.3}}),
               std::invalid_argument);
}

TEST(WeakMeasurementGrover, EvaluatesTheMarkedOutcomeAsSuccess) {
  const WeakMeasurementGrover benchmark{{.markedBitstring = "01"}};
  const auto evaluation = benchmark.evaluate({{"00", 1}, {"01", 9}});
  EXPECT_NEAR(evaluation.totalVariationDistance, 0.1, 1e-15);
  EXPECT_NEAR(evaluation.squaredHellingerFidelity, 0.9, 1e-15);
  ASSERT_TRUE(evaluation.successProbability);
  EXPECT_NEAR(*evaluation.successProbability, 0.9, 1e-15);
}

} // namespace
