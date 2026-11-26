/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

/// @file
/// Tests for bounded angle arithmetic and named phase classification.

#include "mqt/Dialect/MQT/Utils/Angles.h"
#include "mqt/Dialect/MQT/Utils/Parameters.h"

#include "gtest/gtest.h"

#include <cmath>
#include <limits>
#include <numbers>
#include <utility>

namespace {

using mlir::mqt::addConstantAngles;
using mlir::mqt::classifyPhaseGate;
using mlir::mqt::PARAMETER_COMPARISON_TOLERANCE;
using mlir::mqt::PhaseGate;
using mlir::mqt::reduceAngle;
using mlir::mqt::scaleConstantAngle;
using mlir::mqt::twoSum;

TEST(AngleArithmeticTest, AcceptsExactCancellationAndSmallAdditionError) {
  const auto cancelled = addConstantAngles(1.0e308, -1.0e308);
  ASSERT_TRUE(cancelled);
  EXPECT_DOUBLE_EQ(*cancelled, 0.0);

  const auto small = addConstantAngles(0.1, 0.2);
  ASSERT_TRUE(small);
  EXPECT_DOUBLE_EQ(*small, 0.3);

  const auto boundary = addConstantAngles(16.0, PARAMETER_COMPARISON_TOLERANCE);
  ASSERT_TRUE(boundary);
  EXPECT_DOUBLE_EQ(*boundary, 16.0);
  EXPECT_FALSE(addConstantAngles(
      16.0, std::nextafter(PARAMETER_COMPARISON_TOLERANCE,
                           std::numeric_limits<double>::infinity())));
}

TEST(AngleArithmeticTest, RejectsAbsorbedRotationAndOverflow) {
  EXPECT_FALSE(addConstantAngles(1.0e16, 1.0));
  EXPECT_FALSE(addConstantAngles(1.0, 1.0e16));
  EXPECT_FALSE(addConstantAngles(1.0e308, 1.0e308));
  EXPECT_FALSE(addConstantAngles(-1.0e308, -1.0e308));
}

TEST(AngleArithmeticTest, RejectsOverflowWhileRecoveringAdditionError) {
  // The sum is finite, but recovering its rounding error overflows.
  const double negative = -0x1.8p+971;
  const double maximum = std::numeric_limits<double>::max();
  ASSERT_TRUE(std::isfinite(negative + maximum));
  EXPECT_FALSE(addConstantAngles(negative, maximum));
  EXPECT_FALSE(addConstantAngles(maximum, negative));
}

TEST(AngleArithmeticTest, TwoSumRecoversTheExactRoundingError) {
  const auto [sum, error] = twoSum(1.0, 0x1p-60);
  EXPECT_EQ(sum, 1.0);
  EXPECT_EQ(error, 0x1p-60);
  static_assert(twoSum(0.5, 0.25) == std::pair{0.75, 0.0});
}

TEST(AngleArithmeticTest, AcceptsExactProductsAndSmallProductError) {
  const auto exact = scaleConstantAngle(0.125, 8.0);
  ASSERT_TRUE(exact);
  EXPECT_DOUBLE_EQ(*exact, 1.0);

  const auto zero = scaleConstantAngle(1.0e308, 0.0);
  ASSERT_TRUE(zero);
  EXPECT_DOUBLE_EQ(*zero, 0.0);

  const auto small = scaleConstantAngle(0.1, 0.2);
  ASSERT_TRUE(small);
  EXPECT_DOUBLE_EQ(*small, 0.02);
}

TEST(AngleArithmeticTest, RejectsProductOverflowAndExcessRoundingError) {
  EXPECT_FALSE(scaleConstantAngle(1.0e308, 2.0));
  EXPECT_FALSE(scaleConstantAngle(-1.0e308, 2.0));
  EXPECT_FALSE(scaleConstantAngle(1.0e16, 1.1));
}

TEST(PhaseGateClassificationTest, RecognizesNamedPhasesAndTheirInverses) {
  const double pi = std::numbers::pi;
  for (auto [angle, expected] : {
           std::pair{0.0, PhaseGate::Identity},
           std::pair{pi, PhaseGate::Z},
           std::pair{-pi, PhaseGate::Z},
           std::pair{pi / 2.0, PhaseGate::S},
           std::pair{-pi / 2.0, PhaseGate::Sdg},
           std::pair{pi / 4.0, PhaseGate::T},
           std::pair{-pi / 4.0, PhaseGate::Tdg},
       }) {
    SCOPED_TRACE(angle);
    const auto gate = classifyPhaseGate(angle);
    ASSERT_TRUE(gate);
    EXPECT_EQ(*gate, expected);
  }
}

TEST(PhaseGateClassificationTest, RecognizesEquivalentPhasesModuloTwoPi) {
  const double pi = std::numbers::pi;
  for (auto [angle, expected] : {
           std::pair{2.0 * pi, PhaseGate::Identity},
           std::pair{-2.0 * pi, PhaseGate::Identity},
           std::pair{3.0 * pi, PhaseGate::Z},
           std::pair{5.0 * pi / 2.0, PhaseGate::S},
           std::pair{-5.0 * pi / 2.0, PhaseGate::Sdg},
           std::pair{9.0 * pi / 4.0, PhaseGate::T},
           std::pair{-9.0 * pi / 4.0, PhaseGate::Tdg},
       }) {
    SCOPED_TRACE(angle);
    const auto gate = classifyPhaseGate(angle);
    ASSERT_TRUE(gate);
    EXPECT_EQ(*gate, expected);
  }
}

TEST(PhaseGateClassificationTest, KeepsGeneralAnglesUnclassified) {
  EXPECT_FALSE(classifyPhaseGate(0.3));
  EXPECT_FALSE(classifyPhaseGate(-0.3));
  EXPECT_FALSE(classifyPhaseGate(std::numbers::pi / 8.0));
  EXPECT_FALSE(classifyPhaseGate(2.0 * PARAMETER_COMPARISON_TOLERANCE));
}

TEST(PhaseGateClassificationTest, PreservesPhaseLostByLargeAngleReduction) {
  const double angle = std::ldexp(2.0 * std::numbers::pi, 48);
  ASSERT_GT(std::abs(std::sin(angle)), 0.01);
  EXPECT_FALSE(classifyPhaseGate(angle));
  EXPECT_FALSE(classifyPhaseGate(-angle));
}

TEST(AngleReductionTest, KeepsLargeAnglesAccurate) {
  const double pi = std::numbers::pi;
  // atan2(sin, cos) is an independent accurate reference for the phase.
  for (const double angle :
       {100.0, -100.0, 1.0e6, 1.0e9, -1.0e15, std::ldexp(2.0 * pi, 48)}) {
    SCOPED_TRACE(angle);
    EXPECT_NEAR(reduceAngle(angle, 2.0 * pi),
                std::atan2(std::sin(angle), std::cos(angle)),
                PARAMETER_COMPARISON_TOLERANCE);
    const double half = angle / 2.0;
    EXPECT_NEAR(reduceAngle(angle, 4.0 * pi),
                2.0 * std::atan2(std::sin(half), std::cos(half)),
                PARAMETER_COMPARISON_TOLERANCE);
  }
}

TEST(AngleReductionTest, UsesHalfOpenIntervalsWithAnInclusiveUpperBound) {
  const double pi = std::numbers::pi;
  EXPECT_EQ(reduceAngle(pi, 2.0 * pi), pi);
  EXPECT_EQ(reduceAngle(-pi, 2.0 * pi), pi);
  EXPECT_EQ(reduceAngle(-2.0 * pi, 4.0 * pi), 2.0 * pi);
  EXPECT_EQ(reduceAngle(-pi / 4.0, pi / 2.0), pi / 4.0);
  EXPECT_EQ(reduceAngle(0.3, 2.0 * pi), 0.3);
  EXPECT_NEAR(reduceAngle(1.5 * pi, 2.0 * pi), -pi / 2.0,
              PARAMETER_COMPARISON_TOLERANCE);
  EXPECT_NEAR(reduceAngle(3.0 * pi, 4.0 * pi), -pi,
              PARAMETER_COMPARISON_TOLERANCE);
  EXPECT_TRUE(std::isnan(
      reduceAngle(std::numeric_limits<double>::infinity(), 2.0 * pi)));
}

} // namespace
