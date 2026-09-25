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
#include "bench/QFT.hpp"

#include "support/Diagnostics.hpp"
#include "support/TestSupport.hpp"

#include "gtest/gtest.h"

namespace {

using mqt::bench::Output;
using mqt::bench::QFT;
using mqt::bench::QFTMethod;
using mqt::bench::QFTOptions;

TEST(QFT, UsesTheStandardMethodByDefault) {
  const auto benchmark =
      ::mqt::test::value(QFT::create({.qubits = 3, .periodExponent = 1}));
  EXPECT_EQ(benchmark.options().method, QFTMethod::Standard);
  EXPECT_EQ(benchmark.output(), (Output{"result", 3}));
}

TEST(QFT, ValidatesTheConfiguredInstance) {
  // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange)
  constexpr auto invalidMethod = static_cast<QFTMethod>(2);
  EXPECT_EQ(::mqt::test::errorKind([&] {
              return QFT::create({.qubits = 0, .periodExponent = 0});
            }),
            ::mqt::ErrorCategory::InvalidArgument);
  EXPECT_EQ(::mqt::test::errorKind([&] {
              return QFT::create(
                  {.qubits = QFTOptions::MAX_QUBITS + 1, .periodExponent = 0});
            }),
            ::mqt::ErrorCategory::InvalidArgument);
  EXPECT_EQ(::mqt::test::errorKind([&] {
              return QFT::create({.qubits = 3, .periodExponent = 4});
            }),
            ::mqt::ErrorCategory::InvalidArgument);
  EXPECT_EQ(::mqt::test::errorKind([&] {
              return QFT::create({.qubits = 1075, .periodExponent = 1075});
            }),
            ::mqt::ErrorCategory::InvalidArgument);
  EXPECT_EQ(::mqt::test::errorKind([&] {
              return QFT::create(
                  {.qubits = 3, .periodExponent = 1, .method = invalidMethod});
            }),
            ::mqt::ErrorCategory::InvalidArgument);
}

TEST(QFT, GivesTwoPeaksForPeriodTwo) {
  const auto benchmark =
      ::mqt::test::value(QFT::create({.qubits = 3, .periodExponent = 1}));
  EXPECT_DOUBLE_EQ(::mqt::test::value(benchmark.probability("000")), 0.5);
  EXPECT_DOUBLE_EQ(::mqt::test::value(benchmark.probability("100")), 0.5);
  EXPECT_DOUBLE_EQ(::mqt::test::value(benchmark.probability("010")), 0.);
}

TEST(QFT, GivesFourPeaksForPeriodFour) {
  const auto benchmark = ::mqt::test::value(QFT::create(
      {.qubits = 4, .periodExponent = 2, .method = QFTMethod::Semiclassical}));
  for (const auto* outcome : {"0000", "0100", "1000", "1100"}) {
    EXPECT_DOUBLE_EQ(::mqt::test::value(benchmark.probability(outcome)), 0.25);
  }
  EXPECT_DOUBLE_EQ(::mqt::test::value(benchmark.probability("0001")), 0.);

  const auto evaluation = ::mqt::test::value(benchmark.evaluate(
      {{"0000", 25}, {"0100", 25}, {"1000", 25}, {"1100", 25}}));
  EXPECT_DOUBLE_EQ(evaluation.totalVariationDistance, 0.);
  EXPECT_DOUBLE_EQ(evaluation.squaredHellingerFidelity, 1.);
  EXPECT_FALSE(evaluation.successProbability);
}

} // namespace
