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
#include "bench/GHZ.hpp"
#include "bench/TestUtils.hpp"

#include "gtest/gtest.h"

#include <cstddef>
#include <limits>
#include <string>

namespace test = mqt::bench::test;

namespace {

using mqt::bench::Counts;
using mqt::bench::GHZ;
using mqt::bench::GHZBasis;
using mqt::bench::GHZOptions;
using mqt::bench::GHZTopology;
using mqt::bench::Output;

TEST(GHZ, UsesDocumentedDefaults) {
  const auto ghz = test::value(GHZ::create({.qubits = 3}));
  EXPECT_EQ(ghz.options().topology, GHZTopology::Linear);
  EXPECT_EQ(ghz.options().basis, GHZBasis::Z);
  EXPECT_EQ(ghz.output(), (Output{"result", 3}));
}

TEST(GHZ, RejectsUnsupportedQubitCounts) {
  EXPECT_EQ(test::errorKind(GHZ::create({.qubits = 0})),
            mqt::bench::Error::Kind::InvalidArgument);
  EXPECT_EQ(
      test::errorKind(GHZ::create({.qubits = GHZOptions::MAX_QUBITS + 1})),
      mqt::bench::Error::Kind::InvalidArgument);
  EXPECT_NO_THROW(static_cast<void>(test::value(
      GHZ::create({.qubits = GHZOptions::MAX_X_BASIS_QUBITS + 1}))));
  EXPECT_EQ(
      test::errorKind(GHZ::create({.qubits = GHZOptions::MAX_X_BASIS_QUBITS + 1,
                                   .basis = GHZBasis::X})),
      mqt::bench::Error::Kind::InvalidArgument);
}

TEST(GHZ, RejectsUnknownEnumValues) {
  // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange)
  constexpr auto invalidTopology = static_cast<GHZTopology>(2);
  // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange)
  constexpr auto invalidBasis = static_cast<GHZBasis>(2);
  EXPECT_EQ(
      test::errorKind(GHZ::create({.qubits = 2, .topology = invalidTopology})),
      mqt::bench::Error::Kind::InvalidArgument);
  EXPECT_EQ(test::errorKind(GHZ::create({.qubits = 2, .basis = invalidBasis})),
            mqt::bench::Error::Kind::InvalidArgument);
}

TEST(GHZ, GivesTheZBasisDistribution) {
  const auto ghz =
      test::value(GHZ::create({.qubits = 3, .topology = GHZTopology::Star}));
  EXPECT_DOUBLE_EQ(test::value(ghz.probability("000")), 0.5);
  EXPECT_DOUBLE_EQ(test::value(ghz.probability("111")), 0.5);
  EXPECT_DOUBLE_EQ(test::value(ghz.probability("010")), 0.);
}

TEST(GHZ, GivesTheXBasisDistribution) {
  const auto ghz =
      test::value(GHZ::create({.qubits = 3, .basis = GHZBasis::X}));
  EXPECT_DOUBLE_EQ(test::value(ghz.probability("000")), 0.25);
  EXPECT_DOUBLE_EQ(test::value(ghz.probability("011")), 0.25);
  EXPECT_DOUBLE_EQ(test::value(ghz.probability("111")), 0.);

  const auto largest = test::value(GHZ::create(
      {.qubits = GHZOptions::MAX_X_BASIS_QUBITS, .basis = GHZBasis::X}));
  EXPECT_GT(test::value(largest.probability(
                std::string(GHZOptions::MAX_X_BASIS_QUBITS, '0'))),
            0.);
}

TEST(GHZ, EvaluatesCountsAgainstTheWholeIdealDistribution) {
  const auto ghz = test::value(GHZ::create({.qubits = 2}));
  const auto exact = test::value(ghz.evaluate({{"00", 50}, {"11", 50}}));
  EXPECT_DOUBLE_EQ(exact.totalVariationDistance, 0.);
  EXPECT_DOUBLE_EQ(exact.squaredHellingerFidelity, 1.);
  EXPECT_FALSE(exact.successProbability.has_value());

  const auto incomplete = test::value(ghz.evaluate({{"00", 100}}));
  EXPECT_DOUBLE_EQ(incomplete.totalVariationDistance, 0.5);
  EXPECT_DOUBLE_EQ(incomplete.squaredHellingerFidelity, 0.5);
}

TEST(GHZ, ValidatesOutcomesAndShotCounts) {
  const auto ghz = test::value(GHZ::create({.qubits = 2}));
  EXPECT_EQ(test::errorKind(ghz.probability("0")),
            mqt::bench::Error::Kind::InvalidArgument);
  EXPECT_EQ(test::errorKind(ghz.probability("0x")),
            mqt::bench::Error::Kind::InvalidArgument);
  EXPECT_EQ(test::errorKind(ghz.evaluate({})),
            mqt::bench::Error::Kind::InvalidArgument);
  EXPECT_EQ(test::errorKind(ghz.evaluate({{"00", 0}})),
            mqt::bench::Error::Kind::InvalidArgument);
  EXPECT_EQ(test::errorKind(ghz.evaluate(
                {{"00", std::numeric_limits<size_t>::max()}, {"11", 1}})),
            mqt::bench::Error::Kind::Overflow);
}

} // namespace
