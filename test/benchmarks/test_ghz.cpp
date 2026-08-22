/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "benchmarks/GHZ.hpp"

#include <gtest/gtest.h>

#include <limits>
#include <stdexcept>

namespace {

using mqt::benchmarks::Counts;
using mqt::benchmarks::GHZ;
using mqt::benchmarks::GHZBasis;
using mqt::benchmarks::GHZOptions;
using mqt::benchmarks::GHZTopology;
using mqt::benchmarks::Output;

TEST(GHZ, UsesDocumentedDefaults) {
  const GHZ ghz{{.qubits = 3}};
  EXPECT_EQ(ghz.options().topology, GHZTopology::Linear);
  EXPECT_EQ(ghz.options().basis, GHZBasis::Z);
  EXPECT_EQ(ghz.output(), (Output{"result", 3}));
}

TEST(GHZ, RejectsUnsupportedQubitCounts) {
  EXPECT_THROW(static_cast<void>(GHZ{{.qubits = 0}}), std::invalid_argument);
  EXPECT_THROW(static_cast<void>(GHZ{{.qubits = GHZOptions::MAX_QUBITS + 1}}),
               std::invalid_argument);
  EXPECT_NO_THROW(
      static_cast<void>(GHZ{{.qubits = GHZOptions::MAX_X_BASIS_QUBITS + 1}}));
  EXPECT_THROW(
      static_cast<void>(GHZ{{.qubits = GHZOptions::MAX_X_BASIS_QUBITS + 1,
                             .basis = GHZBasis::X}}),
      std::invalid_argument);
}

TEST(GHZ, RejectsUnknownEnumValues) {
  EXPECT_THROW(static_cast<void>(
                   GHZ{{.qubits = 2, .topology = static_cast<GHZTopology>(2)}}),
               std::invalid_argument);
  EXPECT_THROW(
      static_cast<void>(GHZ{{.qubits = 2, .basis = static_cast<GHZBasis>(2)}}),
      std::invalid_argument);
}

TEST(GHZ, GivesTheZBasisDistribution) {
  const GHZ ghz{{.qubits = 3, .topology = GHZTopology::Star}};
  EXPECT_DOUBLE_EQ(ghz.probability("000"), 0.5);
  EXPECT_DOUBLE_EQ(ghz.probability("111"), 0.5);
  EXPECT_DOUBLE_EQ(ghz.probability("010"), 0.);
}

TEST(GHZ, GivesTheXBasisDistribution) {
  const GHZ ghz{{.qubits = 3, .basis = GHZBasis::X}};
  EXPECT_DOUBLE_EQ(ghz.probability("000"), 0.25);
  EXPECT_DOUBLE_EQ(ghz.probability("011"), 0.25);
  EXPECT_DOUBLE_EQ(ghz.probability("111"), 0.);

  const GHZ largest{
      {.qubits = GHZOptions::MAX_X_BASIS_QUBITS, .basis = GHZBasis::X}};
  EXPECT_GT(
      largest.probability(std::string(GHZOptions::MAX_X_BASIS_QUBITS, '0')),
      0.);
}

TEST(GHZ, EvaluatesCountsAgainstTheWholeIdealDistribution) {
  const GHZ ghz{{.qubits = 2}};
  const auto exact = ghz.evaluate({{"00", 50}, {"11", 50}});
  EXPECT_DOUBLE_EQ(exact.totalVariationDistance, 0.);
  EXPECT_DOUBLE_EQ(exact.squaredHellingerFidelity, 1.);
  EXPECT_FALSE(exact.successProbability.has_value());

  const auto incomplete = ghz.evaluate({{"00", 100}});
  EXPECT_DOUBLE_EQ(incomplete.totalVariationDistance, 0.5);
  EXPECT_DOUBLE_EQ(incomplete.squaredHellingerFidelity, 0.5);
}

TEST(GHZ, ValidatesOutcomesAndShotCounts) {
  const GHZ ghz{{.qubits = 2}};
  EXPECT_THROW(static_cast<void>(ghz.probability("0")), std::invalid_argument);
  EXPECT_THROW(static_cast<void>(ghz.probability("0x")), std::invalid_argument);
  EXPECT_THROW(static_cast<void>(ghz.evaluate({})), std::invalid_argument);
  EXPECT_THROW(static_cast<void>(ghz.evaluate({{"00", 0}})),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(ghz.evaluate(
                   {{"00", std::numeric_limits<size_t>::max()}, {"11", 1}})),
               std::overflow_error);
}

} // namespace
