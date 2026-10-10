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
#include "bench/GHZ.hpp"
#include "bench/JSON.hpp"

#include "JSONTestUtils.hpp"

#include "gtest/gtest.h"

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>

namespace mqt::bench {

using test::expectInvalidJSON;

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
  // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange)
  constexpr auto invalidTopology = static_cast<GHZTopology>(2);
  // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange)
  constexpr auto invalidBasis = static_cast<GHZBasis>(2);
  EXPECT_THROW(
      static_cast<void>(GHZ{{.qubits = 2, .topology = invalidTopology}}),
      std::invalid_argument);
  EXPECT_THROW(static_cast<void>(GHZ{{.qubits = 2, .basis = invalidBasis}}),
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
  EXPECT_THROW(static_cast<void>(ghz.evaluate({{"00", 1}, {"0", 0}})),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(ghz.evaluate({{"00", 1}, {"0x", 0}})),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(ghz.evaluate(
                   {{"00", std::numeric_limits<size_t>::max()}, {"11", 1}})),
               std::overflow_error);
}

TEST(GHZ, RoundTripsJSON) {
  const auto defaults = ghzFromInstanceSpecificationJSON(
      R"({"parameters":{"qubits":3},"benchmark":"ghz","schema_version":1})");
  EXPECT_EQ(defaults.options().topology, GHZTopology::Linear);
  EXPECT_EQ(defaults.options().basis, GHZBasis::Z);
  EXPECT_EQ(
      toInstanceSpecificationJSON(defaults),
      R"({"benchmark":"ghz","parameters":{"basis":"z","qubits":3,"topology":"linear"},"schema_version":1})");

  const GHZ configured{
      {.qubits = 4, .topology = GHZTopology::Star, .basis = GHZBasis::X}};
  const auto manifest = toManifestJSON(configured);
  EXPECT_EQ(toManifestJSON(ghzFromManifestJSON(manifest)), manifest);
  EXPECT_EQ(benchmarkIdFromManifestJSON(manifest), "ghz");
}

TEST(GHZ, UsesSemanticCaseIds) {
  const auto defaults = ghzFromInstanceSpecificationJSON(
      R"({"parameters":{"qubits":3},"benchmark":"ghz","schema_version":1})");
  EXPECT_EQ(caseId(defaults), caseId(GHZ{{.qubits = 3}}));
  EXPECT_NE(caseId(defaults),
            caseId(GHZ{{.qubits = 3, .topology = GHZTopology::Star}}));
  EXPECT_EQ(caseId(defaults), "sha256-a222c0c57bcecb4f5e7ea72bab439683"
                              "92861a52c5cb7c9c13aeaffffa059a65");
}

TEST(GHZ, DescribesJSONSchema) {
  const auto schema = describeBenchmarkJSON("ghz");
  EXPECT_NE(schema.find("\"maximum\":1000000"), std::string::npos);
  EXPECT_NE(schema.find("\"maximum\":1075"), std::string::npos);
}

TEST(GHZ, RejectsInvalidJSONParameters) {
  expectInvalidJSON(
      [] {
        static_cast<void>(ghzFromInstanceSpecificationJSON(
            R"({"schema_version":1,"benchmark":"ghz","parameters":{"qubits":2,"extra":true}})"));
      },
      "unknown key 'extra'");
  expectInvalidJSON(
      [] {
        static_cast<void>(ghzFromInstanceSpecificationJSON(
            R"({"schema_version":1,"benchmark":"ghz","parameters":{"qubits":2.5}})"));
      },
      "encoded as an integer");
  expectInvalidJSON(
      [] {
        static_cast<void>(ghzFromInstanceSpecificationJSON(
            R"({"schema_version":1,"benchmark":"ghz","parameters":{"basis":"x","qubits":1076}})"));
      },
      "between 1 and 1075");
}

} // namespace mqt::bench
