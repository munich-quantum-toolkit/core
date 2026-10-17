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
#include "support/Diagnostics.hpp"
#include "support/TestSupport.hpp"

#include "gtest/gtest.h"

#include <cstddef>
#include <limits>
#include <string>

namespace mqt::bench {

using test::expectInvalidJSON;

TEST(GHZ, UsesDocumentedDefaults) {
  const auto ghz = ::mqt::test::value(GHZ::create({.qubits = 3}));
  EXPECT_EQ(ghz.options().topology, GHZTopology::Linear);
  EXPECT_EQ(ghz.options().basis, GHZBasis::Z);
  EXPECT_EQ(ghz.output(), (Output{"result", 3}));
}

TEST(GHZ, RejectsUnsupportedQubitCounts) {
  EXPECT_EQ(::mqt::test::errorKind([&] { return GHZ::create({.qubits = 0}); }),
            ::mqt::ErrorCategory::InvalidArgument);
  EXPECT_EQ(::mqt::test::errorKind([&] {
              return GHZ::create({.qubits = GHZOptions::MAX_QUBITS + 1});
            }),
            ::mqt::ErrorCategory::InvalidArgument);
  EXPECT_NO_THROW(static_cast<void>(::mqt::test::value(
      GHZ::create({.qubits = GHZOptions::MAX_X_BASIS_QUBITS + 1}))));
  EXPECT_EQ(::mqt::test::errorKind([&] {
              return GHZ::create({.qubits = GHZOptions::MAX_X_BASIS_QUBITS + 1,
                                  .basis = GHZBasis::X});
            }),
            ::mqt::ErrorCategory::InvalidArgument);
}

TEST(GHZ, RejectsUnknownEnumValues) {
  // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange)
  constexpr auto invalidTopology = static_cast<GHZTopology>(2);
  // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange)
  constexpr auto invalidBasis = static_cast<GHZBasis>(2);
  EXPECT_EQ(::mqt::test::errorKind([&] {
              return GHZ::create({.qubits = 2, .topology = invalidTopology});
            }),
            ::mqt::ErrorCategory::InvalidArgument);
  EXPECT_EQ(::mqt::test::errorKind([&] {
              return GHZ::create({.qubits = 2, .basis = invalidBasis});
            }),
            ::mqt::ErrorCategory::InvalidArgument);
}

TEST(GHZ, GivesTheZBasisDistribution) {
  const auto ghz = ::mqt::test::value(
      GHZ::create({.qubits = 3, .topology = GHZTopology::Star}));
  EXPECT_DOUBLE_EQ(::mqt::test::value(ghz.probability("000")), 0.5);
  EXPECT_DOUBLE_EQ(::mqt::test::value(ghz.probability("111")), 0.5);
  EXPECT_DOUBLE_EQ(::mqt::test::value(ghz.probability("010")), 0.);
}

TEST(GHZ, GivesTheXBasisDistribution) {
  const auto ghz =
      ::mqt::test::value(GHZ::create({.qubits = 3, .basis = GHZBasis::X}));
  EXPECT_DOUBLE_EQ(::mqt::test::value(ghz.probability("000")), 0.25);
  EXPECT_DOUBLE_EQ(::mqt::test::value(ghz.probability("011")), 0.25);
  EXPECT_DOUBLE_EQ(::mqt::test::value(ghz.probability("111")), 0.);

  const auto largest = ::mqt::test::value(GHZ::create(
      {.qubits = GHZOptions::MAX_X_BASIS_QUBITS, .basis = GHZBasis::X}));
  EXPECT_GT(::mqt::test::value(largest.probability(
                std::string(GHZOptions::MAX_X_BASIS_QUBITS, '0'))),
            0.);
}

TEST(GHZ, EvaluatesCountsAgainstTheWholeIdealDistribution) {
  const auto ghz = ::mqt::test::value(GHZ::create({.qubits = 2}));
  const auto exact = ::mqt::test::value(ghz.evaluate({{"00", 50}, {"11", 50}}));
  EXPECT_DOUBLE_EQ(exact.totalVariationDistance, 0.);
  EXPECT_DOUBLE_EQ(exact.squaredHellingerFidelity, 1.);
  EXPECT_FALSE(exact.successProbability.has_value());

  const auto incomplete = ::mqt::test::value(ghz.evaluate({{"00", 100}}));
  EXPECT_DOUBLE_EQ(incomplete.totalVariationDistance, 0.5);
  EXPECT_DOUBLE_EQ(incomplete.squaredHellingerFidelity, 0.5);
}

TEST(GHZ, ValidatesOutcomesAndShotCounts) {
  const auto ghz = ::mqt::test::value(GHZ::create({.qubits = 2}));
  EXPECT_EQ(::mqt::test::errorKind([&] { return ghz.probability("0"); }),
            ::mqt::ErrorCategory::InvalidArgument);
  EXPECT_EQ(::mqt::test::errorKind([&] { return ghz.probability("0x"); }),
            ::mqt::ErrorCategory::InvalidArgument);
  for (const auto* invalid : {"0", "0x"}) {
    EXPECT_EQ(::mqt::test::errorKind(
                  [&] { return ghz.evaluate({{"00", 1}, {invalid, 0}}); }),
              ::mqt::ErrorCategory::InvalidArgument);
  }
  EXPECT_EQ(::mqt::test::errorKind([&] { return ghz.evaluate({}); }),
            ::mqt::ErrorCategory::InvalidArgument);
  EXPECT_EQ(::mqt::test::errorKind([&] { return ghz.evaluate({{"00", 0}}); }),
            ::mqt::ErrorCategory::InvalidArgument);
  EXPECT_EQ(::mqt::test::errorKind([&] {
              return ghz.evaluate(
                  {{"00", std::numeric_limits<size_t>::max()}, {"11", 1}});
            }),
            ::mqt::ErrorCategory::Overflow);
}

TEST(GHZ, RoundTripsJSON) {
  const auto defaults = ::mqt::test::value(ghzFromInstanceSpecificationJSON(
      R"({"parameters":{"qubits":3},"benchmark":"ghz","schema_version":1})"));
  EXPECT_EQ(defaults.options().topology, GHZTopology::Linear);
  EXPECT_EQ(defaults.options().basis, GHZBasis::Z);
  EXPECT_EQ(
      toInstanceSpecificationJSON(defaults),
      R"({"benchmark":"ghz","parameters":{"basis":"z","qubits":3,"topology":"linear"},"schema_version":1})");

  const auto configured = ::mqt::test::value(GHZ::create(
      {.qubits = 4, .topology = GHZTopology::Star, .basis = GHZBasis::X}));
  const auto manifest = toManifestJSON(configured);
  EXPECT_EQ(toManifestJSON(::mqt::test::value(ghzFromManifestJSON(manifest))),
            manifest);
  EXPECT_EQ(::mqt::test::value(benchmarkIdFromManifestJSON(manifest)), "ghz");
}

TEST(GHZ, UsesSemanticCaseIds) {
  const auto defaults = ::mqt::test::value(ghzFromInstanceSpecificationJSON(
      R"({"parameters":{"qubits":3},"benchmark":"ghz","schema_version":1})"));
  EXPECT_EQ(caseId(defaults),
            caseId(::mqt::test::value(GHZ::create({.qubits = 3}))));
  EXPECT_NE(caseId(defaults),
            caseId(::mqt::test::value(
                GHZ::create({.qubits = 3, .topology = GHZTopology::Star}))));
  EXPECT_EQ(caseId(defaults), "sha256-a222c0c57bcecb4f5e7ea72bab439683"
                              "92861a52c5cb7c9c13aeaffffa059a65");
}

TEST(GHZ, DescribesJSONSchema) {
  const auto schema = ::mqt::test::value(describeBenchmarkJSON("ghz"));
  EXPECT_NE(schema.find("\"maximum\":1000000"), std::string::npos);
  EXPECT_NE(schema.find("\"maximum\":1075"), std::string::npos);
}

TEST(GHZ, RejectsInvalidJSONParameters) {
  expectInvalidJSON(
      [] {
        return ghzFromInstanceSpecificationJSON(
            R"({"schema_version":1,"benchmark":"ghz","parameters":{"qubits":2,"extra":true}})");
      },
      "unknown key 'extra'");
  expectInvalidJSON(
      [] {
        return ghzFromInstanceSpecificationJSON(
            R"({"schema_version":1,"benchmark":"ghz","parameters":{"qubits":2.5}})");
      },
      "encoded as an integer");
  expectInvalidJSON(
      [] {
        return ghzFromInstanceSpecificationJSON(
            R"({"schema_version":1,"benchmark":"ghz","parameters":{"basis":"x","qubits":1076}})");
      },
      "between 1 and 1075");
}

} // namespace mqt::bench
