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

#include "llvm/Support/Error.h"

#include <cstddef>
#include <limits>
#include <string>
#include <system_error>

namespace mqt::bench {

using test::expectInvalidJSON;

TEST(GHZ, UsesDocumentedDefaults) {
  const auto ghz = llvm::cantFail(GHZ::create({.qubits = 3}));
  EXPECT_EQ(ghz.options().topology, GHZTopology::Linear);
  EXPECT_EQ(ghz.options().basis, GHZBasis::Z);
  EXPECT_EQ(ghz.output(), (Output{"result", 3}));
}

TEST(GHZ, RejectsUnsupportedQubitCounts) {
  test::expectError(GHZ::create({.qubits = 0}), std::errc::invalid_argument);
  test::expectError(GHZ::create({.qubits = GHZOptions::MAX_QUBITS + 1}),
                    std::errc::invalid_argument);
  static_cast<void>(llvm::cantFail(
      GHZ::create({.qubits = GHZOptions::MAX_X_BASIS_QUBITS + 1})));
  test::expectError(GHZ::create({
                        .qubits = GHZOptions::MAX_X_BASIS_QUBITS + 1,
                        .basis = GHZBasis::X,
                    }),
                    std::errc::invalid_argument);
}

TEST(GHZ, RejectsUnknownEnumValues) {
  // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange)
  constexpr auto invalidTopology = static_cast<GHZTopology>(2);
  // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange)
  constexpr auto invalidBasis = static_cast<GHZBasis>(2);
  test::expectError(GHZ::create({.qubits = 2, .topology = invalidTopology}),
                    std::errc::invalid_argument);
  test::expectError(GHZ::create({.qubits = 2, .basis = invalidBasis}),
                    std::errc::invalid_argument);
}

TEST(GHZ, GivesTheZBasisDistribution) {
  const auto ghz =
      llvm::cantFail(GHZ::create({.qubits = 3, .topology = GHZTopology::Star}));
  EXPECT_DOUBLE_EQ(llvm::cantFail(ghz.probability("000")), 0.5);
  EXPECT_DOUBLE_EQ(llvm::cantFail(ghz.probability("111")), 0.5);
  EXPECT_DOUBLE_EQ(llvm::cantFail(ghz.probability("010")), 0.);
}

TEST(GHZ, GivesTheXBasisDistribution) {
  const auto ghz =
      llvm::cantFail(GHZ::create({.qubits = 3, .basis = GHZBasis::X}));
  EXPECT_DOUBLE_EQ(llvm::cantFail(ghz.probability("000")), 0.25);
  EXPECT_DOUBLE_EQ(llvm::cantFail(ghz.probability("011")), 0.25);
  EXPECT_DOUBLE_EQ(llvm::cantFail(ghz.probability("111")), 0.);
}

TEST(GHZ, EvaluatesCountsAgainstTheWholeIdealDistribution) {
  const auto ghz = llvm::cantFail(GHZ::create({.qubits = 2}));
  const auto exact = llvm::cantFail(ghz.evaluate({{"00", 50}, {"11", 50}}));
  EXPECT_DOUBLE_EQ(exact.totalVariationDistance, 0.);
  EXPECT_DOUBLE_EQ(exact.squaredHellingerFidelity, 1.);
  EXPECT_FALSE(exact.successProbability.has_value());

  const auto incomplete = llvm::cantFail(ghz.evaluate({{"00", 100}}));
  EXPECT_DOUBLE_EQ(incomplete.totalVariationDistance, 0.5);
  EXPECT_DOUBLE_EQ(incomplete.squaredHellingerFidelity, 0.5);
}

TEST(GHZ, PreservesRepresentableFidelityForSubnormalProbabilities) {
  const auto benchmark = llvm::cantFail(GHZ::create(
      {.qubits = GHZOptions::MAX_X_BASIS_QUBITS, .basis = GHZBasis::X}));
  const auto allZero = std::string(GHZOptions::MAX_X_BASIS_QUBITS, '0');
  auto twoOnes = allZero;
  twoOnes[0] = '1';
  twoOnes[1] = '1';
  const auto evaluation =
      llvm::cantFail(benchmark.evaluate({{allZero, 1}, {twoOnes, 1}}));
  EXPECT_EQ(evaluation.squaredHellingerFidelity,
            2. * std::numeric_limits<double>::denorm_min());
}

TEST(GHZ, ValidatesOutcomesAndShotCounts) {
  const auto ghz = llvm::cantFail(GHZ::create({.qubits = 2}));
  test::expectError(ghz.probability("0"), std::errc::invalid_argument);
  test::expectError(ghz.probability("0x"), std::errc::invalid_argument);
  test::expectError(ghz.evaluate({}), std::errc::invalid_argument);
  test::expectError(ghz.evaluate({{"00", 0}}), std::errc::invalid_argument);
  test::expectError(ghz.evaluate({{"00", 1}, {"0", 0}}),
                    std::errc::invalid_argument);
  test::expectError(ghz.evaluate({{"00", 1}, {"0x", 0}}),
                    std::errc::invalid_argument);
  test::expectError(
      ghz.evaluate({{"00", std::numeric_limits<size_t>::max()}, {"11", 1}}),
      std::errc::value_too_large);
}

TEST(GHZ, RoundTripsJSON) {
  const auto defaults = llvm::cantFail(ghzFromInstanceSpecificationJSON(
      R"({"parameters":{"qubits":3},"benchmark":"ghz","schema_version":1})"));
  EXPECT_EQ(defaults.options().topology, GHZTopology::Linear);
  EXPECT_EQ(defaults.options().basis, GHZBasis::Z);
  EXPECT_EQ(
      toInstanceSpecificationJSON(defaults),
      R"({"benchmark":"ghz","parameters":{"basis":"z","qubits":3,"topology":"linear"},"schema_version":1})");

  const auto configured = llvm::cantFail(GHZ::create(
      {.qubits = 4, .topology = GHZTopology::Star, .basis = GHZBasis::X}));
  const auto manifest = toManifestJSON(configured);
  EXPECT_EQ(toManifestJSON(llvm::cantFail(ghzFromManifestJSON(manifest))),
            manifest);
  EXPECT_EQ(llvm::cantFail(benchmarkIdFromManifestJSON(manifest)), "ghz");
}

TEST(GHZ, UsesSemanticCaseIds) {
  const auto defaults = llvm::cantFail(ghzFromInstanceSpecificationJSON(
      R"({"parameters":{"qubits":3},"benchmark":"ghz","schema_version":1})"));
  EXPECT_EQ(caseId(defaults),
            caseId(llvm::cantFail(GHZ::create({.qubits = 3}))));
  EXPECT_NE(caseId(defaults),
            caseId(llvm::cantFail(
                GHZ::create({.qubits = 3, .topology = GHZTopology::Star}))));
  EXPECT_EQ(caseId(defaults),
            "sha256-"
            "a222c0c57bcecb4f5e7ea72bab43968392861a52c5cb7c9c13aeaffffa059a65");
}

TEST(GHZ, DescribesJSONSchema) {
  const auto schema = llvm::cantFail(describeBenchmarkJSON("ghz"));
  EXPECT_NE(schema.find("\"maximum\":1000000"), std::string::npos);
  EXPECT_NE(schema.find("\"maximum\":1075"), std::string::npos);
}

TEST(GHZ, RejectsInvalidJSONParameters) {
  expectInvalidJSON(
      [] {
        return ghzFromInstanceSpecificationJSON(
            R"({"schema_version":1,"benchmark":"ghz","parameters":{"qubits":2,"extra":true}})");
      },
      "extra");
  expectInvalidJSON(
      [] {
        return ghzFromInstanceSpecificationJSON(
            R"({"schema_version":1,"benchmark":"ghz","parameters":{"qubits":2.5}})");
      },
      "expected uint64_t");
  expectInvalidJSON(
      [] {
        return ghzFromInstanceSpecificationJSON(
            R"({"schema_version":1,"benchmark":"ghz","parameters":{"basis":"x","qubits":1076}})");
      },
      "between 1 and 1075");
}

} // namespace mqt::bench
