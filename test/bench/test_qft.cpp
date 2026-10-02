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
#include "bench/JSON.hpp"
#include "bench/QFT.hpp"

#include "gtest/gtest.h"

#include <stdexcept>
#include <string>

namespace mqt::bench {

TEST(QFT, UsesTheStandardMethodByDefault) {
  const QFT benchmark{{.qubits = 3, .periodExponent = 1}};
  EXPECT_EQ(benchmark.options().method, QFTMethod::Standard);
  EXPECT_EQ(benchmark.output(), (Output{"result", 3}));
}

TEST(QFT, ValidatesTheConfiguredInstance) {
  // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange)
  constexpr auto invalidMethod = static_cast<QFTMethod>(2);
  EXPECT_THROW(static_cast<void>(QFT{{.qubits = 0, .periodExponent = 0}}),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(QFT{{.qubits = QFTOptions::MAX_QUBITS + 1,
                                      .periodExponent = 0}}),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(QFT{{.qubits = 3, .periodExponent = 4}}),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(QFT{{.qubits = 1075, .periodExponent = 1075}}),
               std::invalid_argument);
  EXPECT_THROW(
      static_cast<void>(
          QFT{{.qubits = 3, .periodExponent = 1, .method = invalidMethod}}),
      std::invalid_argument);
}

TEST(QFT, GivesTwoPeaksForPeriodTwo) {
  const QFT benchmark{{.qubits = 3, .periodExponent = 1}};
  EXPECT_DOUBLE_EQ(benchmark.probability("000"), 0.5);
  EXPECT_DOUBLE_EQ(benchmark.probability("100"), 0.5);
  EXPECT_DOUBLE_EQ(benchmark.probability("010"), 0.);
}

TEST(QFT, GivesFourPeaksForPeriodFour) {
  const QFT benchmark{
      {.qubits = 4, .periodExponent = 2, .method = QFTMethod::Semiclassical}};
  for (const auto* outcome : {"0000", "0100", "1000", "1100"}) {
    EXPECT_DOUBLE_EQ(benchmark.probability(outcome), 0.25);
  }
  EXPECT_DOUBLE_EQ(benchmark.probability("0001"), 0.);

  const auto evaluation = benchmark.evaluate(
      {{"0000", 25}, {"0100", 25}, {"1000", 25}, {"1100", 25}});
  EXPECT_DOUBLE_EQ(evaluation.totalVariationDistance, 0.);
  EXPECT_DOUBLE_EQ(evaluation.squaredHellingerFidelity, 1.);
  EXPECT_FALSE(evaluation.successProbability);
}

TEST(QFT, RoundTripsJSON) {
  const auto defaults = qftFromInstanceSpecificationJSON(
      R"({"schema_version":1,"benchmark":"qft","parameters":{"qubits":4,"period_exponent":2}})");
  EXPECT_EQ(defaults.options().method, QFTMethod::Standard);
  EXPECT_EQ(
      toInstanceSpecificationJSON(defaults),
      R"({"benchmark":"qft","parameters":{"method":"standard","period_exponent":2,"qubits":4},"schema_version":1})");

  const QFT benchmark{
      {.qubits = 4, .periodExponent = 2, .method = QFTMethod::Semiclassical}};
  const auto manifest = toManifestJSON(benchmark);
  EXPECT_EQ(toManifestJSON(qftFromManifestJSON(manifest)), manifest);
  EXPECT_EQ(benchmarkIdFromManifestJSON(manifest), "qft");
}

TEST(QFT, UsesSemanticCaseIds) {
  EXPECT_NE(caseId(QFT{{.qubits = 3, .periodExponent = 1}}),
            caseId(QFT{{.qubits = 3,
                        .periodExponent = 1,
                        .method = QFTMethod::Semiclassical}}));
}

TEST(QFT, DescribesJSONSchema) {
  EXPECT_NE(describeBenchmarkJSON("qft").find("\"period_exponent\""),
            std::string::npos);
}

} // namespace mqt::bench
