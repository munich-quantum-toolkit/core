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
#include "bench/Multiplexer.hpp"

#include "gtest/gtest.h"

#include <functional>
#include <numbers>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

using mqt::bench::benchmarkIdFromManifestJSON;
using mqt::bench::caseId;
using mqt::bench::describeBenchmarkJSON;
using mqt::bench::evaluateJSON;
using mqt::bench::Multiplexer;
using mqt::bench::multiplexerFromInstanceSpecificationJSON;
using mqt::bench::multiplexerFromManifestJSON;
using mqt::bench::MultiplexerOptions;
using mqt::bench::Output;
using mqt::bench::toInstanceSpecificationJSON;
using mqt::bench::toManifestJSON;

void expectInvalidMultiplexerJSON(const std::function<void()>& operation,
                                  const std::string_view diagnostic) {
  try {
    operation();
    FAIL() << "Expected invalid JSON input";
  } catch (const std::invalid_argument& error) {
    EXPECT_NE(std::string(error.what()).find(diagnostic), std::string::npos)
        << error.what();
  }
}

TEST(Multiplexer, StoresTheTotalQubitCountAndOutput) {
  const Multiplexer benchmark{{.qubits = 7}};
  EXPECT_EQ(benchmark.options().qubits, 7);
  EXPECT_EQ(benchmark.output(), (Output{"result", 7}));
}

TEST(Multiplexer, ValidatesTheConfiguredInstance) {
  EXPECT_THROW(static_cast<void>(Multiplexer{{.qubits = 1}}),
               std::invalid_argument);
  EXPECT_NO_THROW(static_cast<void>(Multiplexer{{.qubits = 2}}));
  EXPECT_NO_THROW(static_cast<void>(
      Multiplexer{{.qubits = MultiplexerOptions::MAX_QUBITS}}));
  EXPECT_THROW(static_cast<void>(
                   Multiplexer{{.qubits = MultiplexerOptions::MAX_QUBITS + 1}}),
               std::invalid_argument);
}

TEST(Multiplexer, GivesTheUniformControlDistribution) {
  const Multiplexer benchmark{{.qubits = 3}};
  const auto high = (2. + std::numbers::sqrt2) / 16.;
  const auto low = (2. - std::numbers::sqrt2) / 16.;

  EXPECT_NEAR(benchmark.probability("000"), 0.25, 1e-15);
  EXPECT_NEAR(benchmark.probability("001"), 0., 1e-15);
  EXPECT_NEAR(benchmark.probability("010"), high, 1e-15);
  EXPECT_NEAR(benchmark.probability("011"), low, 1e-15);
  EXPECT_NEAR(benchmark.probability("100"), 0.125, 1e-15);
  EXPECT_NEAR(benchmark.probability("101"), 0.125, 1e-15);
  EXPECT_NEAR(benchmark.probability("110"), low, 1e-15);
  EXPECT_NEAR(benchmark.probability("111"), high, 1e-15);

  double total = 0.;
  for (const auto* outcome :
       {"000", "001", "010", "011", "100", "101", "110", "111"}) {
    total += benchmark.probability(outcome);
  }
  EXPECT_NEAR(total, 1., 1e-15);
  EXPECT_THROW(static_cast<void>(benchmark.probability("00")),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(benchmark.probability("00x")),
               std::invalid_argument);
}

TEST(Multiplexer, EvaluatesTheReferenceWithoutASuccessOutcome) {
  const Multiplexer benchmark{{.qubits = 3}};
  const auto evaluation = benchmark.evaluate({{"000", 100}});
  EXPECT_DOUBLE_EQ(evaluation.totalVariationDistance, 0.75);
  EXPECT_DOUBLE_EQ(evaluation.squaredHellingerFidelity, 0.25);
  EXPECT_FALSE(evaluation.successProbability);
}

TEST(Multiplexer, KeepsTheLargestUniformControlWeightRepresentable) {
  const Multiplexer benchmark{{.qubits = MultiplexerOptions::MAX_QUBITS}};
  EXPECT_GT(
      benchmark.probability(std::string(MultiplexerOptions::MAX_QUBITS, '0')),
      0.);
}

TEST(Multiplexer, RoundTripsJSONAndUsesSemanticCaseIds) {
  const auto parsed = multiplexerFromInstanceSpecificationJSON(
      R"({"schema_version":1,"benchmark":"multiplexer","parameters":{"qubits":7}})");
  EXPECT_EQ(parsed.options().qubits, 7);
  EXPECT_EQ(
      toInstanceSpecificationJSON(parsed),
      R"({"benchmark":"multiplexer","parameters":{"qubits":7},"schema_version":1})");

  const auto manifest = toManifestJSON(parsed);
  EXPECT_EQ(toManifestJSON(multiplexerFromManifestJSON(manifest)), manifest);
  EXPECT_EQ(benchmarkIdFromManifestJSON(manifest), "multiplexer");
  EXPECT_NE(manifest.find("\"model\":\"multiplexer\""), std::string::npos);
  EXPECT_EQ(caseId(parsed), caseId(Multiplexer{{.qubits = 7}}));
  EXPECT_NE(caseId(parsed), caseId(Multiplexer{{.qubits = 6}}));
}

TEST(Multiplexer, RejectsInvalidJSONAndDescribesLimits) {
  expectInvalidMultiplexerJSON(
      [] {
        static_cast<void>(multiplexerFromInstanceSpecificationJSON(
            R"({"schema_version":1,"benchmark":"multiplexer","parameters":{"qubits":1}})"));
      },
      "between 2 and 1024");
  expectInvalidMultiplexerJSON(
      [] {
        static_cast<void>(multiplexerFromInstanceSpecificationJSON(
            R"({"schema_version":1,"benchmark":"multiplexer","parameters":{"qubits":7,"angles":[]}})"));
      },
      "unknown key 'angles'");
  const auto schema = describeBenchmarkJSON("multiplexer");
  EXPECT_NE(schema.find("\"maximum\":1024"), std::string::npos);
  EXPECT_NE(schema.find("\"minimum\":2"), std::string::npos);
}

TEST(Multiplexer, EvaluatesJSONWithoutASuccessOutcome) {
  const auto evaluation =
      evaluateJSON(toManifestJSON(Multiplexer{{.qubits = 2}}),
                   R"({"schema_version":1,"counts":{"00":8,"01":2}})");
  EXPECT_NE(evaluation.find("\"success_probability\":null"), std::string::npos);
  EXPECT_NE(evaluation.find("\"total_variation_distance\":"),
            std::string::npos);
}

} // namespace
