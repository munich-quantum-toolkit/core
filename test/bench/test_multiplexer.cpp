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

#include "JSONTestUtils.hpp"

#include "gtest/gtest.h"

#include "llvm/Support/Error.h"

#include <cmath>
#include <numbers>
#include <stdexcept>
#include <string>

namespace mqt::bench {

using test::expectInvalidJSON;

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

TEST(Multiplexer, PreservesSmallProbabilitiesNearBothEndpoints) {
  constexpr auto controls = 64;
  const Multiplexer benchmark{{.qubits = controls + 1}};
  const auto amplitude = std::sin(std::ldexp(std::numbers::pi, -controls - 1));
  const auto expected = std::ldexp(amplitude * amplitude, -controls);
  const auto nearZero = std::string(controls - 1, '0') + "11";
  const auto nearOne = std::string(controls, '1') + "0";
  EXPECT_NEAR(benchmark.probability(nearZero) / expected, 1., 1e-14);
  EXPECT_NEAR(benchmark.probability(nearOne) / expected, 1., 1e-14);
}

TEST(Multiplexer, RoundTripsJSON) {
  const auto parsed = llvm::cantFail(multiplexerFromInstanceSpecificationJSON(
      R"({"schema_version":1,"benchmark":"multiplexer","parameters":{"qubits":7}})"));
  EXPECT_EQ(parsed.options().qubits, 7);
  EXPECT_EQ(
      toInstanceSpecificationJSON(parsed),
      R"({"benchmark":"multiplexer","parameters":{"qubits":7},"schema_version":1})");

  const auto manifest = toManifestJSON(parsed);
  EXPECT_EQ(
      toManifestJSON(llvm::cantFail(multiplexerFromManifestJSON(manifest))),
      manifest);
  EXPECT_EQ(llvm::cantFail(benchmarkIdFromManifestJSON(manifest)),
            "multiplexer");
  EXPECT_NE(manifest.find("\"model\":\"multiplexer\""), std::string::npos);
}

TEST(Multiplexer, UsesSemanticCaseIds) {
  const auto parsed = llvm::cantFail(multiplexerFromInstanceSpecificationJSON(
      R"({"schema_version":1,"benchmark":"multiplexer","parameters":{"qubits":7}})"));
  EXPECT_EQ(caseId(parsed), caseId(Multiplexer{{.qubits = 7}}));
  EXPECT_NE(caseId(parsed), caseId(Multiplexer{{.qubits = 6}}));
}

TEST(Multiplexer, DescribesJSONSchema) {
  const auto schema = llvm::cantFail(describeBenchmarkJSON("multiplexer"));
  EXPECT_NE(schema.find("\"maximum\":1024"), std::string::npos);
  EXPECT_NE(schema.find("\"minimum\":2"), std::string::npos);
}

TEST(Multiplexer, RejectsInvalidJSONParameters) {
  expectInvalidJSON(
      [] {
        return multiplexerFromInstanceSpecificationJSON(
            R"({"schema_version":1,"benchmark":"multiplexer","parameters":{"qubits":1}})");
      },
      "between 2 and 1024");
  expectInvalidJSON(
      [] {
        return multiplexerFromInstanceSpecificationJSON(
            R"({"schema_version":1,"benchmark":"multiplexer","parameters":{"qubits":7,"angles":[]}})");
      },
      "angles");
}

TEST(Multiplexer, EvaluatesCountsFromJSON) {
  const auto evaluation = llvm::cantFail(
      evaluateJSON(toManifestJSON(Multiplexer{{.qubits = 2}}),
                   R"({"schema_version":1,"counts":{"00":8,"01":2}})"));
  EXPECT_NE(evaluation.find("\"success_probability\":null"), std::string::npos);
  EXPECT_NE(evaluation.find("\"total_variation_distance\":"),
            std::string::npos);
}

} // namespace mqt::bench
