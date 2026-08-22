/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "benchmarks/JSON.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

using mqt::benchmarks::benchmarkIdFromManifestJSON;
using mqt::benchmarks::benchmarkIdFromRequestJSON;
using mqt::benchmarks::caseId;
using mqt::benchmarks::countsFromJSON;
using mqt::benchmarks::describeBenchmarkJSON;
using mqt::benchmarks::Evaluation;
using mqt::benchmarks::evaluationToJSON;
using mqt::benchmarks::GHZ;
using mqt::benchmarks::GHZBasis;
using mqt::benchmarks::ghzFromManifestJSON;
using mqt::benchmarks::ghzFromRequestJSON;
using mqt::benchmarks::GHZTopology;
using mqt::benchmarks::Grover;
using mqt::benchmarks::groverFromManifestJSON;
using mqt::benchmarks::groverFromRequestJSON;
using mqt::benchmarks::listBenchmarksJSON;
using mqt::benchmarks::Phase;
using mqt::benchmarks::QPE;
using mqt::benchmarks::qpeFromManifestJSON;
using mqt::benchmarks::qpeFromRequestJSON;
using mqt::benchmarks::QPEMethod;
using mqt::benchmarks::toManifestJSON;
using mqt::benchmarks::toRequestJSON;

void expectInvalid(const std::function<void()>& operation,
                   const std::string_view diagnostic) {
  try {
    operation();
    FAIL() << "Expected invalid JSON input";
  } catch (const std::invalid_argument& error) {
    EXPECT_NE(std::string(error.what()).find(diagnostic), std::string::npos)
        << error.what();
  }
}

TEST(BenchmarkJSON, ParsesRequestsAndSerializesResolvedParameters) {
  const auto ghz = ghzFromRequestJSON(
      R"({"parameters":{"qubits":3},"benchmark":"ghz","schema_version":1})");
  EXPECT_EQ(ghz.options().topology, GHZTopology::Linear);
  EXPECT_EQ(ghz.options().basis, GHZBasis::Z);
  EXPECT_EQ(
      toRequestJSON(ghz),
      R"({"benchmark":"ghz","parameters":{"basis":"z","qubits":3,"topology":"linear"},"schema_version":1})");

  const auto grover = groverFromRequestJSON(
      R"({"schema_version":1,"benchmark":"grover","parameters":{"marked_bitstring":"10"}})");
  ASSERT_TRUE(grover.options().iterations);
  EXPECT_EQ(*grover.options().iterations, 1);
  EXPECT_EQ(
      toRequestJSON(grover),
      R"({"benchmark":"grover","parameters":{"iterations":1,"marked_bitstring":"10"},"schema_version":1})");

  const auto qpe = qpeFromRequestJSON(
      R"({"schema_version":1,"benchmark":"qpe","parameters":{"precision":4,"phase":{"numerator":10,"denominator":8},"method":"iterative"}})");
  EXPECT_EQ(qpe.options().phase, Phase(1, 4));
  EXPECT_EQ(qpe.options().method, QPEMethod::Iterative);
  EXPECT_EQ(
      toRequestJSON(qpe),
      R"({"benchmark":"qpe","parameters":{"method":"iterative","phase":{"denominator":4,"numerator":1},"precision":4},"schema_version":1})");

  const auto integralFloats = ghzFromRequestJSON(
      R"({"schema_version":1.0,"benchmark":"ghz","parameters":{"qubits":3.0}})");
  EXPECT_EQ(integralFloats.options().qubits, 3);
}

TEST(BenchmarkJSON, RoundTripsSelfCheckingManifests) {
  const GHZ ghz{
      {.qubits = 4, .topology = GHZTopology::Star, .basis = GHZBasis::X}};
  const Grover grover{{.markedBitstring = "001", .iterations = 2}};
  const QPE qpe{
      {.precision = 5, .phase = Phase(1, 3), .method = QPEMethod::Iterative}};

  const auto ghzManifest = toManifestJSON(ghz);
  const auto groverManifest = toManifestJSON(grover);
  const auto qpeManifest = toManifestJSON(qpe);
  EXPECT_EQ(toManifestJSON(ghzFromManifestJSON(ghzManifest)), ghzManifest);
  EXPECT_EQ(toManifestJSON(groverFromManifestJSON(groverManifest)),
            groverManifest);
  EXPECT_EQ(toManifestJSON(qpeFromManifestJSON(qpeManifest)), qpeManifest);
  EXPECT_EQ(benchmarkIdFromManifestJSON(ghzManifest), "ghz");
  EXPECT_EQ(benchmarkIdFromManifestJSON(groverManifest), "grover");
  EXPECT_EQ(benchmarkIdFromManifestJSON(qpeManifest), "qpe");
  EXPECT_NE(ghzManifest.find("\"case_id\":\"" + caseId(ghz) + "\""),
            std::string::npos);
  EXPECT_NE(groverManifest.find("\"success_outcome\":\"001\""),
            std::string::npos);
  EXPECT_EQ(qpeManifest.find("0.333"), std::string::npos);
}

TEST(BenchmarkJSON, UsesStableSemanticCaseIds) {
  const GHZ linear{{.qubits = 3}};
  const GHZ same{{.qubits = 3}};
  const GHZ star{{.qubits = 3, .topology = GHZTopology::Star}};
  EXPECT_EQ(caseId(linear), caseId(same));
  EXPECT_NE(caseId(linear), caseId(star));
  EXPECT_EQ(caseId(linear), "sha256-a222c0c57bcecb4f5e7ea72bab439683"
                            "92861a52c5cb7c9c13aeaffffa059a65");
}

TEST(BenchmarkJSON, RejectsDuplicateUnknownAndMistypedRequestValues) {
  expectInvalid(
      [] {
        static_cast<void>(benchmarkIdFromRequestJSON(
            R"({"schema_version":1,"benchmark":"ghz","benchmark":"qpe","parameters":{"qubits":2}})",
            "duplicate.json"));
      },
      "duplicate key 'benchmark'");
  expectInvalid(
      [] {
        static_cast<void>(ghzFromRequestJSON(
            R"({"schema_version":1,"benchmark":"ghz","parameters":{"qubits":2,"qubits":3}})"));
      },
      "duplicate key 'qubits'");
  expectInvalid(
      [] {
        static_cast<void>(qpeFromRequestJSON(
            R"({"schema_version":1,"benchmark":"qpe","parameters":{"precision":2,"phase":{"numerator":1,"denominator":4,"numerator":2}}})"));
      },
      "duplicate key 'numerator'");
  expectInvalid(
      [] {
        static_cast<void>(ghzFromRequestJSON(
            R"({"schema_version":1,"benchmark":"ghz","parameters":{"qubits":2},"extra":true})"));
      },
      "unknown key 'extra'");
  expectInvalid(
      [] {
        static_cast<void>(ghzFromRequestJSON(
            R"({"schema_version":1,"benchmark":"ghz","parameters":{"qubits":2,"extra":true}})"));
      },
      "unknown key 'extra'");
  expectInvalid(
      [] {
        static_cast<void>(ghzFromRequestJSON(
            R"({"schema_version":1,"benchmark":"ghz","parameters":{"qubits":2.5}})"));
      },
      "non-negative integer");
  expectInvalid(
      [] {
        static_cast<void>(qpeFromRequestJSON(
            R"({"schema_version":1,"benchmark":"qpe","parameters":{"precision":18446744073709551615,"phase":{"numerator":1,"denominator":4}}})"));
      },
      "between 1 and 1000000");
  expectInvalid(
      [] {
        static_cast<void>(ghzFromRequestJSON(
            R"({"schema_version":1,"benchmark":"new","parameters":{}})"));
      },
      "unsupported benchmark 'new'");
  expectInvalid(
      [] {
        static_cast<void>(qpeFromRequestJSON(
            R"({"schema_version":1,"benchmark":"qpe","parameters":{"precision":2,"phase":{"numerator":1,"denominator":0}}})"));
      },
      "denominator must not be zero");
}

TEST(BenchmarkJSON, RejectsARequestForAnotherConcreteType) {
  expectInvalid(
      [] {
        static_cast<void>(ghzFromRequestJSON(
            R"({"schema_version":1,"benchmark":"qpe","parameters":{"precision":2,"phase":{"numerator":1,"denominator":4}}})"));
      },
      "must be 'ghz'");
}

TEST(BenchmarkJSON, RejectsAlteredOrUnresolvedManifestData) {
  const GHZ ghz{{.qubits = 3}};
  auto changedOutput = toManifestJSON(ghz);
  const auto width = changedOutput.find("\"width\":3");
  ASSERT_NE(width, std::string::npos);
  changedOutput.replace(width, std::string("\"width\":3").size(),
                        "\"width\":2");
  expectInvalid([&] { static_cast<void>(ghzFromManifestJSON(changedOutput)); },
                "does not match");

  auto changedNumericKind = toManifestJSON(ghz);
  const auto integerWidth = changedNumericKind.find("\"width\":3");
  ASSERT_NE(integerWidth, std::string::npos);
  changedNumericKind.replace(integerWidth, std::string("\"width\":3").size(),
                             "\"width\":3.0");
  expectInvalid(
      [&] { static_cast<void>(ghzFromManifestJSON(changedNumericKind)); },
      "does not match");

  auto changedId = toManifestJSON(ghz);
  const auto digest = changedId.find("sha256-");
  ASSERT_NE(digest, std::string::npos);
  changedId[digest + 7U] = changedId[digest + 7U] == '0' ? '1' : '0';
  expectInvalid([&] { static_cast<void>(ghzFromManifestJSON(changedId)); },
                "case ID");

  auto unresolved = toManifestJSON(ghz);
  const auto basis = unresolved.find("\"basis\":\"z\",");
  ASSERT_NE(basis, std::string::npos);
  unresolved.erase(basis, std::string("\"basis\":\"z\",").size());
  expectInvalid([&] { static_cast<void>(ghzFromManifestJSON(unresolved)); },
                "resolved benchmark instance");
}

TEST(BenchmarkJSON, ListsBenchmarksAndDescribesStandardSchemas) {
  EXPECT_EQ(
      listBenchmarksJSON(),
      R"({"benchmarks":[{"definition_version":1,"id":"ghz"},{"definition_version":1,"id":"grover"},{"definition_version":1,"id":"qpe"}],"schema_version":1})");
  const auto ghz = describeBenchmarkJSON("ghz");
  const auto grover = describeBenchmarkJSON("grover");
  const auto qpe = describeBenchmarkJSON("qpe");
  EXPECT_NE(ghz.find("https://json-schema.org/draft/2020-12/schema"),
            std::string::npos);
  EXPECT_NE(ghz.find("\"additionalProperties\":false"), std::string::npos);
  EXPECT_NE(ghz.find("\"maximum\":1000000"), std::string::npos);
  EXPECT_NE(grover.find("\"maxLength\":62"), std::string::npos);
  EXPECT_NE(qpe.find("\"iterative\""), std::string::npos);
  EXPECT_THROW(static_cast<void>(describeBenchmarkJSON("unknown")),
               std::invalid_argument);
}

TEST(BenchmarkJSON, ParsesCountsAndSerializesEvaluations) {
  const auto counts =
      countsFromJSON(R"({"counts":{"11":50,"00":50},"schema_version":1})");
  EXPECT_EQ(counts.at("00"), 50);
  EXPECT_EQ(counts.at("11"), 50);

  const GHZ ghz{{.qubits = 2}};
  const auto serialized =
      evaluationToJSON(caseId(ghz), 100, ghz.evaluate(counts));
  EXPECT_NE(serialized.find("\"squared_hellinger_fidelity\":1.0"),
            std::string::npos);
  EXPECT_NE(serialized.find("\"success_probability\":null"), std::string::npos);
  EXPECT_NE(serialized.find("\"total_variation_distance\":0.0"),
            std::string::npos);

  expectInvalid(
      [] {
        static_cast<void>(
            countsFromJSON(R"({"schema_version":1,"counts":{"0":1,"0":2}})"));
      },
      "duplicate key '0'");
  expectInvalid(
      [] {
        static_cast<void>(
            countsFromJSON(R"({"schema_version":1,"counts":{"0x":1}})"));
      },
      "bitstrings");
  expectInvalid(
      [] {
        static_cast<void>(
            countsFromJSON(R"({"schema_version":1,"counts":{"00":0}})"));
      },
      "must be positive");
  EXPECT_THROW(static_cast<void>(evaluationToJSON(
                   "not-a-case", 1, Evaluation{0., 1., std::nullopt})),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(evaluationToJSON(
                   caseId(ghz), 1,
                   Evaluation{std::numeric_limits<double>::quiet_NaN(), 1.,
                              std::nullopt})),
               std::invalid_argument);
}

} // namespace
