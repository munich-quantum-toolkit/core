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

#include <limits>
#include <optional>
#include <stdexcept>
#include <string>

namespace {

using mqt::bench::benchmarkIdFromInstanceSpecificationJSON;
using mqt::bench::caseId;
using mqt::bench::countsFromJSON;
using mqt::bench::describeBenchmarkJSON;
using mqt::bench::Evaluation;
using mqt::bench::evaluationToJSON;
using mqt::bench::GHZ;
using mqt::bench::ghzFromInstanceSpecificationJSON;
using mqt::bench::ghzFromManifestJSON;
using mqt::bench::listBenchmarksJSON;
using mqt::bench::toManifestJSON;
using mqt::bench::test::expectInvalidJSON;

// GHZ is a representative fixture for the shared JSON contracts below.

TEST(BenchmarkJSON, RejectsDuplicateKeys) {
  expectInvalidJSON(
      [] {
        static_cast<void>(benchmarkIdFromInstanceSpecificationJSON(
            R"({"schema_version":1,"benchmark":"ghz","benchmark":"qpe","parameters":{"qubits":2}})",
            "duplicate.json"));
      },
      "duplicate key 'benchmark'");
  expectInvalidJSON(
      [] {
        static_cast<void>(benchmarkIdFromInstanceSpecificationJSON(
            R"({"schema_version":1,"benchmark":"ghz","parameters":{"qubits":2,"qubits":3}})"));
      },
      "duplicate key 'qubits'");
  expectInvalidJSON(
      [] {
        static_cast<void>(benchmarkIdFromInstanceSpecificationJSON(
            R"({"schema_version":1,"benchmark":"qpe","parameters":{"precision":2,"phase":{"numerator":1,"denominator":4,"numerator":2}}})"));
      },
      "duplicate key 'numerator'");
}

TEST(BenchmarkJSON, RejectsInvalidInstanceEnvelopes) {
  expectInvalidJSON(
      [] {
        static_cast<void>(benchmarkIdFromInstanceSpecificationJSON(
            R"({"schema_version":1,"benchmark":"ghz","parameters":{"qubits":2},"extra":true})"));
      },
      "unknown key 'extra'");
  expectInvalidJSON(
      [] {
        static_cast<void>(benchmarkIdFromInstanceSpecificationJSON(
            R"({"schema_version":1,"benchmark":"new","parameters":{}})"));
      },
      "unsupported benchmark 'new'");
  expectInvalidJSON(
      [] {
        static_cast<void>(ghzFromInstanceSpecificationJSON(
            R"({"schema_version":1,"benchmark":"qpe","parameters":{"precision":2,"phase":{"numerator":1,"denominator":4}}})"));
      },
      "must be 'ghz'");
}

TEST(BenchmarkJSON, RejectsAlteredOrUnresolvedManifests) {
  const GHZ ghz{{.qubits = 3}};
  const auto manifest = toManifestJSON(ghz);
  EXPECT_NE(manifest.find("\"case_id\":\"" + caseId(ghz) + "\""),
            std::string::npos);

  auto changedOutput = manifest;
  const auto width = changedOutput.find("\"width\":3");
  ASSERT_NE(width, std::string::npos);
  changedOutput.replace(width, std::string("\"width\":3").size(),
                        "\"width\":2");
  expectInvalidJSON(
      [&] { static_cast<void>(ghzFromManifestJSON(changedOutput)); },
      "does not match");

  auto changedNumericKind = manifest;
  const auto integerWidth = changedNumericKind.find(R"("width":3)");
  ASSERT_NE(integerWidth, std::string::npos);
  changedNumericKind.replace(integerWidth, std::string(R"("width":3)").size(),
                             R"("width":3.0)");
  expectInvalidJSON(
      [&] { static_cast<void>(ghzFromManifestJSON(changedNumericKind)); },
      "does not match");

  auto changedId = manifest;
  const auto digest = changedId.find("sha256-");
  ASSERT_NE(digest, std::string::npos);
  changedId[digest + 7U] = changedId[digest + 7U] == '0' ? '1' : '0';
  expectInvalidJSON([&] { static_cast<void>(ghzFromManifestJSON(changedId)); },
                    "case ID");

  auto unresolved = manifest;
  const auto basis = unresolved.find(R"("basis":"z",)");
  ASSERT_NE(basis, std::string::npos);
  unresolved.erase(basis, std::string(R"("basis":"z",)").size());
  expectInvalidJSON([&] { static_cast<void>(ghzFromManifestJSON(unresolved)); },
                    "resolved benchmark instance");
}

TEST(BenchmarkJSON, ListsBenchmarksAndRejectsUnknownSchemas) {
  EXPECT_EQ(
      listBenchmarksJSON(),
      R"({"benchmarks":[{"definition_version":1,"id":"bv"},{"definition_version":1,"id":"ghz"},{"definition_version":1,"id":"grover"},{"definition_version":1,"id":"grover-weak-measurement"},{"definition_version":1,"id":"magic-state-distillation"},{"definition_version":1,"id":"modular-multiplier"},{"definition_version":1,"id":"multiplexer"},{"definition_version":1,"id":"qft"},{"definition_version":1,"id":"qft-adder"},{"definition_version":1,"id":"qpe"},{"definition_version":1,"id":"repeat-until-success"},{"definition_version":1,"id":"shor"},{"definition_version":1,"id":"teleportation"},{"definition_version":1,"id":"w-state"}],"schema_version":1})");
  EXPECT_THROW(static_cast<void>(describeBenchmarkJSON("unknown")),
               std::invalid_argument);
}

TEST(BenchmarkJSON, DescribesSchemaEnvelope) {
  const auto schema = describeBenchmarkJSON("ghz");
  EXPECT_NE(schema.find("https://json-schema.org/draft/2020-12/schema"),
            std::string::npos);
  EXPECT_NE(schema.find("\"additionalProperties\":false"), std::string::npos);
}

TEST(BenchmarkJSON, ParsesCounts) {
  const auto counts =
      countsFromJSON(R"({"counts":{"11":50,"00":50},"schema_version":1})");
  EXPECT_EQ(counts.at("00"), 50);
  EXPECT_EQ(counts.at("11"), 50);

  expectInvalidJSON(
      [] {
        static_cast<void>(
            countsFromJSON(R"({"schema_version":1,"counts":{"0":1,"0":2}})"));
      },
      "duplicate key '0'");
  expectInvalidJSON(
      [] {
        static_cast<void>(
            countsFromJSON(R"({"schema_version":1,"counts":{"0x":1}})"));
      },
      "bitstrings");
  expectInvalidJSON(
      [] {
        static_cast<void>(
            countsFromJSON(R"({"schema_version":1,"counts":{"00":0}})"));
      },
      "must be positive");
}

TEST(BenchmarkJSON, SerializesEvaluations) {
  const auto serialized =
      evaluationToJSON("sha256-" + std::string(64, '0'), 100,
                       Evaluation{
                           .totalVariationDistance = 0.,
                           .squaredHellingerFidelity = 1.,
                           .successProbability = std::nullopt,
                       });
  EXPECT_NE(serialized.find("\"squared_hellinger_fidelity\":1.0"),
            std::string::npos);
  EXPECT_NE(serialized.find("\"success_probability\":null"), std::string::npos);
  EXPECT_NE(serialized.find("\"total_variation_distance\":0.0"),
            std::string::npos);
}

TEST(BenchmarkJSON, RejectsInvalidEvaluations) {
  const auto validCaseId = "sha256-" + std::string(64, '0');
  EXPECT_THROW(static_cast<void>(evaluationToJSON(
                   "not-a-case", 1,
                   Evaluation{.totalVariationDistance = 0.,
                              .squaredHellingerFidelity = 1.,
                              .successProbability = std::nullopt})),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(evaluationToJSON(
                   validCaseId, 1,
                   Evaluation{.totalVariationDistance =
                                  std::numeric_limits<double>::quiet_NaN(),
                              .squaredHellingerFidelity = 1.,
                              .successProbability = std::nullopt})),
               std::invalid_argument);
}

} // namespace
