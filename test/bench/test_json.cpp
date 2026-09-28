/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "bench/BV.hpp"
#include "bench/Evaluation.hpp"
#include "bench/GHZ.hpp"
#include "bench/JSON.hpp"

#include "gtest/gtest.h"

#include <functional>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

using mqt::bench::benchmarkIdFromInstanceSpecificationJSON;
using mqt::bench::BV;
using mqt::bench::caseId;
using mqt::bench::countsFromJSON;
using mqt::bench::describeBenchmarkJSON;
using mqt::bench::evaluateJSON;
using mqt::bench::Evaluation;
using mqt::bench::evaluationToJSON;
using mqt::bench::GHZ;
using mqt::bench::ghzFromInstanceSpecificationJSON;
using mqt::bench::ghzFromManifestJSON;
using mqt::bench::listBenchmarksJSON;
using mqt::bench::toManifestJSON;

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

TEST(BenchmarkJSON, RejectsMalformedInstanceEnvelopes) {
  expectInvalid(
      [] {
        static_cast<void>(benchmarkIdFromInstanceSpecificationJSON(
            R"({"schema_version":1,"benchmark":"ghz","benchmark":"qpe","parameters":{"qubits":2}})",
            "duplicate.json"));
      },
      "duplicate key 'benchmark'");
  expectInvalid(
      [] {
        static_cast<void>(ghzFromInstanceSpecificationJSON(
            R"({"schema_version":1,"benchmark":"ghz","parameters":{"qubits":2},"extra":true})"));
      },
      "unknown key 'extra'");
  expectInvalid(
      [] {
        static_cast<void>(ghzFromInstanceSpecificationJSON(
            R"({"schema_version":1,"benchmark":"new","parameters":{}})"));
      },
      "unsupported benchmark 'new'");
}

TEST(BenchmarkJSON, RejectsAnInstanceSpecificationForAnotherConcreteType) {
  expectInvalid(
      [] {
        static_cast<void>(ghzFromInstanceSpecificationJSON(
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
  const auto integerWidth = changedNumericKind.find(R"("width":3)");
  ASSERT_NE(integerWidth, std::string::npos);
  changedNumericKind.replace(integerWidth, std::string(R"("width":3)").size(),
                             R"("width":3.0)");
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
  const auto basis = unresolved.find(R"("basis":"z",)");
  ASSERT_NE(basis, std::string::npos);
  unresolved.erase(basis, std::string(R"("basis":"z",)").size());
  expectInvalid([&] { static_cast<void>(ghzFromManifestJSON(unresolved)); },
                "resolved benchmark instance");
}

TEST(BenchmarkJSON, ListsBenchmarksAndDescribesStandardSchemas) {
  EXPECT_EQ(
      listBenchmarksJSON(),
      R"({"benchmarks":[{"definition_version":1,"id":"bv"},{"definition_version":1,"id":"ghz"},{"definition_version":1,"id":"grover"},{"definition_version":1,"id":"grover-weak-measurement"},{"definition_version":1,"id":"magic-state-distillation"},{"definition_version":1,"id":"modular-multiplier"},{"definition_version":1,"id":"multiplexer"},{"definition_version":1,"id":"qft"},{"definition_version":1,"id":"qft-adder"},{"definition_version":1,"id":"qpe"},{"definition_version":1,"id":"repeat-until-success"},{"definition_version":1,"id":"shor"},{"definition_version":1,"id":"teleportation"},{"definition_version":1,"id":"w-state"}],"schema_version":1})");
  const auto schema = describeBenchmarkJSON("ghz");
  EXPECT_NE(schema.find("https://json-schema.org/draft/2020-12/schema"),
            std::string::npos);
  EXPECT_NE(schema.find("\"additionalProperties\":false"), std::string::npos);
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

  const BV bv{{.hiddenBitstring = "11"}};
  const auto generic = evaluateJSON(
      toManifestJSON(bv), R"({"schema_version":1,"counts":{"11":8,"00":2}})");
  EXPECT_NE(generic.find("\"success_probability\":0.8"), std::string::npos);
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
                   "not-a-case", 1,
                   Evaluation{.totalVariationDistance = 0.,
                              .squaredHellingerFidelity = 1.,
                              .successProbability = std::nullopt})),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(evaluationToJSON(
                   caseId(ghz), 1,
                   Evaluation{.totalVariationDistance =
                                  std::numeric_limits<double>::quiet_NaN(),
                              .squaredHellingerFidelity = 1.,
                              .successProbability = std::nullopt})),
               std::invalid_argument);
}

} // namespace
