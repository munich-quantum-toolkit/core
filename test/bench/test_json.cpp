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
#include "bench/WState.hpp"

#include "JSONTestUtils.hpp"
#include "support/JSON.hpp"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"

#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

namespace mqt::bench {

using test::expectInvalidJSON;

// GHZ is a representative fixture for the shared JSON contracts below.

TEST(BenchmarkJSON, ReportsParameterErrorsAtTheirSource) {
  for (
      const auto& [json, pointer] : {
          std::pair{
              R"({"schema_version":1,"benchmark":"multiplexer","parameters":{}})",
              "$/parameters/qubits"},
          std::pair{
              R"({"schema_version":1,"benchmark":"w-state","parameters":{}})",
              "$/parameters/qubits"},
          std::pair{
              R"({"schema_version":1,"benchmark":"modular-multiplier","parameters":{}})",
              "$/parameters/multiplier"},
      }) {
    SCOPED_TRACE(json);
    auto result = parseInstanceSpecificationJSON(json, "case.json");
    ASSERT_FALSE(static_cast<bool>(result));
    const auto message = llvm::toString(result.takeError());
    EXPECT_THAT(message, testing::HasSubstr("case.json"));
    EXPECT_THAT(message, testing::HasSubstr(std::string_view(pointer).substr(
                             std::string_view("$/parameters/").size())));
  }
}

TEST(BenchmarkJSON, RejectsDuplicateKeys) {
  expectInvalidJSON(
      [] {
        return benchmarkIdFromInstanceSpecificationJSON(
            R"({"schema_version":1,"benchmark":"ghz","benchmark":"qpe","parameters":{"qubits":2}})",
            "duplicate.json");
      },
      "duplicate key 'benchmark'");
  expectInvalidJSON(
      [] {
        return benchmarkIdFromInstanceSpecificationJSON(
            R"({"schema_version":1,"benchmark":"ghz","parameters":{"qubits":2,"qubits":3}})");
      },
      "duplicate key 'qubits'");
  expectInvalidJSON(
      [] {
        return benchmarkIdFromInstanceSpecificationJSON(
            R"({"schema_version":1,"benchmark":"qpe","parameters":{"precision":2,"phase":{"numerator":1,"denominator":4,"numerator":2}}})");
      },
      "duplicate key 'numerator'");
}

TEST(BenchmarkJSON, RejectsInvalidInstanceEnvelopes) {
  expectInvalidJSON(
      [] {
        return benchmarkIdFromInstanceSpecificationJSON(
            R"({"schema_version":1,"benchmark":"ghz","parameters":{"qubits":2},"extra":true})");
      },
      "extra");
  expectInvalidJSON(
      [] {
        return benchmarkIdFromInstanceSpecificationJSON(
            R"({"schema_version":1,"benchmark":"new","parameters":{}})");
      },
      "unsupported benchmark 'new'");
  expectInvalidJSON(
      [] {
        return ghzFromInstanceSpecificationJSON(
            R"({"schema_version":1,"benchmark":"qpe","parameters":{"precision":2,"phase":{"numerator":1,"denominator":4}}})");
      },
      "must be 'ghz'");
}

TEST(BenchmarkJSON, RejectsAlteredOrUnresolvedManifests) {
  const GHZ ghz{{.qubits = 3}};
  const auto manifest = toManifestJSON(ghz);
  EXPECT_NE(manifest.find("\"case_id\":\"" + caseId(ghz) + "\""),
            std::string::npos);
  const auto reject = [](const std::string& invalid,
                         const std::string_view diagnostic) {
    expectInvalidJSON(
        [&] { return ghzFromManifestJSON(invalid, "manifest.json"); },
        diagnostic);
    expectInvalidJSON(
        [&] {
          return evaluateJSON(invalid,
                              R"({"schema_version":1,"counts":{"000":1}})",
                              "manifest.json", "counts.json");
        },
        diagnostic);
  };

  auto changedDefinition = manifest;
  const auto version = changedDefinition.find(R"("definition_version":1)");
  ASSERT_NE(version, std::string::npos);
  changedDefinition.replace(version,
                            std::string(R"("definition_version":1)").size(),
                            R"("definition_version":0)");
  reject(changedDefinition, "$/definition_version must be 1");

  auto changedOutput = manifest;
  const auto width = changedOutput.find("\"width\":3");
  ASSERT_NE(width, std::string::npos);
  changedOutput.replace(width, std::string("\"width\":3").size(),
                        "\"width\":2");
  reject(changedOutput, "does not match");

  auto changedId = manifest;
  const auto digest = changedId.find("sha256-");
  ASSERT_NE(digest, std::string::npos);
  changedId[digest + 7U] = changedId[digest + 7U] == '0' ? '1' : '0';
  reject(changedId, "case ID");

  auto unresolved = manifest;
  const auto basis = unresolved.find(R"("basis":"z",)");
  ASSERT_NE(basis, std::string::npos);
  unresolved.erase(basis, std::string(R"("basis":"z",)").size());
  reject(unresolved, "resolved benchmark instance");

  auto invalidParameters = manifest;
  const auto qubits = invalidParameters.find(R"("qubits":3)");
  ASSERT_NE(qubits, std::string::npos);
  invalidParameters.replace(qubits, std::string(R"("qubits":3)").size(),
                            R"("qubits":0)");
  reject(invalidParameters, "manifest.json:$/parameters GHZ qubits");
  expectInvalidJSON(
      [&] {
        return evaluateJSON(invalidParameters, "{}", "manifest.json",
                            "counts.json");
      },
      "counts.json.schema_version");
}

TEST(BenchmarkJSON, ComparesReferenceVersionByValue) {
  const GHZ ghz{{.qubits = 3}};
  auto manifest = llvm::cantFail(mqt::json::parse(toManifestJSON(ghz)));
  auto& version = (*manifest.getAsObject()->getObject("reference"))["version"];
  version = 1.0;
  const auto equivalent = mqt::json::serialize(manifest);
  auto parsed = ghzFromManifestJSON(equivalent);
  ASSERT_TRUE(static_cast<bool>(parsed)) << llvm::toString(parsed.takeError());
  EXPECT_EQ(parsed->output(), ghz.output());
  constexpr std::string_view counts =
      R"({"schema_version":1,"counts":{"000":1}})";
  EXPECT_EQ(llvm::cantFail(evaluateJSON(equivalent, counts)),
            llvm::cantFail(evaluateJSON(toManifestJSON(ghz), counts)));

  for (const auto invalid : {1.5, 0x1p63}) {
    version = invalid;
    const auto changed = mqt::json::serialize(manifest);
    expectInvalidJSON([&] { return ghzFromManifestJSON(changed); },
                      "does not match");
    expectInvalidJSON([&] { return evaluateJSON(changed, counts); },
                      "does not match");
  }
}

TEST(BenchmarkJSON, AcceptsEquivalentOutputWidths) {
  const GHZ benchmark{{.qubits = 3}};
  constexpr std::string_view field = R"("width":3)";
  for (const auto* spelling : {"3", "3.0", "3e0"}) {
    SCOPED_TRACE(spelling);
    auto manifest = toManifestJSON(benchmark);
    const auto position = manifest.find(field);
    ASSERT_NE(position, std::string::npos);
    manifest.replace(position, field.size(),
                     std::string(R"("width":)") + spelling);
    auto parsed = ghzFromManifestJSON(manifest);
    ASSERT_TRUE(static_cast<bool>(parsed))
        << llvm::toString(parsed.takeError());
    EXPECT_EQ(toManifestJSON(*parsed), toManifestJSON(benchmark));
    constexpr std::string_view counts =
        R"({"schema_version":1,"counts":{"000":1}})";
    EXPECT_EQ(llvm::cantFail(evaluateJSON(manifest, counts)),
              llvm::cantFail(evaluateJSON(toManifestJSON(benchmark), counts)));
  }
}

TEST(BenchmarkJSON, RejectsRoundedOutputWidths) {
  for (const auto size : {
           uint64_t{1} << 53U,
           (uint64_t{1} << 53U) + 1,
           uint64_t{std::numeric_limits<int64_t>::max()},
       }) {
    SCOPED_TRACE(size);
    const WState benchmark{{.qubits = size}};
    auto manifest = llvm::cantFail(mqt::json::parse(toManifestJSON(benchmark)));
    auto& output = manifest.getAsObject()->getArray("outputs")->front();
    auto& width = (*output.getAsObject())["width"];
    EXPECT_EQ(llvm::cantFail(wStateFromManifestJSON(toManifestJSON(benchmark)))
                  .output(),
              benchmark.output());
    const auto reject = [&] {
      const auto changed = mqt::json::serialize(manifest);
      expectInvalidJSON([&] { return wStateFromManifestJSON(changed); },
                        "does not match");
      expectInvalidJSON(
          [&] {
            return evaluateJSON(changed,
                                R"({"schema_version":1,"counts":{"0":1}})");
          },
          "does not match");
    };
    width = size + 1;
    reject();
    width = static_cast<double>(size);
    if (size == uint64_t{1} << 53U) {
      EXPECT_EQ(
          llvm::cantFail(wStateFromManifestJSON(mqt::json::serialize(manifest)))
              .output(),
          benchmark.output());
    } else {
      reject();
    }
  }
}

TEST(BenchmarkJSON, EvaluatesAResolvedManifest) {
  const GHZ ghz{{.qubits = 2}};
  const Counts counts{{"00", 5}, {"11", 3}};
  EXPECT_EQ(
      llvm::cantFail(
          evaluateJSON(toManifestJSON(ghz),
                       R"({"schema_version":1,"counts":{"00":5,"11":3}})")),
      llvm::cantFail(evaluationToJSON(caseId(ghz), 8, ghz.evaluate(counts))));
}

TEST(BenchmarkJSON, ListsBenchmarksAndRejectsUnknownSchemas) {
  EXPECT_EQ(
      listBenchmarksJSON(),
      R"({"benchmarks":[{"definition_version":1,"id":"bv"},{"definition_version":1,"id":"fan-out"},{"definition_version":1,"id":"ghz"},{"definition_version":1,"id":"grover"},{"definition_version":1,"id":"grover-weak-measurement"},{"definition_version":1,"id":"magic-state-distillation"},{"definition_version":1,"id":"modular-multiplier"},{"definition_version":1,"id":"multiplexer"},{"definition_version":1,"id":"qaoa"},{"definition_version":1,"id":"qft"},{"definition_version":1,"id":"qft-adder"},{"definition_version":1,"id":"qpe"},{"definition_version":1,"id":"repeat-until-success"},{"definition_version":1,"id":"shor"},{"definition_version":1,"id":"teleportation"},{"definition_version":1,"id":"w-state"}],"schema_version":1})");
  test::expectInvalidJSON([&] { return describeBenchmarkJSON("unknown"); }, "");
}

TEST(BenchmarkJSON, DescribesSchemaEnvelope) {
  const auto schema = llvm::cantFail(describeBenchmarkJSON("ghz"));
  EXPECT_NE(schema.find("https://json-schema.org/draft/2020-12/schema"),
            std::string::npos);
  EXPECT_NE(schema.find("\"additionalProperties\":false"), std::string::npos);
}

TEST(BenchmarkJSON, ParsesCounts) {
  const auto counts = llvm::cantFail(
      countsFromJSON(R"({"counts":{"11":50,"00":50},"schema_version":1})"));
  EXPECT_EQ(counts.at("00"), 50);
  EXPECT_EQ(counts.at("11"), 50);

  expectInvalidJSON(
      [] {
        return countsFromJSON(R"({"schema_version":1,"counts":{"0":1,"0":2}})");
      },
      "duplicate key '0'");
  expectInvalidJSON(
      [] {
        return countsFromJSON(R"({"schema_version":1,"counts":{"0x":1}})");
      },
      "bitstrings");
  expectInvalidJSON(
      [] {
        return countsFromJSON(R"({"schema_version":1,"counts":{"00":0}})");
      },
      "must be a positive integer");
}

TEST(BenchmarkJSON, SerializesEvaluations) {
  const auto serialized =
      llvm::cantFail(evaluationToJSON("sha256-" + std::string(64, '0'), 100,
                                      Evaluation{
                                          .totalVariationDistance = 0.,
                                          .squaredHellingerFidelity = 1.,
                                          .successProbability = std::nullopt,
                                      }));
  EXPECT_NE(serialized.find("\"squared_hellinger_fidelity\":1.0"),
            std::string::npos);
  EXPECT_NE(serialized.find("\"success_probability\":null"), std::string::npos);
  EXPECT_NE(serialized.find("\"total_variation_distance\":0.0"),
            std::string::npos);
}

TEST(BenchmarkJSON, RejectsInvalidEvaluations) {
  const auto validCaseId = "sha256-" + std::string(64, '0');
  test::expectInvalidJSON(
      [&] {
        return evaluationToJSON("not-a-case", 1,
                                Evaluation{
                                    .totalVariationDistance = 0.,
                                    .squaredHellingerFidelity = 1.,
                                    .successProbability = std::nullopt,
                                });
      },
      "");
  test::expectInvalidJSON(
      [&] {
        return evaluationToJSON(
            validCaseId, 1,
            Evaluation{
                .totalVariationDistance =
                    std::numeric_limits<double>::quiet_NaN(),
                .squaredHellingerFidelity = 1.,
                .successProbability = std::nullopt,
            });
      },
      "");
}

TEST(BenchmarkJSON, RequiresCurrentManifestEncoding) {
  auto manifest = toManifestJSON(GHZ{{.qubits = 2}});
  const auto version = manifest.find("\"schema_version\":2");
  ASSERT_NE(version, std::string::npos);
  manifest.replace(version, std::string("\"schema_version\":2").size(),
                   "\"schema_version\":1");
  expectInvalidJSON([&] { return ghzFromManifestJSON(manifest); },
                    "regenerate version 1");
}

} // namespace mqt::bench

TEST(BenchmarkJSON, PreservesConditionalAndIntegerSchemaConstraints) {
  auto adder = llvm::cantFail(mqt::json::parse(
      llvm::cantFail(mqt::bench::describeBenchmarkJSON("qft-adder"))));
  const auto* constraints = adder.getAsObject()
                                ->getObject("properties")
                                ->getObject("parameters")
                                ->getArray("allOf");
  ASSERT_NE(constraints, nullptr);
  ASSERT_EQ(constraints->size(), 2U);
  const auto* method = (*constraints)[0]
                           .getAsObject()
                           ->getObject("if")
                           ->getObject("properties")
                           ->getObject("method");
  ASSERT_NE(method, nullptr);
  EXPECT_EQ(method->getString("const"), "constant");
  auto qpe = llvm::cantFail(mqt::json::parse(
      llvm::cantFail(mqt::bench::describeBenchmarkJSON("qpe"))));
  const auto* denominator = qpe.getAsObject()
                                ->getObject("properties")
                                ->getObject("parameters")
                                ->getObject("properties")
                                ->getObject("phase")
                                ->getObject("properties")
                                ->getObject("denominator");
  ASSERT_NE(denominator, nullptr);
  const auto* maximum = denominator->get("maximum");
  ASSERT_NE(maximum, nullptr);
  EXPECT_EQ(maximum->getAsUINT64(), std::numeric_limits<uint64_t>::max());
}
