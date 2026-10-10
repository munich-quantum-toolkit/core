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
#include "bench/Grover.hpp"
#include "bench/JSON.hpp"
#include "bench/ModularMultiplier.hpp"
#include "bench/Multiplexer.hpp"
#include "bench/QFT.hpp"
#include "bench/QFTAdder.hpp"
#include "bench/QPE.hpp"
#include "bench/RepeatUntilSuccess.hpp"
#include "bench/Teleportation.hpp"

#include "JSONTestUtils.hpp"
#include "support/Diagnostics.hpp"
#include "support/TestSupport.hpp"

#include "gtest/gtest.h"
#include "nlohmann/json.hpp"
#include "nlohmann/json_fwd.hpp"

#include "llvm/Support/LogicalResult.h"

#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace mqt::bench {

using test::expectInvalidJSON;

TEST(BenchmarkJSON, ReturnsNormalizedInstancesAndInputDiagnostics) {
  const auto benchmark = ::mqt::test::value(GHZ::create({.qubits = 3}));
  auto result = mqt::bench::parseInstanceSpecificationJSON(
      toInstanceSpecificationJSON(benchmark));
  ASSERT_TRUE(llvm::succeeded(result));
  const auto& parsed = *result;
  EXPECT_TRUE(std::holds_alternative<GHZ>(parsed.instance));
  EXPECT_EQ(parsed.benchmarkId, "ghz");
  EXPECT_EQ(parsed.caseId, caseId(benchmark));
  EXPECT_EQ(nlohmann::json::parse(parsed.manifestJSON)["case_id"],
            parsed.caseId);
  EXPECT_EQ(parsed.manifestJSON, toManifestJSON(benchmark));

  for (
      const auto* invalid : {
          "{",
          R"({"schema_version":1,"benchmark":"ghz","parameters":{"qubits":0}})",
      }) {
    expectInvalidJSON(
        [&] {
          return mqt::bench::parseInstanceSpecificationJSON(invalid,
                                                            "input.json");
        },
        "input.json");
  }
  expectInvalidJSON([] { return mqt::bench::describeBenchmarkJSON("unknown"); },
                    "unknown");
  expectInvalidJSON(
      [&] {
        return mqt::bench::evaluateJSON(toManifestJSON(benchmark), "{}",
                                        "manifest.json", "counts.json");
      },
      "counts.json");
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
        return ghzFromInstanceSpecificationJSON(
            R"({"schema_version":1,"benchmark":"ghz","parameters":{"qubits":2,"qubits":3}})");
      },
      "duplicate key 'qubits'");
  expectInvalidJSON(
      [] {
        return qpeFromInstanceSpecificationJSON(
            R"({"schema_version":1,"benchmark":"qpe","parameters":{"precision":2,"phase":{"numerator":1,"denominator":4,"numerator":2}}})");
      },
      "duplicate key 'numerator'");
}

TEST(BenchmarkJSON, RejectsInvalidInstanceEnvelopes) {
  expectInvalidJSON(
      [] {
        return ghzFromInstanceSpecificationJSON(
            R"({"schema_version":1,"benchmark":"ghz","parameters":{"qubits":2},"extra":true})");
      },
      "unknown key 'extra'");
  expectInvalidJSON(
      [] {
        return ghzFromInstanceSpecificationJSON(
            R"({"schema_version":1,"benchmark":"new","parameters":{}})");
      },
      "unsupported benchmark 'new'");
}

TEST(BenchmarkJSON, RejectsAnInstanceSpecificationForAnotherConcreteType) {
  expectInvalidJSON(
      [] {
        return ghzFromInstanceSpecificationJSON(
            R"({"schema_version":1,"benchmark":"qpe","parameters":{"precision":2,"phase":{"numerator":1,"denominator":4}}})");
      },
      "must be 'ghz'");
}

TEST(BenchmarkJSON, PreservesFieldDiagnosticsAcrossBenchmarkFamilies) {
  using Json = nlohmann::json;
  for (const auto& specification : std::vector<std::string>{
           toInstanceSpecificationJSON(
               ::mqt::test::value(BV::create({.hiddenBitstring = "01"}))),
           toInstanceSpecificationJSON(
               ::mqt::test::value(GHZ::create({.qubits = 2}))),
           toInstanceSpecificationJSON(
               ::mqt::test::value(Grover::create({.markedBitstring = "01"}))),
           toInstanceSpecificationJSON(
               ::mqt::test::value(ModularMultiplier::create({
                   .multiplier = "011",
                   .modulus = "101",
                   .multiplicand = "+++",
               }))),
           toInstanceSpecificationJSON(
               ::mqt::test::value(Multiplexer::create({.qubits = 2}))),
           toInstanceSpecificationJSON(::mqt::test::value(
               QFT::create({.qubits = 2, .periodExponent = 1}))),
           toInstanceSpecificationJSON(::mqt::test::value(
               QFTAdder::create({.addend = "01", .accumulator = "00"}))),
           toInstanceSpecificationJSON(::mqt::test::value(QPE::create({
               .precision = 2,
               .phase = ::mqt::test::value(Phase::create(1, 4)),
           }))),
           toInstanceSpecificationJSON(
               ::mqt::test::value(RepeatUntilSuccess::create())),
           toInstanceSpecificationJSON(Teleportation{}),
       }) {
    SCOPED_TRACE(specification);
    const auto root = Json::parse(specification);
    const auto schema = Json::parse(::mqt::test::value(
        describeBenchmarkJSON(root["benchmark"].get<std::string>())));
    const auto check = [](const Json& invalid, const std::string& pointer) {
      expectInvalidJSON(
          [&] {
            return mqt::bench::parseInstanceSpecificationJSON(invalid.dump(),
                                                              "instance.json");
          },
          "instance.json:$" + pointer);
    };
    for (const auto& [key, value] : root["parameters"].items()) {
      auto invalid = root;
      invalid["parameters"][key] = nullptr;
      check(invalid, "/parameters/" + key);
      if (value.is_string() &&
          (key == "method" || key == "basis" || key == "topology" ||
           key == "overflow" || key == "control")) {
        invalid["parameters"][key] = "unknown";
        check(invalid, "/parameters/" + key);
      }
    }
    auto invalid = root;
    invalid["parameters"]["unknown"] = true;
    check(invalid, "/parameters");
    for (const auto& name : schema["properties"]["parameters"].value(
             "required", std::vector<std::string>{})) {
      invalid = root;
      invalid["parameters"].erase(name);
      check(invalid, "/parameters/" + name);
    }
  }
}

TEST(BenchmarkJSON, RejectsMalformedEnvelopesAndPhaseFields) {
  using Json = nlohmann::json;
  const auto ghz = ::mqt::test::value(GHZ::create({.qubits = 2}));
  const auto manifest = Json::parse(toManifestJSON(ghz));
  for (const auto& item : manifest.items()) {
    const auto& key = item.key();
    SCOPED_TRACE(key);
    auto invalid = manifest;
    invalid.erase(key);
    expectInvalidJSON(
        [&] {
          return benchmarkIdFromManifestJSON(invalid.dump(), "manifest.json");
        },
        "manifest.json:$/" + key);
    invalid[key] = nullptr;
    expectInvalidJSON(
        [&] {
          return benchmarkIdFromManifestJSON(invalid.dump(), "manifest.json");
        },
        "manifest.json:$/" + key);
  }
  for (const auto* key : {"schema_version", "definition_version"}) {
    auto invalid = manifest;
    invalid[key] = 2;
    expectInvalidJSON(
        [&] { return benchmarkIdFromManifestJSON(invalid.dump()); }, key);
  }
  for (const auto* input : {
           "[]",
           R"({"values":[null,true,-1,1,1.5,"text",{}]} trailing)",
           "{\"num\":1e400}",
       }) {
    expectInvalidJSON(
        [&] {
          return benchmarkIdFromInstanceSpecificationJSON(input, "syntax.json");
        },
        "syntax.json:");
  }
  const auto qpe =
      Json::parse(toInstanceSpecificationJSON(::mqt::test::value(QPE::create({
          .precision = 2,
          .phase = ::mqt::test::value(Phase::create(1, 4)),
      }))));
  for (const auto* key : {"numerator", "denominator"}) {
    auto invalid = qpe;
    invalid["parameters"]["phase"].erase(key);
    expectInvalidJSON(
        [&] { return qpeFromInstanceSpecificationJSON(invalid.dump()); },
        std::string("$/parameters/phase/") + key);
    invalid["parameters"]["phase"][key] = nullptr;
    expectInvalidJSON(
        [&] { return qpeFromInstanceSpecificationJSON(invalid.dump()); },
        std::string("$/parameters/phase/") + key);
  }
  auto invalid = qpe;
  invalid["parameters"]["phase"]["unknown"] = 1;
  expectInvalidJSON(
      [&] { return qpeFromInstanceSpecificationJSON(invalid.dump()); },
      "unknown key");
}

TEST(BenchmarkJSON, RejectsAlteredOrUnresolvedManifestData) {
  const auto ghz = ::mqt::test::value(GHZ::create({.qubits = 3}));
  const auto manifest = toManifestJSON(ghz);
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

  auto changedNumericKind = manifest;
  const auto integerWidth = changedNumericKind.find(R"("width":3)");
  ASSERT_NE(integerWidth, std::string::npos);
  changedNumericKind.replace(integerWidth, std::string(R"("width":3)").size(),
                             R"("width":3.0)");
  reject(changedNumericKind, "does not match");

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

  auto invalidParameters = nlohmann::json::parse(manifest);
  invalidParameters["parameters"]["qubits"] = 0;
  reject(invalidParameters.dump(), "manifest.json:$/parameters GHZ qubits");
  expectInvalidJSON(
      [&] {
        return evaluateJSON(invalidParameters.dump(), "{}", "manifest.json",
                            "counts.json");
      },
      "counts.json:$/schema_version");
}

TEST(BenchmarkJSON, ListsBenchmarksAndRejectsUnknownSchemas) {
  EXPECT_EQ(
      listBenchmarksJSON(),
      R"({"benchmarks":[{"definition_version":1,"id":"bv"},{"definition_version":1,"id":"ghz"},{"definition_version":1,"id":"grover"},{"definition_version":1,"id":"grover-weak-measurement"},{"definition_version":1,"id":"magic-state-distillation"},{"definition_version":1,"id":"modular-multiplier"},{"definition_version":1,"id":"multiplexer"},{"definition_version":1,"id":"qft"},{"definition_version":1,"id":"qft-adder"},{"definition_version":1,"id":"qpe"},{"definition_version":1,"id":"repeat-until-success"},{"definition_version":1,"id":"shor"},{"definition_version":1,"id":"teleportation"},{"definition_version":1,"id":"w-state"}],"schema_version":1})");
  const auto ghz = ::mqt::test::value(describeBenchmarkJSON("ghz"));
  EXPECT_NE(ghz.find("https://json-schema.org/draft/2020-12/schema"),
            std::string::npos);
  EXPECT_NE(ghz.find("\"additionalProperties\":false"), std::string::npos);
  EXPECT_EQ(
      ::mqt::test::errorKind([&] { return describeBenchmarkJSON("unknown"); }),
      ::mqt::ErrorCategory::InvalidArgument);
}

TEST(BenchmarkJSON, RejectsMalformedCountsAndPropagatesEvaluationErrors) {
  for (const auto* input : {
           "{",
           "[]",
           R"({"schema_version":2,"counts":{"0":1}})",
           R"({"schema_version":1,"counts":{"0":1},"extra":1})",
           R"({"schema_version":1})",
           R"({"schema_version":1,"counts":[]})",
           R"({"schema_version":1,"counts":{}})",
           R"({"schema_version":1,"counts":{"0":null}})",
           R"({"schema_version":1,"counts":{"0":18446744073709551615,"1":1}})",
       }) {
    SCOPED_TRACE(input);
    expectInvalidJSON([&] { return countsFromJSON(input, "counts.json"); },
                      "counts.json:");
  }
  const auto ghz = ::mqt::test::value(GHZ::create({.qubits = 2}));
  const auto manifest = toManifestJSON(ghz);
  expectInvalidJSON(
      [&] { return evaluateJSON("{}", "{}", "manifest.json", "counts.json"); },
      "manifest.json:");
  expectInvalidJSON(
      [&] {
        return evaluateJSON(manifest, "{}", "manifest.json", "counts.json");
      },
      "counts.json:");
  expectInvalidJSON(
      [&] { return evaluationToJSON(caseId(ghz), 0, Evaluation{}); },
      "at least one shot");
}

TEST(BenchmarkJSON, ParsesCountsAndSerializesEvaluations) {
  const auto counts = ::mqt::test::value(
      countsFromJSON(R"({"counts":{"11":50,"00":50},"schema_version":1})"));
  EXPECT_EQ(counts.at("00"), 50);
  EXPECT_EQ(counts.at("11"), 50);

  const auto ghz = ::mqt::test::value(GHZ::create({.qubits = 2}));
  const auto serialized = ::mqt::test::value(evaluationToJSON(
      caseId(ghz), 100, ::mqt::test::value(ghz.evaluate(counts))));
  EXPECT_NE(serialized.find("\"squared_hellinger_fidelity\":1.0"),
            std::string::npos);
  EXPECT_NE(serialized.find("\"success_probability\":null"), std::string::npos);
  EXPECT_NE(serialized.find("\"total_variation_distance\":0.0"),
            std::string::npos);

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
      "must be positive");
  EXPECT_EQ(::mqt::test::errorKind([&] {
              return evaluationToJSON(
                  "not-a-case", 1,
                  Evaluation{.totalVariationDistance = 0.,
                             .squaredHellingerFidelity = 1.,
                             .successProbability = std::nullopt});
            }),
            ::mqt::ErrorCategory::InvalidArgument);
  EXPECT_EQ(::mqt::test::errorKind([&] {
              return evaluationToJSON(
                  caseId(ghz), 1,
                  Evaluation{.totalVariationDistance =
                                 std::numeric_limits<double>::quiet_NaN(),
                             .squaredHellingerFidelity = 1.,
                             .successProbability = std::nullopt});
            }),
            ::mqt::ErrorCategory::InvalidArgument);
}

} // namespace mqt::bench
