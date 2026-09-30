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
#include "bench/RepeatUntilSuccess.hpp"

#include "JSONTestUtils.hpp"

#include "gtest/gtest.h"

#include <numbers>
#include <stdexcept>
#include <string>

namespace mqt::bench {

using test::expectInvalidJSON;

TEST(RepeatUntilSuccess, HasTheOutput) {
  const RepeatUntilSuccess benchmark;
  EXPECT_EQ(benchmark.output(), (Output{"result", 1}));
}

TEST(RepeatUntilSuccess, ValidatesDataWidth) {
  EXPECT_EQ(RepeatUntilSuccess{}.options().dataQubits, 1U);
  EXPECT_EQ(RepeatUntilSuccess({.dataQubits = 5}).options().dataQubits, 5U);
  EXPECT_NO_THROW(RepeatUntilSuccess(
      {.dataQubits = RepeatUntilSuccessOptions::MAX_DATA_QUBITS}));
  EXPECT_THROW(RepeatUntilSuccess({.dataQubits = 0}), std::invalid_argument);
  EXPECT_THROW(
      RepeatUntilSuccess(
          {.dataQubits = RepeatUntilSuccessOptions::MAX_DATA_QUBITS + 1}),
      std::invalid_argument);
}

TEST(RepeatUntilSuccess, HasThePhaseSensitiveReference) {
  const RepeatUntilSuccess benchmark;
  constexpr auto bias = std::numbers::sqrt2 / 3.;
  EXPECT_DOUBLE_EQ(benchmark.probability("0"), 0.5 + bias);
  EXPECT_DOUBLE_EQ(benchmark.probability("1"), 0.5 - bias);
  EXPECT_THROW(static_cast<void>(benchmark.probability("")),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(benchmark.probability("00")),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(benchmark.probability("x")),
               std::invalid_argument);
}

TEST(RepeatUntilSuccess, EvaluatesTheReferenceWithoutASuccessOutcome) {
  const RepeatUntilSuccess benchmark;
  constexpr auto bias = std::numbers::sqrt2 / 3.;
  const auto allZero = benchmark.evaluate({{"0", 10}});
  EXPECT_NEAR(allZero.totalVariationDistance, 0.5 - bias, 1e-15);
  EXPECT_NEAR(allZero.squaredHellingerFidelity, 0.5 + bias, 1e-15);
  EXPECT_FALSE(allZero.successProbability);
}

TEST(RepeatUntilSuccess, RoundTripsJSON) {
  const auto defaults = repeatUntilSuccessFromInstanceSpecificationJSON(
      R"({"schema_version":1,"benchmark":"repeat-until-success","parameters":{}})");
  EXPECT_EQ(
      toInstanceSpecificationJSON(defaults),
      R"({"benchmark":"repeat-until-success","parameters":{"data_qubits":1},"schema_version":1})");

  const RepeatUntilSuccess benchmark;
  const auto manifest = toManifestJSON(benchmark);
  EXPECT_EQ(toManifestJSON(repeatUntilSuccessFromManifestJSON(manifest)),
            manifest);
  EXPECT_EQ(benchmarkIdFromManifestJSON(manifest), "repeat-until-success");
  EXPECT_NE(manifest.find("\"model\":\"repeat_until_success\""),
            std::string::npos);
  EXPECT_NE(manifest.find("\"parameters\":{\"data_qubits\":1}"),
            std::string::npos);
  EXPECT_EQ(repeatUntilSuccessFromManifestJSON(
                toManifestJSON(RepeatUntilSuccess({.dataQubits = 32})))
                .options()
                .dataQubits,
            32U);
}

TEST(RepeatUntilSuccess, UsesSemanticCaseIds) {
  EXPECT_EQ(caseId(RepeatUntilSuccess{}),
            caseId(RepeatUntilSuccess{{.dataQubits = 1}}));
  EXPECT_NE(caseId(RepeatUntilSuccess{}),
            caseId(RepeatUntilSuccess{{.dataQubits = 5}}));
}

TEST(RepeatUntilSuccess, DescribesJSONSchema) {
  EXPECT_NE(
      describeBenchmarkJSON("repeat-until-success")
          .find(
              R"("data_qubits":{"default":1,"maximum":1000000,"minimum":1,"type":"integer"})"),
      std::string::npos);
}

TEST(RepeatUntilSuccess, RejectsInvalidJSONParameters) {
  for (const auto* width : {"0", "1000001", "-1", "1.5", "true", "\"5\""}) {
    EXPECT_THROW(
        static_cast<void>(repeatUntilSuccessFromInstanceSpecificationJSON(
            std::string(
                R"({"schema_version":1,"benchmark":"repeat-until-success","parameters":{"data_qubits":)") +
            width + "}}")),
        std::invalid_argument);
  }
  expectInvalidJSON(
      [] {
        static_cast<void>(repeatUntilSuccessFromInstanceSpecificationJSON(
            R"({"schema_version":1,"benchmark":"repeat-until-success","parameters":{"attempts":1}})"));
      },
      "unknown key 'attempts'");
}

TEST(RepeatUntilSuccess, EvaluatesCountsFromJSON) {
  const auto evaluation =
      evaluateJSON(toManifestJSON(RepeatUntilSuccess{}),
                   R"({"schema_version":1,"counts":{"0":993,"1":7}})");
  EXPECT_NE(evaluation.find("\"success_probability\":null"), std::string::npos);
  EXPECT_NE(evaluation.find("\"total_variation_distance\":"),
            std::string::npos);
}

} // namespace mqt::bench
