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
#include "support/Diagnostics.hpp"
#include "support/TestSupport.hpp"

#include "gtest/gtest.h"

#include <numbers>
#include <string>

namespace mqt::bench {

using test::expectInvalidJSON;

TEST(RepeatUntilSuccess, HasTheOutput) {
  const auto benchmark = ::mqt::test::value(RepeatUntilSuccess::create());
  EXPECT_EQ(benchmark.output(), (Output{"result", 1}));
}

TEST(RepeatUntilSuccess, ValidatesDataWidth) {
  EXPECT_EQ(
      ::mqt::test::value(RepeatUntilSuccess::create()).options().dataQubits,
      1U);
  EXPECT_EQ(::mqt::test::value(RepeatUntilSuccess::create({.dataQubits = 5}))
                .options()
                .dataQubits,
            5U);
  EXPECT_NO_THROW(::mqt::test::value(RepeatUntilSuccess::create(
      {.dataQubits = RepeatUntilSuccessOptions::MAX_DATA_QUBITS})));
  EXPECT_EQ(::mqt::test::errorKind(
                [&] { return RepeatUntilSuccess::create({.dataQubits = 0}); }),
            ::mqt::ErrorCategory::InvalidArgument);
  EXPECT_EQ(
      ::mqt::test::errorKind([&] {
        return RepeatUntilSuccess::create(
            {.dataQubits = RepeatUntilSuccessOptions::MAX_DATA_QUBITS + 1});
      }),
      ::mqt::ErrorCategory::InvalidArgument);
}

TEST(RepeatUntilSuccess, HasThePhaseSensitiveReference) {
  const auto benchmark = ::mqt::test::value(RepeatUntilSuccess::create());
  constexpr auto bias = std::numbers::sqrt2 / 3.;
  EXPECT_DOUBLE_EQ(::mqt::test::value(benchmark.probability("0")), 0.5 + bias);
  EXPECT_DOUBLE_EQ(::mqt::test::value(benchmark.probability("1")), 0.5 - bias);
  EXPECT_EQ(::mqt::test::errorKind([&] { return benchmark.probability(""); }),
            ::mqt::ErrorCategory::InvalidArgument);
  EXPECT_EQ(::mqt::test::errorKind([&] { return benchmark.probability("00"); }),
            ::mqt::ErrorCategory::InvalidArgument);
  EXPECT_EQ(::mqt::test::errorKind([&] { return benchmark.probability("x"); }),
            ::mqt::ErrorCategory::InvalidArgument);
}

TEST(RepeatUntilSuccess, EvaluatesTheReferenceWithoutASuccessOutcome) {
  const auto benchmark = ::mqt::test::value(RepeatUntilSuccess::create());
  constexpr auto bias = std::numbers::sqrt2 / 3.;
  const auto allZero = ::mqt::test::value(benchmark.evaluate({{"0", 10}}));
  EXPECT_NEAR(allZero.totalVariationDistance, 0.5 - bias, 1e-15);
  EXPECT_NEAR(allZero.squaredHellingerFidelity, 0.5 + bias, 1e-15);
  EXPECT_FALSE(allZero.successProbability);
}

TEST(RepeatUntilSuccess, RoundTripsJSON) {
  const auto defaults =
      ::mqt::test::value(repeatUntilSuccessFromInstanceSpecificationJSON(
          R"({"schema_version":1,"benchmark":"repeat-until-success","parameters":{}})"));
  EXPECT_EQ(
      toInstanceSpecificationJSON(defaults),
      R"({"benchmark":"repeat-until-success","parameters":{"data_qubits":1},"schema_version":1})");

  const auto benchmark = ::mqt::test::value(RepeatUntilSuccess::create());
  const auto manifest = toManifestJSON(benchmark);
  EXPECT_EQ(toManifestJSON(::mqt::test::value(
                repeatUntilSuccessFromManifestJSON(manifest))),
            manifest);
  EXPECT_EQ(::mqt::test::value(benchmarkIdFromManifestJSON(manifest)),
            "repeat-until-success");
  EXPECT_NE(manifest.find("\"model\":\"repeat_until_success\""),
            std::string::npos);
  EXPECT_NE(manifest.find("\"parameters\":{\"data_qubits\":1}"),
            std::string::npos);
  EXPECT_EQ(
      ::mqt::test::value(
          repeatUntilSuccessFromManifestJSON(toManifestJSON(::mqt::test::value(
              RepeatUntilSuccess::create({.dataQubits = 32})))))
          .options()
          .dataQubits,
      32U);
}

TEST(RepeatUntilSuccess, UsesSemanticCaseIds) {
  EXPECT_EQ(caseId(::mqt::test::value(RepeatUntilSuccess::create())),
            caseId(::mqt::test::value(
                RepeatUntilSuccess::create({.dataQubits = 1}))));
  EXPECT_NE(caseId(::mqt::test::value(RepeatUntilSuccess::create())),
            caseId(::mqt::test::value(
                RepeatUntilSuccess::create({.dataQubits = 5}))));
}

TEST(RepeatUntilSuccess, DescribesJSONSchema) {
  EXPECT_NE(
      ::mqt::test::value(describeBenchmarkJSON("repeat-until-success"))
          .find(
              R"("data_qubits":{"default":1,"maximum":1000000,"minimum":1,"type":"integer"})"),
      std::string::npos);
}

TEST(RepeatUntilSuccess, RejectsInvalidJSONParameters) {
  for (const auto* width : {"0", "1000001", "-1", "1.5", "true", "\"5\""}) {
    EXPECT_EQ(
        ::mqt::test::errorKind([&] {
          return repeatUntilSuccessFromInstanceSpecificationJSON(
              std::string(
                  R"({"schema_version":1,"benchmark":"repeat-until-success","parameters":{"data_qubits":)") +
              width + "}}");
        }),
        ::mqt::ErrorCategory::InvalidArgument);
  }
  expectInvalidJSON(
      [] {
        static_cast<void>(repeatUntilSuccessFromInstanceSpecificationJSON(
            R"({"schema_version":1,"benchmark":"repeat-until-success","parameters":{"attempts":1}})"));
      },
      "unknown key 'attempts'");
}

TEST(RepeatUntilSuccess, EvaluatesCountsFromJSON) {
  const auto evaluation = ::mqt::test::value(evaluateJSON(
      toManifestJSON(::mqt::test::value(RepeatUntilSuccess::create())),
      R"({"schema_version":1,"counts":{"0":993,"1":7}})"));
  EXPECT_NE(evaluation.find("\"success_probability\":null"), std::string::npos);
  EXPECT_NE(evaluation.find("\"total_variation_distance\":"),
            std::string::npos);
}

} // namespace mqt::bench
