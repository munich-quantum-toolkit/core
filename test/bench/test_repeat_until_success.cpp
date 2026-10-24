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

#include "llvm/Support/Error.h"

#include <numbers>
#include <string>
#include <system_error>

namespace mqt::bench {

using test::expectInvalidJSON;

TEST(RepeatUntilSuccess, HasTheOutput) {
  const auto benchmark = llvm::cantFail(RepeatUntilSuccess::create());
  EXPECT_EQ(benchmark.output(), (Output{"result", 1}));
}

TEST(RepeatUntilSuccess, ValidatesDataWidth) {
  EXPECT_EQ(llvm::cantFail(RepeatUntilSuccess::create()).options().dataQubits,
            1U);
  EXPECT_EQ(llvm::cantFail(RepeatUntilSuccess::create({.dataQubits = 5}))
                .options()
                .dataQubits,
            5U);
  static_cast<void>(llvm::cantFail(RepeatUntilSuccess::create(
      {.dataQubits = RepeatUntilSuccessOptions::MAX_DATA_QUBITS})));
  test::expectError(RepeatUntilSuccess::create({.dataQubits = 0}),
                    std::errc::invalid_argument);
  test::expectError(
      RepeatUntilSuccess::create(
          {.dataQubits = RepeatUntilSuccessOptions::MAX_DATA_QUBITS + 1}),
      std::errc::invalid_argument);
}

TEST(RepeatUntilSuccess, HasThePhaseSensitiveReference) {
  const auto benchmark = llvm::cantFail(RepeatUntilSuccess::create());
  constexpr auto bias = std::numbers::sqrt2 / 3.;
  EXPECT_DOUBLE_EQ(llvm::cantFail(benchmark.probability("0")), 0.5 + bias);
  EXPECT_DOUBLE_EQ(llvm::cantFail(benchmark.probability("1")), 0.5 - bias);
  test::expectError(benchmark.probability(""), std::errc::invalid_argument);
  test::expectError(benchmark.probability("00"), std::errc::invalid_argument);
  test::expectError(benchmark.probability("x"), std::errc::invalid_argument);
}

TEST(RepeatUntilSuccess, EvaluatesTheReferenceWithoutASuccessOutcome) {
  const auto benchmark = llvm::cantFail(RepeatUntilSuccess::create());
  constexpr auto bias = std::numbers::sqrt2 / 3.;
  const auto allZero = llvm::cantFail(benchmark.evaluate({{"0", 10}}));
  EXPECT_NEAR(allZero.totalVariationDistance, 0.5 - bias, 1e-15);
  EXPECT_NEAR(allZero.squaredHellingerFidelity, 0.5 + bias, 1e-15);
  EXPECT_FALSE(allZero.successProbability);
}

TEST(RepeatUntilSuccess, RoundTripsJSON) {
  const auto defaults = llvm::cantFail(repeatUntilSuccessFromInstanceSpecificationJSON(
      R"({"schema_version":1,"benchmark":"repeat-until-success","parameters":{}})"));
  EXPECT_EQ(
      toInstanceSpecificationJSON(defaults),
      R"({"benchmark":"repeat-until-success","parameters":{"data_qubits":1},"schema_version":1})");

  const auto benchmark = llvm::cantFail(RepeatUntilSuccess::create());
  const auto manifest = toManifestJSON(benchmark);
  EXPECT_EQ(toManifestJSON(
                llvm::cantFail(repeatUntilSuccessFromManifestJSON(manifest))),
            manifest);
  EXPECT_EQ(llvm::cantFail(benchmarkIdFromManifestJSON(manifest)),
            "repeat-until-success");
  EXPECT_NE(manifest.find("\"model\":\"repeat_until_success\""),
            std::string::npos);
  EXPECT_NE(manifest.find("\"parameters\":{\"data_qubits\":1}"),
            std::string::npos);
  EXPECT_EQ(
      llvm::cantFail(
          repeatUntilSuccessFromManifestJSON(toManifestJSON(
              llvm::cantFail(RepeatUntilSuccess::create({.dataQubits = 32})))))
          .options()
          .dataQubits,
      32U);
}

TEST(RepeatUntilSuccess, UsesSemanticCaseIds) {
  EXPECT_EQ(
      caseId(llvm::cantFail(RepeatUntilSuccess::create())),
      caseId(llvm::cantFail(RepeatUntilSuccess::create({.dataQubits = 1}))));
  EXPECT_NE(
      caseId(llvm::cantFail(RepeatUntilSuccess::create())),
      caseId(llvm::cantFail(RepeatUntilSuccess::create({.dataQubits = 5}))));
}

TEST(RepeatUntilSuccess, DescribesJSONSchema) {
  EXPECT_NE(
      llvm::cantFail(describeBenchmarkJSON("repeat-until-success"))
          .find(
              R"("data_qubits":{"default":1,"maximum":1000000,"minimum":1,"type":"integer"})"),
      std::string::npos);
}

TEST(RepeatUntilSuccess, RejectsInvalidJSONParameters) {
  for (const auto* width : {"0", "1000001", "-1", "1.5", "true", "\"5\""}) {
    test::expectInvalidJSON(
        [&] {
          return repeatUntilSuccessFromInstanceSpecificationJSON(
              std::string(
                  R"({"schema_version":1,"benchmark":"repeat-until-success","parameters":{"data_qubits":)") +
              width + "}}");
        },
        "");
  }
  expectInvalidJSON(
      [] {
        return repeatUntilSuccessFromInstanceSpecificationJSON(
            R"({"schema_version":1,"benchmark":"repeat-until-success","parameters":{"attempts":1}})");
      },
      "attempts");
}

TEST(RepeatUntilSuccess, EvaluatesCountsFromJSON) {
  const auto evaluation = llvm::cantFail(
      evaluateJSON(toManifestJSON(llvm::cantFail(RepeatUntilSuccess::create())),
                   R"({"schema_version":1,"counts":{"0":993,"1":7}})"));
  EXPECT_NE(evaluation.find("\"success_probability\":null"), std::string::npos);
  EXPECT_NE(evaluation.find("\"total_variation_distance\":"),
            std::string::npos);
}

} // namespace mqt::bench
