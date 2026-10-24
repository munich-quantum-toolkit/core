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
#include "bench/FanOut.hpp"
#include "bench/JSON.hpp"

#include "JSONTestUtils.hpp"

#include "gtest/gtest.h"

#include "llvm/Support/Error.h"

#include <limits>
#include <string>
#include <system_error>

namespace mqt::bench {

using test::expectInvalidJSON;

TEST(FanOut, StoresTheControlCountAndOutput) {
  const auto benchmark = llvm::cantFail(FanOut::create({.controls = 7}));
  EXPECT_EQ(benchmark.options().controls, 7);
  EXPECT_EQ(benchmark.output(), (Output{"result", 8}));
}

TEST(FanOut, ValidatesTheConfiguredInstance) {
  test::expectError(FanOut::create({.controls = 0}),
                    std::errc::invalid_argument);
  static_cast<void>(llvm::cantFail(FanOut::create({.controls = 1})));
  static_cast<void>(llvm::cantFail(
      FanOut::create({.controls = FanOutOptions::MAX_CONTROLS})));
  test::expectError(
      FanOut::create({.controls = FanOutOptions::MAX_CONTROLS + 1}),
      std::errc::invalid_argument);
}

TEST(FanOut, GivesTheHammingWeightRotationDistribution) {
  const auto benchmark = llvm::cantFail(FanOut::create({.controls = 4}));
  EXPECT_DOUBLE_EQ(llvm::cantFail(benchmark.probability("00000")), 1. / 16.);
  EXPECT_DOUBLE_EQ(llvm::cantFail(benchmark.probability("00001")), 0.);
  EXPECT_DOUBLE_EQ(llvm::cantFail(benchmark.probability("00010")), 1. / 32.);
  EXPECT_DOUBLE_EQ(llvm::cantFail(benchmark.probability("00011")), 1. / 32.);
  EXPECT_DOUBLE_EQ(llvm::cantFail(benchmark.probability("00110")), 0.);
  EXPECT_DOUBLE_EQ(llvm::cantFail(benchmark.probability("00111")), 1. / 16.);
  EXPECT_DOUBLE_EQ(llvm::cantFail(benchmark.probability("01110")), 1. / 32.);
  EXPECT_DOUBLE_EQ(llvm::cantFail(benchmark.probability("01111")), 1. / 32.);
  EXPECT_DOUBLE_EQ(llvm::cantFail(benchmark.probability("11110")), 1. / 16.);
  EXPECT_DOUBLE_EQ(llvm::cantFail(benchmark.probability("11111")), 0.);
  test::expectError(benchmark.probability("0000"), std::errc::invalid_argument);
  test::expectError(benchmark.probability("0000x"),
                    std::errc::invalid_argument);
}

TEST(FanOut, EvaluatesTheReferenceWithoutASuccessOutcome) {
  const auto benchmark = llvm::cantFail(FanOut::create({.controls = 2}));
  const auto evaluation = llvm::cantFail(benchmark.evaluate({
      {"000", 2},
      {"010", 1},
      {"011", 1},
      {"100", 1},
      {"101", 1},
      {"111", 2},
  }));
  EXPECT_DOUBLE_EQ(evaluation.totalVariationDistance, 0.);
  EXPECT_DOUBLE_EQ(evaluation.squaredHellingerFidelity, 1.);
  EXPECT_FALSE(evaluation.successProbability);
}

TEST(FanOut, KeepsTheSmallestNonzeroProbabilityRepresentable) {
  const auto benchmark =
      llvm::cantFail(FanOut::create({.controls = FanOutOptions::MAX_CONTROLS}));
  auto outcome = std::string(benchmark.output().width, '0');
  EXPECT_EQ(llvm::cantFail(benchmark.probability(outcome)),
            2. * std::numeric_limits<double>::denorm_min());
  outcome.front() = '1';
  EXPECT_EQ(llvm::cantFail(benchmark.probability(outcome)),
            std::numeric_limits<double>::denorm_min());
  outcome.back() = '1';
  EXPECT_EQ(llvm::cantFail(benchmark.probability(outcome)),
            std::numeric_limits<double>::denorm_min());
}

TEST(FanOut, RoundTripsJSON) {
  const auto parsed = llvm::cantFail(fanOutFromInstanceSpecificationJSON(
      R"({"schema_version":1,"benchmark":"fan-out","parameters":{"controls":7}})"));
  EXPECT_EQ(parsed.options().controls, 7);
  EXPECT_EQ(
      toInstanceSpecificationJSON(parsed),
      R"({"benchmark":"fan-out","parameters":{"controls":7},"schema_version":1})");

  const auto manifest = toManifestJSON(parsed);
  EXPECT_EQ(toManifestJSON(llvm::cantFail(fanOutFromManifestJSON(manifest))),
            manifest);
  EXPECT_EQ(llvm::cantFail(benchmarkIdFromManifestJSON(manifest)), "fan-out");
  EXPECT_NE(manifest.find(R"("model":"fan_out")"), std::string::npos);
  EXPECT_EQ(caseId(parsed),
            caseId(llvm::cantFail(FanOut::create({.controls = 7}))));
  EXPECT_NE(caseId(parsed),
            caseId(llvm::cantFail(FanOut::create({.controls = 6}))));
}

TEST(FanOut, DescribesJSONSchema) {
  const auto schema = llvm::cantFail(describeBenchmarkJSON("fan-out"));
  EXPECT_NE(schema.find(R"("maximum":1073)"), std::string::npos);
  EXPECT_NE(schema.find(R"("minimum":1)"), std::string::npos);
  EXPECT_NE(schema.find(R"("required":["controls"])"), std::string::npos);
}

TEST(FanOut, RejectsInvalidJSONParameters) {
  expectInvalidJSON(
      [] {
        return fanOutFromInstanceSpecificationJSON(
            R"({"schema_version":1,"benchmark":"fan-out","parameters":{"controls":0}})");
      },
      "between 1 and 1073");
  expectInvalidJSON(
      [] {
        return fanOutFromInstanceSpecificationJSON(
            R"({"schema_version":1,"benchmark":"fan-out","parameters":{"controls":7,"angle":1}})");
      },
      "angle");
}

TEST(FanOut, ReportsTheParameterLocationOnce) {
  expectInvalidJSON(
      [] {
        return fanOutFromInstanceSpecificationJSON(
            R"({"schema_version":1,"benchmark":"fan-out","parameters":{}})",
            "case.json");
      },
      "case.json:$/parameters.controls");
}

TEST(FanOut, EvaluatesCountsFromJSON) {
  const auto evaluation = llvm::cantFail(evaluateJSON(
      toManifestJSON(llvm::cantFail(FanOut::create({.controls = 1}))),
      R"({"schema_version":1,"counts":{"00":2,"10":1,"11":1}})"));
  EXPECT_NE(evaluation.find(R"("squared_hellinger_fidelity":1)"),
            std::string::npos);
  EXPECT_NE(evaluation.find(R"("success_probability":null)"),
            std::string::npos);
}

} // namespace mqt::bench
