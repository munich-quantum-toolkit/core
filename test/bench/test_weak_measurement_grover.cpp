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
#include "bench/WeakMeasurementGrover.hpp"

#include "gtest/gtest.h"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace mqt::bench {

TEST(WeakMeasurementGrover, ResolvesTheDefaultMeasurementStrength) {
  const WeakMeasurementGrover benchmark{{.markedBitstring = "101"}};
  ASSERT_TRUE(benchmark.options().measurementStrength);
  EXPECT_DOUBLE_EQ(*benchmark.options().measurementStrength,
                   std::exp2(-3. / 2.));
  EXPECT_EQ(benchmark.qubits(), 3);
  EXPECT_EQ(benchmark.output(), (Output{"result", 3}));
  EXPECT_DOUBLE_EQ(benchmark.probability("101"), 1.);
  EXPECT_DOUBLE_EQ(benchmark.probability("001"), 0.);
}

TEST(WeakMeasurementGrover, AcceptsStrengthsInTheProvenRegime) {
  const WeakMeasurementGrover boundary{
      {.markedBitstring = "10", .measurementStrength = 0.5}};
  EXPECT_DOUBLE_EQ(*boundary.options().measurementStrength, 0.5);

  const WeakMeasurementGrover weaker{
      {.markedBitstring = "0000", .measurementStrength = 0.125}};
  EXPECT_DOUBLE_EQ(*weaker.options().measurementStrength, 0.125);

  const WeakMeasurementGrover largest{{
      .markedBitstring =
          std::string(WeakMeasurementGroverOptions::MAX_QUBITS, '0'),
  }};
  ASSERT_TRUE(largest.options().measurementStrength);
  EXPECT_DOUBLE_EQ(*largest.options().measurementStrength,
                   std::numeric_limits<double>::min());
}

TEST(WeakMeasurementGrover, RejectsUnsupportedOptions) {
  for (const auto& marked : {
           std::string{"0"},
           std::string{"0x"},
           std::string(WeakMeasurementGroverOptions::MAX_QUBITS + 1, '0'),
       }) {
    EXPECT_THROW(
        static_cast<void>(WeakMeasurementGrover{{.markedBitstring = marked}}),
        std::invalid_argument);
  }

  for (const auto strength : {
           0.,
           -0.1,
           0.51,
           std::numeric_limits<double>::infinity(),
           std::numeric_limits<double>::quiet_NaN(),
       }) {
    EXPECT_THROW(
        static_cast<void>(WeakMeasurementGrover{
            {.markedBitstring = "00", .measurementStrength = strength}}),
        std::invalid_argument);
  }
  EXPECT_THROW(static_cast<void>(WeakMeasurementGrover{
                   {.markedBitstring = "0000", .measurementStrength = 0.3}}),
               std::invalid_argument);
}

TEST(WeakMeasurementGrover, EvaluatesTheMarkedOutcomeAsSuccess) {
  const WeakMeasurementGrover benchmark{{.markedBitstring = "01"}};
  const auto evaluation = benchmark.evaluate({{"00", 1}, {"01", 9}});
  EXPECT_NEAR(evaluation.totalVariationDistance, 0.1, 1e-15);
  EXPECT_NEAR(evaluation.squaredHellingerFidelity, 0.9, 1e-15);
  ASSERT_TRUE(evaluation.successProbability);
  EXPECT_NEAR(*evaluation.successProbability, 0.9, 1e-15);
}

TEST(WeakMeasurementGrover, RoundTripsJSON) {
  const auto defaults = weakMeasurementGroverFromInstanceSpecificationJSON(
      R"({"schema_version":1,"benchmark":"grover-weak-measurement","parameters":{"marked_bitstring":"10"}})");
  ASSERT_TRUE(defaults.options().measurementStrength);
  EXPECT_DOUBLE_EQ(*defaults.options().measurementStrength, 0.5);
  EXPECT_EQ(
      toInstanceSpecificationJSON(defaults),
      R"({"benchmark":"grover-weak-measurement","parameters":{"marked_bitstring":"10","measurement_strength":0.5},"schema_version":1})");
  const WeakMeasurementGrover weaker{
      {.markedBitstring = "10", .measurementStrength = 0.25}};
  const auto manifest = toManifestJSON(weaker);
  EXPECT_EQ(toManifestJSON(weakMeasurementGroverFromManifestJSON(manifest)),
            manifest);
  EXPECT_EQ(benchmarkIdFromManifestJSON(manifest), "grover-weak-measurement");
  EXPECT_NE(manifest.find(R"("model":"grover_weak_measurement")"),
            std::string::npos);
  EXPECT_NE(manifest.find(R"("success_outcome":"10")"), std::string::npos);
}

TEST(WeakMeasurementGrover, UsesSemanticCaseIds) {
  const auto defaults = weakMeasurementGroverFromInstanceSpecificationJSON(
      R"({"schema_version":1,"benchmark":"grover-weak-measurement","parameters":{"marked_bitstring":"10"}})");
  EXPECT_EQ(caseId(defaults),
            caseId(WeakMeasurementGrover{
                {.markedBitstring = "10", .measurementStrength = 0.5}}));
  EXPECT_NE(caseId(defaults),
            caseId(WeakMeasurementGrover{
                {.markedBitstring = "10", .measurementStrength = 0.25}}));
}

TEST(WeakMeasurementGrover, DescribesJSONSchema) {
  const auto schema = describeBenchmarkJSON("grover-weak-measurement");
  EXPECT_NE(schema.find(R"("exclusiveMinimum":0)"), std::string::npos);
  EXPECT_NE(schema.find(R"("maximum":0.5)"), std::string::npos);
  EXPECT_NE(schema.find(R"("maxLength":2044)"), std::string::npos);
}

TEST(WeakMeasurementGrover, RejectsInvalidJSONParameters) {
  for (const auto* strength : {"0", "0.3", "true", "\"0.25\""}) {
    const auto instance =
        std::string{
            R"({"schema_version":1,"benchmark":"grover-weak-measurement","parameters":{"marked_bitstring":"0000","measurement_strength":)"} +
        strength + "}}";
    EXPECT_THROW(
        static_cast<void>(
            weakMeasurementGroverFromInstanceSpecificationJSON(instance)),
        std::invalid_argument);
  }
}

TEST(WeakMeasurementGrover, EvaluatesCountsFromJSON) {
  const WeakMeasurementGrover benchmark{{.markedBitstring = "10"}};
  const auto evaluation =
      evaluateJSON(toManifestJSON(benchmark),
                   R"({"schema_version":1,"counts":{"10":8,"00":2}})");
  EXPECT_NE(evaluation.find(R"("success_probability":0.8)"), std::string::npos);
}

} // namespace mqt::bench
