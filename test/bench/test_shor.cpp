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
#include "bench/Shor.hpp"

#include "gtest/gtest.h"

#include <array>
#include <bitset>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace mqt::bench;

TEST(Shor, ValidatesParametersAndPhaseWidth) {
  const Shor benchmark({.number = 21});
  EXPECT_EQ(benchmark.output(), (Output{"phase", 10}));
  EXPECT_EQ(benchmark.options().base, 2);
  EXPECT_FALSE(benchmark.options().qftCutoff);
  for (const uint64_t number : {0ULL, 1ULL, 2ULL, 4ULL, 2147483648ULL}) {
    EXPECT_THROW(Shor({.number = number}), std::invalid_argument);
  }
  for (const uint64_t base : {0ULL, 1ULL, 3ULL, 7ULL, 21ULL, 22ULL}) {
    EXPECT_THROW(Shor({.number = 21, .base = base}), std::invalid_argument);
  }
  EXPECT_THROW(Shor({.number = 21, .qftCutoff = 0}), std::invalid_argument);
  EXPECT_EQ(Shor({.number = ShorOptions::MAX_NUMBER}).output().width, 62U);
  EXPECT_NO_THROW(Shor({.number = 7, .qftCutoff = 100}));
}

TEST(Shor, RecoversOnlyVerifiedFactorsAndWeightsShots) {
  const Shor benchmark({.number = 21});
  const auto evaluation =
      benchmark.evaluate({{"0010101011", 3}, {"0000000000", 1}});
  EXPECT_EQ(evaluation.factors, (FactorPair{3, 7}));
  EXPECT_DOUBLE_EQ(evaluation.successProbability, 0.75);
  for (const auto* outcome :
       {"0000000000", "0101010101", "1000000000", "1111111111"}) {
    const auto unsuccessful = benchmark.evaluate({{outcome, 2}});
    EXPECT_FALSE(unsuccessful.factors);
    EXPECT_DOUBLE_EQ(unsuccessful.successProbability, 0.);
  }
  const auto zeroWeight =
      benchmark.evaluate({{"0010101011", 0}, {"0000000000", 1}});
  EXPECT_FALSE(zeroWeight.factors);
  EXPECT_DOUBLE_EQ(zeroWeight.successProbability, 0.);
  EXPECT_FALSE(
      Shor({.number = 15, .base = 14}).evaluate({{"10000000", 1}}).factors);
  EXPECT_FALSE(Shor({.number = 7}).evaluate({{"010101", 1}}).factors);
}

TEST(Shor, HandlesMaximalFractionsWithoutWideIntegers) {
  const Shor benchmark({.number = 2147483645, .base = 858993459});
  const auto phase = std::bitset<62>((uint64_t{1} << 61U) + 1).to_string();
  EXPECT_EQ(benchmark.evaluate({{phase, 1}}).factors,
            (FactorPair{5, 429496729}));
  EXPECT_FALSE(benchmark.evaluate({{std::string(62, '1'), 1}}).factors);
  EXPECT_FALSE(benchmark.evaluate({{std::string(61, '0') + "1", 1}}).factors);
}

TEST(Shor, RejectsInvalidCounts) {
  const Shor benchmark({.number = 21});
  EXPECT_THROW(benchmark.evaluate({}), std::invalid_argument);
  EXPECT_THROW(benchmark.evaluate({{"0000000000", 0}}), std::invalid_argument);
  EXPECT_THROW(benchmark.evaluate({{"000000000", 1}}), std::invalid_argument);
  EXPECT_THROW(benchmark.evaluate({{"000000000x", 1}}), std::invalid_argument);
  EXPECT_THROW(
      benchmark.evaluate({{"0000000000", std::numeric_limits<size_t>::max()},
                          {"0010101011", 1}}),
      std::overflow_error);
}

TEST(Shor, PerformsClassicalPrechecksWithoutQuantumRuns) {
  const auto unused = [](const Shor&) -> Counts {
    throw std::runtime_error("unexpected quantum run");
  };
  for (const uint64_t number : {2ULL, 3ULL, 41ULL, 2147483647ULL}) {
    const auto result = factor(number, unused);
    EXPECT_EQ(result.status, FactorStatus::Prime);
    EXPECT_FALSE(result.factors);
    EXPECT_EQ(result.attempts, 0U);
  }
  for (const uint64_t number :
       {4ULL, 12ULL, 27ULL, 81ULL, 121ULL, 1162261467ULL, 1220703125ULL}) {
    const auto result = factor(number, unused);
    EXPECT_EQ(result.status, FactorStatus::Success);
    ASSERT_TRUE(result.factors);
    const auto [first, second] = *result.factors;
    EXPECT_GE(first, 2U);
    EXPECT_LE(first, second);
    EXPECT_EQ(first * second, number);
    EXPECT_EQ(result.attempts, 0U);
  }
}

TEST(Shor, ExecutesTheConfiguredCallbackAndBoundsAttempts) {
  size_t calls = 0;
  const auto result = factor(21,
                             [&](const Shor& benchmark) {
                               ++calls;
                               EXPECT_EQ(benchmark.options().number, 21U);
                               EXPECT_EQ(benchmark.options().base, 2U);
                               EXPECT_EQ(benchmark.options().qftCutoff, 3U);
                               return Counts{{"0010101011", 1}};
                             },
                             {.qftCutoff = 3});
  EXPECT_EQ(calls, 1U);
  EXPECT_EQ(result.status, FactorStatus::Success);
  EXPECT_EQ(result.factors, (FactorPair{3, 7}));
  EXPECT_EQ(result.attempts, 1U);

  for (const uint64_t number : {21ULL, 35ULL, 1373653ULL}) {
    calls = 0;
    const auto unsuccessful =
        factor(number,
               [&](const Shor& benchmark) {
                 ++calls;
                 return Counts{{std::string(benchmark.output().width, '0'), 1}};
               },
               {.maxAttempts = 1});
    EXPECT_EQ(unsuccessful.status, FactorStatus::AttemptsExhausted);
    EXPECT_FALSE(unsuccessful.factors);
    EXPECT_EQ(unsuccessful.attempts, 1U);
    EXPECT_EQ(calls, 1U);
  }
}

TEST(Shor, SelectsRepeatableBasesAndUsesTheGcdShortcut) {
  std::array<std::vector<uint64_t>, 2> observed;
  for (auto& bases : observed) {
    const auto result =
        factor(21,
               [&](const Shor& benchmark) {
                 bases.push_back(benchmark.options().base);
                 return Counts{{std::string(benchmark.output().width, '0'), 1}};
               },
               {.maxAttempts = 64, .seed = 17});
    EXPECT_EQ(result.status, FactorStatus::Success);
    EXPECT_EQ(result.factors, (FactorPair{3, 7}));
    EXPECT_EQ(result.attempts, bases.size() + 1);
    EXPECT_LE(result.attempts, 64U);
  }
  EXPECT_EQ(observed[0], observed[1]);
}

TEST(Shor, PropagatesCallbackFailuresAndRejectsInvalidDriverInputs) {
  const auto run = [](const Shor&) -> Counts {
    throw std::runtime_error("device failed");
  };
  EXPECT_THROW(factor(21, run), std::runtime_error);
  EXPECT_THROW(factor(21, [](const Shor&) { return Counts{}; }),
               std::invalid_argument);
  EXPECT_THROW(factor(1, run), std::invalid_argument);
  EXPECT_THROW(factor(ShorOptions::MAX_NUMBER + 1, run), std::invalid_argument);
  EXPECT_THROW(factor(21, run, {.maxAttempts = 0}), std::invalid_argument);
  EXPECT_THROW(factor(21, run, {.qftCutoff = 0}), std::invalid_argument);
  EXPECT_THROW(factor(21, {}), std::invalid_argument);
}

TEST(Shor, RoundTripsJsonAndUsesAVerificationReference) {
  const auto benchmark = shorFromInstanceSpecificationJSON(
      R"({"schema_version":1,"benchmark":"shor","parameters":{"number":21}})");
  EXPECT_EQ(caseId(benchmark), caseId(Shor({.number = 21, .base = 2})));
  EXPECT_NE(caseId(benchmark), caseId(Shor({.number = 21, .base = 4})));
  EXPECT_NE(caseId(benchmark), caseId(Shor({.number = 21, .qftCutoff = 2})));
  EXPECT_EQ(caseId(benchmark),
            caseId(shorFromManifestJSON(toManifestJSON(benchmark))));
  EXPECT_NE(toManifestJSON(benchmark).find("\"kind\":\"verification\""),
            std::string::npos);
  const auto result = evaluateJSON(
      toManifestJSON(benchmark),
      R"({"schema_version":1,"counts":{"0010101011":3,"0000000000":1}})");
  EXPECT_NE(result.find("\"success_probability\":0.75"), std::string::npos);
  EXPECT_NE(result.find("\"factors\":[3,7]"), std::string::npos);
  EXPECT_EQ(result.find("total_variation_distance"), std::string::npos);
  EXPECT_EQ(result.find("squared_hellinger_fidelity"), std::string::npos);
}

TEST(Shor, RejectsInvalidJsonTypesAndParameters) {
  for (const auto* parameters : {
           R"({"number":true})",
           R"({"number":21.0})",
           R"({"number":"21"})",
           R"({"number":21,"base":false})",
           R"({"number":21,"qft_cutoff":null})",
           R"({"number":21,"qft_cutoff":0})",
           R"({"number":21,"qft_cutoff":1.5})",
           R"({"number":21,"extra":0})",
           R"({"number":21,"base":7})",
           R"({})",
       }) {
    EXPECT_THROW(
        shorFromInstanceSpecificationJSON(
            std::string(
                R"({"schema_version":1,"benchmark":"shor","parameters":)") +
            parameters + "}"),
        std::invalid_argument);
  }
}

} // namespace
