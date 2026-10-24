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

#include "JSONTestUtils.hpp"

#include "gtest/gtest.h"

#include "llvm/Support/Error.h"

#include <array>
#include <bitset>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <system_error>
#include <vector>

namespace mqt::bench {

TEST(Shor, ValidatesParametersAndPhaseWidth) {
  const auto benchmark = llvm::cantFail(Shor::create({.number = 21}));
  EXPECT_EQ(benchmark.output(), (Output{"result", 10}));
  EXPECT_EQ(benchmark.options().base, 2);
  for (const uint64_t number : {0ULL, 1ULL, 2ULL, 4ULL, 2147483648ULL}) {
    test::expectError(Shor::create({.number = number}),
                      std::errc::invalid_argument);
  }
  for (const uint64_t base : {0ULL, 1ULL, 3ULL, 7ULL, 21ULL, 22ULL}) {
    test::expectError(Shor::create({.number = 21, .base = base}),
                      std::errc::invalid_argument);
  }
  EXPECT_EQ(llvm::cantFail(Shor::create({.number = ShorOptions::MAX_NUMBER}))
                .output()
                .width,
            62U);
}

TEST(Shor, RecoversOnlyVerifiedFactorsAndWeightsShots) {
  const auto benchmark = llvm::cantFail(Shor::create({.number = 21}));
  const auto evaluation = llvm::cantFail(
      benchmark.evaluate({{"0010101011", 3}, {"0000000000", 1}}));
  EXPECT_EQ(evaluation.factors, (FactorPair{3, 7}));
  EXPECT_DOUBLE_EQ(evaluation.successProbability, 0.75);
  for (const auto* outcome :
       {"0000000000", "0101010101", "1000000000", "1111111111"}) {
    const auto unsuccessful =
        llvm::cantFail(benchmark.evaluate({{outcome, 2}}));
    EXPECT_FALSE(unsuccessful.factors);
    EXPECT_DOUBLE_EQ(unsuccessful.successProbability, 0.);
  }
  const auto zeroWeight = llvm::cantFail(
      benchmark.evaluate({{"0010101011", 0}, {"0000000000", 1}}));
  EXPECT_FALSE(zeroWeight.factors);
  EXPECT_DOUBLE_EQ(zeroWeight.successProbability, 0.);
  EXPECT_FALSE(
      llvm::cantFail(llvm::cantFail(Shor::create({.number = 15, .base = 14}))
                         .evaluate({{"10000000", 1}}))
          .factors);
  EXPECT_FALSE(
      llvm::cantFail(
          llvm::cantFail(Shor::create({.number = 7})).evaluate({{"010101", 1}}))
          .factors);
}

TEST(Shor, HandlesMaximalFractionsWithoutWideIntegers) {
  const auto benchmark =
      llvm::cantFail(Shor::create({.number = 2147483645, .base = 858993459}));
  const auto phase = std::bitset<62>((uint64_t{1} << 61U) + 1).to_string();
  EXPECT_EQ(llvm::cantFail(benchmark.evaluate({{phase, 1}})).factors,
            (FactorPair{5, 429496729}));
  EXPECT_FALSE(
      llvm::cantFail(benchmark.evaluate({{std::string(62, '1'), 1}})).factors);
  EXPECT_FALSE(
      llvm::cantFail(benchmark.evaluate({{std::string(61, '0') + "1", 1}}))
          .factors);
}

TEST(Shor, RejectsInvalidCounts) {
  const auto benchmark = llvm::cantFail(Shor::create({.number = 21}));
  test::expectError(benchmark.evaluate({}), std::errc::invalid_argument);
  test::expectError(benchmark.evaluate({{"0000000000", 0}}),
                    std::errc::invalid_argument);
  test::expectError(benchmark.evaluate({{"000000000", 1}}),
                    std::errc::invalid_argument);
  test::expectError(benchmark.evaluate({{"000000000x", 1}}),
                    std::errc::invalid_argument);
  test::expectError(benchmark.evaluate({{"0000000000", 1}, {"000000000", 0}}),
                    std::errc::invalid_argument);
  test::expectError(benchmark.evaluate({{"0000000000", 1}, {"000000000x", 0}}),
                    std::errc::invalid_argument);
  test::expectError(benchmark.evaluate({
                        {"0000000000", std::numeric_limits<size_t>::max()},
                        {"0010101011", 1},
                    }),
                    std::errc::value_too_large);
}

TEST(Shor, PerformsClassicalPrechecksWithoutQuantumRuns) {
  const auto unused = [](const Shor&) -> Counts {
    ADD_FAILURE() << "unexpected quantum run";
    return {};
  };
  for (const uint64_t number : {2ULL, 3ULL, 41ULL, 2147483647ULL}) {
    const auto result = llvm::cantFail(factor(number, unused));
    EXPECT_EQ(result.status, FactorStatus::Prime);
    EXPECT_FALSE(result.factors);
    EXPECT_EQ(result.attempts, 0U);
  }
  for (const uint64_t number :
       {4ULL, 12ULL, 27ULL, 81ULL, 121ULL, 1162261467ULL, 1220703125ULL}) {
    const auto result = llvm::cantFail(factor(number, unused));
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
  const auto result = llvm::cantFail(factor(21, [&](const Shor& benchmark) {
    ++calls;
    EXPECT_EQ(benchmark.options().number, 21U);
    EXPECT_EQ(benchmark.options().base, 2U);
    return Counts{{"0010101011", 1}};
  }));
  EXPECT_EQ(calls, 1U);
  EXPECT_EQ(result.status, FactorStatus::Success);
  EXPECT_EQ(result.factors, (FactorPair{3, 7}));
  EXPECT_EQ(result.attempts, 1U);

  for (const uint64_t number : {21ULL, 35ULL, 1373653ULL}) {
    calls = 0;
    const auto unsuccessful = llvm::cantFail(
        factor(number,
               [&](const Shor& benchmark) {
                 ++calls;
                 return Counts{{std::string(benchmark.output().width, '0'), 1}};
               },
               {.maxAttempts = 1}));
    EXPECT_EQ(unsuccessful.status, FactorStatus::AttemptsExhausted);
    EXPECT_FALSE(unsuccessful.factors);
    EXPECT_EQ(unsuccessful.attempts, 1U);
    EXPECT_EQ(calls, 1U);
  }
}

TEST(Shor, SelectsRepeatableBasesAndUsesTheGcdShortcut) {
  std::array<std::vector<uint64_t>, 2> observed;
  for (auto& bases : observed) {
    const auto result = llvm::cantFail(
        factor(21,
               [&](const Shor& benchmark) {
                 bases.push_back(benchmark.options().base);
                 return Counts{{std::string(benchmark.output().width, '0'), 1}};
               },
               {.maxAttempts = 64, .seed = 17}));
    EXPECT_EQ(result.status, FactorStatus::Success);
    EXPECT_EQ(result.factors, (FactorPair{3, 7}));
    EXPECT_EQ(result.attempts, bases.size() + 1);
    EXPECT_LE(result.attempts, 64U);
  }
  EXPECT_EQ(observed[0], observed[1]);
}

TEST(Shor, PropagatesCallbackFailuresAndRejectsInvalidDriverInputs) {
  const auto run = [](const Shor&) -> llvm::Expected<Counts> {
    return llvm::createStringError(std::errc::io_error, "device failed");
  };
  test::expectError(factor(21, run), std::errc::io_error);
  test::expectError(factor(21, [](const Shor&) { return Counts{}; }),
                    std::errc::invalid_argument);
  test::expectError(factor(1, run), std::errc::invalid_argument);
  test::expectError(factor(ShorOptions::MAX_NUMBER + 1, run),
                    std::errc::invalid_argument);
  test::expectError(factor(21, run, {.maxAttempts = 0}),
                    std::errc::invalid_argument);
  test::expectError(factor(21, {}), std::errc::invalid_argument);
}

TEST(Shor, RoundTripsJSON) {
  const auto benchmark = llvm::cantFail(shorFromInstanceSpecificationJSON(
      R"({"schema_version":1,"benchmark":"shor","parameters":{"number":21}})"));
  const auto manifest = toManifestJSON(benchmark);
  EXPECT_EQ(toManifestJSON(llvm::cantFail(shorFromManifestJSON(manifest))),
            manifest);
  EXPECT_NE(manifest.find("\"kind\":\"verification\""), std::string::npos);
}

TEST(Shor, UsesSemanticCaseIds) {
  const auto benchmark = llvm::cantFail(shorFromInstanceSpecificationJSON(
      R"({"schema_version":1,"benchmark":"shor","parameters":{"number":21}})"));
  EXPECT_EQ(caseId(benchmark),
            caseId(llvm::cantFail(Shor::create({.number = 21, .base = 2}))));
  EXPECT_NE(caseId(benchmark),
            caseId(llvm::cantFail(Shor::create({.number = 21, .base = 4}))));
}

TEST(Shor, DescribesJSONSchema) {
  const auto schema = llvm::cantFail(describeBenchmarkJSON("shor"));
  EXPECT_NE(schema.find(R"("minimum":3)"), std::string::npos);
  EXPECT_NE(schema.find(R"("required":["number"])"), std::string::npos);
}

TEST(Shor, RejectsInvalidJSONParameters) {
  for (const auto* parameters : {
           R"({"number":true})",
           R"({"number":21.0})",
           R"({"number":"21"})",
           R"({"number":21,"base":false})",
           R"({"number":21,"extra":0})",
           R"({"number":21,"base":7})",
           R"({})",
       }) {
    test::expectInvalidJSON(
        [&] {
          return shorFromInstanceSpecificationJSON(
              std::string(
                  R"({"schema_version":1,"benchmark":"shor","parameters":)") +
              parameters + "}");
        },
        "");
  }
}

TEST(Shor, EvaluatesCountsFromJSON) {
  const auto result = llvm::cantFail(evaluateJSON(
      toManifestJSON(llvm::cantFail(Shor::create({.number = 21}))),
      R"({"schema_version":1,"counts":{"0010101011":3,"0000000000":1}})"));
  EXPECT_NE(result.find("\"success_probability\":0.75"), std::string::npos);
  EXPECT_NE(result.find("\"factors\":[3,7]"), std::string::npos);
  EXPECT_EQ(result.find("total_variation_distance"), std::string::npos);
  EXPECT_EQ(result.find("squared_hellinger_fidelity"), std::string::npos);
}

} // namespace mqt::bench
