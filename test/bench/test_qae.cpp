/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "bench/JSON.hpp"
#include "bench/QAE.hpp"

#include "JSONTestUtils.hpp"

#include "gtest/gtest.h"

#include "llvm/Support/Error.h"

#include <cmath>
#include <cstddef>
#include <limits>
#include <string>
#include <system_error>
#include <vector>

namespace mqt::bench {
namespace {
QAEProblem problem() { return {.goodStates = {"10", "01"}, .amplitude = 0.25}; }
} // namespace

TEST(QAE, ScoresBothGroverEigenphasesAndAmplifiedData) {
  const auto phase =
      llvm::cantFail(QAE::create({.problem = problem(), .precision = 2}));
  EXPECT_NEAR(phase.probability("00"), 3. / 16., 1e-14);
  EXPECT_NEAR(phase.probability("01"), 3. / 8., 1e-14);
  EXPECT_NEAR(phase.probability("10"), 1. / 16., 1e-14);
  EXPECT_NEAR(phase.probability("11"), 3. / 8., 1e-14);
  // A single query bit measures a Bernoulli variable with probability a.
  const auto tiny = llvm::cantFail(QAE::create({
      .problem = {.goodStates = {"0"}, .amplitude = 1e-34},
      .precision = 1,
  }));
  EXPECT_NEAR(tiny.probability("1"), 1e-34, 1e-49);
  const auto amplified =
      llvm::cantFail(QAE::create({.problem = problem(), .iterations = 1}));
  EXPECT_NEAR(amplified.probability("01"), 0.5, 1e-14);
  EXPECT_NEAR(llvm::cantFail(amplified.evaluate({{"01", 17}, {"10", 17}}))
                  .totalVariationDistance,
              0., 1e-14);
  for (const auto amplitude : {0., 1.}) {
    const auto exact = llvm::cantFail(QAE::create({
        .problem = {.goodStates = {"0"}, .amplitude = amplitude},
        .precision = 3,
    }));
    EXPECT_DOUBLE_EQ(exact.probability(amplitude == 0. ? "000" : "100"), 1.);
  }
}

TEST(QAE, CanonicalizesTheGoodSetAndDistinguishesExperimentPowers) {
  const auto parsed = llvm::cantFail(qaeFromInstanceSpecificationJSON(
      R"({"schema_version":1,"benchmark":"qae","parameters":{"good_states":["10","01"],"amplitude":0.25,"iterations":0}})"));
  const auto sorted = llvm::cantFail(QAE::create({
      .problem = {.goodStates = {"01", "10"}, .amplitude = 0.25},
      .iterations = 0,
  }));
  EXPECT_EQ(caseId(parsed), caseId(sorted));
  EXPECT_NE(caseId(parsed), caseId(llvm::cantFail(QAE::create(
                                {.problem = problem(), .iterations = 1}))));
  EXPECT_EQ(toManifestJSON(
                llvm::cantFail(qaeFromManifestJSON(toManifestJSON(parsed)))),
            toManifestJSON(sorted));
}

TEST(QAE, RejectsAmbiguousAndInvalidExperiments) {
  for (const auto& states : std::vector<std::vector<std::string>>{
           {},
           {""},
           {"1", "00"},
           {"x"},
           {"01", "01"},
           {"0", "1"},
           {std::string(64, '0')},
       }) {
    test::expectError(QAE::create({
                          .problem = {.goodStates = states, .amplitude = 0.4},
                          .iterations = 1,
                      }),
                      std::errc::invalid_argument);
  }
  for (const auto amplitude :
       {-0.1, 1.1, std::numeric_limits<double>::quiet_NaN()}) {
    test::expectError(
        QAE::create({
            .problem = {.goodStates = {"0"}, .amplitude = amplitude},
            .iterations = 0,
        }),
        std::errc::invalid_argument);
  }
  for (const auto& options : {
           QAEOptions{.problem = problem()},
           QAEOptions{.problem = problem(), .precision = 0},
           QAEOptions{.problem = problem(), .precision = 32},
           QAEOptions{.problem = problem(), .precision = 2, .iterations = 0},
           QAEOptions{
               .problem = problem(),
               .iterations = QAEOptions::MAX_ITERATIONS + 1,
           },
       }) {
    test::expectError(QAE::create(options), std::errc::invalid_argument);
  }
}

TEST(QAE, EstimatesFromMeasurementCounts) {
  const auto estimate =
      llvm::cantFail(phaseEstimation(problem(), 3, [](const QAE&) {
        return Counts{{"001", 5}, {"111", 5}, {"100", 9}};
      }));
  EXPECT_NEAR(estimate, (2. - std::sqrt(2.)) / 4., 1e-14);
  // Supply data for a different probability to detect use of the preparation
  // parameter.
  constexpr double measured = 0.78;
  std::vector<size_t> powers;
  const auto run = [&](const QAE& experiment) {
    const auto power = *experiment.options().iterations;
    powers.push_back(power);
    const auto angle = ((2. * static_cast<double>(power)) + 1.) *
                       std::asin(std::sqrt(measured));
    const auto good =
        static_cast<size_t>(std::round(4096. * std::pow(std::sin(angle), 2)));
    return Counts{
        {"01", good / 2},
        {"10", good - (good / 2)},
        {"00", 4096 - good},
    };
  };
  EXPECT_NEAR(llvm::cantFail(maximumLikelihood(problem(), 3, run)), measured,
              1e-4);
  EXPECT_EQ(powers, (std::vector<size_t>{1, 2, 4}));
  powers.clear();
  const auto [low, high] = llvm::cantFail(
      iterativeAmplitudeEstimation(problem(), 0.001, 0.99, 50, run));
  EXPECT_EQ(powers.front(), 0U);
  EXPECT_GT(powers.back(), 0U);
  EXPECT_LE(low, measured);
  EXPECT_GE(high, measured);
  EXPECT_LE(high - low, 0.002);
}

TEST(QAE, ReportsExhaustedBudgetAndInvalidCounts) {
  const auto oneShot = [](const QAE&) { return Counts{{"01", 1}}; };
  test::expectError(
      iterativeAmplitudeEstimation(problem(), 0.01, 0.95, 1, oneShot),
      std::errc::result_out_of_range);
  test::expectError(
      maximumLikelihood(problem(), 2,
                        [](const QAE&) { return Counts{{"1", 1}}; }),
      std::errc::invalid_argument);
}
} // namespace mqt::bench
