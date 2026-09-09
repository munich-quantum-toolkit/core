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
#include "bench/ModularMultiplier.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

using mqt::bench::Counts;
using mqt::bench::ModularMultiplier;
using mqt::bench::ModularMultiplierOptions;
using mqt::bench::Output;

TEST(ModularMultiplier, StoresParametersAndOutput) {
  const ModularMultiplier benchmark{{
      .multiplier = "011",
      .modulus = "101",
      .multiplicand = "+++",
      .control = '+',
  }};
  EXPECT_EQ(benchmark.options().multiplier, "011");
  EXPECT_EQ(benchmark.options().modulus, "101");
  EXPECT_EQ(benchmark.output(), (Output{"result", 8}));
}

TEST(ModularMultiplier, HasAnExactBasisReference) {
  const ModularMultiplier benchmark(
      {.multiplier = "011", .modulus = "101", .multiplicand = "111"});
  EXPECT_EQ(benchmark.expectedResult(), "11110001");
  EXPECT_DOUBLE_EQ(benchmark.probability("11110001"), 1.);
  EXPECT_DOUBLE_EQ(benchmark.probability("10000000"), 0.);
  EXPECT_DOUBLE_EQ(benchmark.probability("00000000"), 0.);
  EXPECT_EQ(
      benchmark.evaluate({{"11110001", 3}, {"00000000", 1}}).successProbability,
      0.75);
  const ModularMultiplier inactive({
      .multiplier = "011",
      .modulus = "101",
      .multiplicand = "111",
      .control = '0',
  });
  EXPECT_EQ(inactive.expectedResult(), "01110000");
  for (const auto* input : {"", "11", "1111", "11x"}) {
    EXPECT_THROW(
        ModularMultiplier(
            {.multiplier = "011", .modulus = "101", .multiplicand = input}),
        std::invalid_argument);
  }
  EXPECT_THROW(ModularMultiplier({.multiplier = "011",
                                  .modulus = "101",
                                  .multiplicand = "111",
                                  .control = 'x'}),
               std::invalid_argument);
}

TEST(ModularMultiplier, ConstrainsPartialSuperpositions) {
  const ModularMultiplier benchmark(
      {.multiplier = "011", .modulus = "101", .multiplicand = "1+0"});
  EXPECT_FALSE(benchmark.expectedResult());
  EXPECT_DOUBLE_EQ(benchmark.probability("11000010"), 0.5);
  EXPECT_DOUBLE_EQ(benchmark.probability("11100011"), 0.5);
  EXPECT_DOUBLE_EQ(benchmark.probability("10000000"), 0.);
  EXPECT_DOUBLE_EQ(benchmark.probability("01000000"), 0.);
}

TEST(ModularMultiplier, ValidatesTheConfiguredInstance) {
  const auto maximumMultiplier =
      std::string(ModularMultiplierOptions::MAX_BITS - 1U, '0') + "1";
  const auto maximumModulus =
      "1" + std::string(ModularMultiplierOptions::MAX_BITS - 1U, '0');
  EXPECT_NO_THROW(
      static_cast<void>(ModularMultiplier{{.multiplier = maximumMultiplier,
                                           .modulus = maximumModulus,
                                           .multiplicand = std::string(63, '+'),
                                           .control = '+'}}));

  EXPECT_THROW(static_cast<void>(ModularMultiplier{{.multiplier = "0",
                                                    .modulus = "1",
                                                    .multiplicand = "+++",
                                                    .control = '+'}}),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(ModularMultiplier{{.multiplier = "001",
                                                    .modulus = "1000",
                                                    .multiplicand = "+++",
                                                    .control = '+'}}),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(ModularMultiplier{{.multiplier = "00x",
                                                    .modulus = "101",
                                                    .multiplicand = "+++",
                                                    .control = '+'}}),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(ModularMultiplier{{.multiplier = "001",
                                                    .modulus = "10x",
                                                    .multiplicand = "+++",
                                                    .control = '+'}}),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(ModularMultiplier{{.multiplier = "001",
                                                    .modulus = "011",
                                                    .multiplicand = "+++",
                                                    .control = '+'}}),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(ModularMultiplier{{.multiplier = "000",
                                                    .modulus = "101",
                                                    .multiplicand = "+++",
                                                    .control = '+'}}),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(ModularMultiplier{{.multiplier = "101",
                                                    .modulus = "101",
                                                    .multiplicand = "+++",
                                                    .control = '+'}}),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(ModularMultiplier{{.multiplier = "110",
                                                    .modulus = "101",
                                                    .multiplicand = "+++",
                                                    .control = '+'}}),
               std::invalid_argument);

  const auto tooLongMultiplier =
      std::string(ModularMultiplierOptions::MAX_BITS, '0') + "1";
  const auto tooLongModulus =
      "1" + std::string(ModularMultiplierOptions::MAX_BITS, '0');
  EXPECT_THROW(
      static_cast<void>(ModularMultiplier{{.multiplier = tooLongMultiplier,
                                           .modulus = tooLongModulus,
                                           .multiplicand = std::string(64, '+'),
                                           .control = '+'}}),
      std::invalid_argument);
}

TEST(ModularMultiplier, GivesUniformWeightToTheExactControlledProducts) {
  const ModularMultiplier benchmark{{
      .multiplier = "011",
      .modulus = "101",
      .multiplicand = "+++",
      .control = '+',
  }};
  constexpr std::array<std::string_view, 8> multiplicands{
      "000", "001", "010", "011", "100", "101", "110", "111",
  };
  constexpr std::array<std::string_view, 8> products{
      "000", "011", "001", "100", "010", "000", "011", "001",
  };

  Counts exact;
  for (size_t index = 0; index < multiplicands.size(); ++index) {
    const auto inactive =
        std::string{"0"} + std::string{multiplicands[index]} + "0000";
    const auto active = std::string{"1"} + std::string{multiplicands[index]} +
                        "0" + std::string{products[index]};
    EXPECT_DOUBLE_EQ(benchmark.probability(inactive), 1. / 16.);
    EXPECT_DOUBLE_EQ(benchmark.probability(active), 1. / 16.);
    exact.emplace(inactive, 1U);
    exact.emplace(active, 1U);
  }

  EXPECT_DOUBLE_EQ(benchmark.probability("10010000"), 0.);
  EXPECT_DOUBLE_EQ(benchmark.probability("01110001"), 0.);
  EXPECT_THROW(static_cast<void>(benchmark.probability("1001001")),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(benchmark.probability("100100x1")),
               std::invalid_argument);

  const auto evaluation = benchmark.evaluate(exact);
  EXPECT_DOUBLE_EQ(evaluation.totalVariationDistance, 0.);
  EXPECT_DOUBLE_EQ(evaluation.squaredHellingerFidelity, 1.);
  EXPECT_EQ(evaluation.successProbability, 1.);
}

TEST(ModularMultiplier, ScoresTheArithmeticRelationByShotCount) {
  const ModularMultiplier benchmark{{
      .multiplier = "011",
      .modulus = "101",
      .multiplicand = "+++",
      .control = '+',
  }};
  EXPECT_EQ(
      benchmark.evaluate({{"10010011", 3}, {"10010010", 1}}).successProbability,
      0.75);
  EXPECT_EQ(
      benchmark.evaluate({{"01110001", 7}, {"10010011", 0}}).successProbability,
      0.);
  EXPECT_THROW(static_cast<void>(benchmark.evaluate({})),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(benchmark.evaluate({{"00000000", 0}})),
               std::invalid_argument);
}

TEST(ModularMultiplier, SeparatesRelationSuccessFromDistributionFit) {
  const ModularMultiplier benchmark{
      {
          .multiplier = std::string(19, '0') + "1",
          .modulus = std::string(20, '1'),
          .multiplicand = std::string(20, '+'),
          .control = '+',
      },
  };
  const auto result = benchmark.evaluate({{std::string(42, '0'), 16'384}});
  EXPECT_EQ(result.successProbability, 1.);
  EXPECT_DOUBLE_EQ(result.totalVariationDistance, 1. - std::ldexp(1., -21));
}

TEST(ModularMultiplier, SupportsNonCoprimeInputsAndMultiplicandsAtLeastN) {
  const ModularMultiplier nonCoprime{{
      .multiplier = "010",
      .modulus = "100",
      .multiplicand = "+++",
      .control = '+',
  }};
  EXPECT_DOUBLE_EQ(nonCoprime.probability("10110010"), 1. / 16.);

  const ModularMultiplier benchmark{{
      .multiplier = "011",
      .modulus = "101",
      .multiplicand = "+++",
      .control = '+',
  }};
  EXPECT_DOUBLE_EQ(benchmark.probability("11110001"), 1. / 16.);
}

TEST(ModularMultiplier, KeepsTheLargestReferenceWeightRepresentable) {
  constexpr auto width = ModularMultiplierOptions::MAX_BITS;
  const auto multiplier = std::string(width - 1U, '1') + "0";
  const auto modulus = std::string(width, '1');
  const ModularMultiplier benchmark{{
      .multiplier = multiplier,
      .modulus = modulus,
      .multiplicand = std::string(width, '+'),
      .control = '+',
  }};
  const auto multiplicand = std::string(width - 2U, '0') + "10";
  const auto accumulator = "0" + std::string(width - 2U, '1') + "01";
  const auto outcome = "1" + multiplicand + accumulator;
  const ModularMultiplier basis({
      .multiplier = multiplier,
      .modulus = modulus,
      .multiplicand = multiplicand,
  });
  EXPECT_EQ(basis.expectedResult(), outcome);
  EXPECT_GT(benchmark.probability(outcome), 0.);
  EXPECT_EQ(benchmark.evaluate({{outcome, 1}}).successProbability, 1.);
}

} // namespace
