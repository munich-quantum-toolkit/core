/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "bench/Error.hpp"
#include "bench/Evaluation.hpp"
#include "bench/ModularMultiplier.hpp"
#include "bench/TestUtils.hpp"

#include "gtest/gtest.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <string_view>

namespace test = mqt::bench::test;

namespace {

using mqt::bench::Counts;
using mqt::bench::ModularMultiplier;
using mqt::bench::ModularMultiplierOptions;
using mqt::bench::Output;

TEST(ModularMultiplier, StoresParametersAndOutput) {
  const auto benchmark = test::value(ModularMultiplier::create({
      .multiplier = "011",
      .modulus = "101",
      .multiplicand = "+++",
      .control = '+',
  }));
  EXPECT_EQ(benchmark.options().multiplier, "011");
  EXPECT_EQ(benchmark.options().modulus, "101");
  EXPECT_EQ(benchmark.output(), (Output{"result", 8}));
}

TEST(ModularMultiplier, HasAnExactBasisReference) {
  const auto benchmark = test::value(ModularMultiplier::create(
      {.multiplier = "011", .modulus = "101", .multiplicand = "111"}));
  EXPECT_EQ(benchmark.expectedResult(), "11110001");
  EXPECT_DOUBLE_EQ(test::value(benchmark.probability("11110001")), 1.);
  EXPECT_DOUBLE_EQ(test::value(benchmark.probability("10000000")), 0.);
  EXPECT_DOUBLE_EQ(test::value(benchmark.probability("00000000")), 0.);
  EXPECT_EQ(test::value(benchmark.evaluate({{"11110001", 3}, {"00000000", 1}}))
                .successProbability,
            0.75);
  const auto inactive = test::value(ModularMultiplier::create({
      .multiplier = "011",
      .modulus = "101",
      .multiplicand = "111",
      .control = '0',
  }));
  EXPECT_EQ(inactive.expectedResult(), "01110000");
  for (const auto* input : {"", "11", "1111", "11x"}) {
    EXPECT_EQ(
        test::errorKind(ModularMultiplier::create(
            {.multiplier = "011", .modulus = "101", .multiplicand = input})),
        mqt::bench::Error::Kind::InvalidArgument);
  }
  EXPECT_EQ(test::errorKind(ModularMultiplier::create({.multiplier = "011",
                                                       .modulus = "101",
                                                       .multiplicand = "111",
                                                       .control = 'x'})),
            mqt::bench::Error::Kind::InvalidArgument);
}

TEST(ModularMultiplier, ConstrainsPartialSuperpositions) {
  const auto benchmark = test::value(ModularMultiplier::create(
      {.multiplier = "011", .modulus = "101", .multiplicand = "1+0"}));
  EXPECT_FALSE(benchmark.expectedResult());
  EXPECT_DOUBLE_EQ(test::value(benchmark.probability("11000010")), 0.5);
  EXPECT_DOUBLE_EQ(test::value(benchmark.probability("11100011")), 0.5);
  EXPECT_DOUBLE_EQ(test::value(benchmark.probability("10000000")), 0.);
  EXPECT_DOUBLE_EQ(test::value(benchmark.probability("01000000")), 0.);
}

TEST(ModularMultiplier, ValidatesTheConfiguredInstance) {
  const auto maximumMultiplier =
      std::string(ModularMultiplierOptions::MAX_BITS - 1U, '0') + "1";
  const auto maximumModulus =
      "1" + std::string(ModularMultiplierOptions::MAX_BITS - 1U, '0');
  EXPECT_NO_THROW(static_cast<void>(test::value(
      ModularMultiplier::create({.multiplier = maximumMultiplier,
                                 .modulus = maximumModulus,
                                 .multiplicand = std::string(63, '+'),
                                 .control = '+'}))));

  EXPECT_EQ(test::errorKind(ModularMultiplier::create({.multiplier = "0",
                                                       .modulus = "1",
                                                       .multiplicand = "+++",
                                                       .control = '+'})),
            mqt::bench::Error::Kind::InvalidArgument);
  EXPECT_EQ(test::errorKind(ModularMultiplier::create({.multiplier = "001",
                                                       .modulus = "1000",
                                                       .multiplicand = "+++",
                                                       .control = '+'})),
            mqt::bench::Error::Kind::InvalidArgument);
  EXPECT_EQ(test::errorKind(ModularMultiplier::create({.multiplier = "00x",
                                                       .modulus = "101",
                                                       .multiplicand = "+++",
                                                       .control = '+'})),
            mqt::bench::Error::Kind::InvalidArgument);
  EXPECT_EQ(test::errorKind(ModularMultiplier::create({.multiplier = "001",
                                                       .modulus = "10x",
                                                       .multiplicand = "+++",
                                                       .control = '+'})),
            mqt::bench::Error::Kind::InvalidArgument);
  EXPECT_EQ(test::errorKind(ModularMultiplier::create({.multiplier = "001",
                                                       .modulus = "011",
                                                       .multiplicand = "+++",
                                                       .control = '+'})),
            mqt::bench::Error::Kind::InvalidArgument);
  EXPECT_EQ(test::errorKind(ModularMultiplier::create({.multiplier = "000",
                                                       .modulus = "101",
                                                       .multiplicand = "+++",
                                                       .control = '+'})),
            mqt::bench::Error::Kind::InvalidArgument);
  EXPECT_EQ(test::errorKind(ModularMultiplier::create({.multiplier = "101",
                                                       .modulus = "101",
                                                       .multiplicand = "+++",
                                                       .control = '+'})),
            mqt::bench::Error::Kind::InvalidArgument);
  EXPECT_EQ(test::errorKind(ModularMultiplier::create({.multiplier = "110",
                                                       .modulus = "101",
                                                       .multiplicand = "+++",
                                                       .control = '+'})),
            mqt::bench::Error::Kind::InvalidArgument);

  const auto tooLongMultiplier =
      std::string(ModularMultiplierOptions::MAX_BITS, '0') + "1";
  const auto tooLongModulus =
      "1" + std::string(ModularMultiplierOptions::MAX_BITS, '0');
  EXPECT_EQ(test::errorKind(
                ModularMultiplier::create({.multiplier = tooLongMultiplier,
                                           .modulus = tooLongModulus,
                                           .multiplicand = std::string(64, '+'),
                                           .control = '+'})),
            mqt::bench::Error::Kind::InvalidArgument);
}

TEST(ModularMultiplier, GivesUniformWeightToTheExactControlledProducts) {
  const auto benchmark = test::value(ModularMultiplier::create({
      .multiplier = "011",
      .modulus = "101",
      .multiplicand = "+++",
      .control = '+',
  }));
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
    EXPECT_DOUBLE_EQ(test::value(benchmark.probability(inactive)), 1. / 16.);
    EXPECT_DOUBLE_EQ(test::value(benchmark.probability(active)), 1. / 16.);
    exact.emplace(inactive, 1U);
    exact.emplace(active, 1U);
  }

  EXPECT_DOUBLE_EQ(test::value(benchmark.probability("10010000")), 0.);
  EXPECT_DOUBLE_EQ(test::value(benchmark.probability("01110001")), 0.);
  EXPECT_EQ(test::errorKind(benchmark.probability("1001001")),
            mqt::bench::Error::Kind::InvalidArgument);
  EXPECT_EQ(test::errorKind(benchmark.probability("100100x1")),
            mqt::bench::Error::Kind::InvalidArgument);

  const auto evaluation = test::value(benchmark.evaluate(exact));
  EXPECT_DOUBLE_EQ(evaluation.totalVariationDistance, 0.);
  EXPECT_DOUBLE_EQ(evaluation.squaredHellingerFidelity, 1.);
  EXPECT_EQ(evaluation.successProbability, 1.);
}

TEST(ModularMultiplier, ScoresTheArithmeticRelationByShotCount) {
  const auto benchmark = test::value(ModularMultiplier::create({
      .multiplier = "011",
      .modulus = "101",
      .multiplicand = "+++",
      .control = '+',
  }));
  EXPECT_EQ(test::value(benchmark.evaluate({{"10010011", 3}, {"10010010", 1}}))
                .successProbability,
            0.75);
  EXPECT_EQ(test::value(benchmark.evaluate({{"01110001", 7}, {"10010011", 0}}))
                .successProbability,
            0.);
  EXPECT_EQ(test::errorKind(benchmark.evaluate({})),
            mqt::bench::Error::Kind::InvalidArgument);
  EXPECT_EQ(test::errorKind(benchmark.evaluate({{"00000000", 0}})),
            mqt::bench::Error::Kind::InvalidArgument);
}

TEST(ModularMultiplier, SeparatesRelationSuccessFromDistributionFit) {
  const auto benchmark = test::value(ModularMultiplier::create({
      .multiplier = std::string(19, '0') + "1",
      .modulus = std::string(20, '1'),
      .multiplicand = std::string(20, '+'),
      .control = '+',
  }));
  const auto result =
      test::value(benchmark.evaluate({{std::string(42, '0'), 16'384}}));
  EXPECT_EQ(result.successProbability, 1.);
  EXPECT_DOUBLE_EQ(result.totalVariationDistance, 1. - std::ldexp(1., -21));
}

TEST(ModularMultiplier, SupportsNonCoprimeInputsAndMultiplicandsAtLeastN) {
  const auto nonCoprime = test::value(ModularMultiplier::create({
      .multiplier = "010",
      .modulus = "100",
      .multiplicand = "+++",
      .control = '+',
  }));
  EXPECT_DOUBLE_EQ(test::value(nonCoprime.probability("10110010")), 1. / 16.);

  const auto benchmark = test::value(ModularMultiplier::create({
      .multiplier = "011",
      .modulus = "101",
      .multiplicand = "+++",
      .control = '+',
  }));
  EXPECT_DOUBLE_EQ(test::value(benchmark.probability("11110001")), 1. / 16.);
}

TEST(ModularMultiplier, KeepsTheLargestReferenceWeightRepresentable) {
  constexpr auto width = ModularMultiplierOptions::MAX_BITS;
  const auto multiplier = std::string(width - 1U, '1') + "0";
  const auto modulus = std::string(width, '1');
  const auto benchmark = test::value(ModularMultiplier::create({
      .multiplier = multiplier,
      .modulus = modulus,
      .multiplicand = std::string(width, '+'),
      .control = '+',
  }));
  const auto multiplicand = std::string(width - 2U, '0') + "10";
  const auto accumulator = "0" + std::string(width - 2U, '1') + "01";
  const auto outcome = "1" + multiplicand + accumulator;
  const auto basis = test::value(ModularMultiplier::create({
      .multiplier = multiplier,
      .modulus = modulus,
      .multiplicand = multiplicand,
  }));
  EXPECT_EQ(basis.expectedResult(), outcome);
  EXPECT_GT(test::value(benchmark.probability(outcome)), 0.);
  EXPECT_EQ(test::value(benchmark.evaluate({{outcome, 1}})).successProbability,
            1.);
}

} // namespace
