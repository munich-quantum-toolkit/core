/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "bench/ControlledMultiplicationModuloN.hpp"
#include "bench/Evaluation.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

using mqt::bench::ControlledMultiplicationModuloN;
using mqt::bench::ControlledMultiplicationModuloNOptions;
using mqt::bench::Counts;
using mqt::bench::Output;

TEST(ControlledMultiplicationModuloN, StoresParametersAndOutput) {
  const ControlledMultiplicationModuloN benchmark{
      {.multiplier = "011", .modulus = "101"}};
  EXPECT_EQ(benchmark.options().multiplier, "011");
  EXPECT_EQ(benchmark.options().modulus, "101");
  EXPECT_EQ(benchmark.output(), (Output{"result", 8}));
}

TEST(ControlledMultiplicationModuloN, ValidatesTheConfiguredInstance) {
  const auto maximumMultiplier =
      std::string(ControlledMultiplicationModuloNOptions::MAX_BITS - 1U, '0') +
      "1";
  const auto maximumModulus =
      "1" +
      std::string(ControlledMultiplicationModuloNOptions::MAX_BITS - 1U, '0');
  EXPECT_NO_THROW(static_cast<void>(ControlledMultiplicationModuloN{
      {.multiplier = maximumMultiplier, .modulus = maximumModulus}}));

  EXPECT_THROW(static_cast<void>(ControlledMultiplicationModuloN{
                   {.multiplier = "0", .modulus = "1"}}),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(ControlledMultiplicationModuloN{
                   {.multiplier = "001", .modulus = "1000"}}),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(ControlledMultiplicationModuloN{
                   {.multiplier = "00x", .modulus = "101"}}),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(ControlledMultiplicationModuloN{
                   {.multiplier = "001", .modulus = "10x"}}),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(ControlledMultiplicationModuloN{
                   {.multiplier = "001", .modulus = "011"}}),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(ControlledMultiplicationModuloN{
                   {.multiplier = "000", .modulus = "101"}}),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(ControlledMultiplicationModuloN{
                   {.multiplier = "101", .modulus = "101"}}),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(ControlledMultiplicationModuloN{
                   {.multiplier = "110", .modulus = "101"}}),
               std::invalid_argument);

  const auto tooLongMultiplier =
      std::string(ControlledMultiplicationModuloNOptions::MAX_BITS, '0') + "1";
  const auto tooLongModulus =
      "1" + std::string(ControlledMultiplicationModuloNOptions::MAX_BITS, '0');
  EXPECT_THROW(
      static_cast<void>(ControlledMultiplicationModuloN{
          {.multiplier = tooLongMultiplier, .modulus = tooLongModulus}}),
      std::invalid_argument);
}

TEST(ControlledMultiplicationModuloN,
     GivesUniformWeightToTheExactControlledProducts) {
  const ControlledMultiplicationModuloN benchmark{
      {.multiplier = "011", .modulus = "101"}};
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
  EXPECT_FALSE(evaluation.successProbability);
}

TEST(ControlledMultiplicationModuloN,
     SupportsNonCoprimeInputsAndMultiplicandsAtLeastN) {
  const ControlledMultiplicationModuloN nonCoprime{
      {.multiplier = "010", .modulus = "100"}};
  EXPECT_DOUBLE_EQ(nonCoprime.probability("10110010"), 1. / 16.);

  const ControlledMultiplicationModuloN benchmark{
      {.multiplier = "011", .modulus = "101"}};
  EXPECT_DOUBLE_EQ(benchmark.probability("11110001"), 1. / 16.);
}

TEST(ControlledMultiplicationModuloN,
     KeepsTheLargestReferenceWeightRepresentable) {
  constexpr auto width = ControlledMultiplicationModuloNOptions::MAX_BITS;
  const auto multiplier = std::string(width - 1U, '0') + "1";
  const auto modulus = "1" + std::string(width - 1U, '0');
  const ControlledMultiplicationModuloN benchmark{
      {.multiplier = multiplier, .modulus = modulus}};
  const auto outcome =
      "1" + std::string(width, '1') + "00" + std::string(width - 1U, '1');
  EXPECT_GT(benchmark.probability(outcome), 0.);
}

} // namespace
