/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "bench/QFTAdder.hpp"

#include "support/Diagnostics.hpp"
#include "support/TestSupport.hpp"

#include "gtest/gtest.h"

#include <string>

namespace {

using mqt::bench::QFTAdder;
using mqt::bench::QFTAdderMethod;
using mqt::bench::QFTAdderOptions;
using mqt::bench::QFTAdderOverflow;

TEST(QFTAdder, PreservesConfiguredOperandsAndOverflow) {
  for (const auto method :
       {QFTAdderMethod::Register, QFTAdderMethod::Constant}) {
    for (const auto overflow :
         {QFTAdderOverflow::Wrap, QFTAdderOverflow::Carry}) {
      const auto benchmark = ::mqt::test::value(QFTAdder::create({
          .addend = "011",
          .accumulator = "110",
          .method = method,
          .overflow = overflow,
      }));
      const auto* const sum =
          overflow == QFTAdderOverflow::Carry ? "1001" : "001";
      const auto expected =
          (method == QFTAdderMethod::Register ? std::string{"011"}
                                              : std::string{}) +
          sum;
      EXPECT_EQ(benchmark.options().addend, "011");
      EXPECT_EQ(benchmark.options().accumulator, "110");
      EXPECT_EQ(benchmark.output().width, expected.size());
      EXPECT_EQ(benchmark.expectedResult(), expected);
      EXPECT_DOUBLE_EQ(::mqt::test::value(benchmark.probability(expected)), 1.);
      EXPECT_EQ(::mqt::test::value(benchmark.evaluate({{expected, 16}}))
                    .successProbability,
                1.);
    }
  }
}

TEST(QFTAdder, KeepsLeadingZerosAndRejectsUnsupportedInputs) {
  const auto zero = ::mqt::test::value(QFTAdder::create({
      .addend = "000",
      .accumulator = "000",
      .method = QFTAdderMethod::Constant,
      .overflow = QFTAdderOverflow::Carry,
  }));
  EXPECT_EQ(zero.expectedResult(), "0000");
  for (const auto& options : {
           QFTAdderOptions{.addend = "", .accumulator = ""},
           QFTAdderOptions{.addend = "01", .accumulator = "1"},
           QFTAdderOptions{.addend = "x", .accumulator = "0"},
           QFTAdderOptions{.addend = "1", .accumulator = "+"},
           QFTAdderOptions{
               .addend = "+",
               .accumulator = "0",
               .method = QFTAdderMethod::Constant,
           },
       }) {
    EXPECT_EQ(::mqt::test::errorKind([&] { return QFTAdder::create(options); }),
              ::mqt::ErrorCategory::InvalidArgument);
  }
  EXPECT_EQ(::mqt::test::errorKind([&] { return zero.probability("000"); }),
            ::mqt::ErrorCategory::InvalidArgument);
  EXPECT_EQ(::mqt::test::errorKind([&] { return zero.probability("000x"); }),
            ::mqt::ErrorCategory::InvalidArgument);
}

TEST(QFTAdder, ScoresTheCorrelatedSuperposition) {
  const auto benchmark = ::mqt::test::value(
      QFTAdder::create({.addend = "1+0", .accumulator = "001"}));
  EXPECT_FALSE(benchmark.expectedResult());
  EXPECT_DOUBLE_EQ(::mqt::test::value(benchmark.probability("100101")), 0.5);
  EXPECT_DOUBLE_EQ(::mqt::test::value(benchmark.probability("110111")), 0.5);
  EXPECT_DOUBLE_EQ(::mqt::test::value(benchmark.probability("000001")), 0.);
  EXPECT_DOUBLE_EQ(::mqt::test::value(benchmark.probability("100100")), 0.);
  const auto exact =
      ::mqt::test::value(benchmark.evaluate({{"100101", 8}, {"110111", 8}}));
  EXPECT_DOUBLE_EQ(exact.totalVariationDistance, 0.);
  EXPECT_DOUBLE_EQ(exact.squaredHellingerFidelity, 1.);
  EXPECT_FALSE(exact.successProbability);
  const auto biased = ::mqt::test::value(benchmark.evaluate({{"100101", 16}}));
  EXPECT_DOUBLE_EQ(biased.totalVariationDistance, 0.5);
  EXPECT_DOUBLE_EQ(biased.squaredHellingerFidelity, 0.5);
}

TEST(QFTAdder, BoundsTheSumWidthAndKeepsReferenceWeightsRepresentable) {
  const auto width = QFTAdderOptions::MAX_QUBITS;
  const auto accumulator = std::string(width - 1, '0') + "1";
  const auto maximum = ::mqt::test::value(QFTAdder::create(
      {.addend = std::string(width, '+'), .accumulator = accumulator}));
  EXPECT_GT(::mqt::test::value(maximum.probability(std::string(width, '1') +
                                                   std::string(width, '0'))),
            0.);
  EXPECT_EQ(::mqt::test::errorKind([&] {
              return QFTAdder::create({.addend = std::string(width, '1'),
                                       .accumulator = accumulator,
                                       .overflow = QFTAdderOverflow::Carry});
            }),
            ::mqt::ErrorCategory::InvalidArgument);
  EXPECT_NO_THROW(static_cast<void>(::mqt::test::value(
      QFTAdder::create({.addend = std::string(width - 1, '1'),
                        .accumulator = accumulator.substr(1),
                        .overflow = QFTAdderOverflow::Carry}))));
  EXPECT_EQ(::mqt::test::errorKind([&] {
              return QFTAdder::create(
                  {.addend = std::string(width + 1, '0'),
                   .accumulator = std::string(width + 1, '0')});
            }),
            ::mqt::ErrorCategory::InvalidArgument);
}

} // namespace
