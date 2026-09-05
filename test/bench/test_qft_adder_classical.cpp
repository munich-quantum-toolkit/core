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
#include "bench/QFTAdderClassical.hpp"

#include <gtest/gtest.h>

#include <stdexcept>
#include <string>

namespace {

using mqt::bench::Output;
using mqt::bench::QFTAdderClassical;
using mqt::bench::QFTAdderClassicalOptions;

TEST(QFTAdderClassical, StoresTheAddendAndResult) {
  const QFTAdderClassical benchmark{{.addend = "00101"}};
  EXPECT_EQ(benchmark.options().addend, "00101");
  EXPECT_EQ(benchmark.output(), (Output{"result", 6}));
  EXPECT_EQ(benchmark.expectedResult(), "000110");
}

TEST(QFTAdderClassical, ValidatesTheConfiguredInstance) {
  const auto maximum =
      std::string(QFTAdderClassicalOptions::MAX_ADDEND_BITS, '1');
  const auto tooLong =
      std::string(QFTAdderClassicalOptions::MAX_ADDEND_BITS + 1U, '0');
  const QFTAdderClassical maximumBenchmark{{.addend = maximum}};
  EXPECT_THROW(static_cast<void>(QFTAdderClassical{{.addend = ""}}),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(QFTAdderClassical{{.addend = "10x"}}),
               std::invalid_argument);
  EXPECT_EQ(maximumBenchmark.expectedResult(),
            "1" + std::string(QFTAdderClassicalOptions::MAX_ADDEND_BITS, '0'));
  EXPECT_THROW(static_cast<void>(QFTAdderClassical{{.addend = tooLong}}),
               std::invalid_argument);
}

TEST(QFTAdderClassical, AddsOneWithoutTruncatingOverflow) {
  const QFTAdderClassical zero{{.addend = "0"}};
  const QFTAdderClassical one{{.addend = "1"}};
  const QFTAdderClassical leadingZeros{{.addend = "001"}};
  const QFTAdderClassical five{{.addend = "101"}};
  const QFTAdderClassical six{{.addend = "110"}};
  const QFTAdderClassical seven{{.addend = "111"}};

  EXPECT_DOUBLE_EQ(zero.probability("01"), 1.);
  EXPECT_DOUBLE_EQ(one.probability("10"), 1.);
  EXPECT_DOUBLE_EQ(leadingZeros.probability("0010"), 1.);
  EXPECT_DOUBLE_EQ(five.probability("0110"), 1.);
  EXPECT_DOUBLE_EQ(six.probability("0111"), 1.);
  EXPECT_DOUBLE_EQ(seven.probability("1000"), 1.);
  EXPECT_DOUBLE_EQ(seven.probability("0111"), 0.);
  EXPECT_THROW(static_cast<void>(seven.probability("000")),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(seven.probability("000x")),
               std::invalid_argument);
}

TEST(QFTAdderClassical, EvaluatesTheDeterministicResult) {
  const QFTAdderClassical benchmark{{.addend = "101"}};
  const auto evaluation = benchmark.evaluate({{"0110", 80}, {"0101", 20}});
  EXPECT_DOUBLE_EQ(evaluation.totalVariationDistance, 0.2);
  EXPECT_DOUBLE_EQ(evaluation.squaredHellingerFidelity, 0.8);
  ASSERT_TRUE(evaluation.successProbability);
  EXPECT_DOUBLE_EQ(*evaluation.successProbability, 0.8);
}

} // namespace
