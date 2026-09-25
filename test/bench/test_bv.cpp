/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "bench/BV.hpp"
#include "bench/Evaluation.hpp"

#include "support/Diagnostics.hpp"
#include "support/TestSupport.hpp"

#include "gtest/gtest.h"

#include <string>

namespace {

using mqt::bench::BV;
using mqt::bench::BVMethod;
using mqt::bench::BVOptions;
using mqt::bench::Output;

TEST(BV, UsesTheStaticMethodByDefault) {
  const auto benchmark =
      ::mqt::test::value(BV::create({.hiddenBitstring = "101"}));
  EXPECT_EQ(benchmark.options().method, BVMethod::Static);
  EXPECT_EQ(benchmark.output(), (Output{"result", 3}));
}

TEST(BV, ValidatesTheConfiguredInstance) {
  // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange)
  constexpr auto invalidMethod = static_cast<BVMethod>(2);
  EXPECT_EQ(::mqt::test::errorKind(
                [&] { return BV::create({.hiddenBitstring = ""}); }),
            ::mqt::ErrorCategory::InvalidArgument);
  EXPECT_EQ(::mqt::test::errorKind(
                [&] { return BV::create({.hiddenBitstring = "10x"}); }),
            ::mqt::ErrorCategory::InvalidArgument);
  EXPECT_EQ(::mqt::test::errorKind([&] {
              return BV::create({.hiddenBitstring = std::string(
                                     BVOptions::MAX_BITS + 1, '0')});
            }),
            ::mqt::ErrorCategory::InvalidArgument);
  EXPECT_EQ(::mqt::test::errorKind([&] {
              return BV::create(
                  {.hiddenBitstring = "1", .method = invalidMethod});
            }),
            ::mqt::ErrorCategory::InvalidArgument);
}

TEST(BV, GivesTheHiddenBitstringAsASelectedOutcome) {
  for (const auto method : {BVMethod::Static, BVMethod::Dynamic}) {
    const auto benchmark = ::mqt::test::value(
        BV::create({.hiddenBitstring = "101", .method = method}));
    EXPECT_DOUBLE_EQ(::mqt::test::value(benchmark.probability("101")), 1.);
    EXPECT_DOUBLE_EQ(::mqt::test::value(benchmark.probability("011")), 0.);

    const auto evaluation =
        ::mqt::test::value(benchmark.evaluate({{"101", 80}, {"011", 20}}));
    EXPECT_DOUBLE_EQ(evaluation.totalVariationDistance, 0.2);
    EXPECT_DOUBLE_EQ(evaluation.squaredHellingerFidelity, 0.8);
    ASSERT_TRUE(evaluation.successProbability);
    EXPECT_DOUBLE_EQ(*evaluation.successProbability, 0.8);
  }
}

} // namespace
