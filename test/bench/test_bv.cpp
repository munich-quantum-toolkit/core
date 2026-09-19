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
#include "bench/Error.hpp"
#include "bench/Evaluation.hpp"
#include "bench/TestUtils.hpp"

#include "gtest/gtest.h"

#include <string>

namespace test = mqt::bench::test;

namespace {

using mqt::bench::BV;
using mqt::bench::BVMethod;
using mqt::bench::BVOptions;
using mqt::bench::Output;

TEST(BV, UsesTheStaticMethodByDefault) {
  const auto benchmark = test::value(BV::create({.hiddenBitstring = "101"}));
  EXPECT_EQ(benchmark.options().method, BVMethod::Static);
  EXPECT_EQ(benchmark.output(), (Output{"result", 3}));
}

TEST(BV, ValidatesTheConfiguredInstance) {
  // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange)
  constexpr auto invalidMethod = static_cast<BVMethod>(2);
  EXPECT_EQ(test::errorKind(BV::create({.hiddenBitstring = ""})),
            mqt::bench::Error::Kind::InvalidArgument);
  EXPECT_EQ(test::errorKind(BV::create({.hiddenBitstring = "10x"})),
            mqt::bench::Error::Kind::InvalidArgument);
  EXPECT_EQ(test::errorKind(BV::create({.hiddenBitstring = std::string(
                                            BVOptions::MAX_BITS + 1, '0')})),
            mqt::bench::Error::Kind::InvalidArgument);
  EXPECT_EQ(test::errorKind(
                BV::create({.hiddenBitstring = "1", .method = invalidMethod})),
            mqt::bench::Error::Kind::InvalidArgument);
}

TEST(BV, GivesTheHiddenBitstringAsASelectedOutcome) {
  for (const auto method : {BVMethod::Static, BVMethod::Dynamic}) {
    const auto benchmark =
        test::value(BV::create({.hiddenBitstring = "101", .method = method}));
    EXPECT_DOUBLE_EQ(test::value(benchmark.probability("101")), 1.);
    EXPECT_DOUBLE_EQ(test::value(benchmark.probability("011")), 0.);

    const auto evaluation =
        test::value(benchmark.evaluate({{"101", 80}, {"011", 20}}));
    EXPECT_DOUBLE_EQ(evaluation.totalVariationDistance, 0.2);
    EXPECT_DOUBLE_EQ(evaluation.squaredHellingerFidelity, 0.8);
    ASSERT_TRUE(evaluation.successProbability);
    EXPECT_DOUBLE_EQ(*evaluation.successProbability, 0.8);
  }
}

} // namespace
