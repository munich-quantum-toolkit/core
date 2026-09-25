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
#include "bench/JSON.hpp"

#include "support/Diagnostics.hpp"
#include "support/TestSupport.hpp"

#include "gtest/gtest.h"

#include <string>

namespace mqt::bench {

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

TEST(BV, RoundTripsJSON) {
  const auto defaults = ::mqt::test::value(bvFromInstanceSpecificationJSON(
      R"({"schema_version":1,"benchmark":"bv","parameters":{"hidden_bitstring":"101"}})"));
  EXPECT_EQ(defaults.options().method, BVMethod::Static);
  EXPECT_EQ(
      toInstanceSpecificationJSON(defaults),
      R"({"benchmark":"bv","parameters":{"hidden_bitstring":"101","method":"static"},"schema_version":1})");

  const auto dynamic = ::mqt::test::value(
      BV::create({.hiddenBitstring = "101", .method = BVMethod::Dynamic}));
  const auto manifest = toManifestJSON(dynamic);
  EXPECT_EQ(toManifestJSON(::mqt::test::value(bvFromManifestJSON(manifest))),
            manifest);
  EXPECT_EQ(::mqt::test::value(benchmarkIdFromManifestJSON(manifest)), "bv");
}

TEST(BV, UsesSemanticCaseIds) {
  EXPECT_NE(caseId(::mqt::test::value(BV::create({.hiddenBitstring = "1"}))),
            caseId(::mqt::test::value(BV::create(
                {.hiddenBitstring = "1", .method = BVMethod::Dynamic}))));
}

TEST(BV, DescribesJSONSchema) {
  EXPECT_NE(::mqt::test::value(describeBenchmarkJSON("bv")).find("\"dynamic\""),
            std::string::npos);
}

TEST(BV, EvaluatesCountsFromJSON) {
  const auto evaluation = ::mqt::test::value(evaluateJSON(
      toManifestJSON(::mqt::test::value(BV::create({.hiddenBitstring = "11"}))),
      R"({"schema_version":1,"counts":{"11":8,"00":2}})"));
  EXPECT_NE(evaluation.find("\"success_probability\":0.8"), std::string::npos);
}

} // namespace mqt::bench
