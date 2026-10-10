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
#include "bench/QPE.hpp"

#include "JSONTestUtils.hpp"
#include "support/Diagnostics.hpp"
#include "support/TestSupport.hpp"

#include "gtest/gtest.h"

#include <cstddef>
#include <numbers>
#include <string>

namespace mqt::bench {

using test::expectInvalidJSON;

TEST(Phase, NormalizesTurns) {
  EXPECT_EQ(::mqt::test::value(Phase::create(10, 8)),
            ::mqt::test::value(Phase::create(1, 4)));
  EXPECT_EQ(::mqt::test::value(Phase::create(9, 8)),
            ::mqt::test::value(Phase::create(1, 8)));
  EXPECT_EQ(::mqt::test::value(Phase::create(0, 42)),
            ::mqt::test::value(Phase::create(0, 1)));
  EXPECT_EQ(::mqt::test::errorKind([&] { return Phase::create(1, 0); }),
            ::mqt::ErrorCategory::InvalidArgument);
}

TEST(QPE, UsesDocumentedDefaults) {
  const auto qpe = ::mqt::test::value(QPE::create(
      {.precision = 3, .phase = ::mqt::test::value(Phase::create(3, 8))}));
  EXPECT_EQ(qpe.options().method, QPEMethod::Standard);
  EXPECT_EQ(qpe.output(), (Output{"result", 3}));
}

TEST(QPE, RejectsUnsupportedPrecision) {
  EXPECT_EQ(
      ::mqt::test::errorKind([&] {
        return QPE::create(
            {.precision = 0, .phase = ::mqt::test::value(Phase::create(0, 1))});
      }),
      ::mqt::ErrorCategory::InvalidArgument);
  EXPECT_EQ(::mqt::test::errorKind([&] {
              return QPE::create(
                  {.precision = mqt::bench::QPEOptions::MAX_PRECISION + 1,
                   .phase = ::mqt::test::value(Phase::create(0, 1))});
            }),
            ::mqt::ErrorCategory::InvalidArgument);
}

TEST(QPE, RejectsAnUnknownMethod) {
  // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange)
  constexpr auto invalidMethod = static_cast<QPEMethod>(2);
  EXPECT_EQ(::mqt::test::errorKind([&] {
              return QPE::create(
                  {.precision = 2,
                   .phase = ::mqt::test::value(Phase::create(0, 1)),
                   .method = invalidMethod});
            }),
            ::mqt::ErrorCategory::InvalidArgument);
}

TEST(QPE, GivesAnExactDistribution) {
  const auto qpe = ::mqt::test::value(QPE::create(
      {.precision = 3, .phase = ::mqt::test::value(Phase::create(3, 8))}));
  EXPECT_DOUBLE_EQ(::mqt::test::value(qpe.probability("011")), 1.);
  EXPECT_DOUBLE_EQ(::mqt::test::value(qpe.probability("010")), 0.);
  EXPECT_DOUBLE_EQ(::mqt::test::value(qpe.probability("111")), 0.);

  const auto evaluation = ::mqt::test::value(qpe.evaluate({{"011", 100}}));
  EXPECT_DOUBLE_EQ(evaluation.totalVariationDistance, 0.);
  EXPECT_DOUBLE_EQ(evaluation.squaredHellingerFidelity, 1.);
  EXPECT_FALSE(evaluation.successProbability.has_value());
}

TEST(QPE, GivesTheInexactDistribution) {
  const auto qpe = ::mqt::test::value(QPE::create(
      {.precision = 2, .phase = ::mqt::test::value(Phase::create(1, 8))}));
  const auto high = (2. + std::numbers::sqrt2) / 8.;
  const auto low = (2. - std::numbers::sqrt2) / 8.;
  EXPECT_NEAR(::mqt::test::value(qpe.probability("00")), high, 1e-15);
  EXPECT_NEAR(::mqt::test::value(qpe.probability("01")), high, 1e-15);
  EXPECT_NEAR(::mqt::test::value(qpe.probability("10")), low, 1e-15);
  EXPECT_NEAR(::mqt::test::value(qpe.probability("11")), low, 1e-15);
}

TEST(QPE, WrapsTheDistributionAtOneTurn) {
  const auto qpe = ::mqt::test::value(QPE::create(
      {.precision = 2, .phase = ::mqt::test::value(Phase::create(7, 8))}));
  const auto high = (2. + std::numbers::sqrt2) / 8.;
  const auto low = (2. - std::numbers::sqrt2) / 8.;
  EXPECT_NEAR(::mqt::test::value(qpe.probability("00")), high, 1e-15);
  EXPECT_NEAR(::mqt::test::value(qpe.probability("11")), high, 1e-15);
  EXPECT_NEAR(::mqt::test::value(qpe.probability("01")), low, 1e-15);
  EXPECT_NEAR(::mqt::test::value(qpe.probability("10")), low, 1e-15);
}

TEST(QPE, UsesTheNegativeHalfTurnRepresentative) {
  const auto qpe = ::mqt::test::value(QPE::create(
      {.precision = 1, .phase = ::mqt::test::value(Phase::create(7, 8))}));
  EXPECT_NEAR(::mqt::test::value(qpe.probability("0")),
              (2. + std::numbers::sqrt2) / 4., 1e-15);
  EXPECT_NEAR(::mqt::test::value(qpe.probability("1")),
              (2. - std::numbers::sqrt2) / 4., 1e-15);
}

TEST(QPE, SupportsArbitraryWidthOutcomes) {
  constexpr size_t precision = 1025;
  const auto qpe = ::mqt::test::value(QPE::create({
      .precision = precision,
      .phase = ::mqt::test::value(Phase::create(1, 3)),
      .method = QPEMethod::Iterative,
  }));
  auto lower = std::string{};
  lower.reserve(precision);
  for (size_t index = 0; index < precision; ++index) {
    lower.push_back(index % 2 == 0 ? '0' : '1');
  }
  auto upper = lower;
  upper.back() = '1';

  const auto pi = std::numbers::pi;
  EXPECT_NEAR(::mqt::test::value(qpe.probability(lower)), 27. / (16. * pi * pi),
              1e-15);
  EXPECT_NEAR(::mqt::test::value(qpe.probability(upper)), 27. / (4. * pi * pi),
              1e-15);
}

TEST(QPE, RoundTripsJSON) {
  const auto parsed = ::mqt::test::value(qpeFromInstanceSpecificationJSON(
      R"({"schema_version":1,"benchmark":"qpe","parameters":{"precision":4,"phase":{"numerator":10,"denominator":8},"method":"iterative"}})"));
  EXPECT_EQ(parsed.options().phase, ::mqt::test::value(Phase::create(1, 4)));
  EXPECT_EQ(parsed.options().method, QPEMethod::Iterative);
  EXPECT_EQ(
      toInstanceSpecificationJSON(parsed),
      R"({"benchmark":"qpe","parameters":{"method":"iterative","phase":{"denominator":4,"numerator":1},"precision":4},"schema_version":1})");

  const auto benchmark = ::mqt::test::value(QPE::create({
      .precision = 5,
      .phase = ::mqt::test::value(Phase::create(1, 3)),
      .method = QPEMethod::Iterative,
  }));
  const auto manifest = toManifestJSON(benchmark);
  EXPECT_EQ(toManifestJSON(::mqt::test::value(qpeFromManifestJSON(manifest))),
            manifest);
  EXPECT_EQ(::mqt::test::value(benchmarkIdFromManifestJSON(manifest)), "qpe");
  EXPECT_EQ(manifest.find("0.333"), std::string::npos);
}

TEST(QPE, DescribesJSONSchema) {
  EXPECT_NE(
      ::mqt::test::value(describeBenchmarkJSON("qpe")).find("\"iterative\""),
      std::string::npos);
}

TEST(QPE, RejectsInvalidJSONParameters) {
  expectInvalidJSON(
      [] {
        return qpeFromInstanceSpecificationJSON(
            R"({"schema_version":1,"benchmark":"qpe","parameters":{"precision":2,"phase":{"numerator":9007199254740993.0,"denominator":9007199254740994}}})");
      },
      "encoded as an integer");
  expectInvalidJSON(
      [] {
        return qpeFromInstanceSpecificationJSON(
            R"({"schema_version":1,"benchmark":"qpe","parameters":{"precision":18446744073709551615,"phase":{"numerator":1,"denominator":4}}})");
      },
      "between 1 and 1000000");
  expectInvalidJSON(
      [] {
        return qpeFromInstanceSpecificationJSON(
            R"({"schema_version":1,"benchmark":"qpe","parameters":{"precision":2,"phase":{"numerator":1,"denominator":0}}})");
      },
      "denominator must not be zero");
}

} // namespace mqt::bench
