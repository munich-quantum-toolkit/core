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
#include "bench/MagicStateDistillation.hpp"

#include "support/Diagnostics.hpp"
#include "support/TestSupport.hpp"

#include "gtest/gtest.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>

namespace mqt::bench {

TEST(MagicStateDistillation, ValidatesLevelsAndTwoBitReference) {
  EXPECT_EQ(
      ::mqt::test::value(MagicStateDistillation::create()).options().levels,
      1U);
  const auto maxLevels =
      static_cast<size_t>(std::numeric_limits<int64_t>::max()) / 5;
  for (const size_t levels :
       {size_t{1}, size_t{2}, size_t{3}, size_t{4}, size_t{8}, maxLevels}) {
    const auto benchmark =
        ::mqt::test::value(MagicStateDistillation::create({.levels = levels}));
    EXPECT_EQ(benchmark.output(), (Output{"result", 2}));
    EXPECT_DOUBLE_EQ(::mqt::test::value(benchmark.probability("00")), 1.);
    for (const auto* outcome : {"01", "10", "11"}) {
      EXPECT_DOUBLE_EQ(::mqt::test::value(benchmark.probability(outcome)), 0.);
    }
  }
  for (const size_t levels :
       {size_t{0}, maxLevels + 1, std::numeric_limits<size_t>::max()}) {
    EXPECT_EQ(::mqt::test::errorKind([&] {
                return MagicStateDistillation::create({.levels = levels});
              }),
              ::mqt::ErrorCategory::InvalidArgument);
  }
  const auto benchmark = ::mqt::test::value(MagicStateDistillation::create());
  for (const auto* outcome : {"", "0", "000", "0x"}) {
    EXPECT_EQ(
        ::mqt::test::errorKind([&] { return benchmark.probability(outcome); }),
        ::mqt::ErrorCategory::InvalidArgument);
    EXPECT_EQ(::mqt::test::errorKind(
                  [&] { return benchmark.evaluate({{outcome, 1}}); }),
              ::mqt::ErrorCategory::InvalidArgument);
  }
  EXPECT_EQ(::mqt::test::errorKind([&] { return benchmark.evaluate({}); }),
            ::mqt::ErrorCategory::InvalidArgument);
  EXPECT_EQ(
      ::mqt::test::errorKind([&] { return benchmark.evaluate({{"00", 0}}); }),
      ::mqt::ErrorCategory::InvalidArgument);
}

TEST(MagicStateDistillation, EvaluatesAcceptanceAndRootStateTogether) {
  const auto benchmark = ::mqt::test::value(MagicStateDistillation::create());
  const auto exact = ::mqt::test::value(benchmark.evaluate({{"00", 256}}));
  EXPECT_DOUBLE_EQ(exact.totalVariationDistance, 0.);
  EXPECT_DOUBLE_EQ(exact.squaredHellingerFidelity, 1.);
  EXPECT_EQ(exact.successProbability, 1.);
  const auto mixed = ::mqt::test::value(
      benchmark.evaluate({{"00", 5}, {"01", 1}, {"10", 1}, {"11", 1}}));
  EXPECT_DOUBLE_EQ(mixed.totalVariationDistance, 0.375);
  EXPECT_DOUBLE_EQ(mixed.squaredHellingerFidelity, 0.625);
  EXPECT_EQ(mixed.successProbability, 0.625);
}

TEST(MagicStateDistillation, RoundTripsJSON) {
  const auto defaults =
      ::mqt::test::value(magicStateDistillationFromInstanceSpecificationJSON(
          R"({"schema_version":1,"benchmark":"magic-state-distillation","parameters":{}})"));
  EXPECT_EQ(
      toInstanceSpecificationJSON(defaults),
      R"({"benchmark":"magic-state-distillation","parameters":{"levels":1},"schema_version":1})");
  for (const size_t levels : {1U, 2U, 3U, 4U, 8U}) {
    const auto benchmark =
        ::mqt::test::value(MagicStateDistillation::create({.levels = levels}));
    const auto instance = toInstanceSpecificationJSON(benchmark);
    EXPECT_EQ(
        toInstanceSpecificationJSON(::mqt::test::value(
            magicStateDistillationFromInstanceSpecificationJSON(instance))),
        instance);
    const auto manifest = toManifestJSON(benchmark);
    EXPECT_EQ(toManifestJSON(::mqt::test::value(
                  magicStateDistillationFromManifestJSON(manifest))),
              manifest);
  }
}

TEST(MagicStateDistillation, UsesSemanticCaseIds) {
  const auto defaults =
      ::mqt::test::value(magicStateDistillationFromInstanceSpecificationJSON(
          R"({"schema_version":1,"benchmark":"magic-state-distillation","parameters":{}})"));
  EXPECT_EQ(caseId(defaults),
            caseId(::mqt::test::value(
                MagicStateDistillation::create({.levels = 1}))));
  for (const size_t levels : {2U, 3U, 4U, 8U}) {
    EXPECT_NE(caseId(::mqt::test::value(
                  MagicStateDistillation::create({.levels = levels}))),
              caseId(defaults));
  }
}

TEST(MagicStateDistillation, DescribesJSONSchema) {
  EXPECT_NE(
      ::mqt::test::value(describeBenchmarkJSON("magic-state-distillation"))
          .find("\"maximum\":" +
                std::to_string(std::numeric_limits<int64_t>::max() / 5)),
      std::string::npos);
}

TEST(MagicStateDistillation, RejectsInvalidJSONParameters) {
  for (const auto* parameters : {
           R"({"levels":0})",
           R"({"levels":18446744073709551615})",
           R"({"levels":-1})",
           R"({"levels":1.0})",
           R"({"levels":true})",
           R"({"noise":0})",
       }) {
    EXPECT_EQ(
        ::mqt::test::errorKind([&] {
          return magicStateDistillationFromInstanceSpecificationJSON(
              std::string(
                  R"({"schema_version":1,"benchmark":"magic-state-distillation","parameters":)") +
              parameters + "}");
        }),
        ::mqt::ErrorCategory::InvalidArgument);
  }
}

TEST(MagicStateDistillation, EvaluatesCountsFromJSON) {
  const auto evaluation = ::mqt::test::value(evaluateJSON(
      toManifestJSON(::mqt::test::value(MagicStateDistillation::create())),
      R"({"schema_version":1,"counts":{"00":256}})"));
  EXPECT_NE(evaluation.find(R"("success_probability":1.0)"), std::string::npos);
}

} // namespace mqt::bench
