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

#include "gtest/gtest.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace mqt::bench {

TEST(MagicStateDistillation, ValidatesLevelsAndTwoBitReference) {
  EXPECT_EQ(MagicStateDistillation{}.options().levels, 1U);
  const auto maxLevels =
      static_cast<size_t>(std::numeric_limits<int64_t>::max()) / 5;
  for (const size_t levels :
       {size_t{1}, size_t{2}, size_t{3}, size_t{4}, size_t{8}, maxLevels}) {
    const MagicStateDistillation benchmark({.levels = levels});
    EXPECT_EQ(benchmark.output(), (Output{"result", 2}));
    EXPECT_DOUBLE_EQ(benchmark.probability("00"), 1.);
    for (const auto* outcome : {"01", "10", "11"}) {
      EXPECT_DOUBLE_EQ(benchmark.probability(outcome), 0.);
    }
  }
  for (const size_t levels :
       {size_t{0}, maxLevels + 1, std::numeric_limits<size_t>::max()}) {
    EXPECT_THROW(MagicStateDistillation({.levels = levels}),
                 std::invalid_argument);
  }
  const MagicStateDistillation benchmark;
  for (const auto* outcome : {"", "0", "000", "0x"}) {
    EXPECT_THROW(static_cast<void>(benchmark.probability(outcome)),
                 std::invalid_argument);
    EXPECT_THROW(static_cast<void>(benchmark.evaluate({{outcome, 1}})),
                 std::invalid_argument);
  }
  EXPECT_THROW(static_cast<void>(benchmark.evaluate({})),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(benchmark.evaluate({{"00", 0}})),
               std::invalid_argument);
}

TEST(MagicStateDistillation, EvaluatesAcceptanceAndRootStateTogether) {
  const MagicStateDistillation benchmark;
  const auto exact = benchmark.evaluate({{"00", 256}});
  EXPECT_DOUBLE_EQ(exact.totalVariationDistance, 0.);
  EXPECT_DOUBLE_EQ(exact.squaredHellingerFidelity, 1.);
  EXPECT_EQ(exact.successProbability, 1.);
  const auto mixed =
      benchmark.evaluate({{"00", 5}, {"01", 1}, {"10", 1}, {"11", 1}});
  EXPECT_DOUBLE_EQ(mixed.totalVariationDistance, 0.375);
  EXPECT_DOUBLE_EQ(mixed.squaredHellingerFidelity, 0.625);
  EXPECT_EQ(mixed.successProbability, 0.625);
}

TEST(MagicStateDistillation, RoundTripsJSON) {
  const auto defaults = magicStateDistillationFromInstanceSpecificationJSON(
      R"({"schema_version":1,"benchmark":"magic-state-distillation","parameters":{}})");
  EXPECT_EQ(
      toInstanceSpecificationJSON(defaults),
      R"({"benchmark":"magic-state-distillation","parameters":{"levels":1},"schema_version":1})");
  for (const size_t levels : {1U, 2U, 3U, 4U, 8U}) {
    const MagicStateDistillation benchmark({.levels = levels});
    const auto instance = toInstanceSpecificationJSON(benchmark);
    EXPECT_EQ(
        toInstanceSpecificationJSON(
            magicStateDistillationFromInstanceSpecificationJSON(instance)),
        instance);
    const auto manifest = toManifestJSON(benchmark);
    EXPECT_EQ(toManifestJSON(magicStateDistillationFromManifestJSON(manifest)),
              manifest);
  }
}

TEST(MagicStateDistillation, UsesSemanticCaseIds) {
  const auto defaults = magicStateDistillationFromInstanceSpecificationJSON(
      R"({"schema_version":1,"benchmark":"magic-state-distillation","parameters":{}})");
  EXPECT_EQ(caseId(defaults), caseId(MagicStateDistillation({.levels = 1})));
  for (const size_t levels : {2U, 3U, 4U, 8U}) {
    EXPECT_NE(caseId(MagicStateDistillation({.levels = levels})),
              caseId(defaults));
  }
}

TEST(MagicStateDistillation, DescribesJSONSchema) {
  EXPECT_NE(describeBenchmarkJSON("magic-state-distillation")
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
    EXPECT_THROW(
        static_cast<void>(magicStateDistillationFromInstanceSpecificationJSON(
            std::string(
                R"({"schema_version":1,"benchmark":"magic-state-distillation","parameters":)") +
            parameters + "}")),
        std::invalid_argument);
  }
}

TEST(MagicStateDistillation, EvaluatesCountsFromJSON) {
  const auto evaluation =
      evaluateJSON(toManifestJSON(MagicStateDistillation{}),
                   R"({"schema_version":1,"counts":{"00":256}})");
  EXPECT_NE(evaluation.find(R"("success_probability":1.0)"), std::string::npos);
}

} // namespace mqt::bench
