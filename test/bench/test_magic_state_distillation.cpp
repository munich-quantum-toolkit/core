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
#include <limits>
#include <stdexcept>
#include <string>

namespace mqt::bench {

TEST(MagicStateDistillation, ValidatesLevelsAndTwoBitReference) {
  EXPECT_EQ(MagicStateDistillation{}.options().levels, 1U);
  for (const size_t levels : {1U, 2U, 3U, 4U}) {
    const MagicStateDistillation benchmark({.levels = levels});
    EXPECT_EQ(benchmark.output(), (Output{"result", 2}));
    EXPECT_DOUBLE_EQ(benchmark.probability("00"), 1.);
    for (const auto* outcome : {"01", "10", "11"}) {
      EXPECT_DOUBLE_EQ(benchmark.probability(outcome), 0.);
    }
  }
  for (const size_t levels :
       {size_t{0}, size_t{5}, std::numeric_limits<size_t>::max()}) {
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

TEST(MagicStateDistillation, RoundTripsStrictJSONAndSemanticCaseIds) {
  const auto defaults = magicStateDistillationFromInstanceSpecificationJSON(
      R"({"schema_version":1,"benchmark":"magic-state-distillation","parameters":{}})");
  EXPECT_EQ(caseId(defaults), caseId(MagicStateDistillation({.levels = 1})));
  EXPECT_EQ(
      toInstanceSpecificationJSON(defaults),
      R"({"benchmark":"magic-state-distillation","parameters":{"levels":1},"schema_version":1})");
  for (const size_t levels : {1U, 2U, 3U, 4U}) {
    const MagicStateDistillation benchmark({.levels = levels});
    EXPECT_EQ(caseId(magicStateDistillationFromInstanceSpecificationJSON(
                  toInstanceSpecificationJSON(benchmark))),
              caseId(benchmark));
    EXPECT_EQ(caseId(magicStateDistillationFromManifestJSON(
                  toManifestJSON(benchmark))),
              caseId(benchmark));
    if (levels != 1) {
      EXPECT_NE(caseId(benchmark), caseId(defaults));
    }
  }
  for (const auto* parameters : {
           R"({"levels":0})",
           R"({"levels":5})",
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
  EXPECT_NE(
      describeBenchmarkJSON("magic-state-distillation").find(R"("maximum":4)"),
      std::string::npos);
  const auto evaluation = evaluateJSON(
      toManifestJSON(defaults), R"({"schema_version":1,"counts":{"00":256}})");
  EXPECT_NE(evaluation.find(R"("success_probability":1.0)"), std::string::npos);
}

} /* namespace mqt::bench */
