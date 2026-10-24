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

#include "JSONTestUtils.hpp"

#include "gtest/gtest.h"

#include "llvm/Support/Error.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <system_error>

namespace mqt::bench {

TEST(MagicStateDistillation, ValidatesLevelsAndTwoBitReference) {
  EXPECT_EQ(llvm::cantFail(MagicStateDistillation::create()).options().levels,
            1U);
  const auto maxLevels =
      static_cast<size_t>(std::numeric_limits<int64_t>::max()) / 5;
  for (const size_t levels : {size_t{1}, size_t{8}, maxLevels}) {
    const auto benchmark =
        llvm::cantFail(MagicStateDistillation::create({.levels = levels}));
    EXPECT_EQ(benchmark.options().levels, levels);
  }
  for (const size_t levels :
       {size_t{0}, maxLevels + 1, std::numeric_limits<size_t>::max()}) {
    test::expectError(MagicStateDistillation::create({.levels = levels}),
                      std::errc::invalid_argument);
  }
  const auto benchmark = llvm::cantFail(MagicStateDistillation::create());
  EXPECT_EQ(benchmark.output(), (Output{"result", 2}));
  EXPECT_DOUBLE_EQ(llvm::cantFail(benchmark.probability("00")), 1.);
  for (const auto* outcome : {"01", "10", "11"}) {
    EXPECT_DOUBLE_EQ(llvm::cantFail(benchmark.probability(outcome)), 0.);
  }
  for (const auto* outcome : {"", "0", "000", "0x"}) {
    test::expectError(benchmark.probability(outcome),
                      std::errc::invalid_argument);
    test::expectError(benchmark.evaluate({{outcome, 1}}),
                      std::errc::invalid_argument);
  }
  test::expectError(benchmark.evaluate({}), std::errc::invalid_argument);
  test::expectError(benchmark.evaluate({{"00", 0}}),
                    std::errc::invalid_argument);
}

TEST(MagicStateDistillation, EvaluatesAcceptanceAndRootStateTogether) {
  const auto benchmark = llvm::cantFail(MagicStateDistillation::create());
  const auto exact = llvm::cantFail(benchmark.evaluate({{"00", 256}}));
  EXPECT_DOUBLE_EQ(exact.totalVariationDistance, 0.);
  EXPECT_DOUBLE_EQ(exact.squaredHellingerFidelity, 1.);
  EXPECT_EQ(exact.successProbability, 1.);
  const auto mixed = llvm::cantFail(
      benchmark.evaluate({{"00", 5}, {"01", 1}, {"10", 1}, {"11", 1}}));
  EXPECT_DOUBLE_EQ(mixed.totalVariationDistance, 0.375);
  EXPECT_DOUBLE_EQ(mixed.squaredHellingerFidelity, 0.625);
  EXPECT_EQ(mixed.successProbability, 0.625);
}

TEST(MagicStateDistillation, RoundTripsJSON) {
  const auto defaults =
      llvm::cantFail(magicStateDistillationFromInstanceSpecificationJSON(
          R"({"schema_version":1,"benchmark":"magic-state-distillation","parameters":{}})"));
  EXPECT_EQ(
      toInstanceSpecificationJSON(defaults),
      R"({"benchmark":"magic-state-distillation","parameters":{"levels":1},"schema_version":1})");
  for (const size_t levels : {2U, 8U}) {
    const auto benchmark =
        llvm::cantFail(MagicStateDistillation::create({.levels = levels}));
    const auto instance = toInstanceSpecificationJSON(benchmark);
    EXPECT_EQ(
        toInstanceSpecificationJSON(llvm::cantFail(
            magicStateDistillationFromInstanceSpecificationJSON(instance))),
        instance);
    const auto manifest = toManifestJSON(benchmark);
    EXPECT_EQ(toManifestJSON(llvm::cantFail(
                  magicStateDistillationFromManifestJSON(manifest))),
              manifest);
    EXPECT_NE(caseId(benchmark), caseId(defaults));
  }
  EXPECT_EQ(
      caseId(defaults),
      caseId(llvm::cantFail(MagicStateDistillation::create({.levels = 1}))));
}

TEST(MagicStateDistillation, DescribesJSONSchema) {
  EXPECT_NE(llvm::cantFail(describeBenchmarkJSON("magic-state-distillation"))
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
    test::expectInvalidJSON(
        [&] {
          return magicStateDistillationFromInstanceSpecificationJSON(
              std::string(
                  R"({"schema_version":1,"benchmark":"magic-state-distillation","parameters":)") +
              parameters + "}");
        },
        "");
  }
}

TEST(MagicStateDistillation, EvaluatesCountsFromJSON) {
  const auto evaluation = llvm::cantFail(evaluateJSON(
      toManifestJSON(llvm::cantFail(MagicStateDistillation::create())),
      R"({"schema_version":1,"counts":{"00":256}})"));
  EXPECT_NE(evaluation.find(R"("success_probability":1)"), std::string::npos);
}

} // namespace mqt::bench
