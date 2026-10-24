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

#include "JSONTestUtils.hpp"

#include "gtest/gtest.h"

#include "llvm/Support/Error.h"

#include <string>
#include <system_error>

namespace mqt::bench {

TEST(BV, UsesTheStaticMethodByDefault) {
  const auto benchmark = llvm::cantFail(BV::create({.hiddenBitstring = "101"}));
  EXPECT_EQ(benchmark.options().method, BVMethod::Static);
  EXPECT_EQ(benchmark.output(), (Output{"result", 3}));
}

TEST(BV, ValidatesTheConfiguredInstance) {
  // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange)
  constexpr auto invalidMethod = static_cast<BVMethod>(2);
  test::expectError(BV::create({.hiddenBitstring = ""}),
                    std::errc::invalid_argument);
  test::expectError(BV::create({.hiddenBitstring = "10x"}),
                    std::errc::invalid_argument);
  test::expectError(
      BV::create({
          .hiddenBitstring = std::string(BVOptions::MAX_BITS + 1, '0'),
      }),
      std::errc::invalid_argument);
  test::expectError(
      BV::create({.hiddenBitstring = "1", .method = invalidMethod}),
      std::errc::invalid_argument);
}

TEST(BV, GivesTheHiddenBitstringAsASelectedOutcome) {
  for (const auto method : {BVMethod::Static, BVMethod::Dynamic}) {
    const auto benchmark = llvm::cantFail(
        BV::create({.hiddenBitstring = "101", .method = method}));
    EXPECT_DOUBLE_EQ(llvm::cantFail(benchmark.probability("101")), 1.);
    EXPECT_DOUBLE_EQ(llvm::cantFail(benchmark.probability("011")), 0.);

    const auto evaluation =
        llvm::cantFail(benchmark.evaluate({{"101", 80}, {"011", 20}}));
    EXPECT_DOUBLE_EQ(evaluation.totalVariationDistance, 0.2);
    EXPECT_DOUBLE_EQ(evaluation.squaredHellingerFidelity, 0.8);
    ASSERT_TRUE(evaluation.successProbability);
    EXPECT_DOUBLE_EQ(*evaluation.successProbability, 0.8);
  }
}

TEST(BV, RoundTripsJSON) {
  const auto defaults = llvm::cantFail(bvFromInstanceSpecificationJSON(
      R"({"schema_version":1,"benchmark":"bv","parameters":{"hidden_bitstring":"101"}})"));
  EXPECT_EQ(defaults.options().method, BVMethod::Static);
  EXPECT_EQ(
      toInstanceSpecificationJSON(defaults),
      R"({"benchmark":"bv","parameters":{"hidden_bitstring":"101","method":"static"},"schema_version":1})");

  const auto dynamic = llvm::cantFail(
      BV::create({.hiddenBitstring = "101", .method = BVMethod::Dynamic}));
  const auto manifest = toManifestJSON(dynamic);
  EXPECT_EQ(toManifestJSON(llvm::cantFail(bvFromManifestJSON(manifest))),
            manifest);
  EXPECT_EQ(llvm::cantFail(benchmarkIdFromManifestJSON(manifest)), "bv");
}

TEST(BV, UsesSemanticCaseIds) {
  EXPECT_NE(caseId(llvm::cantFail(BV::create({.hiddenBitstring = "1"}))),
            caseId(llvm::cantFail(BV::create(
                {.hiddenBitstring = "1", .method = BVMethod::Dynamic}))));
}

TEST(BV, DescribesJSONSchema) {
  EXPECT_NE(llvm::cantFail(describeBenchmarkJSON("bv")).find("\"dynamic\""),
            std::string::npos);
}

TEST(BV, EvaluatesCountsFromJSON) {
  const auto evaluation = llvm::cantFail(evaluateJSON(
      toManifestJSON(llvm::cantFail(BV::create({.hiddenBitstring = "11"}))),
      R"({"schema_version":1,"counts":{"11":8,"00":2}})"));
  EXPECT_NE(evaluation.find("\"success_probability\":0.8"), std::string::npos);
}

} // namespace mqt::bench
