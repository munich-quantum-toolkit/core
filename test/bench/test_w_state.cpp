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
#include "bench/WState.hpp"

#include "JSONTestUtils.hpp"

#include "gtest/gtest.h"

#include "llvm/Support/Error.h"

#include <bit>
#include <cstddef>
#include <limits>
#include <string>
#include <system_error>

namespace mqt::bench {

TEST(WState, AssignsUniformProbabilityToSingleExcitations) {
  for (const size_t qubits : {1U, 2U, 3U, 7U}) {
    const auto benchmark = llvm::cantFail(WState::create({.qubits = qubits}));
    EXPECT_EQ(benchmark.options().qubits, qubits);
    EXPECT_EQ(benchmark.output(), (Output{"result", qubits}));
    Counts uniform;
    for (size_t index = 0; index < (size_t{1} << qubits); ++index) {
      auto bits = std::string(qubits, '0');
      for (size_t bit = 0; bit < qubits; ++bit) {
        if ((index & (size_t{1} << bit)) != 0) {
          bits[qubits - bit - 1] = '1';
        }
      }
      const auto expected =
          std::popcount(index) == 1 ? 1. / static_cast<double>(qubits) : 0.;
      EXPECT_DOUBLE_EQ(llvm::cantFail(benchmark.probability(bits)), expected);
      if (expected != 0.) {
        uniform.emplace(bits, 10);
      }
    }
    const auto evaluation = llvm::cantFail(benchmark.evaluate(uniform));
    EXPECT_NEAR(evaluation.totalVariationDistance, 0., 1e-12);
    EXPECT_NEAR(evaluation.squaredHellingerFidelity, 1., 1e-12);
  }
  const auto partial = llvm::cantFail(
      llvm::cantFail(WState::create({.qubits = 2})).evaluate({{"01", 20}}));
  EXPECT_NEAR(partial.totalVariationDistance, 0.5, 1e-12);
  EXPECT_NEAR(partial.squaredHellingerFidelity, 0.5, 1e-12);
}

TEST(WState, RejectsInvalidParametersAndOutcomes) {
  test::expectError(WState::create({.qubits = 0}), std::errc::invalid_argument);
  test::expectError(
      WState::create({.qubits = std::numeric_limits<size_t>::max()}),
      std::errc::invalid_argument);
  const auto benchmark = llvm::cantFail(WState::create({.qubits = 2}));
  test::expectError(benchmark.probability("1"), std::errc::invalid_argument);
  test::expectError(benchmark.evaluate({{"0x", 1}}),
                    std::errc::invalid_argument);
}

TEST(WState, RoundTripsJSON) {
  const auto benchmark = llvm::cantFail(WState::create({.qubits = 2}));
  const auto instance = toInstanceSpecificationJSON(benchmark);
  EXPECT_EQ(toInstanceSpecificationJSON(
                llvm::cantFail(wStateFromInstanceSpecificationJSON(instance))),
            instance);
  const auto manifest = toManifestJSON(benchmark);
  EXPECT_EQ(toManifestJSON(llvm::cantFail(wStateFromManifestJSON(manifest))),
            manifest);
  EXPECT_NE(manifest.find(R"("model":"w_state")"), std::string::npos);
}

TEST(WState, UsesSemanticCaseIds) {
  EXPECT_NE(caseId(llvm::cantFail(WState::create({.qubits = 2}))),
            caseId(llvm::cantFail(WState::create({.qubits = 3}))));
}

TEST(WState, DescribesJSONSchema) {
  const auto schema = llvm::cantFail(describeBenchmarkJSON("w-state"));
  EXPECT_NE(schema.find(R"("minimum":1)"), std::string::npos);
  EXPECT_NE(schema.find(R"("required":["qubits"])"), std::string::npos);
}

TEST(WState, RejectsInvalidJSONParameters) {
  test::expectInvalidJSON(
      [&] {
        return wStateFromInstanceSpecificationJSON(
            R"({"schema_version":1,"benchmark":"w-state","parameters":{"qubits":0}})");
      },
      "");
}

TEST(WState, EvaluatesCountsFromJSON) {
  const auto evaluation = llvm::cantFail(evaluateJSON(
      toManifestJSON(llvm::cantFail(WState::create({.qubits = 2}))),
      R"({"schema_version":1,"counts":{"01":10,"10":10}})"));
  EXPECT_NE(evaluation.find(R"("squared_hellinger_fidelity":1)"),
            std::string::npos);
  EXPECT_NE(evaluation.find(R"("shots":20)"), std::string::npos);
}

} // namespace mqt::bench
