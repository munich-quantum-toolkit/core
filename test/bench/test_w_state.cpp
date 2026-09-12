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

#include "gtest/gtest.h"

#include <bit>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>

namespace mqt::bench {

TEST(WState, AssignsUniformProbabilityToSingleExcitations) {
  for (const size_t qubits : {1U, 2U, 3U, 7U}) {
    const WState benchmark{{.qubits = qubits}};
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
      EXPECT_DOUBLE_EQ(benchmark.probability(bits), expected);
      if (expected != 0.) {
        uniform.emplace(bits, 10);
      }
    }
    const auto evaluation = benchmark.evaluate(uniform);
    EXPECT_NEAR(evaluation.totalVariationDistance, 0., 1e-12);
    EXPECT_NEAR(evaluation.squaredHellingerFidelity, 1., 1e-12);
  }
  const auto partial = WState{{.qubits = 2}}.evaluate({{"01", 20}});
  EXPECT_NEAR(partial.totalVariationDistance, 0.5, 1e-12);
  EXPECT_NEAR(partial.squaredHellingerFidelity, 0.5, 1e-12);
}

TEST(WState, RejectsInvalidParametersAndOutcomes) {
  EXPECT_THROW(static_cast<void>(WState{{.qubits = 0}}), std::invalid_argument);
  EXPECT_THROW(
      static_cast<void>(WState{{.qubits = std::numeric_limits<size_t>::max()}}),
      std::invalid_argument);
  const WState benchmark{{.qubits = 2}};
  EXPECT_THROW(static_cast<void>(benchmark.probability("1")),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(benchmark.evaluate({{"0x", 1}})),
               std::invalid_argument);
}

TEST(WState, RoundTripsStrictManifestsAndEvaluatesCounts) {
  const WState benchmark{{.qubits = 2}};
  const auto manifest = toManifestJSON(benchmark);
  EXPECT_EQ(caseId(wStateFromInstanceSpecificationJSON(
                toInstanceSpecificationJSON(benchmark))),
            caseId(benchmark));
  EXPECT_EQ(toManifestJSON(wStateFromManifestJSON(manifest)), manifest);
  EXPECT_NE(manifest.find(R"("model":"w_state")"), std::string::npos);
  const auto evaluation = evaluateJSON(
      manifest, R"({"schema_version":1,"counts":{"01":10,"10":10}})");
  EXPECT_NE(evaluation.find(R"("squared_hellinger_fidelity":1.0)"),
            std::string::npos);
  EXPECT_NE(evaluation.find(R"("shots":20)"), std::string::npos);
  EXPECT_THROW(
      static_cast<void>(wStateFromInstanceSpecificationJSON(
          R"({"schema_version":1,"benchmark":"w-state","parameters":{"qubits":0}})")),
      std::invalid_argument);
}

} // namespace mqt::bench
