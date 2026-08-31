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
#include "bench/Multiplexer.hpp"

#include <gtest/gtest.h>

#include <stdexcept>
#include <string>

namespace {

using mqt::bench::Multiplexer;
using mqt::bench::MultiplexerOptions;
using mqt::bench::Output;

TEST(Multiplexer, StoresTheTotalQubitCountAndOutput) {
  const Multiplexer benchmark{{.qubits = 7}};
  EXPECT_EQ(benchmark.options().qubits, 7);
  EXPECT_EQ(benchmark.output(), (Output{"result", 7}));
}

TEST(Multiplexer, ValidatesTheConfiguredInstance) {
  EXPECT_THROW(static_cast<void>(Multiplexer{{.qubits = 1}}),
               std::invalid_argument);
  EXPECT_NO_THROW(static_cast<void>(Multiplexer{{.qubits = 2}}));
  EXPECT_NO_THROW(static_cast<void>(
      Multiplexer{{.qubits = MultiplexerOptions::MAX_QUBITS}}));
  EXPECT_THROW(static_cast<void>(
                   Multiplexer{{.qubits = MultiplexerOptions::MAX_QUBITS + 1}}),
               std::invalid_argument);
}

TEST(Multiplexer, HasTheExactZeroInputReference) {
  const Multiplexer benchmark{{.qubits = 3}};
  EXPECT_DOUBLE_EQ(benchmark.probability("000"), 1.);
  EXPECT_DOUBLE_EQ(benchmark.probability("001"), 0.);
  EXPECT_DOUBLE_EQ(benchmark.probability("100"), 0.);
  EXPECT_THROW(static_cast<void>(benchmark.probability("00")),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(benchmark.probability("00x")),
               std::invalid_argument);
}

TEST(Multiplexer, EvaluatesTheReferenceWithoutASuccessOutcome) {
  const Multiplexer benchmark{{.qubits = 3}};
  const auto evaluation = benchmark.evaluate({{"000", 80}, {"001", 20}});
  EXPECT_DOUBLE_EQ(evaluation.totalVariationDistance, 0.2);
  EXPECT_DOUBLE_EQ(evaluation.squaredHellingerFidelity, 0.8);
  EXPECT_FALSE(evaluation.successProbability);
}

} // namespace
