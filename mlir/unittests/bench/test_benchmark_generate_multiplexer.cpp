/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "bench/Multiplexer.hpp"
#include "mqt/bench/Generate.h"

#include "TestUtils.h"

#include "gtest/gtest.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/LogicalResult.h"

#include <utility>

namespace mqt::bench {

TEST(GenerateProgramTest, SerializesTheLargestQuantumMultiplexer) {
  auto program = generate(llvm::cantFail(
      Multiplexer::create({.qubits = MultiplexerOptions::MAX_QUBITS})));
  ASSERT_TRUE(mlir::succeeded(program));
  EXPECT_LT(test::countOperations(program->module()), 150U);

  test::expectJeffRoundTrip(std::move(*program));
}

TEST(GenerateProgramTest, SamplesMultiplexerAgainstReference) {
  test::expectSamplingMatchesReference(
      llvm::cantFail(Multiplexer::create({.qubits = 3})));
}

} // namespace mqt::bench
