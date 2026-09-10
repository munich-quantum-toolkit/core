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
#include "bench/GHZ.hpp"
#include "bench/Grover.hpp"
#include "bench/ModularMultiplier.hpp"
#include "bench/Multiplexer.hpp"
#include "bench/QFT.hpp"
#include "bench/QFTAdder.hpp"
#include "bench/QPE.hpp"
#include "bench/RepeatUntilSuccess.hpp"
#include "bench/Teleportation.hpp"
#include "mqt/bench/Generate.h"

#include "TestUtils.h"
#include "gtest/gtest.h"

#include <utility>

namespace mqt::bench {

using namespace mlir;

template <class Benchmark>
static void expectQCAndJeff(const Benchmark& benchmark) {
  auto program = generate(benchmark);
  ASSERT_TRUE(program);
  test::expectJeffRoundTrip(std::move(*program));
}

TEST(GenerateProgramTest, GeneratesEveryBenchmarkMethodAsQCAndJeff) {
  expectQCAndJeff(BV{{.hiddenBitstring = "101"}});
  expectQCAndJeff(BV{{.hiddenBitstring = "101", .method = BVMethod::Dynamic}});
  expectQCAndJeff(ModularMultiplier{{
      .multiplier = "011",
      .modulus = "101",
      .multiplicand = "+++",
      .control = '+',
  }});
  expectQCAndJeff(GHZ{{.qubits = 3}});
  expectQCAndJeff(Grover{{.markedBitstring = "101"}});
  expectQCAndJeff(Multiplexer{{.qubits = 3}});
  expectQCAndJeff(QFT{{.qubits = 3, .periodExponent = 1}});
  expectQCAndJeff(QFT{
      {.qubits = 3, .periodExponent = 1, .method = QFTMethod::Semiclassical}});
  expectQCAndJeff(QFTAdder{{
      .addend = "101",
      .accumulator = "001",
      .method = QFTAdderMethod::Constant,
      .overflow = QFTAdderOverflow::Carry,
  }});
  expectQCAndJeff(QFTAdder{{.addend = "+++", .accumulator = "001"}});
  expectQCAndJeff(QPE{{.precision = 3, .phase = Phase(3, 8)}});
  expectQCAndJeff(QPE{
      {.precision = 3, .phase = Phase(3, 8), .method = QPEMethod::Iterative}});
  expectQCAndJeff(RepeatUntilSuccess{});
  expectQCAndJeff(Teleportation{});
}

} // namespace mqt::bench
