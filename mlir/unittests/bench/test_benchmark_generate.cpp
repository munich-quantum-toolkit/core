/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "TestUtils.h"
#include "bench/BV.hpp"
#include "bench/GHZ.hpp"
#include "bench/Grover.hpp"
#include "bench/Multiplexer.hpp"
#include "bench/QFT.hpp"
#include "bench/QFTAdder.hpp"
#include "bench/QPE.hpp"
#include "bench/Teleportation.hpp"
#include "mlir/bench/Generate.h"

#include <gtest/gtest.h>

#include <utility>

namespace mqt::bench {

using namespace mlir;

template <class Benchmark>
static void expectValidQCAndJeff(const Benchmark& benchmark) {
  auto program = generate(benchmark);
  ASSERT_TRUE(program);
  EXPECT_TRUE(program->isValid());
  test::expectJeffRoundTrip(std::move(*program));
}

TEST(GenerateProgramTest, GeneratesEveryBenchmarkMethodAsQCAndJeff) {
  expectValidQCAndJeff(BV{{.hiddenBitstring = "101"}});
  expectValidQCAndJeff(
      BV{{.hiddenBitstring = "101", .method = BVMethod::Dynamic}});
  expectValidQCAndJeff(GHZ{{.qubits = 3}});
  expectValidQCAndJeff(Grover{{.markedBitstring = "101"}});
  expectValidQCAndJeff(Multiplexer{{.qubits = 3}});
  expectValidQCAndJeff(QFT{{.qubits = 3, .periodExponent = 1}});
  expectValidQCAndJeff(QFT{
      {.qubits = 3, .periodExponent = 1, .method = QFTMethod::Semiclassical}});
  expectValidQCAndJeff(QFTAdder{{
      .addend = "101",
      .accumulator = "001",
      .method = QFTAdderMethod::Constant,
      .overflow = QFTAdderOverflow::Carry,
  }});
  expectValidQCAndJeff(QFTAdder{{.addend = "+++", .accumulator = "001"}});
  expectValidQCAndJeff(QPE{{.precision = 3, .phase = Phase(3, 8)}});
  expectValidQCAndJeff(QPE{
      {.precision = 3, .phase = Phase(3, 8), .method = QPEMethod::Iterative}});
  expectValidQCAndJeff(Teleportation{});
}

} // namespace mqt::bench
