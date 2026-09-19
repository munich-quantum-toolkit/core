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
#include "bench/TestUtils.hpp"
#include "mqt/bench/Generate.h"

#include "TestUtils.h"

#include "gtest/gtest.h"

#include <string>
#include <utility>

namespace mqt::bench {

using namespace mlir;

template <class Benchmark>
static void expectQCAndJeff(const Benchmark& benchmark) {
  auto program = generate(benchmark);
  ASSERT_TRUE(static_cast<bool>(program))
      << llvm::toString(program.takeError());
  test::expectJeffRoundTrip(std::move(*program));
}

TEST(GenerateProgramTest, GeneratesEveryBenchmarkMethodAsQCAndJeff) {
  expectQCAndJeff(test::value(BV::create({.hiddenBitstring = "101"})));
  expectQCAndJeff(test::value(
      BV::create({.hiddenBitstring = "101", .method = BVMethod::Dynamic})));
  expectQCAndJeff(test::value(ModularMultiplier::create({
      .multiplier = "011",
      .modulus = "101",
      .multiplicand = "+++",
      .control = '+',
  })));
  expectQCAndJeff(test::value(GHZ::create({.qubits = 3})));
  expectQCAndJeff(test::value(Grover::create({.markedBitstring = "101"})));
  expectQCAndJeff(test::value(Multiplexer::create({.qubits = 3})));
  expectQCAndJeff(test::value(QFT::create({.qubits = 3, .periodExponent = 1})));
  expectQCAndJeff(test::value(QFT::create(
      {.qubits = 3, .periodExponent = 1, .method = QFTMethod::Semiclassical})));
  expectQCAndJeff(test::value(QFTAdder::create({
      .addend = "101",
      .accumulator = "001",
      .method = QFTAdderMethod::Constant,
      .overflow = QFTAdderOverflow::Carry,
  })));
  expectQCAndJeff(
      test::value(QFTAdder::create({.addend = "+++", .accumulator = "001"})));
  expectQCAndJeff(test::value(QPE::create(
      {.precision = 3, .phase = test::value(Phase::create(3, 8))})));
  expectQCAndJeff(test::value(QPE::create({
      .precision = 3,
      .phase = test::value(Phase::create(3, 8)),
      .method = QPEMethod::Iterative,
  })));
  expectQCAndJeff(test::value(RepeatUntilSuccess::create()));
  expectQCAndJeff(Teleportation{});
}

TEST(GenerateProgramTest, ReturnsSourceDiagnosticsAndRecovers) {
  auto invalid = generate(
      R"({"schema_version":1,"benchmark":"ghz","parameters":{"qubits":0}})",
      "invalid.json");
  ASSERT_FALSE(static_cast<bool>(invalid));
  const auto diagnostic = llvm::toString(invalid.takeError());
  EXPECT_NE(diagnostic.find("invalid.json:$/parameters"), std::string::npos);
  EXPECT_NE(diagnostic.find("GHZ qubits"), std::string::npos);

  auto valid = generate(
      R"({"schema_version":1,"benchmark":"ghz","parameters":{"qubits":2}})");
  ASSERT_TRUE(static_cast<bool>(valid)) << llvm::toString(valid.takeError());
  EXPECT_EQ(valid->benchmarkId, "ghz");
}

} // namespace mqt::bench
