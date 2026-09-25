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
#include "support/TestSupport.hpp"

#include "gtest/gtest.h"

#include <string>
#include <utility>

namespace mqt::bench {

using namespace mlir;

template <class Benchmark>
static void expectQCAndJeff(const Benchmark& benchmark) {
  auto program = generate(benchmark);
  ASSERT_TRUE(succeeded(program));
  test::expectJeffRoundTrip(std::move(*program));
}

TEST(GenerateProgramTest, GeneratesEveryBenchmarkMethodAsQCAndJeff) {
  expectQCAndJeff(::mqt::test::value(BV::create({.hiddenBitstring = "101"})));
  expectQCAndJeff(::mqt::test::value(
      BV::create({.hiddenBitstring = "101", .method = BVMethod::Dynamic})));
  expectQCAndJeff(::mqt::test::value(ModularMultiplier::create({
      .multiplier = "011",
      .modulus = "101",
      .multiplicand = "+++",
      .control = '+',
  })));
  expectQCAndJeff(::mqt::test::value(GHZ::create({.qubits = 3})));
  expectQCAndJeff(
      ::mqt::test::value(Grover::create({.markedBitstring = "101"})));
  expectQCAndJeff(::mqt::test::value(Multiplexer::create({.qubits = 3})));
  expectQCAndJeff(
      ::mqt::test::value(QFT::create({.qubits = 3, .periodExponent = 1})));
  expectQCAndJeff(::mqt::test::value(QFT::create(
      {.qubits = 3, .periodExponent = 1, .method = QFTMethod::Semiclassical})));
  expectQCAndJeff(::mqt::test::value(QFTAdder::create({
      .addend = "101",
      .accumulator = "001",
      .method = QFTAdderMethod::Constant,
      .overflow = QFTAdderOverflow::Carry,
  })));
  expectQCAndJeff(::mqt::test::value(
      QFTAdder::create({.addend = "+++", .accumulator = "001"})));
  expectQCAndJeff(::mqt::test::value(QPE::create(
      {.precision = 3, .phase = ::mqt::test::value(Phase::create(3, 8))})));
  expectQCAndJeff(::mqt::test::value(QPE::create({
      .precision = 3,
      .phase = ::mqt::test::value(Phase::create(3, 8)),
      .method = QPEMethod::Iterative,
  })));
  expectQCAndJeff(::mqt::test::value(RepeatUntilSuccess::create()));
  expectQCAndJeff(Teleportation{});
}

TEST(GenerateProgramTest, ReturnsSourceDiagnosticsAndRecovers) {
  ::mqt::test::DiagnosticCapture invalidDiagnostics;
  auto invalid = generate(
      R"({"schema_version":1,"benchmark":"ghz","parameters":{"qubits":0}})",
      "invalid.json");
  ASSERT_FALSE(succeeded(invalid));
  const auto diagnostic = invalidDiagnostics.error->message;
  EXPECT_NE(diagnostic.find("invalid.json:$/parameters"), std::string::npos);
  EXPECT_NE(diagnostic.find("GHZ qubits"), std::string::npos);

  auto valid = generate(
      R"({"schema_version":1,"benchmark":"ghz","parameters":{"qubits":2}})");
  ASSERT_TRUE(succeeded(valid));
  EXPECT_EQ(valid->benchmarkId, "ghz");
}

} // namespace mqt::bench
