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
#include "bench/ControlledMultiplicationModuloN.hpp"
#include "bench/GHZ.hpp"
#include "bench/Grover.hpp"
#include "bench/Multiplexer.hpp"
#include "bench/QFT.hpp"
#include "bench/QFTAdderClassical.hpp"
#include "bench/QFTAdderQuantum.hpp"
#include "bench/QPE.hpp"
#include "bench/RepeatUntilSuccess.hpp"
#include "bench/Teleportation.hpp"
#include "mlir/Dialect/QC/IR/QCOps.h"
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
  expectValidQCAndJeff(
      ControlledMultiplicationModuloN{{.multiplier = "011", .modulus = "101"}});
  expectValidQCAndJeff(GHZ{{.qubits = 3}});
  expectValidQCAndJeff(Grover{{.markedBitstring = "101"}});
  expectValidQCAndJeff(Multiplexer{{.qubits = 3}});
  expectValidQCAndJeff(QFT{{.qubits = 3, .periodExponent = 1}});
  expectValidQCAndJeff(QFT{
      {.qubits = 3, .periodExponent = 1, .method = QFTMethod::Semiclassical}});
  expectValidQCAndJeff(QFTAdderClassical{{.addend = "101"}});
  expectValidQCAndJeff(QFTAdderQuantum{{.qubits = 3}});
  expectValidQCAndJeff(QPE{{.precision = 3, .phase = Phase(3, 8)}});
  expectValidQCAndJeff(QPE{
      {.precision = 3, .phase = Phase(3, 8), .method = QPEMethod::Iterative}});
  expectValidQCAndJeff(RepeatUntilSuccess{});
  expectValidQCAndJeff(Teleportation{});
}

TEST(GenerateProgramTest, OmitsAllocationAdjacentResets) {
  EXPECT_EQ(test::countOps<qc::ResetOp>(generate(GHZ{{.qubits = 3}})->module()),
            0U);
  EXPECT_EQ(test::countOps<qc::ResetOp>(
                generate(Grover{{.markedBitstring = "101"}})->module()),
            0U);
  EXPECT_EQ(test::countOps<qc::ResetOp>(
                generate(Multiplexer{{.qubits = 3}})->module()),
            0U);
  EXPECT_EQ(test::countOps<qc::ResetOp>(
                generate(BV{{.hiddenBitstring = "101"}})->module()),
            0U);
  EXPECT_EQ(test::countOps<qc::ResetOp>(
                generate(ControlledMultiplicationModuloN{
                             {.multiplier = "011", .modulus = "101"}})
                    ->module()),
            0U);
  EXPECT_EQ(test::countOps<qc::ResetOp>(
                generate(QFT{{.qubits = 3, .periodExponent = 1}})->module()),
            0U);
  EXPECT_EQ(test::countOps<qc::ResetOp>(
                generate(QFTAdderClassical{{.addend = "101"}})->module()),
            0U);
  EXPECT_EQ(test::countOps<qc::ResetOp>(
                generate(QFTAdderQuantum{{.qubits = 3}})->module()),
            0U);
  EXPECT_EQ(
      test::countOps<qc::ResetOp>(
          generate(QPE{{.precision = 3, .phase = Phase(3, 8)}})->module()),
      0U);
  EXPECT_EQ(
      test::countOps<qc::ResetOp>(generate(RepeatUntilSuccess{})->module()),
      0U);
  EXPECT_EQ(test::countOps<qc::ResetOp>(generate(Teleportation{})->module()),
            0U);

  EXPECT_GT(
      test::countOps<qc::ResetOp>(
          generate(BV{{.hiddenBitstring = "101", .method = BVMethod::Dynamic}})
              ->module()),
      0U);
  EXPECT_GT(test::countOps<qc::ResetOp>(
                generate(QFT{{.qubits = 3,
                              .periodExponent = 1,
                              .method = QFTMethod::Semiclassical}})
                    ->module()),
            0U);
  EXPECT_GT(test::countOps<qc::ResetOp>(
                generate(QPE{{.precision = 3,
                              .phase = Phase(3, 8),
                              .method = QPEMethod::Iterative}})
                    ->module()),
            0U);
}

} // namespace mqt::bench
