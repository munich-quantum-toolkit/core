/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Benchmark/BenchmarkUtils.h"
#include "mlir/Benchmark/Programs.h"
#include "mlir/Dialect/QC/Builder/QCProgramBuilder.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>

#include <cstdint>
#include <numbers>

namespace mqt::benchmark {

using namespace mlir;

namespace {
/// The base of the modular exponentiation.
constexpr int64_t SHOR_BASE = 3;
/// The modulus that the exponentiation reduces by.
constexpr int64_t SHOR_MODULUS = 5;
/// The inverse of the base modulo the modulus, because 3 * 2 = 1 mod 5. The
/// inverse of a square is the square of the inverse, so squaring this value
/// alongside the base keeps the pair in step.
constexpr int64_t SHOR_INVERSE = 2;
} // namespace

SmallVector<Value> shor(qc::QCProgramBuilder& b, const uint64_t n) {
  // Beauregard fits the circuit into 2n+3 qubits: the multiplier register, an
  // accumulator with one extra qubit, one ancilla, and one control qubit that
  // every round reuses.
  const auto bits = (static_cast<int64_t>(n) - 3) / 2;
  const auto width = bits + 1;
  const auto rounds = 2 * bits;
  auto ctrl = b.allocQubit();
  auto x = b.allocQubitRegister(bits, "x");
  auto acc = b.allocQubitRegister(width, "acc");
  auto anc = b.allocQubit();
  auto c = b.allocClassicalBitRegister(rounds, "c");

  b.reset(ctrl);
  resetRegister(b, x.value, bits);
  resetRegister(b, acc.value, width);
  b.reset(anc);

  // The multiplier register starts at one, the neutral element.
  b.x(x[0]);

  auto zero = b.indexConstant(0);
  auto one = b.indexConstant(1);
  auto last = b.indexConstant(rounds - 1);

  // Phase estimation reads one control qubit over and over. Round k applies
  // the controlled multiplication by a^(2^k) mod N, so the round carries that
  // value and its inverse, squaring both for the next round.
  auto loop = scf::ForOp::create(
      b, zero, b.indexConstant(rounds), one,
      ValueRange{b.intConstant(SHOR_BASE % SHOR_MODULUS),
                 b.intConstant(SHOR_INVERSE % SHOR_MODULUS)});
  {
    const OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(loop.getBody());
    auto power = loop.getRegionIterArg(0);
    auto inverse = loop.getRegionIterArg(1);
    auto step = loop.getInductionVar();
    auto index = arith::SubIOp::create(b, last, step);

    b.reset(ctrl);
    b.h(ctrl);

    // The controlled multiplication by a leaves the product in the
    // accumulator. Swapping the registers moves it into place, and
    // subtracting the product of the inverse returns the accumulator to zero.
    modularMultiply(b, ctrl, x.value, acc.value, anc, bits, power, SHOR_MODULUS,
                    1.0);
    b.scfFor(0, bits, 1, [&](Value i) {
      b.cswap(ctrl, b.loadQubit(x.value, i), b.loadQubit(acc.value, i));
    });
    modularMultiply(b, ctrl, x.value, acc.value, anc, bits, inverse,
                    SHOR_MODULUS, -1.0);

    // The inverse transform of the phase estimation runs on the classical
    // side: every bit measured so far removes its share of the phase.
    auto innerLower = arith::AddIOp::create(b, index, one);
    scfForWithAngle(b, innerLower, b.indexConstant(rounds),
                    -std::numbers::pi / 2.0, 0.5, [&](Value angle, Value bit) {
                      b.scfIf(c, bit, [&] { b.p(angle, ctrl); });
                    });

    b.h(ctrl);
    b.measure(ctrl, c, index);

    auto squared = arith::MulIOp::create(b, power, power);
    auto squaredInverse = arith::MulIOp::create(b, inverse, inverse);
    auto modulus = b.intConstant(SHOR_MODULUS);
    scf::YieldOp::create(
        b, ValueRange{arith::RemSIOp::create(b, squared, modulus),
                      arith::RemSIOp::create(b, squaredInverse, modulus)});
  }

  return {c};
}

} // namespace mqt::benchmark
