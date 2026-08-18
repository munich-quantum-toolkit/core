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

#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>

#include <cstdint>
#include <numbers>

namespace mqt::benchmark {

using namespace mlir;

namespace {
/// The multiplier of the modular multiplication.
constexpr int64_t MULTIPLIER = 3;
/// The modulus of the modular multiplication.
constexpr int64_t MODULUS = 5;

} // namespace

SmallVector<Value> controlledMultiplyModN(qc::QCProgramBuilder& b,
                                          const uint64_t n) {
  const auto total = static_cast<int64_t>(n) - 2;
  const auto factor = total / 2;
  const auto size = total - factor;
  auto ctrl = b.allocQubit();
  auto x = b.allocQubitRegister(factor, "x");
  auto acc = b.allocQubitRegister(size, "acc");
  auto anc = b.allocQubit();
  auto c = b.allocClassicalBitRegister(size, "c");

  b.reset(ctrl);
  resetRegister(b, x.value, factor);
  resetRegister(b, acc.value, size);
  b.reset(anc);

  b.h(ctrl);
  b.scfFor(0, factor, 1, [&](Value i) { b.h(b.loadQubit(x.value, i)); });

  fourierTransform(b, acc.value, size, 1.0);

  auto zero = b.indexConstant(0);
  auto factors = b.indexConstant(factor);
  auto width = b.indexConstant(size);

  // Each multiplier qubit adds a shifted copy of the multiplier into the
  // accumulator, and every addition is a layer of phases in the Fourier basis.
  // The shift doubles the phase from one multiplier qubit to the next, and the
  // phase halves down the accumulator, so both angles come from the loops.
  scfForWithAngle(
      b, zero, factors, std::numbers::pi * static_cast<double>(MULTIPLIER), 2.0,
      [&](Value shifted, Value i) {
        const SmallVector<Value> controls{ctrl, b.loadQubit(x.value, i)};
        scfForWithAngle(b, zero, width, shifted, 0.5,
                        [&](Value angle, Value j) {
                          b.mcp(angle, controls, b.loadQubit(acc.value, j));
                        });
      });

  // The product is reduced modulo the modulus. Each round tests the top qubit
  // of the accumulator and subtracts the modulus once when it is set, so the
  // number of rounds depends on the product.
  b.scfWhile(
      [&] {
        // The top qubit only carries the overflow in the computational basis,
        // so the accumulator leaves and re-enters the Fourier basis around the
        // copy onto the ancilla.
        fourierTransform(b, acc.value, size, -1.0);
        b.reset(anc);
        b.cx(acc[size - 1], anc);
        fourierTransform(b, acc.value, size, 1.0);
        auto overflow = b.measure(anc);
        b.scfCondition(overflow);
      },
      [&] {
        scfForWithAngle(b, zero, width,
                        -std::numbers::pi * static_cast<double>(MODULUS), 0.5,
                        [&](Value angle, Value i) {
                          b.p(angle, b.loadQubit(acc.value, i));
                        });
      });

  fourierTransform(b, acc.value, size, -1.0);

  measureRegister(b, acc.value, size, c);

  return {c};
}

} // namespace mqt::benchmark
