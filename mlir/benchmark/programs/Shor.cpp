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
#include <mlir/IR/Value.h>
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

} // namespace

SmallVector<Value> shor(qc::QCProgramBuilder& b, const uint64_t n) {
  const auto size = (static_cast<int64_t>(n) - 2) / 2;
  const auto steps = 2 * size;
  auto ctrl = b.allocQubit();
  auto x = b.allocQubitRegister(size, "x");
  auto acc = b.allocQubitRegister(size, "acc");
  auto anc = b.allocQubit();
  auto c = b.allocClassicalBitRegister(steps, "c");

  b.reset(ctrl);
  resetRegister(b, x.value, size);
  resetRegister(b, acc.value, size);
  b.reset(anc);

  // The work register starts in the neutral element of the multiplication.
  b.x(x[0]);

  auto zero = b.indexConstant(0);
  auto one = b.indexConstant(1);
  auto width = b.indexConstant(size);
  auto rounds = b.indexConstant(steps);

  // Phase estimation reads one control qubit over and over. Each round
  // multiplies the accumulator by the next power of the base, corrects the
  // control from the results of the earlier rounds, and then reuses the qubit.
  scfForWithAngle(
      b, zero, rounds, std::numbers::pi * static_cast<double>(SHOR_BASE), 2.0,
      [&](Value power, Value k) {
        b.reset(ctrl);
        b.h(ctrl);

        // The multiplication runs in the Fourier basis, so each multiplier
        // qubit contributes a phase to every accumulator qubit. The phase
        // doubles from one multiplier qubit to the next and halves down the
        // accumulator.
        fourierTransform(b, acc.value, size, 1.0);
        scfForWithAngle(
            b, zero, width, power, 2.0, [&](Value shifted, Value i) {
              const SmallVector<Value> controls{ctrl, b.loadQubit(x.value, i)};
              scfForWithAngle(
                  b, zero, width, shifted, 0.5, [&](Value angle, Value j) {
                    b.mcp(angle, controls, b.loadQubit(acc.value, j));
                  });
            });

        // The product is reduced modulo the modulus. The number of rounds
        // depends on the product, so the loop bound is only known at runtime.
        b.scfWhile(
            [&] {
              fourierTransform(b, acc.value, size, -1.0);
              b.reset(anc);
              b.cx(acc[size - 1], anc);
              fourierTransform(b, acc.value, size, 1.0);
              auto overflow = b.measure(anc);
              b.scfCondition(overflow);
            },
            [&] {
              scfForWithAngle(b, zero, width,
                              -std::numbers::pi *
                                  static_cast<double>(SHOR_MODULUS),
                              0.5, [&](Value angle, Value i) {
                                b.p(angle, b.loadQubit(acc.value, i));
                              });
            });
        fourierTransform(b, acc.value, size, -1.0);

        // The inverse transform of the phase estimation runs on the classical
        // side: every result measured so far rotates the control qubit, and
        // the rotation halves the further back the result lies.
        scfForWithAngle(b, zero, k, -std::numbers::pi / 2.0, 0.5,
                        [&](Value angle, Value step) {
                          auto previous = arith::SubIOp::create(
                              b, arith::SubIOp::create(b, k, one), step);
                          b.scfIf(c, previous, [&] { b.p(angle, ctrl); });
                        });

        b.h(ctrl);
        b.measure(ctrl, c, k);
      });

  return {c};
}

} // namespace mqt::benchmark
