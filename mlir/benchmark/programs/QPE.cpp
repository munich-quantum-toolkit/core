/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "BenchmarkUtils.h"
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
/// The phase of the unitary whose eigenvalue is estimated.
constexpr double QPE_PHASE = 3.0 * std::numbers::pi / 8.0;
} // namespace

SmallVector<Value> qpe(qc::QCProgramBuilder& b, const uint64_t n) {
  const auto counting = static_cast<int64_t>(n) - 1;
  auto q = b.allocQubitRegister(counting, "q");
  auto anc = b.allocQubit();
  auto c = b.allocClassicalBitRegister(counting, "c");

  resetRegister(b, q.value, counting);
  b.reset(anc);

  b.scfFor(0, counting, 1, [&](Value iv) { b.h(b.loadQubit(q.value, iv)); });
  b.x(anc);

  auto zero = b.indexConstant(0);
  auto one = b.indexConstant(1);
  auto upper = b.indexConstant(counting);
  auto last = b.indexConstant(counting - 1);

  // Repeating a phase gate 2^i times multiplies its angle by 2^i, so each
  // controlled power is one rotation whose angle doubles.
  phaseRotationLoop(b, zero, upper, QPE_PHASE, 2.0, [&](Value angle, Value i) {
    b.cp(angle, b.loadQubit(q.value, i), anc);
  });

  // Inverse quantum Fourier transform on the counting register.
  b.scfFor(0, counting / 2, 1, [&](Value i) {
    auto mirrored = arith::SubIOp::create(b, last, i);
    b.swap(b.loadQubit(q.value, i), b.loadQubit(q.value, mirrored));
  });

  b.scfFor(0, counting, 1, [&](Value i) {
    auto lower = arith::AddIOp::create(b, i, one);
    phaseRotationLoop(b, lower, upper, -std::numbers::pi / 2.0, 0.5,
                      [&](Value angle, Value j) {
                        b.cp(angle, b.loadQubit(q.value, j),
                             b.loadQubit(q.value, i));
                      });
    b.h(b.loadQubit(q.value, i));
  });

  measureRegister(b, q.value, counting, c);

  return {c};
}

} // namespace mqt::benchmark
