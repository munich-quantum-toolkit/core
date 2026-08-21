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

SmallVector<Value> qft(qc::QCProgramBuilder& b, const uint64_t n) {
  const auto size = static_cast<int64_t>(n);
  auto q = b.allocQubitRegister(size, "q");
  auto c = b.allocClassicalBitRegister(size, "c");

  resetRegister(b, q.value, size);

  auto one = b.indexConstant(1);
  auto upper = b.indexConstant(size);
  auto last = b.indexConstant(size - 1);

  // Every qubit takes a Hadamard and then a controlled phase from each qubit
  // above it. The angle halves as the distance grows.
  b.scfFor(0, size, 1, [&](Value i) {
    b.h(b.loadQubit(q.value, i));

    auto lower = arith::AddIOp::create(b, i, one);
    phaseRotationLoop(b, lower, upper, std::numbers::pi / 2.0, 0.5,
                      [&](Value angle, Value j) {
                        b.cp(angle, b.loadQubit(q.value, j),
                             b.loadQubit(q.value, i));
                      });
  });

  // Reverse the bit order. The count is floor(size / 2).
  b.scfFor(0, size / 2, 1, [&](Value i) {
    auto mirrored = arith::SubIOp::create(b, last, i);
    b.swap(b.loadQubit(q.value, i), b.loadQubit(q.value, mirrored));
  });

  measureRegister(b, q.value, size, c);

  return {c};
}

} // namespace mqt::benchmark
