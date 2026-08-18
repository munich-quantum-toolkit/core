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

namespace mqt::benchmark {

using namespace mlir;

namespace {
/// The number of cost and mixer layers.
constexpr int64_t QAOA_LAYERS = 2;
/// The cost angle of one layer.
constexpr double QAOA_GAMMA = 0.8;
/// The mixer angle of one layer.
constexpr double QAOA_BETA = 0.4;
} // namespace

SmallVector<Value> qaoa(qc::QCProgramBuilder& b, const uint64_t n) {
  const auto size = static_cast<int64_t>(n);
  auto q = b.allocQubitRegister(size, "q");
  auto c = b.allocClassicalBitRegister(size, "c");

  resetRegister(b, q.value, size);
  b.scfFor(0, size, 1, [&](Value iv) { b.h(b.loadQubit(q.value, iv)); });

  // Each layer applies the cost operator of a ring of couplings and then the
  // mixer. The problem graph is fixed, so the layer count is a constant.
  auto one = b.indexConstant(1);
  b.scfFor(0, QAOA_LAYERS, 1, [&](Value) {
    b.scfFor(0, size - 1, 1, [&](Value i) {
      auto next = arith::AddIOp::create(b, i, one);
      b.rzz(QAOA_GAMMA, b.loadQubit(q.value, i), b.loadQubit(q.value, next));
    });
    b.rzz(QAOA_GAMMA, q[size - 1], q[0]);
    b.scfFor(0, size, 1,
             [&](Value i) { b.rx(QAOA_BETA, b.loadQubit(q.value, i)); });
  });

  measureRegister(b, q.value, size, c);

  return {c};
}

} // namespace mqt::benchmark
