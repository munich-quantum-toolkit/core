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

namespace mqt::benchmark {

using namespace mlir;

SmallVector<Value> fanOut(qc::QCProgramBuilder& b, const uint64_t n) {
  const auto size = static_cast<int64_t>(n);
  auto q = b.allocQubitRegister(size, "q");
  auto c = b.allocClassicalBitRegister(size, "c");

  resetRegister(b, q.value, size);

  // The first qubit holds the value that is copied. The fan-out spreads it
  // over the register so that the following layer acts on every qubit at once.
  b.h(q[0]);
  b.scfFor(1, size, 1, [&](Value iv) { b.cx(q[0], b.loadQubit(q.value, iv)); });

  b.scfFor(1, size, 1, [&](Value iv) { b.t(b.loadQubit(q.value, iv)); });

  // The copies are uncomputed so that only the first qubit stays entangled.
  b.scfFor(1, size, 1, [&](Value iv) { b.cx(q[0], b.loadQubit(q.value, iv)); });

  measureRegister(b, q.value, size, c);

  return {c};
}

} // namespace mqt::benchmark
