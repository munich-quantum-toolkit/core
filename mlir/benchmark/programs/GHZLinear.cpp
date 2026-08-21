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

namespace mqt::benchmark {

using namespace mlir;

SmallVector<Value> ghzLinear(qc::QCProgramBuilder& b, const uint64_t n) {
  const auto size = static_cast<int64_t>(n);
  auto q = b.allocQubitRegister(size, "q");
  auto c = b.allocClassicalBitRegister(size, "c");

  resetRegister(b, q.value, size);

  b.h(q[0]);
  auto one = b.indexConstant(1);
  b.scfFor(1, size, 1, [&](Value iv) {
    auto previous = arith::SubIOp::create(b, iv, one);
    b.cx(b.loadQubit(q.value, previous), b.loadQubit(q.value, iv));
  });

  measureRegister(b, q.value, size, c);

  return {c};
}

} // namespace mqt::benchmark
