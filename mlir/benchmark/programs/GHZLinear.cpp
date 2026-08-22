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
#include "benchmarks/GHZ.hpp"
#include "mlir/Benchmark/Programs.h"
#include "mlir/Dialect/QC/Builder/QCProgramBuilder.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>

#include <cstddef>
#include <cstdint>

namespace mqt::benchmark {

using namespace mlir;

SmallVector<Value> ghz(qc::QCProgramBuilder& b,
                       const benchmarks::GHZ& benchmark) {
  const auto& options = benchmark.options();
  const auto size = static_cast<int64_t>(options.qubits);
  auto q = b.allocQubitRegisterStorage(size, "q");
  auto result = b.allocClassicalBitRegister(size, benchmark.output().name);

  resetRegister(b, q, size);

  auto root = b.loadQubit(q, b.indexConstant(0));
  b.h(root);
  if (options.topology == benchmarks::GHZTopology::Linear) {
    auto one = b.indexConstant(1);
    b.scfFor(1, size, 1, [&](Value iv) {
      auto previous = arith::SubIOp::create(b, iv, one);
      b.cx(b.loadQubit(q, previous), b.loadQubit(q, iv));
    });
  } else {
    b.scfFor(1, size, 1, [&](Value iv) { b.cx(root, b.loadQubit(q, iv)); });
  }

  if (options.basis == benchmarks::GHZBasis::X) {
    b.scfFor(0, size, 1, [&](Value iv) { b.h(b.loadQubit(q, iv)); });
  }
  measureRegister(b, q, size, result);

  return {result};
}

SmallVector<Value> ghzLinear(qc::QCProgramBuilder& b, const uint64_t n) {
  return ghz(b, benchmarks::GHZ({.qubits = static_cast<size_t>(n)}));
}

} // namespace mqt::benchmark
