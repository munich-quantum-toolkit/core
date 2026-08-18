/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Benchmark/Programs.h"
#include "mlir/Dialect/QC/Builder/QCProgramBuilder.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>

#include <cstdint>
#include <numbers>

namespace mqt::benchmark {

using namespace mlir;

SmallVector<Value> toffoliHeavy(qc::QCProgramBuilder& b, const uint64_t n) {
  const auto size = static_cast<int64_t>(n);
  auto q = b.allocQubitRegister(size, "q");
  auto c = b.allocClassicalBitRegister(size, "c");

  b.scfFor(0, size, 1, [&](Value iv) { b.reset(b.loadQubit(q.value, iv)); });
  b.scfFor(0, size, 1, [&](Value iv) { b.h(b.loadQubit(q.value, iv)); });

  // A chain of Toffoli gates. Each gate takes its two controls from the
  // preceding qubits, so the chain carries a running AND along the register.
  auto one = b.indexConstant(1);
  auto two = b.indexConstant(2);
  b.scfFor(0, size - 2, 1, [&](Value iv) {
    auto second = arith::AddIOp::create(b, iv, one);
    auto target = arith::AddIOp::create(b, iv, two);
    const SmallVector<Value> controls{b.loadQubit(q.value, iv),
                                      b.loadQubit(q.value, second)};
    b.mcx(controls, b.loadQubit(q.value, target));
  });

  b.scfFor(0, size, 1,
           [&](Value iv) { b.measure(b.loadQubit(q.value, iv), c, iv); });

  return {c};
}

} // namespace mqt::benchmark
