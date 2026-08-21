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

SmallVector<Value> syndromeMeasurement(qc::QCProgramBuilder& b,
                                       const uint64_t n) {
  const auto size = static_cast<int64_t>(n) - 1;
  auto q = b.allocQubitRegister(size, "q");
  auto anc = b.allocQubit();
  auto syndrome = b.allocClassicalBitRegister(size, "syndrome");
  auto c = b.allocClassicalBitRegister(size, "c");

  resetRegister(b, q.value, size);
  b.reset(anc);

  auto one = b.indexConstant(1);

  // The block starts in an encoded state.
  b.h(q[0]);
  b.scfFor(0, size - 1, 1, [&](Value i) {
    auto next = arith::AddIOp::create(b, i, one);
    b.cx(b.loadQubit(q.value, i), b.loadQubit(q.value, next));
  });

  // A round reads one stabilizer per neighbouring pair and flips the qubits
  // that the syndrome marks. Rounds repeat while the first stabilizer still
  // reports an error, so their number depends on the measurements.
  b.scfWhile(
      [&] {
        b.scfFor(0, size - 1, 1, [&](Value i) {
          auto next = arith::AddIOp::create(b, i, one);
          b.reset(anc);
          b.cx(b.loadQubit(q.value, i), anc);
          b.cx(b.loadQubit(q.value, next), anc);
          b.measure(anc, syndrome, i);
        });

        b.scfFor(0, size - 1, 1, [&](Value i) {
          auto next = arith::AddIOp::create(b, i, one);
          b.scfIf(syndrome, i, [&] { b.x(b.loadQubit(q.value, next)); });
        });

        b.scfCondition(syndrome, 0);
      },
      [] {});

  measureRegister(b, q.value, size, c);

  return {syndrome, c};
}

} // namespace mqt::benchmark
