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

SmallVector<Value> magicStateDistillation(qc::QCProgramBuilder& b,
                                          const uint64_t n) {
  const auto size = static_cast<int64_t>(n) - 1;
  auto q = b.allocQubitRegister(size, "q");
  auto anc = b.allocQubit();
  auto c = b.allocClassicalBitRegister(size, "c");

  // A round prepares noisy magic states, folds their parity onto the check
  // qubit, and reads it. The distillation discards a rejected round and starts
  // over, so the number of rounds depends on the measurements.
  b.scfWhile(
      [&] {
        resetRegister(b, q.value, size);
        b.reset(anc);

        b.scfFor(0, size, 1, [&](Value i) {
          auto qubit = b.loadQubit(q.value, i);
          b.h(qubit);
          b.t(qubit);
        });

        b.h(anc);
        b.scfFor(0, size, 1,
                 [&](Value i) { b.cx(b.loadQubit(q.value, i), anc); });
        b.h(anc);

        auto rejected = b.measure(anc);
        b.scfCondition(rejected);
      },
      [] {});

  measureRegister(b, q.value, size, c);

  return {c};
}

} // namespace mqt::benchmark
