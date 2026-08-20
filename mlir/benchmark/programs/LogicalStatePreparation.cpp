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

SmallVector<Value> logicalStatePreparation(qc::QCProgramBuilder& b,
                                           const uint64_t n) {
  const auto size = static_cast<int64_t>(n) - 1;
  auto q = b.allocQubitRegister(size, "q");
  auto anc = b.allocQubit();
  auto c = b.allocClassicalBitRegister(size, "c");

  auto one = b.indexConstant(1);

  // The encoding is checked before it is accepted. A failed check throws the
  // block away and prepares it again, so the number of attempts is only known
  // at runtime.
  b.scfWhile(
      [&] {
        resetRegister(b, q.value, size);
        b.reset(anc);

        b.h(q[0]);
        b.scfFor(0, size - 1, 1, [&](Value i) {
          auto next = arith::AddIOp::create(b, i, one);
          b.cx(b.loadQubit(q.value, i), b.loadQubit(q.value, next));
        });

        // The check reads the parity of the encoded block.
        b.scfFor(0, size, 1,
                 [&](Value i) { b.cx(b.loadQubit(q.value, i), anc); });

        auto failed = b.measure(anc);
        b.scfCondition(failed);
      },
      [] {});

  measureRegister(b, q.value, size, c);

  return {c};
}

} // namespace mqt::benchmark
