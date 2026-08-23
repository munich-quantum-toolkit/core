/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "bench/Grover.hpp"

#include "Programs.h"
#include "mlir/Dialect/QC/Builder/QCProgramBuilder.h"

#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>

#include <cstddef>

namespace mqt::bench {

using namespace mlir;

SmallVector<Value> grover(qc::QCProgramBuilder& b, const Grover& benchmark) {
  const auto& options = benchmark.options();
  const auto search = static_cast<int64_t>(benchmark.qubits());
  auto q = b.allocQubitRegister(search, "q");
  auto result = b.allocClassicalBitRegister(search, benchmark.output().name);

  b.scfFor(0, search, 1, [&](Value iv) { b.h(b.loadQubit(q.value, iv)); });

  const llvm::ArrayRef<Value> qubits(q.qubits);
  const auto controls = qubits.drop_back();
  const auto target = qubits.back();
  const auto iterations = static_cast<int64_t>(*options.iterations);

  b.scfFor(0, iterations, 1, [&](Value) {
    for (size_t i = 0; i < benchmark.qubits(); ++i) {
      if (options.markedBitstring[benchmark.qubits() - 1 - i] == '0') {
        b.x(q[i]);
      }
    }
    b.mcz(controls, target);
    for (size_t i = 0; i < benchmark.qubits(); ++i) {
      if (options.markedBitstring[benchmark.qubits() - 1 - i] == '0') {
        b.x(q[i]);
      }
    }

    b.scfFor(0, search, 1, [&](Value iv) { b.h(b.loadQubit(q.value, iv)); });
    b.scfFor(0, search, 1, [&](Value iv) { b.x(b.loadQubit(q.value, iv)); });
    b.mcz(controls, target);
    b.scfFor(0, search, 1, [&](Value iv) { b.x(b.loadQubit(q.value, iv)); });
    b.scfFor(0, search, 1, [&](Value iv) { b.h(b.loadQubit(q.value, iv)); });
  });

  b.measureQubitRegister(q.value, result, search);

  return {result};
}
} // namespace mqt::bench
