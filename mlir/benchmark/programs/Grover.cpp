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

#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>

#include <cmath>
#include <cstdint>
#include <numbers>

namespace mqt::benchmark {

using namespace mlir;

SmallVector<Value> grover(qc::QCProgramBuilder& b, const uint64_t n) {
  const auto search = static_cast<int64_t>(n) - 1;
  auto q = b.allocQubitRegister(search, "q");
  auto flag = b.allocQubit();
  auto c = b.allocClassicalBitRegister(search, "c");

  b.scfFor(0, search, 1, [&](Value iv) { b.reset(b.loadQubit(q.value, iv)); });
  b.reset(flag);

  b.scfFor(0, search, 1, [&](Value iv) { b.h(b.loadQubit(q.value, iv)); });
  b.x(flag);

  const auto iterations = static_cast<int64_t>(
      std::ceil(std::numbers::pi / 4.0 *
                std::sqrt(std::pow(2.0, static_cast<double>(search)))));

  // The oracle marks the all-ones state. The diffusion operator reflects about
  // the uniform superposition.
  const llvm::ArrayRef<Value> oracleControls(q.qubits);
  const auto diffusionControls = oracleControls.drop_back();

  b.scfFor(0, iterations, 1, [&](Value) {
    b.mcz(oracleControls, flag);

    b.scfFor(0, search, 1, [&](Value iv) { b.h(b.loadQubit(q.value, iv)); });
    b.scfFor(0, search, 1, [&](Value iv) { b.x(b.loadQubit(q.value, iv)); });
    b.mcz(diffusionControls, q[search - 1]);
    b.scfFor(0, search, 1, [&](Value iv) { b.x(b.loadQubit(q.value, iv)); });
    b.scfFor(0, search, 1, [&](Value iv) { b.h(b.loadQubit(q.value, iv)); });
  });

  b.scfFor(0, search, 1,
           [&](Value iv) { b.measure(b.loadQubit(q.value, iv), c, iv); });

  return {c};
}

} // namespace mqt::benchmark
