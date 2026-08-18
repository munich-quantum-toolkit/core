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
#include <numbers>

namespace mqt::benchmark {

using namespace mlir;

SmallVector<Value> qftAdderQuantum(qc::QCProgramBuilder& b, const uint64_t n) {
  const auto size = static_cast<int64_t>(n) / 2;
  auto x = b.allocQubitRegister(size, "x");
  auto y = b.allocQubitRegister(size, "y");
  auto c = b.allocClassicalBitRegister(size, "c");

  resetRegister(b, x.value, size);
  resetRegister(b, y.value, size);
  b.scfFor(0, size, 1, [&](Value iv) { b.h(b.loadQubit(x.value, iv)); });

  // The Draper adder writes the sum into the Fourier basis of `y`. The phase
  // that qubit `x[j - d]` contributes to `y[j]` halves as the distance grows.
  fourierTransform(b, y.value, size, 1.0);

  auto one = b.indexConstant(1);
  b.scfFor(0, size, 1, [&](Value j) {
    auto count = arith::AddIOp::create(b, j, one);
    auto lower = b.indexConstant(0);

    scfForWithAngle(
        b, lower, count, std::numbers::pi, 0.5, [&](Value angle, Value step) {
          auto source = arith::SubIOp::create(b, j, step);
          b.cp(angle, b.loadQubit(x.value, source), b.loadQubit(y.value, j));
        });
  });

  fourierTransform(b, y.value, size, -1.0);

  measureRegister(b, y.value, size, c);

  return {c};
}

} // namespace mqt::benchmark
