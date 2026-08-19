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
#include <numbers>

namespace mqt::benchmark {

using namespace mlir;

namespace {
/// The constant that the adder adds to the register.
constexpr int64_t ADDEND = 5;
} // namespace

SmallVector<Value> qftAdderClassical(qc::QCProgramBuilder& b,
                                     const uint64_t n) {
  const auto size = static_cast<int64_t>(n);
  auto q = b.allocQubitRegister(size, "q");
  auto c = b.allocClassicalBitRegister(size, "c");

  resetRegister(b, q.value, size);
  b.scfFor(0, size, 1, [&](Value iv) { b.h(b.loadQubit(q.value, iv)); });

  // Adding a classical constant in the Fourier basis needs no controls. Each
  // qubit takes one phase that depends only on the constant, so the whole
  // addition is a single layer of phase gates between the two transforms.
  fourierTransform(b, q.value, size, 1.0);

  for (int64_t i = 0; i < size; ++i) {
    const auto angle = std::numbers::pi * static_cast<double>(ADDEND) /
                       static_cast<double>(int64_t{1} << i);
    b.p(angle, q[size - 1 - i]);
  }

  fourierTransform(b, q.value, size, -1.0);

  measureRegister(b, q.value, size, c);

  return {c};
}

} // namespace mqt::benchmark
