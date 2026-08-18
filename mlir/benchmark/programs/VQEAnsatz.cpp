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

namespace {
/// The number of ansatz layers.
constexpr int64_t VQE_REPETITIONS = 3;
/// The rotation angle of one ansatz layer.
constexpr double VQE_ANGLE = 0.37;
} // namespace

SmallVector<Value> vqeAnsatz(qc::QCProgramBuilder& b, const uint64_t n) {
  const auto size = static_cast<int64_t>(n);
  auto q = b.allocQubitRegister(size, "q");
  auto c = b.allocClassicalBitRegister(size, "c");

  b.scfFor(0, size, 1, [&](Value iv) { b.reset(b.loadQubit(q.value, iv)); });

  // A hardware-efficient ansatz: every layer applies a rotation to each qubit
  // and then entangles neighbouring qubits along a chain.
  auto one = arith::ConstantIndexOp::create(b, 1);
  b.scfFor(0, VQE_REPETITIONS, 1, [&](Value) {
    b.scfFor(0, size, 1,
             [&](Value i) { b.ry(VQE_ANGLE, b.loadQubit(q.value, i)); });
    b.scfFor(0, size - 1, 1, [&](Value i) {
      auto next = arith::AddIOp::create(b, i, one);
      b.cx(b.loadQubit(q.value, i), b.loadQubit(q.value, next));
    });
  });

  b.scfFor(0, size, 1,
           [&](Value iv) { b.measure(b.loadQubit(q.value, iv), c, iv); });

  return {c};
}

} // namespace mqt::benchmark
