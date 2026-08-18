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
/// The rotation that sets the amplitude the estimation recovers.
constexpr double MLQAE_ANGLE = std::numbers::pi / 5.0;
} // namespace

SmallVector<Value> mlqae(qc::QCProgramBuilder& b, const uint64_t n) {
  const auto size = static_cast<int64_t>(n) - 1;
  auto q = b.allocQubitRegister(size, "q");
  auto flag = b.allocQubit();
  auto c = b.allocClassicalBitRegister(size, "c");

  const SmallVector<Value> controls(q.qubits);
  auto one = b.indexConstant(1);

  // Every round prepares the state again and then applies the Grover operator
  // a number of times that doubles from round to round. The schedule turns the
  // bound of the inner loop into a runtime value.
  b.scfFor(0, size, 1, [&](Value k) {
    b.scfFor(0, size, 1, [&](Value i) { b.reset(b.loadQubit(q.value, i)); });
    b.reset(flag);

    b.scfFor(0, size, 1, [&](Value i) { b.h(b.loadQubit(q.value, i)); });
    b.mcry(MLQAE_ANGLE, controls, flag);

    auto power = arith::ShLIOp::create(b, one, k);
    b.scfFor(0, power, 1, [&](Value) {
      // Mark the good subspace, then reflect about the prepared state.
      b.z(flag);
      b.mcry(-MLQAE_ANGLE, controls, flag);
      b.scfFor(0, size, 1, [&](Value i) { b.h(b.loadQubit(q.value, i)); });
      b.scfFor(0, size, 1, [&](Value i) { b.x(b.loadQubit(q.value, i)); });
      b.x(flag);
      b.mcz(controls, flag);
      b.x(flag);
      b.scfFor(0, size, 1, [&](Value i) { b.x(b.loadQubit(q.value, i)); });
      b.scfFor(0, size, 1, [&](Value i) { b.h(b.loadQubit(q.value, i)); });
      b.mcry(MLQAE_ANGLE, controls, flag);
    });

    b.measure(flag, c, k);
  });

  return {c};
}

} // namespace mqt::benchmark
