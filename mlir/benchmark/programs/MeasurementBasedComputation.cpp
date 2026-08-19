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

namespace {
/// The number of qubits in the cluster chain.
constexpr int64_t MBQC_CLUSTER = 4;
/// The measurement angle that the earlier outcomes flip.
constexpr double MBQC_ANGLE = std::numbers::pi / 4.0;
} // namespace

SmallVector<Value> measurementBasedComputation(qc::QCProgramBuilder& b,
                                               const uint64_t /*n*/) {
  auto q = b.allocQubitRegister(MBQC_CLUSTER, "q");
  auto c = b.allocClassicalBitRegister(MBQC_CLUSTER, "c");

  resetRegister(b, q.value, MBQC_CLUSTER);

  // The cluster state is a chain of qubits in the plus state that CZ joins.
  b.scfFor(0, MBQC_CLUSTER, 1, [&](Value i) { b.h(b.loadQubit(q.value, i)); });
  auto one = b.indexConstant(1);
  b.scfFor(0, MBQC_CLUSTER - 1, 1, [&](Value i) {
    auto next = arith::AddIOp::create(b, i, one);
    b.cz(b.loadQubit(q.value, i), b.loadQubit(q.value, next));
  });

  // The computation travels along the chain: every qubit is measured in the
  // X-Y plane, and the outcome picks the basis of the next measurement. The
  // sign of the rotation therefore follows the result of the round before it.
  b.rz(MBQC_ANGLE, q[0]);
  for (int64_t i = 0; i < MBQC_CLUSTER - 1; ++i) {
    b.h(q[i]);
    b.measure(q[i], c, i);
    b.scfIf(
        c, i, [&] { b.rz(-MBQC_ANGLE, q[i + 1]); },
        [&] { b.rz(MBQC_ANGLE, q[i + 1]); });
  }

  // The last qubit carries the result and takes the final correction.
  b.scfIf(c, 0, [&] { b.x(q[MBQC_CLUSTER - 1]); });
  b.measure(q[MBQC_CLUSTER - 1], c, MBQC_CLUSTER - 1);

  return {c};
}

} // namespace mqt::benchmark
