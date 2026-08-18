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

namespace mqt::benchmarks {

using namespace mlir;

namespace {
/// The rotation that prepares the weights of the linear combination.
constexpr double ENCODING_ANGLE = std::numbers::pi / 3.0;
} // namespace

SmallVector<Value> blockEncoding(qc::QCProgramBuilder& b, const uint64_t n) {
  const auto size = static_cast<int64_t>(n) - 2;
  auto anc = b.allocQubitRegister(2, "anc");
  auto q = b.allocQubitRegister(size, "q");
  auto flag = b.allocClassicalBitRegister(2, "flag");
  auto c = b.allocClassicalBitRegister(size, "c");

  b.scfFor(0, 2, 1, [&](Value iv) { b.reset(b.loadQubit(anc.value, iv)); });
  b.scfFor(0, size, 1, [&](Value iv) { b.reset(b.loadQubit(q.value, iv)); });
  b.scfFor(0, size, 1, [&](Value iv) { b.h(b.loadQubit(q.value, iv)); });

  // The ancillas hold the weights of the linear combination of unitaries.
  b.ry(ENCODING_ANGLE, anc[0]);
  b.ry(ENCODING_ANGLE, anc[1]);

  // Each ancilla state selects one term. The controls are flipped so that the
  // selected term fires on exactly one state of the ancilla register.
  for (int64_t term = 0; term < 4; ++term) {
    for (int64_t bit = 0; bit < 2; ++bit) {
      if (((term >> bit) & 1) == 0) {
        b.x(anc[bit]);
      }
    }
    const SmallVector<Value> controls{anc[0], anc[1]};
    b.scfFor(0, size, 1,
             [&](Value i) { b.mcz(controls, b.loadQubit(q.value, i)); });
    for (int64_t bit = 0; bit < 2; ++bit) {
      if (((term >> bit) & 1) == 0) {
        b.x(anc[bit]);
      }
    }
  }

  b.ry(-ENCODING_ANGLE, anc[1]);
  b.ry(-ENCODING_ANGLE, anc[0]);

  // The encoding succeeds when both ancillas return to zero. A failed round is
  // corrected on the system register.
  b.measure(anc[0], flag, 0);
  b.measure(anc[1], flag, 1);
  b.scfIf(flag, 0, [&] { b.z(q[0]); });
  b.scfIf(flag, 1, [&] { b.z(q[0]); });

  b.scfFor(0, size, 1,
           [&](Value iv) { b.measure(b.loadQubit(q.value, iv), c, iv); });

  return {flag, c};
}

} // namespace mqt::benchmarks
