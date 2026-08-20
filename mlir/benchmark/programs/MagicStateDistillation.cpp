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

namespace {
/// The number of noisy copies that one round consumes.
constexpr int64_t MAGIC_COPIES = 5;
/// The number of stabilizers of the five-qubit code.
constexpr int64_t MAGIC_STABILIZERS = 4;
/// The stabilizers of the five-qubit code, which are the cyclic shifts of
/// XZZXI. A dot marks the identity.
constexpr char MAGIC_GENERATORS[MAGIC_STABILIZERS][MAGIC_COPIES] = {
    {'X', 'Z', 'Z', 'X', '.'},
    {'.', 'X', 'Z', 'Z', 'X'},
    {'X', '.', 'X', 'Z', 'Z'},
    {'Z', 'X', '.', 'X', 'Z'},
};
} // namespace

SmallVector<Value> magicStateDistillation(qc::QCProgramBuilder& b,
                                          const uint64_t /*n*/) {
  // The protocol consumes a fixed number of copies, so the size is fixed too.
  auto q = b.allocQubitRegister(MAGIC_COPIES, "q");
  auto anc = b.allocQubit();
  auto syndrome = b.allocClassicalBitRegister(MAGIC_STABILIZERS, "syndrome");
  auto c = b.allocClassicalBitRegister(MAGIC_COPIES, "c");

  resetRegister(b, q.value, MAGIC_COPIES);
  b.reset(anc);

  // A round prepares five noisy copies and reads the four stabilizers of the
  // five-qubit code. It is accepted only when every syndrome bit is trivial,
  // and a rejected round throws the copies away and starts over, so the number
  // of rounds depends on the measurements.
  b.scfWhile(
      [&] {
        resetRegister(b, q.value, MAGIC_COPIES);
        b.scfFor(0, MAGIC_COPIES, 1, [&](Value i) {
          auto qubit = b.loadQubit(q.value, i);
          b.h(qubit);
          b.t(qubit);
        });

        for (int64_t s = 0; s < MAGIC_STABILIZERS; ++s) {
          b.reset(anc);
          b.h(anc);
          for (int64_t j = 0; j < MAGIC_COPIES; ++j) {
            if (MAGIC_GENERATORS[s][j] == 'X') {
              b.cx(anc, q[j]);
            } else if (MAGIC_GENERATORS[s][j] == 'Z') {
              b.cz(anc, q[j]);
            }
          }
          b.h(anc);
          b.measure(anc, syndrome, s);
        }

        auto rejected = b.loadClassicalBit(syndrome, 0);
        for (int64_t s = 1; s < MAGIC_STABILIZERS; ++s) {
          rejected = arith::OrIOp::create(b, rejected,
                                          b.loadClassicalBit(syndrome, s));
        }
        b.scfCondition(rejected);
      },
      [] {});

  measureRegister(b, q.value, MAGIC_COPIES, c);

  return {syndrome, c};
}

} // namespace mqt::benchmark
