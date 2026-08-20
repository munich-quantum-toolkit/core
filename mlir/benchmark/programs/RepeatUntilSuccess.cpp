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
/// The number of ancillas that one round measures.
constexpr int64_t RUS_ANCILLAS = 2;
} // namespace

SmallVector<Value> repeatUntilSuccess(qc::QCProgramBuilder& b,
                                      const uint64_t /*n*/) {
  // The circuit couples a fixed number of ancillas to one target qubit, so the
  // size is fixed too.
  auto anc = b.allocQubitRegister(RUS_ANCILLAS, "anc");
  auto q = b.allocQubit();
  auto flag = b.allocClassicalBitRegister(RUS_ANCILLAS, "flag");
  auto c = b.allocClassicalBitRegister(1, "c");

  resetRegister(b, anc.value, RUS_ANCILLAS);
  b.reset(q);
  b.h(q);

  // A round applies a block of Clifford and T gates that entangles the
  // ancillas with the target, then reads both ancillas. The round succeeds
  // only when both read zero. A failed round leaves a Pauli correction on the
  // target, which the next round undoes before it tries again, so the number
  // of rounds depends on the measurements.
  b.scfWhile(
      [&] {
        resetRegister(b, anc.value, RUS_ANCILLAS);

        b.h(anc[0]);
        b.h(anc[1]);
        b.t(anc[0]);
        b.t(anc[1]);
        b.cx(anc[0], q);
        b.tdg(anc[1]);
        b.cz(anc[1], q);
        b.t(anc[0]);
        b.cx(anc[1], anc[0]);
        b.h(anc[0]);
        b.h(anc[1]);

        b.measure(anc[0], flag, 0);
        b.measure(anc[1], flag, 1);

        auto failed = arith::OrIOp::create(b, b.loadClassicalBit(flag, 0),
                                           b.loadClassicalBit(flag, 1));
        b.scfCondition(failed);
      },
      [&] { b.z(q); });

  b.measure(q, c, 0);

  return {flag, c};
}

} // namespace mqt::benchmark
