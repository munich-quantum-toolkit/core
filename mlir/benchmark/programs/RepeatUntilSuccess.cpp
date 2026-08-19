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

#include <cstdint>
#include <numbers>

namespace mqt::benchmark {

using namespace mlir;

namespace {
/// The rotation that a failed round leaves on the target qubit.
constexpr double RUS_ANGLE = std::numbers::pi / 4.0;
} // namespace

SmallVector<Value> repeatUntilSuccess(qc::QCProgramBuilder& b,
                                      const uint64_t /*n*/) {
  auto anc = b.allocQubit();
  auto q = b.allocQubit();
  auto c = b.allocClassicalBitRegister(1, "c");

  b.reset(anc);
  b.reset(q);
  b.h(q);

  // The gadget applies the wanted rotation for one measurement outcome only.
  // A failed round leaves a known rotation behind, which the next round undoes
  // before it tries again. The round count follows from the measurements.
  b.scfWhile(
      [&] {
        b.reset(anc);
        b.h(anc);
        b.t(anc);
        b.cx(q, anc);
        b.tdg(anc);
        b.h(anc);
        auto failed = b.measure(anc);
        b.scfCondition(failed);
      },
      [&] { b.rz(-RUS_ANGLE, q); });

  b.measure(q, c, 0);

  return {c};
}

} // namespace mqt::benchmark
