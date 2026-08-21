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

namespace mqt::benchmark {

using namespace mlir;

namespace {
/// The multiplier of the modular multiplication.
constexpr int64_t MULTIPLIER = 3;
/// The modulus of the modular multiplication. It is coprime to the multiplier
/// and fits in the width that the minimum size guarantees.
constexpr int64_t MODULUS = 5;
} // namespace

SmallVector<Value> controlledMultiplyModN(qc::QCProgramBuilder& b,
                                          const uint64_t n) {
  // The register layout follows Beauregard: one control qubit, the multiplier,
  // an accumulator that carries one extra qubit for the overflow, and one
  // ancilla.
  const auto bits = (static_cast<int64_t>(n) - 3) / 2;
  const auto width = bits + 1;
  auto ctrl = b.allocQubit();
  auto x = b.allocQubitRegister(bits, "x");
  auto acc = b.allocQubitRegister(width, "acc");
  auto anc = b.allocQubit();
  auto c = b.allocClassicalBitRegister(width, "c");

  b.reset(ctrl);
  resetRegister(b, x.value, bits);
  resetRegister(b, acc.value, width);
  b.reset(anc);

  b.h(ctrl);
  b.scfFor(0, bits, 1, [&](Value i) { b.h(b.loadQubit(x.value, i)); });

  modularMultiply(b, ctrl, x.value, acc.value, anc, bits,
                  b.intConstant(MULTIPLIER % MODULUS), MODULUS, 1.0);

  measureRegister(b, acc.value, width, c);

  return {c};
}

} // namespace mqt::benchmark
