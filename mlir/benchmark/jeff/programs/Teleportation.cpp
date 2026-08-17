/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "Programs.h"
#include "mlir/Dialect/QC/Builder/QCProgramBuilder.h"

#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>

#include <cstdint>

namespace mqt::jeff::benchmarks {

using namespace mlir;

SmallVector<Value> teleportation(qc::QCProgramBuilder& b,
                                 const uint64_t /*n*/) {
  auto msg = b.allocQubit();
  auto alice = b.allocQubit();
  auto bob = b.allocQubit();
  auto c = b.allocClassicalBitRegister(3, "c");

  b.reset(msg);
  b.reset(alice);
  b.reset(bob);

  // Alice holds a qubit in an unknown state.
  b.h(msg);

  // Alice and Bob share an entangled pair.
  b.h(alice);
  b.cx(alice, bob);

  // Alice prepares and measures.
  b.cx(msg, alice);
  b.h(msg);
  b.measure(msg, c, 0);
  b.measure(alice, c, 1);

  // Bob corrects from the measurement results.
  b.scfIf(c, 1, [&] { b.x(bob); });
  b.scfIf(c, 0, [&] { b.z(bob); });

  b.measure(bob, c, 2);

  return {c};
}

} // namespace mqt::jeff::benchmarks
