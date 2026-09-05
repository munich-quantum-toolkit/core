/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "bench/Teleportation.hpp"

#include "Programs.h"
#include "mlir/Dialect/QC/Builder/QCProgramBuilder.h"

#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>

#include <cstdint>

namespace mqt::bench {

using namespace mlir;

SmallVector<Value> teleportation(qc::QCProgramBuilder& builder,
                                 const Teleportation& benchmark) {
  auto msg = builder.allocQubit();
  auto alice = builder.allocQubit();
  auto bob = builder.allocQubit();
  auto result = builder.allocClassicalBitRegister(
      static_cast<int64_t>(benchmark.output().width), benchmark.output().name);

  builder.h(msg);
  builder.h(alice);
  builder.cx(alice, bob);

  builder.cx(msg, alice);
  builder.h(msg);

  auto a = builder.measure(msg, result, 0);
  auto b1 = builder.measure(alice, result, 1);
  builder.scfIf(b1, [&] { builder.x(bob); });
  builder.scfIf(a, [&] { builder.z(bob); });

  builder.measure(bob, result, 2);
  return {result};
}

} // namespace mqt::bench
