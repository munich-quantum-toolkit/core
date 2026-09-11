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

#include "mqt/Dialect/QC/Builder/QCProgramBuilder.h"

#include "Programs.h"

#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"

#include <cstdint>

namespace mqt::bench {

using namespace mlir;

SmallVector<Value> teleportation(qc::QCProgramBuilder& builder,
                                 const Teleportation& benchmark) {
  auto message = builder.allocQubit();
  auto alice = builder.allocQubit();
  auto bob = builder.allocQubit();
  auto result = builder.allocClassicalBitRegister(
      static_cast<int64_t>(benchmark.output().width), benchmark.output().name);

  builder.h(message);
  builder.h(alice);
  builder.cx(alice, bob);

  builder.cx(message, alice);
  builder.h(message);

  auto messageMeasurement = builder.measure(message);
  auto aliceMeasurement = builder.measure(alice);
  builder.scfIf(aliceMeasurement, [&] { builder.x(bob); });
  builder.scfIf(messageMeasurement, [&] { builder.z(bob); });

  builder.h(bob);
  builder.measure(bob, result, 0);
  return {result};
}

} // namespace mqt::bench
