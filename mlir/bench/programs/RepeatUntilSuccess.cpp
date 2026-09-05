/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "bench/RepeatUntilSuccess.hpp"

#include "Programs.h"
#include "mlir/Dialect/QC/Builder/QCProgramBuilder.h"

#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>

#include <cstdint>

namespace mqt::bench {

using namespace mlir;

SmallVector<Value> repeatUntilSuccess(qc::QCProgramBuilder& builder,
                                      const RepeatUntilSuccess& benchmark) {
  auto ancilla = builder.allocQubit();
  auto data = builder.allocQubit();
  auto result = builder.allocClassicalBitRegister(
      static_cast<int64_t>(benchmark.output().width), benchmark.output().name);

  builder.scfWhile(
      [&] {
        // Paetznick--Svore, Figure 8.
        builder.h(ancilla);
        builder.t(ancilla);
        builder.cx(ancilla, data);
        builder.h(ancilla);
        builder.cx(ancilla, data);
        builder.t(ancilla);
        builder.h(ancilla);

        // Outcome one is failure, so it is also the continuation condition.
        auto failure = builder.measure(ancilla);
        builder.scfCondition(failure);
      },
      [&] {
        // Failure leaves the data unchanged and the ancilla in |1>.
        builder.x(ancilla);
      });

  // Read the relative phase of (I + i sqrt(2) X)|0> / sqrt(3).
  builder.sdg(data);
  builder.h(data);
  builder.measure(data, result, 0);
  return {result};
}

} // namespace mqt::bench
