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

#include "mqt/Dialect/QC/Builder/QCProgramBuilder.h"

#include "Programs.h"

#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"

#include <cstdint>

namespace mqt::bench {

using namespace mlir;

SmallVector<Value> repeatUntilSuccess(qc::QCProgramBuilder& builder,
                                      const RepeatUntilSuccess& benchmark) {
  auto ancilla = builder.allocQubit();
  const auto size = static_cast<int64_t>(benchmark.options().dataQubits);
  auto data = builder.allocQubitRegisterStorage(size, "data");
  auto first = builder.loadQubit(data, builder.indexConstant(0));
  auto result = builder.allocClassicalBitRegister(
      static_cast<int64_t>(benchmark.output().width), benchmark.output().name);

  builder.scfWhile(
      [&] {
        /// Paetznick--Svore, Figure 8, with X replaced by X tensor ... tensor
        /// X.
        builder.h(ancilla);
        builder.t(ancilla);
        builder.scfFor(0, size, 1, [&](Value iv) {
          builder.cx(ancilla, builder.loadQubit(data, iv));
        });
        builder.h(ancilla);
        builder.scfFor(0, size, 1, [&](Value iv) {
          builder.cx(ancilla, builder.loadQubit(data, iv));
        });
        builder.t(ancilla);
        builder.h(ancilla);

        /// Outcome one is failure, so it is also the continuation condition.
        auto failure = builder.measure(ancilla);
        builder.scfCondition(failure);
      },
      [&] {
        /// Failure leaves the data unchanged and the ancilla in |1>.
        builder.x(ancilla);
      });

  /// Measure Y on the first data qubit and X on the rest, returning parity.
  builder.sdg(first);
  builder.scfFor(0, size, 1,
                 [&](Value iv) { builder.h(builder.loadQubit(data, iv)); });
  builder.scfFor(1, size, 1, [&](Value iv) {
    builder.cx(builder.loadQubit(data, iv), first);
  });
  builder.measure(first, result, 0);
  return {result};
}

} // namespace mqt::bench
