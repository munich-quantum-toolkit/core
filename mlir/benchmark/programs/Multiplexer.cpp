/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "BenchmarkUtils.h"
#include "mlir/Benchmark/Programs.h"
#include "mlir/Dialect/QC/Builder/QCProgramBuilder.h"

#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>

#include <cstdint>
#include <numbers>

namespace mqt::benchmark {

using namespace mlir;

SmallVector<Value> multiplexer(qc::QCProgramBuilder& b, const uint64_t n) {
  const auto numControls = static_cast<int64_t>(n) - 1;
  const auto numStates = int64_t{1} << numControls;

  auto controls = b.allocQubitRegister(numControls, "controls");
  auto target = b.allocQubit();
  auto c = b.allocClassicalBitRegister(numControls, "c");
  auto outcome = b.allocClassicalBitRegister(1, "outcome");

  resetRegister(b, controls.value, numControls);
  b.reset(target);

  // Each control state selects its own rotation. The controls are flipped so
  // that the fully controlled gate fires exactly on the current state.
  for (int64_t state = 0; state < numStates; ++state) {
    const auto angle = static_cast<double>(state) * std::numbers::pi /
                       static_cast<double>(numStates);

    for (int64_t bit = 0; bit < numControls; ++bit) {
      if (((state >> bit) & 1) == 0) {
        b.x(controls[bit]);
      }
    }

    b.mcry(angle, controls.qubits, target);

    for (int64_t bit = 0; bit < numControls; ++bit) {
      if (((state >> bit) & 1) == 0) {
        b.x(controls[bit]);
      }
    }
  }

  measureRegister(b, controls.value, numControls, c);
  b.measure(target, outcome, 0);

  return {c, outcome};
}

} // namespace mqt::benchmark
