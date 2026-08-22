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

#include <mlir/Dialect/Arith/IR/Arith.h>
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

  auto zero = b.indexConstant(0);
  auto one = b.indexConstant(1);
  auto states = b.indexConstant(numStates);

  // The fully controlled gate fires on the all-ones state, so every control
  // that the current state holds at zero is flipped before the gate and back
  // afterwards.
  const auto flipZeroControls = [&](Value state) {
    b.scfFor(0, numControls, 1, [&](Value bit) {
      auto lowBit =
          arith::AndIOp::create(b, arith::ShRSIOp::create(b, state, bit), one);
      auto isZero =
          arith::CmpIOp::create(b, arith::CmpIPredicate::eq, lowBit, zero)
              .getResult();
      b.scfIf(isZero, [&] { b.x(b.loadQubit(controls.value, bit)); });
    });
  };

  // Each control state selects its own rotation, and the angles are spaced
  // evenly over the states.
  const auto increment = std::numbers::pi / static_cast<double>(numStates);
  uniformRotationLoop(b, zero, states, 0.0, increment,
                      [&](Value angle, Value state) {
                        flipZeroControls(state);
                        b.mcry(angle, controls.qubits, target);
                        flipZeroControls(state);
                      });

  measureRegister(b, controls.value, numControls, c);
  b.measure(target, outcome, 0);

  return {c, outcome};
}

} // namespace mqt::benchmark
