/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "bench/Multiplexer.hpp"

#include "Programs.h"
#include "mlir/Dialect/QC/Builder/QCProgramBuilder.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>

#include <cstdint>
#include <numbers>

namespace mqt::bench {

using namespace mlir;

SmallVector<Value> multiplexer(qc::QCProgramBuilder& builder,
                               const Multiplexer& benchmark) {
  const auto numControls = static_cast<int64_t>(benchmark.options().qubits - 1);
  const auto numStates = int64_t{1} << numControls;
  auto controls = builder.allocQubitRegister(numControls, "controls");
  auto target = builder.allocQubit();
  auto result = builder.allocClassicalBitRegister(
      static_cast<int64_t>(benchmark.output().width), benchmark.output().name);

  auto zero = builder.indexConstant(0);
  auto one = builder.indexConstant(1);

  const auto flipZeroControls = [&](Value state) {
    builder.scfFor(0, numControls, 1, [&](Value bitPosition) {
      auto shifted = arith::ShRSIOp::create(builder, state, bitPosition);
      auto bit = arith::AndIOp::create(builder, shifted, one);
      auto isZero =
          arith::CmpIOp::create(builder, arith::CmpIPredicate::eq, bit, zero);
      builder.scfIf(isZero, [&] {
        builder.x(builder.loadQubit(controls.value, bitPosition));
      });
    });
  };

  auto states = builder.indexConstant(numStates);
  auto firstAngle = builder.floatConstant(0.);
  auto angleIncrement =
      builder.floatConstant(std::numbers::pi / static_cast<double>(numStates));
  auto stateLoop =
      scf::ForOp::create(builder, zero, states, one, ValueRange{firstAngle});
  {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(stateLoop.getBody());
    auto angle = stateLoop.getRegionIterArg(0);
    auto state = stateLoop.getInductionVar();
    flipZeroControls(state);
    builder.mcry(angle, controls.qubits, target);
    flipZeroControls(state);
    auto nextAngle = arith::AddFOp::create(builder, angle, angleIncrement);
    scf::YieldOp::create(builder, ValueRange{nextAngle});
  }

  builder.measure(target, result, 0);
  builder.scfFor(0, numControls, 1, [&](Value index) {
    auto resultIndex = arith::AddIOp::create(builder, index, one);
    builder.measure(builder.loadQubit(controls.value, index), result,
                    resultIndex);
  });
  return {result};
}

} // namespace mqt::bench
