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

#include "mqt/Dialect/QC/Builder/QCProgramBuilder.h"

#include "Programs.h"
#include "QFTUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"

#include <cstdint>
#include <numbers>

namespace mqt::bench {

using namespace mlir;

SmallVector<Value> multiplexer(qc::QCProgramBuilder& builder,
                               const Multiplexer& benchmark) {
  const auto numControls = static_cast<int64_t>(benchmark.options().qubits - 1);
  auto controls = builder.allocQubitRegisterStorage(numControls, "controls");
  auto target = builder.allocQubit();
  auto result = builder.allocClassicalBitRegister(
      static_cast<int64_t>(benchmark.output().width), benchmark.output().name);

  auto zero = builder.indexConstant(0);
  auto one = builder.indexConstant(1);

  builder.scfFor(0, numControls, 1, [&](Value index) {
    builder.h(builder.loadQubit(controls, index));
  });

  auto upper = builder.indexConstant(numControls);
  auto last = builder.indexConstant(numControls - 1);
  auto firstAngle = builder.floatConstant(std::numbers::pi / 2.);
  auto half = builder.floatConstant(0.5);
  detail::phaseRotationLoop(
      builder, zero, upper, one, firstAngle, half,
      [&](Value angle, Value index) {
        auto control = arith::SubIOp::create(builder, last, index);
        builder.cry(angle, builder.loadQubit(controls, control), target);
      });

  builder.measure(target, result, 0);
  builder.scfFor(0, numControls, 1, [&](Value index) {
    auto resultIndex = arith::AddIOp::create(builder, index, one);
    builder.measure(builder.loadQubit(controls, index), result, resultIndex);
  });
  return {result};
}

} // namespace mqt::bench
