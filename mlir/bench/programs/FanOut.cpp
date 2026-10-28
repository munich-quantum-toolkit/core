/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "bench/FanOut.hpp"

#include "mqt/Dialect/QC/Builder/QCProgramBuilder.h"

#include "Programs.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"

#include <cstdint>

namespace mqt::bench {

using namespace mlir;

SmallVector<Value> fanOut(qc::QCProgramBuilder& builder,
                          const FanOut& benchmark) {
  const auto size = static_cast<int64_t>(benchmark.options().controls);
  auto controls = builder.allocQubitRegisterStorage(size, "controls");
  auto workspace = builder.allocQubitRegisterStorage(size, "workspace");
  auto result =
      builder.allocClassicalBitRegister(size + 1, benchmark.output().name);
  auto target = builder.loadQubit(workspace, builder.indexConstant(0));

  builder.h(target);

  const auto fanOutTarget = [&] {
    builder.scfFor(1, size, 1, [&](Value iv) {
      builder.cx(target, builder.loadQubit(workspace, iv));
    });
  };
  fanOutTarget();
  // Figure 5's fixed phase is S, not the usual Rz(pi/2) gate.
  builder.scfFor(0, size, 1, [&](Value iv) {
    auto control = builder.loadQubit(controls, iv);
    builder.h(control);
    builder.cs(control, builder.loadQubit(workspace, iv));
  });
  fanOutTarget();

  builder.h(target);
  builder.measure(target, result, 0);
  auto one = builder.indexConstant(1);
  builder.scfFor(0, size, 1, [&](Value iv) {
    auto bit = arith::AddIOp::create(builder, iv, one);
    builder.measure(builder.loadQubit(controls, iv), result, bit);
  });
  return {result};
}

} // namespace mqt::bench
