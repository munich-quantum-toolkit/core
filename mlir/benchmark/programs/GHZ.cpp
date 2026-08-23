/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "bench/GHZ.hpp"

#include "Programs.h"
#include "mlir/Dialect/QC/Builder/QCProgramBuilder.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>

#include <cstdint>

namespace mqt::bench {

using namespace mlir;

SmallVector<Value> ghz(qc::QCProgramBuilder& b, const GHZ& benchmark) {
  const auto& options = benchmark.options();
  const auto size = static_cast<int64_t>(options.qubits);
  auto q = b.allocQubitRegisterStorage(size, "q");
  auto result = b.allocClassicalBitRegister(size, benchmark.output().name);

  auto root = b.loadQubit(q, arith::ConstantIndexOp::create(b, 0).getResult());
  b.h(root);
  if (options.topology == GHZTopology::Linear) {
    auto one = arith::ConstantIndexOp::create(b, 1).getResult();
    b.scfFor(1, size, 1, [&](Value iv) {
      auto previous = arith::SubIOp::create(b, iv, one);
      b.cx(b.loadQubit(q, previous), b.loadQubit(q, iv));
    });
  } else {
    b.scfFor(1, size, 1, [&](Value iv) { b.cx(root, b.loadQubit(q, iv)); });
  }

  if (options.basis == GHZBasis::X) {
    b.scfFor(0, size, 1, [&](Value iv) { b.h(b.loadQubit(q, iv)); });
  }
  b.measureQubitRegister(q, result, size);

  return {result};
}
} // namespace mqt::bench
