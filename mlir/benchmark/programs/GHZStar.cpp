/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "benchmarks/GHZ.hpp"
#include "mlir/Benchmark/Programs.h"

#include <mlir/Support/LLVM.h>

#include <cstddef>
#include <cstdint>

namespace mqt::benchmark {

using namespace mlir;

SmallVector<Value> ghzStar(qc::QCProgramBuilder& b, const uint64_t n) {
  return ghz(b, benchmarks::GHZ({.qubits = static_cast<size_t>(n),
                                 .topology = benchmarks::GHZTopology::Star}));
}

} // namespace mqt::benchmark
