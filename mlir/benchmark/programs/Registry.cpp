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
#include "benchmarks/QPE.hpp"
#include "mlir/Benchmark/Programs.h"

#include <mlir/Support/LLVM.h>

namespace mqt::benchmark {

SmallVector<Benchmark> benchmarks() {
  return {
      {.name = "ghz-linear",
       .build = &ghzLinear,
       .minimumSize = 1,
       .maximumSize = benchmarks::GHZOptions::MAX_QUBITS},
      {.name = "ghz-star",
       .build = &ghzStar,
       .minimumSize = 1,
       .maximumSize = benchmarks::GHZOptions::MAX_QUBITS},
      {.name = "grover", .build = &grover, .minimumSize = 3, .maximumSize = 63},
      {.name = "iqft", .build = &iqft, .minimumSize = 1},
      {.name = "iqpe",
       .build = &iqpe,
       .minimumSize = 1,
       .maximumSize = benchmarks::QPEOptions::MAX_PRECISION},
      // Limit n since the loop runs over the 2^(n-1) control states
      {.name = "multiplexer",
       .build = &multiplexer,
       .minimumSize = 2,
       .maximumSize = 20},
      {.name = "qft", .build = &qft, .minimumSize = 1},
      {.name = "qpe",
       .build = &qpe,
       .minimumSize = 2,
       .maximumSize = benchmarks::QPEOptions::MAX_PRECISION + 1},
      {.name = "teleportation", .build = &teleportation, .minimumSize = 1},
  };
}

} // namespace mqt::benchmark
