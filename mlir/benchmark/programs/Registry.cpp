/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Benchmark/Programs.h"

#include <mlir/Support/LLVM.h>

namespace mqt::benchmark {

SmallVector<Benchmark> benchmarks() {
  return {
      {.name = "ghz-linear", .build = &ghzLinear, .minimumSize = 1},
      {.name = "ghz-star", .build = &ghzStar, .minimumSize = 1},
      {.name = "grover", .build = &grover, .minimumSize = 2},
      {.name = "iqft", .build = &iqft, .minimumSize = 1},
      {.name = "iqpe", .build = &iqpe, .minimumSize = 1},
      // Every control state applies its own rotation, so the program emits
      // 2^(n-1) of them. Beyond this size the emission exhausts memory.
      {.name = "multiplexer",
       .build = &multiplexer,
       .minimumSize = 2,
       .maximumSize = 20},
      {.name = "qft", .build = &qft, .minimumSize = 1},
      {.name = "qpe", .build = &qpe, .minimumSize = 2},
      {.name = "teleportation", .build = &teleportation, .minimumSize = 1},
  };
}

} // namespace mqt::benchmark
