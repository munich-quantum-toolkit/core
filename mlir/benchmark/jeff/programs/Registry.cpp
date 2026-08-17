/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "Programs.h"

#include <llvm/ADT/SmallVector.h>

namespace mqt::jeff::benchmarks {

llvm::SmallVector<Benchmark> benchmarks() {
  return {
      {"ghz-linear", &ghzLinear, 1},
      {"ghz-star", &ghzStar, 1},
      {"grover", &grover, 2},
      {"iqft", &iqft, 1},
      {"iqpe", &iqpe, 1},
      {"multiplexer", &multiplexer, 2},
      {"qft", &qft, 1},
      {"qpe", &qpe, 2},
      {"teleportation", &teleportation, 1},
  };
}

} // namespace mqt::jeff::benchmarks
