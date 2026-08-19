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
      {.name = "block-encoding", .build = &blockEncoding, .minimumSize = 3},
      {.name = "controlled-mult-mod-n",
       .build = &controlledMultiplyModN,
       .minimumSize = 4},
      {.name = "fan-out", .build = &fanOut, .minimumSize = 2},
      {.name = "grover-weak-measurement",
       .build = &groverWeakMeasurement,
       .minimumSize = 4},
      {.name = "logical-state-preparation",
       .build = &logicalStatePreparation,
       .minimumSize = 3},
      {.name = "magic-state-distillation",
       .build = &magicStateDistillation,
       .minimumSize = 2},
      {.name = "measurement-based-computation",
       .build = &measurementBasedComputation,
       .minimumSize = 1},
      {.name = "ml-qae", .build = &mlqae, .minimumSize = 2},
      {.name = "repeat-until-success",
       .build = &repeatUntilSuccess,
       .minimumSize = 1},
      {.name = "qaoa", .build = &qaoa, .minimumSize = 3},
      {.name = "qft-adder-classical",
       .build = &qftAdderClassical,
       .minimumSize = 1},
      {.name = "qft-adder-quantum",
       .build = &qftAdderQuantum,
       .minimumSize = 2},
      {.name = "shor", .build = &shor, .minimumSize = 4},
      {.name = "syndrome-measurement",
       .build = &syndromeMeasurement,
       .minimumSize = 3},
      {.name = "toffoli-heavy", .build = &toffoliHeavy, .minimumSize = 3},
      {.name = "vqe", .build = &vqe, .minimumSize = 2},
      {.name = "vqe-ansatz", .build = &vqeAnsatz, .minimumSize = 2},
      {.name = "ghz-linear", .build = &ghzLinear, .minimumSize = 1},
      {.name = "ghz-star", .build = &ghzStar, .minimumSize = 1},
      {.name = "grover", .build = &grover, .minimumSize = 2},
      {.name = "iqft", .build = &iqft, .minimumSize = 1},
      {.name = "iqpe", .build = &iqpe, .minimumSize = 1},
      {.name = "multiplexer", .build = &multiplexer, .minimumSize = 2},
      {.name = "qft", .build = &qft, .minimumSize = 1},
      {.name = "qpe", .build = &qpe, .minimumSize = 2},
      {.name = "teleportation", .build = &teleportation, .minimumSize = 1},
  };
}

} // namespace mqt::benchmark
