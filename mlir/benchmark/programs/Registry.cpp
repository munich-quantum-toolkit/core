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
      // The modulus needs three bits, and the layout is 2 * bits + 3. The
      // rounds derive their addend at run time, and `jeff` cannot turn that
      // integer into a rotation angle.
      {.name = "controlled-mult-mod-n",
       .build = &controlledMultiplyModN,
       .minimumSize = 9,
       .lowersToJeff = false},
      {.name = "fan-out", .build = &fanOut, .minimumSize = 2},
      {.name = "grover-weak-measurement",
       .build = &groverWeakMeasurement,
       .minimumSize = 4},
      {.name = "logical-state-preparation",
       .build = &logicalStatePreparation,
       .minimumSize = 3},
      // The protocol consumes five copies, so the size is fixed.
      {.name = "magic-state-distillation",
       .build = &magicStateDistillation,
       .minimumSize = 1},
      {.name = "measurement-based-computation",
       .build = &measurementBasedComputation,
       .minimumSize = 1},
      {.name = "ml-qae", .build = &mlqae, .minimumSize = 2},
      // The circuit couples two ancillas to one target, so the size is fixed.
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
      // The layout is 2 * bits + 3 and the modulus needs three bits. The
      // rounds derive their multiplier at run time, and `jeff` cannot turn
      // that integer into a rotation angle.
      {.name = "shor", .build = &shor, .minimumSize = 9, .lowersToJeff = false},
      {.name = "syndrome-measurement",
       .build = &syndromeMeasurement,
       .minimumSize = 3},
      {.name = "toffoli-heavy", .build = &toffoliHeavy, .minimumSize = 3},
      // The optimizer counts the energy from the measured bits, and `jeff`
      // cannot widen a measured bit into an integer.
      {.name = "vqe", .build = &vqe, .minimumSize = 2, .lowersToJeff = false},
      {.name = "vqe-ansatz", .build = &vqeAnsatz, .minimumSize = 2},
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
