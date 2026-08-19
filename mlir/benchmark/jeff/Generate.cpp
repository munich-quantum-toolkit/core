/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Benchmark/Jeff/Generate.h"

#include "mlir/Benchmark/Compile.h"
#include "mlir/Compiler/Programs.h"

#include <cstdint>
#include <optional>
#include <utility>

namespace mqt::benchmark {

using namespace mlir;

std::optional<JeffProgram> buildJeffProgram(const Benchmark& benchmark,
                                            const uint64_t n) {
  auto program = buildQCProgram(benchmark, n);
  if (!program) {
    return std::nullopt;
  }

  auto qco = std::move(*program).intoQCO();
  if (!qco) {
    return std::nullopt;
  }
  // jeff represents a modifier as attributes on a single gate, so modifiers
  // that wrap several operations are unrolled first. The optimization pipeline
  // can empty a modifier body, for example when it folds a zero-angle rotation
  // away, so the cleanup runs afterwards to erase the modifiers left behind.
  if (!qco->runPassPipeline("unroll-modifiers") ||
      !qco->runPassPipeline("mqt-qco-default") || !qco->cleanup()) {
    return std::nullopt;
  }

  return std::move(*qco).intoJeff();
}

} // namespace mqt::benchmark
