/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

#include "mqt/Compiler/Programs.h"

#include "llvm/Support/LogicalResult.h"

#include <string>
#include <string_view>

namespace mqt::bench {
/// A generated program and the normalized semantic instance that produced it.
struct GeneratedBenchmark {
  std::string benchmarkId;
  std::string caseId;
  std::string manifestJSON;
  mlir::QCProgram program;
};

// Class names in forward declarations cannot be parenthesized.
// NOLINTBEGIN(bugprone-macro-parentheses)
#define MQT_BENCHMARK_FAMILY(TYPE, STEM, ID, DEFINITION_VERSION)               \
  class TYPE;                                                                  \
  /** Generate the QC program for a configured benchmark. */                   \
  [[nodiscard]] llvm::FailureOr<mlir::QCProgram> generate(                     \
      const TYPE& benchmark);
// NOLINTEND(bugprone-macro-parentheses)
#include "bench/BenchmarkFamilies.inc"

/// Parse a benchmark instance specification and generate the benchmark.
[[nodiscard]] llvm::FailureOr<GeneratedBenchmark>
generate(std::string_view instanceSpecificationJSON,
         std::string_view source = "<instance-specification>");

} // namespace mqt::bench
