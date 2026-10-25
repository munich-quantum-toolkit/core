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

#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"

namespace mlir::qc {
class QCProgramBuilder;
} // namespace mlir::qc

namespace mqt::bench {

using namespace mlir;

// Class names in forward declarations cannot be parenthesized.
// NOLINTBEGIN(bugprone-macro-parentheses)
#define MQT_BENCHMARK_FAMILY(TYPE, STEM, ID, DEFINITION_VERSION)               \
  class TYPE;                                                                  \
  /** Emit the configured benchmark. */                                        \
  SmallVector<Value> STEM(qc::QCProgramBuilder& builder, const TYPE& benchmark);
// NOLINTEND(bugprone-macro-parentheses)
#include "bench/BenchmarkFamilies.inc"

} // namespace mqt::bench
