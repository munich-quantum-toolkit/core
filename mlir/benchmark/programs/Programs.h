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

#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>

namespace mlir::qc {
class QCProgramBuilder;
} // namespace mlir::qc

namespace mqt::benchmarks {
class GHZ;
class Grover;
class QPE;
} // namespace mqt::benchmarks

namespace mqt::benchmark {

using namespace mlir;

/// Emit one configured GHZ benchmark.
SmallVector<Value> ghz(qc::QCProgramBuilder& b,
                       const benchmarks::GHZ& benchmark);

/// Emit one configured Grover benchmark.
SmallVector<Value> grover(qc::QCProgramBuilder& b,
                          const benchmarks::Grover& benchmark);

/// Emit one configured QPE benchmark.
SmallVector<Value> qpe(qc::QCProgramBuilder& b,
                       const benchmarks::QPE& benchmark);

} // namespace mqt::benchmark
