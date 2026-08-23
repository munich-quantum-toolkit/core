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

#include "mlir/Compiler/Programs.h"

#include <optional>

namespace mqt::benchmarks {
class GHZ;
class Grover;
class QPE;
} // namespace mqt::benchmarks

namespace mqt::benchmark {

/// Generate the QC program for a configured GHZ benchmark.
[[nodiscard]] std::optional<mlir::QCProgram>
generateProgram(const benchmarks::GHZ& benchmark);

/// Generate the QC program for a configured Grover benchmark.
[[nodiscard]] std::optional<mlir::QCProgram>
generateProgram(const benchmarks::Grover& benchmark);

/// Generate the QC program for a configured QPE benchmark.
[[nodiscard]] std::optional<mlir::QCProgram>
generateProgram(const benchmarks::QPE& benchmark);

} // namespace mqt::benchmark
