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

#include <cstdint>
#include <optional>

namespace mqt::benchmarks {
class GHZ;
class Grover;
class QPE;
} // namespace mqt::benchmarks

namespace mqt::benchmark {

struct Benchmark;

/**
 * @brief Generates a benchmark as a QC program.
 *
 * @details The program is built with the `QCProgramBuilder` for size @p n and
 * then cleaned up. Returns no program when @p n lies outside the sizes the
 * benchmark accepts or when the build fails, and writes the reason to the
 * error stream. The context holds every dialect that the later conversions
 * need, so the result can enter any backend pipeline.
 */
[[nodiscard]] std::optional<mlir::QCProgram>
generateProgram(const Benchmark& benchmark, uint64_t n);

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
