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

namespace mqt::jeff::benchmarks {

struct Benchmark;

/**
 * @brief Builds a benchmark and lowers it to a `jeff` program.
 *
 * @details The program is built with the `QCProgramBuilder` for size @p n and
 * then lowered through QCO. Returns no program when @p n is below the
 * benchmark's minimum size or when a stage of the pipeline fails.
 */
[[nodiscard]] std::optional<mlir::JeffProgram>
buildJeffProgram(const Benchmark& benchmark, uint64_t n);

} // namespace mqt::jeff::benchmarks
