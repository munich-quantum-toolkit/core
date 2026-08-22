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

#include "mlir/Compiler/ProgramFormat.h"

namespace mlir {

class CompilerTarget;
class OpPassManager;

/**
 * @brief Populate the canonical compiler-target pipeline.
 *
 * @details Decomposes supported multi-controlled gates, performs
 * target-independent optimization, maps to the target topology, synthesizes
 * native operations, performs a final local cleanup, and verifies target
 * conformance.
 */
void populateTargetCompilationPipeline(OpPassManager& pm,
                                       const CompilerTarget& target);

/**
 * @brief Populate target compilation for the selected output profile.
 */
void populateTargetCompilationPipeline(OpPassManager& pm,
                                       const CompilerTarget& target,
                                       ProgramFormat format);

/**
 * @brief Populate target compilation for an exact payload and output stage.
 */
void populateTargetCompilationPipeline(OpPassManager& pm,
                                       const CompilerTarget& target,
                                       ProgramFormat format,
                                       const PayloadDescriptor& descriptor);

} // namespace mlir
