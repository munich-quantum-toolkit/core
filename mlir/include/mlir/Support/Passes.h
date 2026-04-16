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

#include "mlir/Support/LogicalResult.h"

namespace mlir {
class ModuleOp;
class PassManager;
} // namespace mlir

/**
 * @brief Populate a QC-oriented cleanup pipeline on the given pass manager.
 * @details Adds generic cleanup and QC qubit-register shrinking.
 */
void populateQCCleanupPipeline(mlir::PassManager& pm);

/**
 * @brief Populate a QCO-oriented cleanup pipeline on the given pass manager.
 * @details Adds generic cleanup and qtensor shrink-to-fit.
 */
void populateQCOCleanupPipeline(mlir::PassManager& pm);

/**
 * @brief Populate a QIR-oriented cleanup pipeline on the given pass manager.
 * @details Adds generic cleanup and QIR-specific simplifications.
 */
void populateQIRCleanupPipeline(mlir::PassManager& pm);

/**
 * @brief Run the QC-oriented cleanup pipeline on a module.
 */
[[nodiscard]] mlir::LogicalResult runQCCleanupPipeline(mlir::ModuleOp module);

/**
 * @brief Run the QCO-oriented cleanup pipeline on a module.
 */
[[nodiscard]] mlir::LogicalResult runQCOCleanupPipeline(mlir::ModuleOp module);

/**
 * @brief Run the QIR-oriented cleanup pipeline on a module.
 */
[[nodiscard]] mlir::LogicalResult runQIRCleanupPipeline(mlir::ModuleOp module);
