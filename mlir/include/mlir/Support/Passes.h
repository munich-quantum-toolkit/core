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

#include <mlir/Support/LLVM.h>

namespace mlir {
class ModuleOp;
class OpPassManager;
class PassManager;
} // namespace mlir

/**
 * @brief Populate the pass manager and run it on the module.
 */
mlir::LogicalResult runWithPassManager(
    mlir::ModuleOp module,
    mlir::function_ref<void(mlir::OpPassManager&)> populatePasses,
    mlir::StringRef errorMessage);

/** @brief Register the QCO passes and named compiler pipelines. */
void registerMQTCompilerPasses();

/** @brief Populate the default QCO optimization pipeline. */
void populateDefaultQCOOptimizationPipeline(mlir::OpPassManager& pm);

/** @brief Parse and run a module-level MLIR textual pass pipeline. */
[[nodiscard]] mlir::LogicalResult
runPassPipeline(mlir::ModuleOp module, mlir::StringRef pipeline,
                bool enableTiming = false, bool enableStatistics = false);

/**
 * @brief Populate a QC-oriented cleanup pipeline on the given pass manager.
 * @details Adds generic cleanup and QC qubit-register shrinking.
 */
void populateQCCleanupPipeline(mlir::OpPassManager& pm);

/**
 * @brief Populate a QCO-oriented cleanup pipeline on the given pass manager.
 * @details Adds generic cleanup and qtensor shrink-to-fit.
 */
void populateQCOCleanupPipeline(mlir::OpPassManager& pm);

/**
 * @brief Populate a QIR-oriented cleanup pipeline on the given pass manager.
 * @details Adds generic cleanup and QIR-specific simplifications. Updates the
 * meta data accordingly.
 */
void populateQIRCleanupPipeline(mlir::OpPassManager& pm, bool useAdaptive);

/**
 * @brief Populate a `jeff`-oriented cleanup pipeline on the given pass manager.
 * @details Adds generic cleanup and dead-value removal. This matches the QCO
 * cleanup minus the QTensor-specific shrink pass, as QTensor operations no
 * longer exist once lowered into the `jeff` dialect.
 */
void populateJeffCleanupPipeline(mlir::OpPassManager& pm);

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
[[nodiscard]] mlir::LogicalResult runQIRCleanupPipeline(mlir::ModuleOp module,
                                                        bool useAdaptive);

/**
 * @brief Run the `jeff`-oriented cleanup pipeline on a module.
 */
[[nodiscard]] mlir::LogicalResult runJeffCleanupPipeline(mlir::ModuleOp module);
