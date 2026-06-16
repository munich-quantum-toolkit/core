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

namespace mlir {
class ModuleOp;
class PassManager;

/**
 * @brief Populate the QIR conversion pipeline on the given pass manager.
 */
void populateQIRConversionPipeline(mlir::PassManager& pm,
                                   bool useAdaptive = false);
} // namespace mlir
