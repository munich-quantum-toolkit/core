/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

/**
 * @file
 * Compatibility macro for applying rewrite patterns across different MLIR
 * versions.
 *
 * MLIR deprecated `applyPatternsAndFoldGreedily` in version 20 and introduced
 * `applyPatternsGreedily` as the replacement. To maintain compatibility across
 * both MLIR 19 and MLIR 20+, this macro maps to the correct function
 * automatically based on the MLIR version used during compilation.
 *
 * Usage:
 * ```
 * if (mlir::failed(APPLY_PATTERNS_GREEDILY(op, std::move(patterns)))) {
 *   signalPassFailure();
 * }
 * ```
 */

#pragma once

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#if MLIR_VERSION_MAJOR >= 20
/// Use `applyPatternsGreedily` for MLIR >= 20
#define APPLY_PATTERNS_GREEDILY(op, patterns)                                  \
  mlir::applyPatternsGreedily(op, patterns)
#else
/// Use `applyPatternsAndFoldGreedily` for MLIR < 20
#define APPLY_PATTERNS_GREEDILY(op, patterns)                                  \
  mlir::applyPatternsAndFoldGreedily(op, patterns)
#endif
