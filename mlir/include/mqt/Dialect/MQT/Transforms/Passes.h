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

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

#include <string>

namespace mlir {
class ModuleOp;
class RewriterBase;
} // namespace mlir

namespace mlir::mqt {

/// Discard entry-point QC and QCO phases outside quantum modifier bodies.
///
/// Preserves probabilities, not exact amplitudes. Helper functions are
/// unchanged.
void dropGlobalPhases(ModuleOp moduleOp, RewriterBase& rewriter);

/// Propagate scoped QCO branch conditions.
void simplifyClassicalControl(Operation* operation);

#define GEN_PASS_DECL
#include "mqt/Dialect/MQT/Transforms/Passes.h.inc" // IWYU pragma: export

#define GEN_PASS_REGISTRATION
#include "mqt/Dialect/MQT/Transforms/Passes.h.inc" // IWYU pragma: export

} // namespace mlir::mqt
