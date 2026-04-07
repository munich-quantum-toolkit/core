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

#include "mlir/Dialect/QTensor/IR/QTensorDialect.h"

#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassRegistry.h>

/**
 * Include-generated QTensor pass declarations.
 *
 * Defining `GEN_PASS_DECL` before including the generated header emits the
 * public pass declaration symbols for the QTensor dialect.
 */
/**
 * Include-generated QTensor pass registration.
 *
 * Defining `GEN_PASS_REGISTRATION` before including the generated header emits
 * the code that registers QTensor transformation passes with the MLIR pass
 * registry.
 */
namespace mlir::qtensor {

#define GEN_PASS_DECL
#include "mlir/Dialect/QTensor/Transforms/Passes.h.inc" // IWYU pragma: export

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "mlir/Dialect/QTensor/Transforms/Passes.h.inc" // IWYU pragma: export

} // namespace mlir::qtensor
