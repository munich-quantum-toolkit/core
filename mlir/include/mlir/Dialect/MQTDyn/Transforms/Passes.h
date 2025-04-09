/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

#include <mlir/Pass/Pass.h>

namespace mlir {

class RewritePatternSet;

} // namespace mlir

namespace mqt::ir::dyn {

#define GEN_PASS_DECL
#include "mlir/Dialect/MQTDyn/Transforms/Passes.h.inc" // IWYU pragma: export

void populateConstantFoldExtractQubitPatterns(
    mlir::RewritePatternSet& patterns);

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "mlir/Dialect/MQTDyn/Transforms/Passes.h.inc" // IWYU pragma: export
} // namespace mqt::ir::dyn
