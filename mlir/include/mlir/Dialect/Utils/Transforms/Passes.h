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

#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassRegistry.h>

namespace mlir::quantum {

#define GEN_PASS_DECL
#include "mlir/Dialect/Utils/Transforms/Passes.h.inc" // IWYU pragma: export

#define GEN_PASS_REGISTRATION
#include "mlir/Dialect/Utils/Transforms/Passes.h.inc" // IWYU pragma: export

} // namespace mlir::quantum
