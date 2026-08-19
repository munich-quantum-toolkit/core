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

namespace mlir {
#define GEN_PASS_DECL_CONVERTCBITTOMEMREF
#include "mlir/Conversion/CBitToMemRef/CBitToMemRef.h.inc"

#define GEN_PASS_REGISTRATION
#include "mlir/Conversion/CBitToMemRef/CBitToMemRef.h.inc"
} // namespace mlir
