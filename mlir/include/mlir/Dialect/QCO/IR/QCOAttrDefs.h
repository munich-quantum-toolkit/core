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

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"

// clang-format:off
#include "mlir/Dialect/QCO/IR/QCOEnums.h.inc"
#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/QCO/IR/QCOAttributes.h.inc"
// clang-format:on
