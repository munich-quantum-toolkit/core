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

#include <mlir/IR/Dialect.h>
#include <mlir/IR/OpDefinition.h>

#define DIALECT_NAME_QTensor "qtensor"

//===----------------------------------------------------------------------===//
// QTensor Dialect
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/QTensor/IR/QTensorOpsDialect.h.inc" // IWYU pragma: export

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/QTensor/IR/QTensorOpsTypes.h.inc" // IWYU pragma: export
