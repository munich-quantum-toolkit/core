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

#include <mlir/Bytecode/BytecodeOpInterface.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

// Suppress warnings about ambiguous reversed operators in MLIR
// (see https://github.com/llvm/llvm-project/issues/45853)
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wambiguous-reversed-operator"
#endif
#include <mlir/Interfaces/InferTypeOpInterface.h>
#if defined(__clang__)
#pragma clang diagnostic pop
#endif

#define DIALECT_NAME_FLUX "quartz"

//===----------------------------------------------------------------------===//
// Dialect
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Quartz/IR/QuartzOpsDialect.h.inc"

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/Quartz/IR/QuartzOpsTypes.h.inc"

//===----------------------------------------------------------------------===//
// Interfaces
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Quartz/IR/QuartzInterfaces.h.inc" // IWYU pragma: export

//===----------------------------------------------------------------------===//
// Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/Quartz/IR/QuartzOps.h.inc" // IWYU pragma: export
