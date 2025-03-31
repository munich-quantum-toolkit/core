/*
 * Copyright (c) 2025 Chair for Design Automation, TUM
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

#include <mlir/Bytecode/BytecodeOpInterface.h>

#define DIALECT_NAME_MQTOPT "mqtopt"

//===----------------------------------------------------------------------===//
// Dialect
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MQTOpt/IR/MQTOptOpsDialect.h.inc" // IWYU pragma: export

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/MQTOpt/IR/MQTOptOpsTypes.h.inc" // IWYU pragma: export

//===----------------------------------------------------------------------===//
// Interfaces
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MQTOpt/IR/MQTOptInterfaces.h.inc" // IWYU pragma: export
#include "mlir/Dialect/MQTOpt/IR/MQTOptTraits.h"         // IWYU pragma: export

//===----------------------------------------------------------------------===//
// Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/MQTOpt/IR/MQTOptOps.h.inc" // IWYU pragma: export
