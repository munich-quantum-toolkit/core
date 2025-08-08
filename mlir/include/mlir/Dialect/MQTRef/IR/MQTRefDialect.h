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
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

#define DIALECT_NAME_MQTREF "mqtref"

//===----------------------------------------------------------------------===//
// Dialect
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MQTRef/IR/MQTRefOpsDialect.h.inc" // IWYU pragma: export

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/MQTRef/IR/MQTRefOpsTypes.h.inc" // IWYU pragma: export

//===----------------------------------------------------------------------===//
// Interfaces
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Common/IR/CommonTraits.h"         // IWYU pragma: export
#include "mlir/Dialect/MQTRef/IR/MQTRefInterfaces.h.inc" // IWYU pragma: export

//===----------------------------------------------------------------------===//
// Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/MQTRef/IR/MQTRefOps.h.inc" // IWYU pragma: export
