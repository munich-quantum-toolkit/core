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

#include "mlir/Dialect/OQ3/IR/OQ3Dialect.h"
#include "mlir/Dialect/QC/IR/QCDialect.h"

#include <mlir/Bytecode/BytecodeOpInterface.h>
#include <mlir/Interfaces/FunctionInterfaces.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

#define GET_OP_CLASSES
#include "mlir/Dialect/OQ3/IR/OQ3Ops.h.inc" // IWYU pragma: export
