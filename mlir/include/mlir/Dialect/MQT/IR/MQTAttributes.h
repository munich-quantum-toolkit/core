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

#include "mlir/Dialect/MQT/IR/MQTDialect.h"

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/StringRef.h>
#include <mlir/IR/Attributes.h>
#include <mlir/Interfaces/DataLayoutInterfaces.h>

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/MQT/IR/MQTAttributes.h.inc" // IWYU pragma: export
