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

#include <llvm/ADT/DenseMapInfo.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/OpImplementation.h>

#include <cstdint>
#include <optional>

//===----------------------------------------------------------------------===//
// Enumerations
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/CBit/IR/CBitOpsEnums.h.inc" // IWYU pragma: export

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/CBit/IR/CBitOpsAttributes.h.inc" // IWYU pragma: export
