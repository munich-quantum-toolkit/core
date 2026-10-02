/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/Dialect/QTensor/IR/QTensorOps.h"

#include "mqt/Dialect/QTensor/IR/QTensorDialect.h" // IWYU pragma: associated

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/DialectImplementation.h" // IWYU pragma: keep (template instantiations)
#include "mlir/Transforms/InliningUtils.h"

#include "llvm/ADT/TypeSwitch.h" // IWYU pragma: keep (template instantiations)

using namespace mlir;
using namespace mlir::qtensor;

namespace {

struct QTensorInlinerInterface final : DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  /// Inlining remaps each linear tensor and extracted qubit to its local value.
  bool isLegalToInline(Operation* /*operation*/, Region* /*destination*/,
                       bool /*wouldBeCloned*/,
                       IRMapping& /*valueMapping*/) const final {
    return true;
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Dialect
//===----------------------------------------------------------------------===//

#include "mqt/Dialect/QTensor/IR/QTensorOpsDialect.cpp.inc"

void QTensorDialect::initialize() {
  addInterfaces<QTensorInlinerInterface>();
  // NOLINTNEXTLINE(clang-analyzer-core.StackAddressEscape)
  addTypes<
#define GET_TYPEDEF_LIST
#include "mqt/Dialect/QTensor/IR/QTensorOpsTypes.cpp.inc"

      >();

  addOperations<
#define GET_OP_LIST
#include "mqt/Dialect/QTensor/IR/QTensorOps.cpp.inc"

      >();
}

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "mqt/Dialect/QTensor/IR/QTensorOpsTypes.cpp.inc"

//===----------------------------------------------------------------------===//
// Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mqt/Dialect/QTensor/IR/QTensorOps.cpp.inc"
