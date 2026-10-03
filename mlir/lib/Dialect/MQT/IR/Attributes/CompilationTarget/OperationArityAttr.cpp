/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/Dialect/MQT/IR/MQTAttributes.h"

#include "mlir/IR/Diagnostics.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/STLFunctionalExtras.h"

#include <cstdint>

using namespace mlir;
using namespace mlir::mqt;

LogicalResult
OperationArityAttr::verify(const function_ref<InFlightDiagnostic()> emitError,
                           const OperationArityKind kind,
                           const uint64_t value) {
  if (kind == OperationArityKind::Variadic && value == 0) {
    return emitError()
           << "compiler target operation variadic minimum must be positive";
  }
  return success();
}
