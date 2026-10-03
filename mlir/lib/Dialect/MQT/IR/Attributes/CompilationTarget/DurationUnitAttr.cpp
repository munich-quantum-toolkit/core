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

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/StringRef.h"

#include <cmath>

using namespace mlir;
using namespace mlir::mqt;

LogicalResult
DurationUnitAttr::verify(const function_ref<InFlightDiagnostic()> emitError,
                         const StringAttr unit, const FloatAttr scaleFactor) {
  if (unit.getValue().trim().empty()) {
    return emitError() << "duration unit must not be empty";
  }
  if (!scaleFactor.getType().isF64()) {
    return emitError() << "duration scale factor must be an f64 value";
  }
  const auto value = scaleFactor.getValueAsDouble();
  if (!std::isfinite(value) || value <= 0.) {
    return emitError() << "duration scale factor must be positive and finite";
  }
  return success();
}
