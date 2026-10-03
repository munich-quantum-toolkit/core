/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "AttributeUtils.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/StringRef.h"

#include <cmath>

namespace mlir::mqt::detail {

LogicalResult
verifyFidelity(const function_ref<InFlightDiagnostic()>& emitError,
               const FloatAttr fidelity, const StringRef description) {
  if (!fidelity) {
    return success();
  }
  if (!fidelity.getType().isF64()) {
    return emitError() << description << " must be an f64 value";
  }
  const auto value = fidelity.getValueAsDouble();
  if (!std::isfinite(value) || value < 0. || value > 1.) {
    return emitError() << description << " must be finite and in [0, 1]";
  }
  return success();
}

} // namespace mlir::mqt::detail
