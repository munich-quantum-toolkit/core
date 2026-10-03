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
#include "mlir/IR/Diagnostics.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>

using namespace mlir;
using namespace mlir::mqt;

LogicalResult ProgramConstraintAttr::verify(
    const function_ref<InFlightDiagnostic()> emitError, const StringAttr id,
    const uint64_t /*value*/) {
  if (id.getValue().empty()) {
    return emitError() << "program constraint ID must not be empty";
  }
  if (id.getValue().contains('\0')) {
    return emitError() << "program constraint ID must not contain a null "
                          "character";
  }
  return success();
}
