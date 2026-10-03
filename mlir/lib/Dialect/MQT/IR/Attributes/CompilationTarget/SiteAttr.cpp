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
#include <optional>

using namespace mlir;
using namespace mlir::mqt;

LogicalResult
SiteAttr::verify(const function_ref<InFlightDiagnostic()> emitError,
                 const int64_t id, const StringAttr name,
                 const std::optional<uint64_t> t1,
                 const std::optional<uint64_t> t2) {
  if (id < 0) {
    return emitError() << "compiler target site ID must be nonnegative";
  }
  if (name && name.getValue().empty()) {
    return emitError()
           << "compiler target site name must not be empty when present";
  }
  if (t1 == 0 || t2 == 0) {
    return emitError()
           << "compiler target site coherence times must be positive";
  }
  return success();
}
