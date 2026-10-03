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
#include "llvm/Support/VersionTuple.h"

using namespace mlir;
using namespace mlir::mqt;

[[nodiscard]] static bool isCanonicalPayloadVersion(const StringRef version) {
  llvm::VersionTuple parsed;
  return !parsed.tryParse(version) && !parsed.getBuild() &&
         parsed.getAsString() == version;
}

LogicalResult
PayloadFormatAttr::verify(const function_ref<InFlightDiagnostic()> emitError,
                          const StringAttr id, const StringAttr version,
                          const StringAttr profile,
                          const PayloadEncoding /*encoding*/) {
  if (id.getValue().empty() || version.getValue().empty()) {
    return emitError() << "payload format requires an ID and version";
  }
  if (id.getValue().contains('\0') || version.getValue().contains('\0') ||
      profile.getValue().contains('\0')) {
    return emitError()
           << "payload format fields must not contain null characters";
  }
  if (!isCanonicalPayloadVersion(version.getValue())) {
    return emitError()
           << "payload format version must use major[.minor[.patch]]";
  }
  return success();
}
