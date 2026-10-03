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

#include "../AttributeUtils.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLFunctionalExtras.h"

#include <cstdint>
#include <optional>

using namespace mlir;
using namespace mlir::mqt;

LogicalResult
SiteTupleAttr::verify(const function_ref<InFlightDiagnostic()> emitError,
                      const ArrayRef<int64_t> sites,
                      const std::optional<uint64_t> /*duration*/,
                      const FloatAttr fidelity) {
  llvm::SmallDenseSet<int64_t> seen;
  seen.reserve(sites.size());
  for (const int64_t site : sites) {
    if (site < 0) {
      return emitError()
             << "compiler target site tuple contains a negative site ID";
    }
    if (!seen.insert(site).second) {
      return emitError()
             << "compiler target site tuple contains a duplicate site";
    }
  }
  return detail::verifyFidelity(emitError, fidelity,
                                "compiler target site-tuple fidelity");
}
