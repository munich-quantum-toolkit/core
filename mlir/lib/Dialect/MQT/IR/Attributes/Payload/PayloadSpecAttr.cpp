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

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <utility>

using namespace mlir;
using namespace mlir::mqt;

LogicalResult
PayloadSpecAttr::verify(const function_ref<InFlightDiagnostic()> emitError,
                        const PayloadFormatAttr /*format*/,
                        const ArrayRef<ProgramCapabilityAttr> capabilities,
                        const bool /*optionalCapabilitiesKnown*/) {
  llvm::SmallDenseSet<std::pair<StringRef, uint64_t>> seen;
  seen.reserve(capabilities.size());
  for (const ProgramCapabilityAttr capability : capabilities) {
    const auto key =
        std::pair(capability.getId().getValue(), capability.getValue());
    if (!seen.insert(key).second) {
      return emitError()
             << "payload specification contains duplicate capability '"
             << capability.getId().getValue() << "' with value "
             << capability.getValue();
    }
  }
  return success();
}
