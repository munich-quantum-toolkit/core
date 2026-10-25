/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/Target/QIR/Capabilities.h"

#include "mqt/Compiler/TargetEnvironment.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/LogicalResult.h"

namespace mlir::qir {

LogicalResult validateCapabilities(ModuleOp moduleOp,
                                   ArrayRef<ProgramCapability> capabilities) {
  const auto requireCapability =
      [&](llvm::StringRef id, bool allowConstraints = false) -> LogicalResult {
    const auto* const capability =
        llvm::find_if(capabilities, [&](const auto& entry) {
          return entry.id == id && entry.value == 0;
        });
    if (capability == capabilities.end()) {
      return moduleOp.emitError()
             << "Selected QIR payload requires capability '" << id
             << "' that the target contract does not permit";
    }
    if (!allowConstraints && !capability->constraints.empty()) {
      return moduleOp.emitError() << "QIR lowering cannot verify constraints "
                                     "on required capability '"
                                  << id << "'";
    }
    return success();
  };
  const auto optionalFeatures = llvm::ArrayRef(OPTIONAL_CAPABILITIES);
  for (auto flags : moduleOp.getOps<LLVM::ModuleFlagsOp>()) {
    for (auto attribute : flags.getFlags()) {
      const auto flag = cast<LLVM::ModuleFlagAttr>(attribute);
      const auto key = flag.getKey().getValue();
      if (const auto value = dyn_cast<IntegerAttr>(flag.getValue());
          value && value.getValue().isZero()) {
        continue;
      }
      if (key == "backwards_branching") {
        const auto branches =
            cast<IntegerAttr>(flag.getValue()).getValue().getZExtValue();
        if (((branches & 1U) != 0U &&
             failed(requireCapability(ProgramCapability::COUNTED_ITERATION))) ||
            ((branches & 2U) != 0U &&
             failed(requireCapability(ProgramCapability::CONDITIONAL_LOOP)))) {
          return failure();
        }
        continue;
      }
      const auto* const feature =
          llvm::find_if(optionalFeatures,
                        [&](const auto& entry) { return key == entry.first; });
      if (feature != optionalFeatures.end() &&
          failed(requireCapability(
              feature->second,
              feature->second == ProgramCapability::MULTIWAY_BRANCHING))) {
        return failure();
      }
    }
  }
  return success();
}

} // namespace mlir::qir
