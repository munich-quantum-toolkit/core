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

#include "mqt/Dialect/MQT/IR/PayloadVersion.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>
#include <utility>

using namespace mlir;
using namespace mlir::mqt;

bool mlir::mqt::preservesProgramQubits(Operation* operation) {
  auto moduleOp = dyn_cast<ModuleOp>(operation);
  if (!moduleOp) {
    moduleOp = operation->getParentOfType<ModuleOp>();
  }
  const auto environment =
      moduleOp ? moduleOp->getAttrOfType<TargetEnvAttr>(TargetEnvAttr::name)
               : TargetEnvAttr{};
  return environment && environment.getCompilationTarget().getGateSet();
}

LogicalResult
PayloadFormatAttr::verify(const function_ref<InFlightDiagnostic()> emitError,
                          const StringAttr id, const StringAttr version,
                          const StringAttr profile,
                          const PayloadEncoding /*encoding*/) {
  if (id.getValue().empty()) {
    return emitError() << "payload format requires an ID";
  }
  if (id.getValue().contains('\0') || version.getValue().contains('\0') ||
      profile.getValue().contains('\0')) {
    return emitError()
           << "payload format fields must not contain null characters";
  }
  if (!version.getValue().empty() &&
      !PayloadVersionRange::parse(version.getValue())) {
    return emitError() << "payload version requires a numeric ceiling or >=, "
                          "<, <= constraints";
  }
  return success();
}

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

LogicalResult ProgramCapabilityAttr::verify(
    const function_ref<InFlightDiagnostic()> emitError, const StringAttr id,
    const uint64_t /*value*/,
    const ArrayRef<ProgramConstraintAttr> constraints) {
  if (id.getValue().empty()) {
    return emitError() << "program capability ID must not be empty";
  }
  if (id.getValue().contains('\0')) {
    return emitError()
           << "program capability ID must not contain a null character";
  }

  llvm::SmallDenseSet<StringRef> seen;
  seen.reserve(constraints.size());
  for (const ProgramConstraintAttr constraint : constraints) {
    if (!seen.insert(constraint.getId().getValue()).second) {
      return emitError() << "program capability contains duplicate constraint '"
                         << constraint.getId().getValue() << "'";
    }
  }
  return success();
}

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
