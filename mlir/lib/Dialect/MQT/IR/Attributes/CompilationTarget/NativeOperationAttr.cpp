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

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"

#include <cstdint>
#include <optional>

using namespace mlir;
using namespace mlir::mqt;

LogicalResult NativeOperationAttr::verify(
    const function_ref<InFlightDiagnostic()> emitError, const StringAttr name,
    const OperationArityAttr arity, const uint64_t numParameters,
    const ArrayRef<SiteTupleAttr> siteTuples,
    const std::optional<uint64_t> /*duration*/, const FloatAttr fidelity,
    const ArrayAttr fixedParameters, const StringAttr canonicalName) {
  if (name.getValue().trim().empty()) {
    return emitError() << "compiler target operation name must not be empty";
  }
  if (canonicalName && canonicalName.getValue().trim().empty()) {
    return emitError()
           << "compiler target canonical operation name must not be empty";
  }
  if (failed(detail::verifyFidelity(emitError, fidelity,
                                    "compiler target operation fidelity"))) {
    return failure();
  }

  if (fixedParameters) {
    if (fixedParameters.size() != numParameters) {
      return emitError() << "compiler target fixed parameters must match its "
                            "parameter count";
    }
    for (Attribute parameter : fixedParameters) {
      if (isa<UnitAttr>(parameter)) {
        continue;
      }
      auto value = dyn_cast<FloatAttr>(parameter);
      if (!value || !value.getType().isF64() || !value.getValue().isFinite()) {
        return emitError() << "compiler target fixed parameters must be finite "
                              "f64 values or unit";
      }
    }
  }

  if (!siteTuples.empty() && arity.getKind() == OperationArityKind::Variadic) {
    return emitError()
           << "compiler target variadic operation cannot contain site tuples";
  }
  if (!siteTuples.empty() && arity.getValue() == 0) {
    return emitError()
           << "compiler target zero-arity operation cannot contain site tuples";
  }

  llvm::SmallDenseSet<ArrayRef<int64_t>> seen;
  seen.reserve(siteTuples.size());
  for (const SiteTupleAttr siteTuple : siteTuples) {
    if (siteTuple.getSites().size() != arity.getValue()) {
      return emitError()
             << "compiler target operation site tuple does not match its arity";
    }
    if (!seen.insert(siteTuple.getSites()).second) {
      return emitError()
             << "compiler target operation contains a duplicate site tuple";
    }
  }

  return success();
}
