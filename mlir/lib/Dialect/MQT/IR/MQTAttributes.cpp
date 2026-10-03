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

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/VersionTuple.h"

#include <cmath>
#include <cstdint>
#include <optional>
#include <utility>

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

LogicalResult
CouplingAttr::verify(const function_ref<InFlightDiagnostic()> emitError,
                     const int64_t source, const int64_t target) {
  if (source < 0 || target < 0) {
    return emitError() << "compiler target coupling sites must be nonnegative";
  }
  if (source == target) {
    return emitError() << "compiler target coupling must join distinct sites";
  }
  return success();
}

[[nodiscard]] static LogicalResult
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
  return verifyFidelity(emitError, fidelity,
                        "compiler target site-tuple fidelity");
}

LogicalResult
OperationArityAttr::verify(const function_ref<InFlightDiagnostic()> emitError,
                           const OperationArityKind kind,
                           const uint64_t value) {
  if (kind == OperationArityKind::Variadic && value == 0) {
    return emitError()
           << "compiler target operation variadic minimum must be positive";
  }
  return success();
}

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
  if (failed(verifyFidelity(emitError, fidelity,
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

LogicalResult CompilationTargetAttr::verify(
    const function_ref<InFlightDiagnostic()> emitError, const StringAttr name,
    const ArrayRef<SiteAttr> sites, const DurationUnitAttr durationUnit,
    const ConnectivityKind connectivity, const ArrayRef<CouplingAttr> couplings,
    const NativeOperationsKind nativeOperations,
    const ArrayRef<NativeOperationAttr> operations) {
  if (name && name.getValue().empty()) {
    return emitError() << "compiler target name must not be empty when present";
  }
  if (sites.empty()) {
    return emitError() << "compiler target must contain at least one site";
  }

  llvm::SmallDenseSet<int64_t> siteIds;
  siteIds.reserve(sites.size());
  for (const SiteAttr site : sites) {
    if (!siteIds.insert(site.getId()).second) {
      return emitError() << "compiler target contains duplicate site IDs";
    }
  }

  if (connectivity != ConnectivityKind::Explicit && !couplings.empty()) {
    return emitError()
           << "compiler target couplings require explicit connectivity";
  }
  if (connectivity == ConnectivityKind::Explicit) {
    llvm::SmallDenseSet<std::pair<int64_t, int64_t>> seen;
    for (const CouplingAttr coupling : couplings) {
      auto source = coupling.getSource();
      auto target = coupling.getTarget();
      if (!siteIds.contains(source) || !siteIds.contains(target)) {
        return emitError()
               << "compiler target coupling references an unknown site";
      }
      if (target < source) {
        std::swap(source, target);
      }
      if (!seen.insert({source, target}).second) {
        return emitError() << "compiler target contains a duplicate coupling";
      }
    }
  }

  if (nativeOperations != NativeOperationsKind::Explicit &&
      !operations.empty()) {
    return emitError()
           << "compiler target operations require explicit native operations";
  }
  for (const NativeOperationAttr operation : operations) {
    if (operation.getArity().getValue() > sites.size()) {
      if (operation.getArity().getKind() == OperationArityKind::Variadic) {
        return emitError() << "compiler target operation variadic minimum "
                              "exceeds its site count";
      }
      return emitError() << "compiler target operation arity exceeds its site "
                            "count";
    }
    for (const SiteTupleAttr siteTuple : operation.getSiteTuples()) {
      if (llvm::any_of(siteTuple.getSites(), [&](const int64_t site) {
            return !siteIds.contains(site);
          })) {
        return emitError() << "compiler target operation site tuple references "
                              "an unknown site";
      }
    }
  }

  const bool hasTiming =
      llvm::any_of(sites,
                   [](const SiteAttr site) {
                     return site.getT1().has_value() ||
                            site.getT2().has_value();
                   }) ||
      llvm::any_of(operations, [](const NativeOperationAttr operation) {
        return operation.getDuration().has_value() ||
               llvm::any_of(operation.getSiteTuples(),
                            [](const SiteTupleAttr siteTuple) {
                              return siteTuple.getDuration().has_value();
                            });
      });
  if (hasTiming && !durationUnit) {
    return emitError()
           << "compiler target timing metadata requires a duration unit";
  }
  return success();
}
