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
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <optional>
#include <utility>

using namespace mlir;
using namespace mlir::mqt;

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
