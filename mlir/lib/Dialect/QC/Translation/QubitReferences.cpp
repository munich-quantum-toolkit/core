/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/Dialect/QC/Translation/QubitReferences.h"

#include "mqt/Dialect/QC/IR/QCOps.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Dominance.h"

#include <cstddef>

namespace mlir::qc {

FailureOr<llvm::SmallVector<Value>>
collectQubitReferences(memref::AllocaOp storage) {
  const auto type = storage.getType();
  if (type.getRank() != 1 || !type.hasStaticShape() ||
      !isa<QubitType>(type.getElementType())) {
    return storage.emitError(
        "qubit reference storage requires a static rank-one buffer");
  }
  llvm::SmallVector<Value> elements(type.getDimSize(0));
  memref::StoreOp lastStore;
  size_t numStores = 0;
  llvm::SmallVector<memref::LoadOp> loads;
  for (auto* user : storage->getUsers()) {
    if (auto store = dyn_cast<memref::StoreOp>(user)) {
      const auto index = getConstantIntValue(store.getIndices().front());
      if (!index || *index < 0 || *index >= type.getDimSize(0) ||
          elements[*index] || store->getBlock() != storage->getBlock()) {
        return store.emitError(
            "qubit references require one constant-index initializer per slot");
      }
      elements[*index] = store.getValue();
      ++numStores;
      if (!lastStore || lastStore->isBeforeInBlock(store)) {
        lastStore = store;
      }
    } else if (auto load = dyn_cast<memref::LoadOp>(user)) {
      loads.push_back(load);
    } else {
      return user->emitError("unsupported use of qubit reference storage");
    }
  }
  if (numStores != elements.size()) {
    return storage.emitError(
        "qubit reference storage is not fully initialized");
  }
  DominanceInfo dominance;
  for (auto load : loads) {
    if (lastStore && !dominance.dominates(lastStore, load)) {
      return load.emitError(
          "qubit reference initialization must dominate its uses");
    }
  }
  return elements;
}

} // namespace mlir::qc
