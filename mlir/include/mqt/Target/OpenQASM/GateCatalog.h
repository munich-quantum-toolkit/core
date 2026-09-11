/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

#include "mqt/Dialect/QC/Translation/StandardGate.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

#include <cstddef>
#include <cstdint>

namespace mlir::openqasm::frontend {

enum class GateAvailability : uint8_t {
  Language,
  StandardLibrary,
  QELib1,
  StandardLibraryAndQELib1,
  Compatibility,
};

struct GateCatalogEntry {
  GateCatalogEntry(llvm::StringRef name, qc::StandardGate gate,
                   size_t controlCount, GateAvailability availability,
                   bool variadicControls = false, bool inverse = false) noexcept
      : name(name), gate(gate),
        parameterCount(qc::getStandardGateDescriptor(gate).parameterCount),
        controlCount(controlCount +
                     qc::getStandardGateDescriptor(gate).controlCount),
        targetCount(qc::getStandardGateDescriptor(gate).targetCount),
        availability(availability), variadicControls(variadicControls),
        inverse(inverse) {}

  llvm::StringRef name;
  qc::StandardGate gate;
  size_t parameterCount;
  size_t controlCount;
  size_t targetCount;
  GateAvailability availability;
  bool variadicControls;
  bool inverse;

  [[nodiscard]] size_t qubitCount() const { return controlCount + targetCount; }
};

[[nodiscard]] llvm::ArrayRef<GateCatalogEntry> getGateCatalog();

[[nodiscard]] const GateCatalogEntry* lookupGate(llvm::StringRef name);

[[nodiscard]] llvm::StringRef canonicalGateName(qc::StandardGate gate);

} // namespace mlir::openqasm::frontend
