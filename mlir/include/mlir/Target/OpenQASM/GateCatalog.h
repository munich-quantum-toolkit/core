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

#include "mlir/Dialect/QC/Translation/StandardGate.h"

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/StringRef.h>

#include <cstddef>
#include <cstdint>

namespace mlir::oq3::frontend {

enum class GateAvailability : uint8_t {
  Language,
  StandardLibrary,
  QELib1,
  StandardLibraryAndQELib1,
  Compatibility,
};

using GateLowering = qc::StandardGate;

struct GateCatalogEntry {
  GateCatalogEntry(llvm::StringRef name, GateLowering lowering,
                   size_t controlCount, GateAvailability availability,
                   bool variadicControls = false, bool inverse = false) noexcept
      : name(name), lowering(lowering),
        parameterCount(qc::getStandardGateDescriptor(lowering).parameterCount),
        controlCount(controlCount +
                     qc::getStandardGateDescriptor(lowering).controlCount),
        targetCount(qc::getStandardGateDescriptor(lowering).targetCount),
        availability(availability), variadicControls(variadicControls),
        inverse(inverse) {}

  llvm::StringRef name;
  GateLowering lowering;
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

[[nodiscard]] llvm::StringRef canonicalGateName(GateLowering lowering);

} // namespace mlir::oq3::frontend
