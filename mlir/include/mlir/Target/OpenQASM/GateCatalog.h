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

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/StringRef.h>

#include <cstddef>
#include <cstdint>

namespace mlir::oq3::frontend {

enum class GateAvailability : std::uint8_t {
  Language,
  StandardLibrary,
  QELib1,
  StandardLibraryAndQELib1,
  Compatibility,
};

enum class GateLowering : std::uint8_t {
  GPhase,
  Id,
  X,
  Y,
  Z,
  H,
  S,
  Sdg,
  T,
  Tdg,
  SX,
  SXdg,
  P,
  RX,
  RY,
  RZ,
  R,
  U2,
  U3,
  BuiltinU,
  CU,
  SWAP,
  ISWAP,
  DCX,
  ECR,
  RCCX,
  RXX,
  RYY,
  RZX,
  RZZ,
  XXPlusYY,
  XXMinusYY,
};

struct GateCatalogEntry {
  llvm::StringRef name;
  GateLowering lowering;
  std::size_t parameterCount;
  std::size_t controlCount;
  std::size_t targetCount;
  GateAvailability availability;
  bool variadicControls = false;
  bool inverse = false;

  [[nodiscard]] std::size_t qubitCount() const {
    return controlCount + targetCount;
  }
};

[[nodiscard]] llvm::ArrayRef<GateCatalogEntry> getGateCatalog();

[[nodiscard]] const GateCatalogEntry* lookupGate(llvm::StringRef name);

[[nodiscard]] llvm::StringRef canonicalGateName(GateLowering lowering);

} // namespace mlir::oq3::frontend
