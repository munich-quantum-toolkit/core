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

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"

#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <random>

namespace mlir {

/// Temporary compilation input, also captured in MLIR crash reproducers.
inline constexpr llvm::StringLiteral COMPILATION_SEED_ATTR =
    "mqt.compilation_seed";

/// Resolve a compilation-wide override before a pass-local seed.
inline uint64_t compilationSeed(ModuleOp moduleOp, uint64_t fallback) {
  while (moduleOp) {
    if (auto seed = moduleOp->getAttrOfType<IntegerAttr>(COMPILATION_SEED_ATTR);
        seed && seed.getValue().getBitWidth() == 64) {
      fallback = seed.getValue().getZExtValue();
    }
    moduleOp = moduleOp->getParentOfType<ModuleOp>();
  }
  return fallback;
}

/// Keep legacy 32-bit sequences while accepting every bit of a compiler seed.
inline std::mt19937 makeMt19937(uint64_t seed) {
  if (seed <= std::mt19937::max()) {
    return std::mt19937(static_cast<std::mt19937::result_type>(seed));
  }
  std::seed_seq words{static_cast<uint32_t>(seed),
                      static_cast<uint32_t>(seed >> 32)};
  return std::mt19937(words);
}

} // namespace mlir
