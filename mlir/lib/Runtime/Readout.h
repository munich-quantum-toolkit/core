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

#include "mqt/Compiler/ResultLayout.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"

#include <cstddef>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace mqt::runtime {

/// Validated logical MSB-first outcomes and their multiplicities.
///
/// Ordered SHOTS additionally retain an outcome index for every shot.
/// Histograms have no order; aggregate consumers sum projected collisions.
struct Readout {
  llvm::SmallVector<std::pair<std::string, size_t>> outcomes;
  llvm::SmallVector<size_t> order;
};

/// Decode distinct raw outcomes once, merging equal logical outcomes.
///
/// Preserve shot order without materializing a logical string for every shot.
[[nodiscard]] llvm::Expected<Readout>
decodeOrderedShots(const std::vector<std::string>& samples,
                   std::optional<size_t> width, size_t shots,
                   const std::optional<mlir::ResultLayout>& layout);

} // namespace mqt::runtime
