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
#include "qdmi/QDMI.hpp"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace mqt::runtime {

/// Validated logical MSB-first outcomes and their multiplicities.
/// Ordered SHOTS remain ordered; aggregate consumers sum projected collisions.
using Readout = llvm::SmallVector<std::pair<std::string, size_t>>;

/// Load logical results, falling back only on unavailable or empty retrieval.
[[nodiscard]] Readout readout(const qdmi::Job& job, size_t width, size_t shots,
                              size_t programIndex,
                              const std::optional<mlir::ResultLayout>& layout,
                              bool samples);

/// Expand ordered readout into a row-major array in logical wire order.
[[nodiscard]] std::vector<int8_t>
decodeSamples(const qdmi::Job& job, size_t width, size_t shots,
              size_t programIndex = 0,
              const std::optional<mlir::ResultLayout>& layout = std::nullopt);

/// Index a validated outcome in the requested wire order, most significant
/// first.
[[nodiscard]] size_t basisIndex(llvm::StringRef bits,
                                llvm::ArrayRef<size_t> wires);

/// Accumulate probabilities in the requested wire order.
[[nodiscard]] std::vector<double>
marginalProbabilities(const Readout& result, llvm::ArrayRef<size_t> wires,
                      size_t shots);

/// Accumulate binary keys in the requested wire order, without a word-size
/// limit.
[[nodiscard]] std::map<std::string, size_t>
marginalCounts(const Readout& result, llvm::ArrayRef<size_t> wires);

} // namespace mqt::runtime
