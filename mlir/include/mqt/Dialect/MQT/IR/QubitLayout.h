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
#include "mlir/IR/Diagnostics.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <optional>
#include <vector>

namespace mlir::mqt {

/// Number of input slots being tracked until placement publishes the layout.
inline constexpr llvm::StringLiteral kSourceQubitCountAttr =
    "mqt.source_qubit_count";

/// Input slot IDs retained through tensor shrinking and consumed by placement.
inline constexpr llvm::StringLiteral kSourceQubitIndicesAttr =
    "mqt.source_qubit_indices";

/// Layout metadata; see the MQT dialect reference for the schema.
///
/// initial maps program qubits to device positions. routing maps those
/// positions to final device positions. inputCount excludes workspace qubits.
struct QubitLayout {
  std::vector<int64_t> initial;
  std::optional<std::vector<int64_t>> routing;
  int64_t inputCount = 0;
  /// Target site IDs in position order; absent for circuit-wire ordering.
  std::optional<std::vector<int64_t>> sites;

  [[nodiscard]] DictionaryAttr toAttr(MLIRContext* context) const;
  [[nodiscard]] static FailureOr<QubitLayout>
  fromAttr(Attribute attribute,
           llvm::function_ref<InFlightDiagnostic()> emitError);
};

} // namespace mlir::mqt
