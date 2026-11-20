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

/// Program-qubit router order mapped to dense target vertices between passes.
///
/// The array length excludes unused target sites.
inline constexpr llvm::StringLiteral kPlacementOrderAttr =
    "mqt.placement_order";

/// Layout metadata; see the MQT dialect reference for the schema.
///
/// initial maps program qubits to device positions. routing maps those
/// positions to final device positions. Thus routing[initial[i]] is the final
/// position of program qubit i when routing is present; otherwise use
/// initial[i]. sites maps a device position to a target site ID when present.
/// inputCount excludes workspace qubits. These maps do not describe classical
/// result order.
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

/// Read the circuit-wire layout of a module with @p count source qubits.
///
/// Return the identity if no layout is attached. Emit an error unless an
/// attached layout has no sites and one position per source qubit.
[[nodiscard]] FailureOr<QubitLayout> getCircuitLayout(ModuleOp moduleOp,
                                                      int64_t count);

} // namespace mlir::mqt
