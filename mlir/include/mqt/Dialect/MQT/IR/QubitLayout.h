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
#include <string>
#include <vector>

namespace mlir::mqt {

/// Input slot IDs retained through tensor shrinking and consumed by placement.
inline constexpr llvm::StringLiteral kSourceQubitIndicesAttr =
    "mqt.source_qubit_indices";

/// Source register slots refer to logical inputs; -1 denotes an absent input.
struct LayoutRegister {
  std::string name;
  std::vector<int64_t> slots;
  bool ancillary = false;
};

/// Circuit-wire provenance; see the MQT dialect reference for the schema.
///
/// initial maps logical inputs to physical positions; outputOrder maps those
/// positions to circuit wires. final[i] = routing[outputOrder[initial[i]]], or
/// initial[i] when routing is absent. Missing assignments use -1.
/// Register slots and ancillas index logical inputs; inputCount excludes
/// workspace.
struct QubitLayout {
  int64_t physicalSize = 0;
  std::vector<int64_t> initial;
  std::optional<std::vector<int64_t>> routing;
  std::vector<int64_t> outputOrder;
  std::optional<int64_t> inputCount;
  std::vector<int64_t> ancillas;
  std::vector<LayoutRegister> registers;

  [[nodiscard]] DictionaryAttr toAttr(MLIRContext* context) const;
  [[nodiscard]] static FailureOr<QubitLayout>
  fromAttr(Attribute attribute,
           llvm::function_ref<InFlightDiagnostic()> emitError);
};

/// Require one entry point directly in the module and no nested entry points.
[[nodiscard]] LogicalResult verifyLayoutEntryPoint(ModuleOp moduleOp);

/// Require any layout provenance to belong to this program's sole entry point.
[[nodiscard]] LogicalResult verifyQubitLayoutOwner(ModuleOp moduleOp);

/// Invalidate retained layout provenance on the program entry point.
void invalidateQubitLayout(ModuleOp moduleOp);
/// Discard retained and invalidated layout provenance on the entry point.
void discardQubitLayout(ModuleOp moduleOp);
/// Reject entry-point layout provenance at an output boundary.
[[nodiscard]] LogicalResult requireNoQubitLayout(ModuleOp moduleOp);

} // namespace mlir::mqt
