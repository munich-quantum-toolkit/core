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

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace mlir::mqt {

/// Source register slots refer to logical inputs; -1 denotes an absent input.
struct LayoutRegister {
  std::string name;
  std::vector<int64_t> slots;
  bool ancillary = false;
};

/// Serialized circuit-wire provenance, independent of any frontend SDK.
///
/// Initial positions index the physical output order. Routing maps physical
/// circuit wires to their final positions; an absent routing map leaves initial
/// positions unchanged. Otherwise final[i] = routing[outputOrder[initial[i]]].
/// Missing initial or routing assignments use -1. Register slots and ancillary
/// inputs refer to the logical input order. inputCount, when present, separates
/// original inputs from subsequently introduced workspace inputs.
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

/// Invalidate retained layout provenance in this module and its nested modules.
void invalidateQubitLayout(ModuleOp moduleOp);
/// Explicitly discard both retained and invalidated layout provenance.
void discardQubitLayout(ModuleOp moduleOp);
/// Reject layout loss or use of stale metadata at a serialization boundary.
[[nodiscard]] LogicalResult requireNoQubitLayout(ModuleOp moduleOp);

} // namespace mlir::mqt
