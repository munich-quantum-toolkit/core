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

#include "mlir/Dialect/QCO/IR/QCOOps.h"

#include <llvm/ADT/SmallVector.h>

#include <variant>

namespace mlir::qco {

/**
 * A controlled gate whose control configuration can never be satisfied in the
 * current state: the body never runs, so the whole CtrlOp is deleted and every
 * qubit passes straight through.
 */
struct DropOp {
  CtrlOp op;
};

/**
 * @brief A controlled gate and the control qubits that provably always hold in
 * the current state.
 *
 * If a real control remains, the op is rebuilt with only those. If
 * dropControlIndices covers *every* control, the gate runs unconditionally and
 * its body is inlined in place of the op.
 *
 * dropControlIndices indexes into op.getInputControls(). Indices, not values,
 * so an earlier rewrite in the same batch cannot invalidate the decision.
 * Classical controls are not considered yet.
 */
struct StripControls {
  CtrlOp op;
  SmallVector<unsigned> dropControlIndices;
};

/// @brief One rewrite the pass has decided to perform.
using Decision = std::variant<DropOp, StripControls>;

} // namespace mlir::qco
