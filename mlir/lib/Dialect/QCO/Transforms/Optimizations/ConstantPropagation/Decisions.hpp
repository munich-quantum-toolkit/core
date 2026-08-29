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
#include <mlir/Support/LLVM.h>

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
// TODO: To remove all controls
/**
 * @brief A controlled gate and *strict subset* of control qubits that provably
 * always hold: the op is rebuilt with only the remaining controls.
 *
 * dropControlIndices indexes into op.getInputControls(). Indices, not values,
 * so an earlier rewrite in the same batch (which may RAUW this op's operands)
 * cannot invalidate the decision. The all-controls-redundant case (which would
 * turn the op into an uncontrolled gate) is out of v2.0 scope and never
 * produced here.
 */
struct StripControls {
  CtrlOp op;
  SmallVector<unsigned> dropControlIndices;
};

/// @brief One rewrite the pass has decided to perform.
using Decision = std::variant<DropOp, StripControls>;

} // namespace mlir::qco
