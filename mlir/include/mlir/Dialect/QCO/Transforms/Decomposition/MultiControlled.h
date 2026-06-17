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

#include <mlir/IR/Builders.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>

namespace mlir::qco::decomposition {

/**
 * @brief Emits a decomposition of a multi-controlled X gate.
 *
 * @details Emits a sequence of one- and two-qubit gates that, taken together,
 * implement the multi-controlled X gate that flips @p target whenever all @p
 * controls are in the @f$|1\rangle@f$ state. The decomposition follows Huang
 * and Palsberg (PLDI 2024), composing Iten et al. (Phys. Rev. A 93, 032318,
 * 2016) dirty-ancilla subcircuits for large control counts. The emitted gates
 * are `h`, `t`, `tdg`, `p`, and `cx`.
 *
 * @note Adapted from ``synth_mcx_noaux_hp24`` and ``synth_mcx_n_dirty_i15`` in
 *       the IBM Qiskit framework
 *       (``crates/synthesis/src/multi_controlled/mcx.rs``).
 *       (C) Copyright IBM 2024
 *
 *       This code is licensed under the Apache License, Version 2.0. You may
 *       obtain a copy of this license in the LICENSE.txt file in the root
 *       directory of this source tree or at
 *       https://www.apache.org/licenses/LICENSE-2.0.
 *
 *       Any modifications or derivative works of this code must retain this
 *       copyright notice, and modified files need to carry a notice
 *       indicating that they have been altered from the originals.
 *
 * @param builder Builder positioned at the desired insertion point.
 * @param loc Location attached to the emitted operations.
 * @param controls Current SSA values of the control qubits.
 * @param target Current SSA value of the target qubit.
 * @return The updated SSA values, ordered as `[controls..., target]`.
 */
[[nodiscard]] SmallVector<Value> synthesizeMcx(OpBuilder& builder, Location loc,
                                               ValueRange controls,
                                               Value target);

} // namespace mlir::qco::decomposition
