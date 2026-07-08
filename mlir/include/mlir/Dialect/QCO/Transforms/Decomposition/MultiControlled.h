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

#include "mlir/Dialect/QCO/Utils/Matrix.h"

#include <mlir/IR/Builders.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>

#include <cstdint>
#include <optional>

namespace mlir::qco::decomposition {

/// Target Pauli or phase gate inside a multi-controlled `qco.ctrl` body.
enum class ControlledTarget : std::uint8_t { X, Z, Phase };

/**
 * @brief Synthesizes a two-controlled Pauli-X, Pauli-Z, or phase gate.
 *
 * @param builder Builder positioned at the desired insertion point.
 * @param loc Location attached to the emitted operations.
 * @param control0 First control qubit (MSB).
 * @param control1 Second control qubit.
 * @param target Target qubit (LSB).
 * @param gate Target gate type (`CCX`, `CCZ`, or `CCPhase`).
 * @param theta Phase angle in radians when @p gate is
 * `ControlledTarget::Phase`.
 * @return The updated SSA values `[control0, control1, target]`.
 */
[[nodiscard]] SmallVector<Value>
synthesizeTwoControlled(OpBuilder& builder, Location loc, Value control0,
                        Value control1, Value target, ControlledTarget gate,
                        std::optional<double> theta = std::nullopt);

/**
 * @brief Synthesizes a compile-time-known two-controlled unitary.
 *
 * @details Dispatches to @ref synthesizeTwoControlled when the unitary matches
 * two-controlled @f$X@f$ or two-controlled phase patterns.
 *
 * @param builder Builder positioned at the desired insertion point.
 * @param loc Location attached to the emitted operations.
 * @param control0 First control qubit (MSB).
 * @param control1 Second control qubit.
 * @param target Target qubit (LSB).
 * @param unitary Compile-time-known @f$8 \times 8@f$ unitary matrix.
 * @return The updated SSA values `[control0, control1, target]`.
 *
 * @pre @p unitary must be @f$8 \times 8@f$.
 */
[[nodiscard]] SmallVector<Value>
synthesizeTwoControlled(OpBuilder& builder, Location loc, Value control0,
                        Value control1, Value target,
                        const DynamicMatrix& unitary);

/**
 * @brief Emits an elementary-gate decomposition of a relative-phase CCX
 * (`RCCX`).
 *
 * @param builder Builder positioned at the desired insertion point.
 * @param loc Location attached to the emitted operations.
 * @param control0 First control qubit.
 * @param control1 Second control qubit.
 * @param target Target qubit.
 * @return The updated SSA values `[control0, control1, target]`.
 */
[[nodiscard]] SmallVector<Value> synthesizeRCCX(OpBuilder& builder,
                                                Location loc, Value control0,
                                                Value control1, Value target);

/**
 * @brief Synthesizes a three-controlled Pauli-X, Pauli-Z, or phase gate.
 *
 * @param builder Builder positioned at the desired insertion point.
 * @param loc Location attached to the emitted operations.
 * @param controls Exactly three control qubits.
 * @param target Target qubit.
 * @param gate Target gate type (`CCCX`, `CCCZ`, or `CCCPhase`).
 * @param theta Phase angle in radians when @p gate is
 * `ControlledTarget::Phase`.
 * @return The updated SSA values, ordered as `[controls..., target]`.
 *
 * @pre @p controls must contain exactly three qubits.
 */
[[nodiscard]] SmallVector<Value>
synthesizeThreeControlled(OpBuilder& builder, Location loc, ValueRange controls,
                          Value target, ControlledTarget gate,
                          std::optional<double> theta = std::nullopt);

/**
 * @brief Emits a decomposition of a multi-controlled Pauli-X, Pauli-Z, or
 * phase gate.
 *
 * @details For exactly three controls, dispatches to @ref
 * synthesizeThreeControlled (used by @c decompose-three-controlled). For four
 * or more controls, applies the Huang-Palsberg (PLDI 2024) no-ancilla MCX core
 * (with target Hadamard bookends for Pauli-X). Subcircuits with two or three
 * controls are expanded via @ref synthesizeTwoControlled and @ref
 * synthesizeThreeControlled when @p minControls is at most the subcircuit
 * control count; otherwise the corresponding `qco.ctrl` building blocks are
 * retained for subsequent passes.
 *
 * @note Adapted from ``synth_mcx_noaux_hp24`` and ``synth_mcx_n_dirty_i15`` in
 *       the IBM Qiskit framework.
 *       (C) Copyright IBM 2025
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
 * @param controls Control qubits (at least three).
 * @param target Target qubit.
 * @param minControls Minimum control count for which subcircuits are expanded
 * rather than retained as building-block `qco.ctrl` operations.
 * @param gate Target gate type.
 * @param theta Phase angle when @p gate is `ControlledTarget::Phase`.
 * @return The updated SSA values, ordered as `[controls..., target]`.
 *
 * @pre @p controls must contain at least three qubits. The HP24 core is used
 * only when @p controls has four or more elements.
 */
[[nodiscard]] SmallVector<Value>
synthesizeMultiControlled(OpBuilder& builder, Location loc, ValueRange controls,
                          Value target, std::uint64_t minControls,
                          ControlledTarget gate,
                          std::optional<double> theta = std::nullopt);

} // namespace mlir::qco::decomposition
