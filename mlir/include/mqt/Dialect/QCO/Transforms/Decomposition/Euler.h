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

#include "mqt/Compiler/Target.h"
#include "mqt/Dialect/QCO/Transforms/Decomposition/Pauli.h"
#include "mqt/Dialect/QCO/Utils/Matrix.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/Support/LLVM.h"

#include "llvm/Support/LogicalResult.h"

#include <array>
#include <cstddef>
#include <optional>
#include <variant>

namespace mlir {
class Operation;
class RewriterBase;
} // namespace mlir

namespace mlir::qco::decomposition {

using SingleQubitBasis = CompilerTarget::SingleQubitBasis;
using RotationParameter = std::variant<double, Value>;

/// Reduce a gate angle modulo 4*pi before adding fixed Euler offsets.
/// This preserves SU(2) phase and avoids losing offsets for large angles.
[[nodiscard]] RotationParameter
normalizeRotationParameter(OpBuilder& builder, Location loc,
                           RotationParameter angle);

/// Parses a basis name (e.g. `zyz`, `zsxx`; case-insensitive).
///
/// @param basis The basis name.
/// @return The parsed basis, or `std::nullopt` if unrecognized.
[[nodiscard]] std::optional<SingleQubitBasis>
parseSingleQubitBasis(StringRef basis);

/// Euler angles `(theta, phi, lambda)` and global phase for a 2x2
/// unitary.
///
/// The decomposition obeys `matrix == e^{i*phase} * K(phi) * A(theta) *
/// K(lambda)` where `(K, A)` are the rotation axes of the chosen @ref
/// SingleQubitBasis.
struct EulerAngles {
  double theta = 0.0;  ///< Middle rotation angle.
  double phi = 0.0;    ///< First outer rotation angle.
  double lambda = 0.0; ///< Second outer rotation angle.
  double phase = 0.0;  ///< Global phase in radians.
};

/// Result of single-qubit synthesis, including its phase correction.
///
/// The caller owns materialization of @ref globalPhase. Compound synthesis can
/// therefore accumulate corrections in C++ and emit one `qco.gphase`.
struct SynthesizedUnitary1Q {
  Value qubit;
  double globalPhase = 0.0;
};

/// Returns whether @p op belongs to @p basis.
[[nodiscard]] bool isSingleQubitBasisGate(Operation* op,
                                          SingleQubitBasis basis);

/// Extracts `(theta, phi, lambda, phase)` of @p matrix in @p basis.
///
/// @param matrix The single-qubit unitary to decompose.
/// @param basis The single-qubit synthesis basis.
/// @return The extracted Euler angles and global phase.
[[nodiscard]] EulerAngles anglesFromUnitary(const Matrix2x2& matrix,
                                            SingleQubitBasis basis);

/// Synthesizes a composed single-qubit unitary as gates in @p basis.
///
/// Returns `std::nullopt` when @p hasNonBasisGate is false and resynthesis
/// would not shorten a run of @p runSize gates; otherwise emits gates and
/// returns their global-phase correction separately.
///
/// @param builder Builder for the emitted operations.
/// @param loc Location for the emitted operations.
/// @param qubit Input qubit value.
/// @param composed Composed unitary to synthesize.
/// @param runSize Number of gates in the run.
/// @param hasNonBasisGate Whether the run contains a gate outside @p basis.
/// @param basis The single-qubit synthesis basis.
/// @return The synthesized qubit and correction, or `std::nullopt` if synthesis
/// is skipped.
[[nodiscard]] std::optional<SynthesizedUnitary1Q>
synthesizeUnitary1QEuler(OpBuilder& builder, Location loc, Value qubit,
                         const Matrix2x2& composed, std::size_t runSize,
                         bool hasNonBasisGate,
                         const CompilerTarget::SynthesisBasis& basis);

/// Emit basis Euler angles `(theta, phi, lambda, phase)`. For the U basis,
/// phase is the correction multiplying U(theta, phi, lambda).
[[nodiscard]] Value
emitParameterizedEulerAngles(OpBuilder& builder, Location loc, Value qubit,
                             const std::array<RotationParameter, 4>& angles,
                             const CompilerTarget::SynthesisBasis& basis);

/// Synthesize an elementary Pauli rotation without evaluating its angle.
/// Constant frame changes and their phases use the ordinary Euler emitter.
[[nodiscard]] Value
synthesizePauliRotation1Q(OpBuilder& builder, Location loc, Value qubit,
                          PauliAxis axis, RotationParameter angle,
                          const CompilerTarget::SynthesisBasis& basis);

/// Materializes one accumulated phase correction when needed.
///
/// @param builder Builder for the operation.
/// @param loc Location of the operation.
/// @param phase Global phase in radians.
void emitGPhaseIfNeeded(OpBuilder& builder, Location loc, double phase);

/// Returns whether @p op supports runtime one-qubit synthesis.
[[nodiscard]] bool canSynthesizeParameterizedUnitary1Q(Operation* op);

/// Synthesizes one supported runtime-parameterized operation in @p basis.
///
/// Leaves operations that already belong to @p basis unchanged.
///
/// @pre `canSynthesizeParameterizedUnitary1Q(op)` is true.
void synthesizeParameterizedUnitary1Q(
    RewriterBase& rewriter, Operation* op,
    const CompilerTarget::SynthesisBasis& basis);

/// Emit equatorial gates while carrying Z rotations through diagonal gates.
/// Flush frames at region boundaries and other operations; preserve all phases.
LogicalResult synthesizeEquatorialGates(RewriterBase& rewriter, Operation* root,
                                        bool nativeRZ);

} // namespace mlir::qco::decomposition
