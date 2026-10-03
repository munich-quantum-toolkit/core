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
#include "mqt/Dialect/QCO/Utils/Matrix.h"

#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <variant>

namespace mlir {
class Operation;
class OpBuilder;
class Location;
class RewriterBase;
} // namespace mlir

namespace mlir::qco::decomposition {

enum class PauliAxis : uint8_t { I, X, Y, Z };

/// A quarter-turn Clifford C that maps from to the requested Pauli axis.
/// The identity axis returns the identity matrix.
[[nodiscard]] Matrix2x2 pauliFrame(PauliAxis axis,
                                   PauliAxis from = PauliAxis::Z);

/// Generator axes of a recognized one- or two-qubit Pauli rotation.
/// A one-qubit rotation has I as its second axis.
[[nodiscard]] std::array<PauliAxis, 2> pauliAxes(CompilerTarget::GateKind gate);

/// Emits RXX, RYY, RZX, or RZZ with a scalar or SSA angle.
[[nodiscard]] Operation* emitPauliRotation2Q(OpBuilder& builder, Location loc,
                                             Value qubit0, Value qubit1,
                                             CompilerTarget::GateKind gate,
                                             std::variant<double, Value> angle);

/// R_P(theta) = exp(i*phase) P^product C R_P(angle) C, where C
/// anticommutes with P when negate is true. The angle lies in [0, pi/2].
struct FoldedPauliAngle {
  double angle;
  double phase;
  bool product;
  bool negate;
};

[[nodiscard]] FoldedPauliAngle foldPauliAngle(double angle);

struct PauliRotation {
  std::array<PauliAxis, 2> axes;
  double angleScale = 1.;
};

/// Commuting rotations sharing one angle, with an explicit global phase.
/// Represents the existing Pauli rotations, P, and their one-control forms.
/// Recognized two-qubit operations contain exactly one entangling rotation.
struct PauliRotationSequence {
  Value angle;
  SmallVector<PauliRotation, 3> rotations;
  double globalPhaseScale = 0.;
};

/// Recognizes exact Pauli-generator decompositions without changing the IR.
[[nodiscard]] std::optional<PauliRotationSequence>
getPauliRotations(Operation* operation);

/// Cost of direct Pauli synthesis. Constant Clifford angles use the matrix
/// planner, which can remove or shorten their entangling part.
[[nodiscard]] std::optional<size_t>
pauliRotationEntanglerCount(const PauliRotationSequence& sequence,
                            CompilerTarget::Entangler entangler);

/// Emits a recognized sequence directly in the target's synthesis basis.
/// Hoists supporting scalar operations from a recognized control body; the
/// caller replaces the original operation with the returned qubits.
[[nodiscard]] SmallVector<Value, 2>
emitPauliRotations(RewriterBase& rewriter, Operation* operation,
                   const PauliRotationSequence& sequence,
                   const CompilerTarget::SynthesisBasis& basis, bool reverse);

} // namespace mlir::qco::decomposition
