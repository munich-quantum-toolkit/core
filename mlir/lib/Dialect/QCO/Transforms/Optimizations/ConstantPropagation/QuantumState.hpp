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

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LogicalResult.h>

#include <memory>
#include <optional>
#include <utility>

namespace mlir::qco {

class QuantumState;

/**
 * @brief One branch of a measurement or reset: the observed bit, its
 * probability, and the normalized post-measurement state.
 */
struct MeasurementOutcome {
  unsigned bit;
  double probability;
  std::unique_ptr<QuantumState> state;
};

/**
 * @brief One entanglement group: a pure quantum state over a set of qubit SSA
 * values.
 *
 * The state is a map from computational-basis index to complex amplitude. The
 * bit at position i of a basis index refers to the qubit at position i of the
 * managed vector.
 *
 * If the number of non-zero amplitudes exceeds the threshold
 * maxNonzeroAmplitudes, or the group holds more than qubits than unsigned
 * int has bits, the state collapses to top.
 */
class QuantumState {
  bool top = false;
  size_t maxNonzeroAmplitudes;
  SmallVector<Value> qubits;
  llvm::DenseMap<uint64_t, Complex> amplitudes;

  explicit QuantumState(const size_t maxNonzeroAmplitudes)
      : maxNonzeroAmplitudes(maxNonzeroAmplitudes) {}

  /// @brief Bitmask of the positions of the given values that are in the group.
  [[nodiscard("QuantumState::maskOf called but ignored")]] uint64_t
  maskOf(ArrayRef<Value> values) const;

  /// @brief Drops negligible amplitudes and collapses to top if the conditions
  /// are met.
  void canonicalize();

public:
  /**
   * @brief Builds the all-zero state |0...0> over the qubits.
   *
   * @param qubits The qubit values of the group, in bit-position order.
   * @param maxNonzeroAmplitudes Amplitude budget before the state becomes top.
   */
  QuantumState(ArrayRef<Value> qubits, size_t maxNonzeroAmplitudes);

  /// @brief Builds a QuantumState for a single-qubit in state |0>.
  static QuantumState singletonZero(Value qubit, size_t maxNonzeroAmplitudes);

  [[nodiscard("QuantumState::isTop called but ignored")]] bool isTop() const {
    return top;
  }
  [[nodiscard("QuantumState::getQubits called but ignored")]] ArrayRef<Value>
  getQubits() const {
    return qubits;
  }

  /// @brief Whether QuantumState contains the qubit.
  [[nodiscard("QuantumState::contains called but ignored")]] bool
  contains(Value q) const {
    return indexOf(q).has_value();
  }
  /// @brief The bit position of a qubit, if present.
  [[nodiscard(
      "QuantumState::indexOf called but ignored")]] std::optional<unsigned>
  indexOf(Value q) const;

  /// @brief Collapses the state to top.
  void markTop();

  /// @brief Changes qubit from to qubit to in place. No-op if QuantumState does
  /// not contain from.
  void forwardQubit(Value from, Value to);

  /// @brief forwardQubit for each from[i] -> to[i]. to is empty (no rename) or
  /// the same length as from.
  void forwardQubits(ArrayRef<Value> from, ArrayRef<Value> to);

  /**
   * @brief Applies a single-qubit unitary to qubit in, renaming it to qubit
   * out.
   *
   * When ctrls is non-empty the matrix is applied only on the subspace where
   * every control qubit is |1>; the rest of the state passes through. Does
   * nothing but the rename when the state is top.
   *
   * @param in The qubit to apply the matrix to.
   * @param out The qubit that in is changed to.
   * @param matrix The matrix to apply to the amplitudes of in.
   * @param ctrlsIn The qubits that have to be |1> to apply the matrix.
   * @param ctrlsOut The controls' post-gate names (empty = unchanged, else same
   * length as ctrlsIn).
   * @return failure() if in or a control is not in this group (a
   * caller/propagation bug - the interpreter must co-locate a gate's targets
   * and controls before applying it), or the control in/out lengths mismatch;
   * success() otherwise.
   */
  [[nodiscard("QuantumState::applyMatrix1Q called but ignored")]] LogicalResult
  applyMatrix1Q(Value in, Value out, const Matrix2x2& matrix,
                ArrayRef<Value> ctrlsIn = {}, ArrayRef<Value> ctrlsOut = {});

  /**
   * @brief Applies a two-qubit unitary to in0 and in1, renaming them to out0,
   * out1.
   *
   * Qubit ordering follows QCO's @ref Matrix4x4 convention: in0 is the high bit
   * of the 4-dimensional local index, in1 the low bit. When ctrls is non-empty
   * the matrix is applied only on the subspace where every control qubit is
   * |1>; the rest of the state passes through. Does nothing but the renames
   * when the state is top.
   *
   * @param in0 The high bit the matrx is applied to.
   * @param in1 The low bit the matrx is applied to.
   * @param out0 The qubit that in0 is changed to.
   * @param out1 The qubit that in1 is changed to.
   * @param matrix The matrix to apply to the amplitudes of in0 and in1.
   * @param ctrlsIn The qubits that have to be |1> to apply the matrix.
   * @param ctrlsOut The controls' post-gate names (empty = unchanged, else same
   * length as ctrlsIn).
   * @return failure() if in0, in1, or a control is not in this group, in0 and
   * in1 are the same bit position, or the control in/out lengths mismatch (a
   * caller/propagation bug); success() otherwise.
   */
  [[nodiscard("QuantumState::applyMatrix2Q called but ignored")]] LogicalResult
  applyMatrix2Q(Value in0, Value in1, Value out0, Value out1,
                const Matrix4x4& matrix, ArrayRef<Value> ctrlsIn = {},
                ArrayRef<Value> ctrlsOut = {});

  /**
   * @brief Multiplies by exp(i*phase) the amplitudes where every control is
   * |1>.
   *
   * This function applies a relative phase when there is a controlled global
   * phase. An uncontrolled (global) phase is physically unobservable and not
   * recoverable from an amplitude map, so it is tracked by HybridState instead
   * and rejected here.
   *
   * @param phase The phase to apply.
   * @param ctrlsIn The qubits that all have to be |1> for the phase to apply.
   * @param ctrlsOut The controls' post-gate names (empty = unchanged, else same
   * length as ctrlsIn).
   * @return failure() if ctrlsIn is empty, a control is not in this group, or
   * the control in/out lengths mismatch.
   */
  [[nodiscard("QuantumState::applyControlledPhase called but ignored")]]
  LogicalResult applyControlledPhase(double phase, ArrayRef<Value> ctrlsIn,
                                     ArrayRef<Value> ctrlsOut = {});

  /**
   * @brief Projective measurement of qubit in in the computational basis.
   *
   * Each branch's state is re-normalized and holds the measured qubit (now
   * definite) under its post-measurement name out.
   *
   * @param in The qubit that is being measured.
   * @param out The measured qubit's post-measurement name.
   * @return failure() if in is not in the group (a caller/propagation bug).
   * Otherwise: an empty list if the state is top, one branch if the outcome is
   * deterministic, two branches otherwise.
   */
  [[nodiscard("QuantumState::measure called but ignored")]]
  FailureOr<SmallVector<MeasurementOutcome>> measure(Value in, Value out);

  /**
   * @brief Reset of qubit in: measure, then force the qubit to |0>.
   *
   * Each branch's state is re-normalized and holds the reset qubit (now |0>)
   * under its post-reset name out.
   *
   * @param in The qubit that is being reset.
   * @param out The reset qubit's post-reset name.
   * @return failure() if in is not in the group (a caller/propagation bug).
   * Otherwise: an empty list if the state is top, one branch if the outcome is
   * deterministic, two branches otherwise.
   */
  [[nodiscard("QuantumState::reset called but ignored")]]
  FailureOr<SmallVector<MeasurementOutcome>> reset(Value in, Value out);

  /**
   * @brief Tensor product of this group with that.
   *
   * The result's qubits are this->getQubits() followed by that.getQubits().
   * Becomes top if either operand is top or the product exceeds this group's
   * maximally allowed amplitude number. The two groups must not share qubits.
   *
   * @param that The QuantumState to unify this with.
   */
  [[nodiscard("QuantumState::unify called but ignored")]] QuantumState
  unify(const QuantumState& that) const;

  /// @brief Whether every non-zero amplitude has q set to zero.
  [[nodiscard("QuantumState::isAlwaysZero called but ignored")]] bool
  isAlwaysZero(Value q) const;

  /// @brief Whether every non-zero amplitude has q set to one.
  [[nodiscard("QuantumState::isAlwaysOne called but ignored")]] bool
  isAlwaysOne(Value q) const;

  /**
   * @brief Whether the given qubit basis never occurs.
   *
   * @param basis Pairs of (qubit, expected bit value); qubits not in the
   * group are ignored. Returns true when no non-zero amplitude matches all
   * the (in-group) pairs simultaneously.
   */
  [[nodiscard("QuantumState::hasZeroAmplitude called but ignored")]] bool
  hasAlwaysZeroAmplitude(ArrayRef<std::pair<Value, bool>> basis) const;

  [[nodiscard("QuantumState::== called but ignored")]] bool
  operator==(const QuantumState& that) const;

  /**
   * @brief Human-readable dump, e.g. "|010> -> 0.71, |110> -> -0.71".
   *
   * Used by the enclosing HybridState / UnionTable / lattice print overrides
   * and for debugging. Basis states are listed in ascending index order; bit i
   * of the printed string (from the right) is getQubits()[i]. Amplitudes use
   * two decimals and an "+ i" / "- i" imaginary part when non-negligible.
   * Prints nothing for a group with no qubits.
   */
  void print(raw_ostream& os) const;
};

} // namespace mlir::qco
