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

#include "QuantumState.hpp"

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/DenseMap.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LogicalResult.h>

#include <optional>

namespace mlir::qco {

/**
 * @brief One correlated subsystem-alternative.
 *
 * Holds a single QuantumState (one entanglement group's qubits, possibly none)
 * together with the classical values correlated with it, this branch's
 * probability, and its accumulated globalPhase.
 *
 * The enclosing UnionTable owns the partition: two HybridStates, whose qubit
 * sets are disjoint, are tensor factors, two whose qubit sets are equal are
 * alternatives of one probabilistic disjunction. A HybridState never sees a
 * qubit outside its own state.
 *
 * Every mutating operation accepts positive and negative classical controls; if
 * they do not hold in this branch, the operation is skipped (the qubit renames
 * still happen). An unresolved classical control is a failure().
 */
class HybridState {
  size_t maxNonzeroAmplitudes;
  double probability;
  Complex globalPhase{1.0, 0.0};
  QuantumState state;
  llvm::DenseMap<Value, Attribute> classical;

  /** Whether the classical controls permit the operation in this branch:
   * failure() if any is unresolved, else true (apply) / false (skip).
   * @param pos The classical values that need to be true (i.e., nonzero).
   * @param neg The classical values that need to be false (i.e., zero).
   *
   * @returns failure if one of the given Values is not present in the state.
   * Success and whether the controls hold if all values are present.
   */
  [[nodiscard(
      "HybridState::classicalControlsHold called but ignored")]] FailureOr<bool>
  classicalControlsHold(ArrayRef<Value> pos, ArrayRef<Value> neg) const;

public:
  /**
   * @param state The quantum state of this subsystem (may hold no qubits).
   * @param maxNonzeroAmplitudes Budget for QuantumStates created here (reset).
   * @param probability This alternative's weight within its slot (1 if sole).
   */
  HybridState(QuantumState state, const size_t maxNonzeroAmplitudes,
              const double probability)
      : maxNonzeroAmplitudes(maxNonzeroAmplitudes), probability(probability),
        state(std::move(state)) {}

  //===--------------------------------------------------------------------===//
  // Observers
  //===--------------------------------------------------------------------===//

  /// @brief Whether the quantum state of this branch is top.
  [[nodiscard("HybridState::isTop called but ignored")]] bool isTop() const {
    return state.isTop();
  }

  [[nodiscard("HybridState::getProbability called but ignored")]] double
  getProbability() const {
    return probability;
  }
  [[nodiscard("HybridState::getGlobalPhase called but ignored")]] Complex
  getGlobalPhase() const {
    return globalPhase;
  }
  // @brief The qubits this subsystem covers (the UnionTable's partition unit).
  [[nodiscard("HybridState::getQubits called but ignored")]] ArrayRef<Value>
  getQubits() const {
    return state.getQubits();
  }
  [[nodiscard("HybridState::hasQubit called but ignored")]] bool
  hasQubit(const Value q) const {
    return state.contains(q);
  }
  [[nodiscard("HybridState::getClassical called but ignored")]]
  std::optional<Attribute> getClassical(Value v) const;

  //===--------------------------------------------------------------------===//
  // Mutation
  //===--------------------------------------------------------------------===//

  /**
   * @brief Records the resolved constant of a classical value (overwrites).
   *
   * @param v The value whose attribute is changed.
   * @param attr The new attribute for v.
   */
  void setClassical(Value v, Attribute attr);

  /**
   * @brief Renames from to to, whether it is this branch's qubit or one of its
   * classical keys. No-op if from is not present.
   *
   * @param from The value being replaced.
   * @param to The value it is replaced with.
   */
  void forwardValue(Value from, Value to);

  /**
   * @brief Multiplies this branch's probability by factor.
   *
   * @param factor The factor to multiply the probability with.
   */
  void scaleProbability(const double factor) { probability *= factor; }

  /**
   * @brief Sets this branch's probability (its weight within its slot).
   *
   * @param newProbability The new probability.
   */
  void setProbability(const double newProbability) {
    probability = newProbability;
  }

  /// @brief Collapses this branch's QuantumState to top; classical facts stay.
  void markStateTop();

  /**
   * @brief Combines this subsystem with a disjoint one into a single
   * HybridState.
   *
   * The qubit sets must be disjointed. Probabilities and global phases
   * multiply; classical maps merge (other wins on a key collision). Becomes top
   * if the tensor product exceeds the amplitude budget.
   *
   * @param other The Hybrid state to merge this HybridState with.
   */
  [[nodiscard("HybridState::tensor called but ignored")]] HybridState
  tensor(const HybridState& other) const;

  //===--------------------------------------------------------------------===//
  // Gate application
  //===--------------------------------------------------------------------===//

  /**
   * @brief Applies a single-qubit unitary to in (renamed to out).
   *
   * in and all quantum controls must already be in this branch's state.
   * quantumCtrlsOut, if non-empty, matches quantumCtrlsIn in size and gives the
   * controls' post-gate values.
   *
   * @param in The qubit to apply the matrix to.
   * @param out The qubit that in is changed to.
   * @param matrix The matrix to apply to the amplitudes of in.
   * @param quantumCtrlsIn The qubits that have to be |1> to apply the matrix.
   * @param quantumCtrlsOut The qubits that quantumCtrlsIn are changed to.
   * @param posClassicalCtrls The classical values that have to be true
   * (nonzero) to apply the matrix.
   * @param negClassicalCtrls The classical values that have to be false (zero)
   * to apply the matrix.
   * @return failure() if a target/control qubit is not in this state, the
   * control in/out lengths mismatch, or a classical control is unresolved.
   */
  [[nodiscard("HybridState::applyMatrix1Q called but ignored")]]
  LogicalResult applyMatrix1Q(Value in, Value out, const Matrix2x2& matrix,
                              ArrayRef<Value> quantumCtrlsIn = {},
                              ArrayRef<Value> quantumCtrlsOut = {},
                              ArrayRef<Value> posClassicalCtrls = {},
                              ArrayRef<Value> negClassicalCtrls = {});

  /**
   * @brief Applies a two-qubit unitary to in (renamed to out).
   *
   * in0, in1, and all quantum controls must already be in this branch's state.
   * quantumCtrlsOut, if non-empty, matches quantumCtrlsIn in size and gives the
   * controls' post-gate values.
   *
   * @param in0 The high qubit to apply the matrix to.
   * @param in1 The low qubit to apply the matrix to.
   * @param out0 The qubit that in0 is changed to.
   * @param out1 The qubit that in1 is changed to.
   * @param matrix The matrix to apply to the amplitudes of in.
   * @param quantumCtrlsIn The qubits that have to be |1> to apply the matrix.
   * @param quantumCtrlsOut The qubits that quantumCtrlsIn are changed to.
   * @param posClassicalCtrls The classical values that have to be true
   * (nonzero) to apply the matrix.
   * @param negClassicalCtrls The classical values that have to be false (zero)
   * to apply the matrix.
   * @return failure() if a target/control qubit is not in this state, the
   * control in/out lengths mismatch, or a classical control is unresolved.
   */
  [[nodiscard("HybridState::applyMatrix2Q called but ignored")]]
  LogicalResult applyMatrix2Q(Value in0, Value in1, Value out0, Value out1,
                              const Matrix4x4& matrix,
                              ArrayRef<Value> quantumCtrlsIn = {},
                              ArrayRef<Value> quantumCtrlsOut = {},
                              ArrayRef<Value> posClassicalCtrls = {},
                              ArrayRef<Value> negClassicalCtrls = {});

  /**
   * @brief Adds a global phase exp(i*theta).
   *
   * Uncontrolled: accumulated into globalPhase. With quantum controls: a
   * relative phase on the subspace where every control is |1>.
   *
   * @param theta The phase to add.
   * @param quantumCtrlsIn The qubits that have to be |1> to apply the matrix.
   * @param quantumCtrlsOut The qubits that quantumCtrlsIn are changed to.
   * @param posClassicalCtrls The classical values that have to be true
   * (nonzero) to apply the matrix.
   * @param negClassicalCtrls The classical values that have to be false (zero)
   * to apply the matrix.
   * @return failure() if a control qubit is not in this state or a classical
   * control is unresolved.
   */
  [[nodiscard("HybridState::addGlobalPhase called but ignored")]]
  LogicalResult addGlobalPhase(double theta,
                               ArrayRef<Value> quantumCtrlsIn = {},
                               ArrayRef<Value> quantumCtrlsOut = {},
                               ArrayRef<Value> posClassicalCtrls = {},
                               ArrayRef<Value> negClassicalCtrls = {});

  //===--------------------------------------------------------------------===//
  // Measurement / reset
  //===--------------------------------------------------------------------===//

  /**
   * @brief Measures in (renamed to out), recording the outcome in
   * classicalResult.
   *
   * If in is deterministic, classicalResult is set to the exact i1 value;
   * otherwise the state is marked top, and classicalResult left unknown.
   *
   * @param in The qubit to be measured,
   * @param out The value to change in to.
   * @param classicalResult The classical value to save the result of the
   * measurement in.
   * @param posClassicalCtrls The classical values that have to be true
   * (nonzero) to apply the matrix.
   * @param negClassicalCtrls The classical values that have to be false (zero)
   * to apply the matrix.
   * @return failure() if in is not in this state or a classical control is
   * unresolved.
   */
  [[nodiscard("HybridState::measureQubit called but ignored")]]
  LogicalResult measureQubit(Value in, Value out, Value classicalResult,
                             ArrayRef<Value> posClassicalCtrls = {},
                             ArrayRef<Value> negClassicalCtrls = {});

  /**
   * @brief Resets in to |0> (renamed to out).
   *
   * Exact when this state is deterministic; otherwise the state is marked top
   * (the reduced state is mixed).
   *
   * @param in The qubit to be measured,
   * @param out The value to change in to.
   * @param posClassicalCtrls The classical values that have to be true
   * (nonzero) to apply the matrix.
   * @param negClassicalCtrls The classical values that have to be false (zero)
   * to apply the matrix.
   * @return failure() if in is not in this state or a classical control is
   * unresolved.
   */
  [[nodiscard("HybridState::resetQubit called but ignored")]] LogicalResult
  resetQubit(Value in, Value out, ArrayRef<Value> posClassicalCtrls = {},
             ArrayRef<Value> negClassicalCtrls = {});

  //===--------------------------------------------------------------------===//
  // Queries
  //===--------------------------------------------------------------------===//

  [[nodiscard("HybridState::isAlwaysZero called but ignored")]] bool
  isQubitAlwaysZero(Value q) const;
  [[nodiscard("HybridState::isAlwaysOne called but ignored")]] bool
  isQubitAlwaysOne(Value q) const;

  /// @brief Whether v is a known non-zero classical constant in this branch.
  [[nodiscard("HybridState::isClassicalTrue called but ignored")]] bool
  isClassicalTrue(Value v) const;
  /// @brief Whether v is a known zero classical constant in this branch.
  [[nodiscard("HybridState::isClassicalFalse called but ignored")]] bool
  isClassicalFalse(Value v) const;

  /**
   * Whether the given controls can all hold simultaneously in this branch
   * (positive classical not provably false, negative not provably true, quantum
   * controls jointly possible).
   *
   * @param quantumCtrls The qubits that have to be |1>.
   * @param posClassicalCtrls The classical values that have to be true
   * (nonzero) to apply the matrix.
   * @param negClassicalCtrls The classical values that have to be false (zero)
   * to apply the matrix.
   */
  [[nodiscard("HybridState::areControlsSatisfiable called but ignored")]] bool
  areControlsSatisfiable(ArrayRef<Value> quantumCtrls,
                         ArrayRef<Value> posClassicalCtrls,
                         ArrayRef<Value> negClassicalCtrls) const;

  //===--------------------------------------------------------------------===//
  // Comparison / dump
  //===--------------------------------------------------------------------===//

  /**
   * Whether the two branches carry the same state, global phase, and classical
   * facts - everything except their probability. The de-dup key when merging
   * alternatives in UnionTable::join.
   *
   * @param other The HybridState to compare this one with.
   */
  [[nodiscard("HybridState::sameConfiguration called but ignored")]] bool
  sameConfiguration(const HybridState& other) const;

  /// @brief sameConfiguration and equal probability, both within
  /// MATRIX_TOLERANCE.
  [[nodiscard("HybridState::== called but ignored")]] bool
  operator==(const HybridState& other) const;

  void print(raw_ostream& os) const;
};

} // namespace mlir::qco
