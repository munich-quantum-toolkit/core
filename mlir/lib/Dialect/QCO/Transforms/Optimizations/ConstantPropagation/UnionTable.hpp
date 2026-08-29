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

#include "HybridState.hpp"
#include "QuantumState.hpp"

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LogicalResult.h>

namespace mlir::qco {

/**
 * @brief Which controls a controlled operation could drop without changing its
 * effect on the current state.
 *
 * completelySuperfluous means the operation can never fire (some control is
 * provably unsatisfiable), so the whole operation is dead. Otherwise, the two
 * sets list the individual controls that always hold and can be stripped.
 */
struct SuperfluousResult {
  bool completelySuperfluous = false;
  llvm::DenseSet<Value> superfluousQubits;
  llvm::DenseSet<Value> superfluousClassicalValues;
};

/**
 * @brief The abstract state of a whole program point: a probability
 * distribution over correlated subsystems.
 *
 * A UnionTable is a flat list of HybridState "HybridStates" partitioned into
 * *slots*:
 * - HybridStates in different slots are unentangled **tensor factors**; the
 * full state is their product.
 * - HybridStates in the same slot are **alternatives** of one probabilistic
 *   disjunction; their probabilities sum to one.
 *
 * The slot of a qubit-bearing HybridState is every HybridState with the exact
 * same qubit set. Each purely classical HybridState has its own slot.
 *
 * Operations take matrix-level arguments (no Operation*); the analysis maps
 * gates to matrices and target/output SSA values. Before a multi-qubit or
 * controlled operation, the touched slots are coalesced into one (alternatives
 * multiply out via HybridState::tensor); if that exceeds maxHybridStates all
 * states in the new slot collapse to top. A target or control value that is
 * absent from the table is a caller/propagation bug and yields failure(); the
 * analysis seeds every qubit before first use.
 */
class UnionTable {
  bool allTop = false;
  size_t maxNonzeroAmplitudes;
  size_t maxHybridStates;
  SmallVector<HybridState> hybridStates;

  /**
   * @brief Indices of the HybridStates that mention v (as a qubit or as a
   * classical key), ascending.
   *
   * @param v The value to be checked for
   * @returns The indices of the states with v
   */
  [[nodiscard(
      "UnionTable::statesWith called but ignored")]] SmallVector<unsigned>
  statesWith(Value v) const;

  /**
   * @brief The slot index belongs to (itself included), ascending.
   *
   * @param index The index to be checked for
   * @returns The indices of the slots that index belongs to.
   */
  [[nodiscard("UnionTable::slotOf called but ignored")]] SmallVector<unsigned>
  slotOf(unsigned index) const;

  /**
   * @brief The distinct slots touched by any of the values, each as an
   * ascending index list.
   *
   * @param values The values whose slots are collected.
   */
  [[nodiscard("UnionTable::slotsTouchedBy called but ignored")]] SmallVector<
      SmallVector<unsigned>>
  slotsTouchedBy(ArrayRef<Value> values) const;

  /**
   * @brief Fuses every slot touched by values into a single slot.
   *
   * The new slot's alternatives are the cartesian product of the fused slots'
   * alternatives, combined with HybridState::tensor (probabilities and global
   * phases multiply). Collapses the table to allTop if the product exceeds
   * maxHybridStates. Values absent from the table are ignored.
   *
   * @param values The values whose entries should be merged.
   */
  void mergeSlots(ArrayRef<Value> values);

public:
  /**
   * @param maxNonzeroAmplitudes Per-QuantumState amplitude budget before it
   * collapses to top.
   * @param maxHybridStates Per-UnionTable HybridState budget before the whole
   * table collapses to allTop.
   */
  UnionTable(const size_t maxNonzeroAmplitudes, const size_t maxHybridStates)
      : maxNonzeroAmplitudes(maxNonzeroAmplitudes),
        maxHybridStates(maxHybridStates) {}

  //===--------------------------------------------------------------------===//
  // Seeding
  //===--------------------------------------------------------------------===//

  /**
   * @brief Adds qubit in state |0> as its own factor. No-op if it is already
   * tracked or the table is allTop.
   *
   * @param qubit The qubit to be added.
   */
  void seedQubit(Value qubit);

  /**
   * @brief Records value as the resolved classical constant attr in its own
   * factor (overwrites an existing entry). No-op if the table is allTop.
   *
   * @param value The value to be saved.
   * @param attr The attribute that value should get.
   */
  void seedClassical(Value value, Attribute attr);

  /**
   * @brief Whether v is tracked as a qubit or a classical value.
   *
   * @param v The value to be checked for.
   * @returns True if v is already tracked.
   */
  [[nodiscard("UnionTable::isTracked called but ignored")]] bool
  isTracked(Value v) const;

  //===--------------------------------------------------------------------===//
  // SSA forwarding
  //===--------------------------------------------------------------------===//

  /**
   * @brief Renames from to to everywhere (qubit or clasical). No-op if from is
   * not present.
   *
   * @param from The value being replaced.
   * @param to The value it is replaced with.
   */
  void forwardValue(Value from, Value to);

  /**
   * @brief forwardValue for each from[i] -> to[i].
   *
   * @param from The values being replaced.
   * @param to The values from are replaced with.
   */
  void forwardValues(ArrayRef<Value> from, ArrayRef<Value> to);

  //===--------------------------------------------------------------------===//
  // Operation propagation
  //===--------------------------------------------------------------------===//

  /**
   * @brief Applies a single-qubit unitary to in (renamed to out).
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
  [[nodiscard("UnionTable::applyMatrix1Q called but ignored")]] LogicalResult
  applyMatrix1Q(Value in, Value out, const Matrix2x2& matrix,
                ArrayRef<Value> quantumCtrlsIn = {},
                ArrayRef<Value> quantumCtrlsOut = {},
                ArrayRef<Value> posClassicalCtrls = {},
                ArrayRef<Value> negClassicalCtrls = {});

  /**
   * @brief Applies a two-qubit unitary to in0, in1 (renamed to out0, out1),
   * following QCO's Matrix4x4 convention (in0 = high bit).
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
  [[nodiscard("UnionTable::applyMatrix2Q called but ignored")]] LogicalResult
  applyMatrix2Q(Value in0, Value in1, Value out0, Value out1,
                const Matrix4x4& matrix, ArrayRef<Value> quantumCtrlsIn = {},
                ArrayRef<Value> quantumCtrlsOut = {},
                ArrayRef<Value> posClassicalCtrls = {},
                ArrayRef<Value> negClassicalCtrls = {});

  /**
   * @brief Adds a global phase exp(i*theta).
   *
   * Uncontrolled: accumulated into one representative HybridState's global
   * phase. With quantum controls: a relative phase on the controlled subspace.
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
  [[nodiscard("UnionTable::addGlobalPhase called but ignored")]] LogicalResult
  addGlobalPhase(double theta, ArrayRef<Value> quantumCtrlsIn = {},
                 ArrayRef<Value> quantumCtrlsOut = {},
                 ArrayRef<Value> posClassicalCtrls = {},
                 ArrayRef<Value> negClassicalCtrls = {});

  /**
   * @brief Measures in (renamed to out), recording the outcome in
   * classicalResult.
   *
   * Per alternative: an exact bit if in is deterministic there, otherwise that
   * alternative's QuantumState collapses to top, and the result stays unknown.
   *
   * @param in The qubit to be measured.
   * @param out The value to change in to.
   * @param classicalResult The classical value to save the result of the
   * measurement in.
   * @param posClassicalCtrls The classical values that have to be true
   * (nonzero) to apply the measurement.
   * @param negClassicalCtrls The classical values that have to be false (zero)
   * to apply the measurement.
   * @return failure() if in is absent or a classical control is unresolved.
   */
  [[nodiscard("UnionTable::measureQubit called but ignored")]] LogicalResult
  measureQubit(Value in, Value out, Value classicalResult,
               ArrayRef<Value> posClassicalCtrls = {},
               ArrayRef<Value> negClassicalCtrls = {});

  /**
   * @brief Resets in to |0> (renamed to out).
   *
   * Exact per alternative when in is deterministic there, otherwise that
   * alternative's QuantumState collapses to top.
   *
   * @param in The qubit to be reset.
   * @param out The value to change in to.
   * @param posClassicalCtrls The classical values that have to be true
   * (nonzero) to apply the reset.
   * @param negClassicalCtrls The classical values that have to be false (zero)
   * to apply the reset.
   * @return failure() if in is absent or a classical control is unresolved.
   */
  [[nodiscard("UnionTable::resetQubit called but ignored")]] LogicalResult
  resetQubit(Value in, Value out, ArrayRef<Value> posClassicalCtrls = {},
             ArrayRef<Value> negClassicalCtrls = {});

  /**
   * @brief Collapses the QuantumState of every alternative in the slots that
   * hold qubits to top. The analysis' fallback for an operation whose effect it
   * cannot represent.
   *
   * @param qubits The qubits whose slots should collapse to top.
   */
  void markQubitsTop(ArrayRef<Value> qubits);

  //===--------------------------------------------------------------------===//
  // Queries
  //===--------------------------------------------------------------------===//

  [[nodiscard("UnionTable::isQubitAlwaysOne called but ignored")]] bool
  isQubitAlwaysOne(Value q) const;
  [[nodiscard("UnionTable::isQubitAlwaysZero called but ignored")]] bool
  isQubitAlwaysZero(Value q) const;
  [[nodiscard("UnionTable::isClassicalAlwaysTrue called but ignored")]] bool
  isClassicalAlwaysTrue(Value v) const;
  [[nodiscard("UnionTable::isClassicalAlwaysFalse called but ignored")]] bool
  isClassicalAlwaysFalse(Value v) const;

  /**
   * @brief Whether the controls can all hold at once somewhere in the
   * distribution.
   *
   * @param quantumCtrls The qubits that have to be |1>.
   * @param posClassicalCtrls The classical values that have to be true
   * (nonzero).
   * @param negClassicalCtrls The classical values that have to be false (zero).
   * @returns True if the control configuration is satisfiable.
   */
  [[nodiscard("UnionTable::areControlsSatisfiable called but ignored")]] bool
  areControlsSatisfiable(ArrayRef<Value> quantumCtrls,
                         ArrayRef<Value> posClassicalCtrls = {},
                         ArrayRef<Value> negClassicalCtrls = {}) const;

  /**
   * @brief Which of the given (positive quantum / positive classical / negative
   * classical) controls are redundant in the current state.
   *
   * @param quantumCtrls The qubits that have to be |1>.
   * @param posClassicalCtrls The classical values that have to be true
   * (nonzero).
   * @param negClassicalCtrls The classical values that have to be false (zero).
   * @returns Whether the whole operation is superfluous (controls will never be
   * satisfied), or if there are parts of the controls that are superfluous.
   */
  [[nodiscard("UnionTable::getSuperfluousControls called but ignored")]]
  SuperfluousResult
  getSuperfluousControls(ArrayRef<Value> quantumCtrls,
                         ArrayRef<Value> posClassicalCtrls = {},
                         ArrayRef<Value> negClassicalCtrls = {}) const;

  //===--------------------------------------------------------------------===//
  // Lattice support
  //===--------------------------------------------------------------------===//

  /**
   * @brief Reconciles this state with other coming from a sibling control-flow
   * path (the two branches of a non-constant qco.if).
   *
   * Slots are matched by qubit set. Matching slots merge their alternatives
   * (probability-weighted, deduplicated, renormalized); a classical-only fact
   * survives only if other asserts it too. The table collapses to allTop if the
   * entanglement structure differs or maxHybridStates is exceeded.
   *
   * The caller aligns yielded SSA names (via forwardValues) before calling.
   *
   * @param other The UnionTable to join this with.
   */
  void join(const UnionTable& other);

  /// @brief Collapses the whole table: no quantum or classical facts survive.
  void markAllTop();

  [[nodiscard("UnionTable::isAllTop called but ignored")]] bool
  isAllTop() const {
    return allTop;
  }

  /// @brief Whether every tracked QuantumState is top (classical facts may
  /// remain).
  [[nodiscard("UnionTable::areStatesAllTop called but ignored")]] bool
  areStatesAllTop() const;

  /// @brief Order-independent structural equality (drives lattice convergence).
  [[nodiscard("UnionTable::== called but ignored")]] bool
  operator==(const UnionTable& other) const;

  void print(raw_ostream& os) const;
};

} // namespace mlir::qco
