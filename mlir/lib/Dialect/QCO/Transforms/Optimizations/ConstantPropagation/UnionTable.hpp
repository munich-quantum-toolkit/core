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

#include <optional>

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
 * A UnionTable is a list of *slots*. Each slot is a non-empty list of
 * @ref HybridState "HybridStates":
 * - Slots are unentangled **tensor factors**; the full state is their product.
 * - The HybridStates within a slot are **alternatives** of one probabilistic
 *   disjunction; they share a qubit set and their probabilities sum to one.
 *
 * Qubit sets of different slots are disjoint. A slot with no qubits is a
 * purely classical factor.
 *
 * Operations take matrix-level arguments (no Operation*); the analysis maps
 * gates to matrices and target/output SSA values. Before a multi-qubit or
 * controlled operation the touched slots are merged into one (alternatives
 * multiply out via HybridState::tensor); if that exceeds maxHybridStates the
 * merged slots collapse to a single top state. A target or control value absent
 * from the table is a caller/propagation bug and yields failure(); the analysis
 * seeds every qubit before first use.
 */
class UnionTable {
public:
  using Slot = SmallVector<HybridState>;

private:
  bool allTop = false;
  size_t maxNonzeroAmplitudes = 0;
  size_t maxHybridStates = 0;
  SmallVector<Slot> slots;

  /// @brief Index of the slot that holds v (as a qubit or a classical key).
  [[nodiscard("UnionTable::slotIndexContaining called but ignored")]]
  std::optional<unsigned> slotIndexContaining(Value v) const;

  /// @brief The distinct slot indices touched by any of values, ascending.
  [[nodiscard("UnionTable::slotsTouchedBy called but ignored")]]
  SmallVector<unsigned> slotsTouchedBy(ArrayRef<Value> values) const;

  /**
   * @brief Merges every slot touched by values into a single slot.
   *
   * The merged slot's alternatives are the cartesian product of the merged
   * slots' alternatives, combined with HybridState::tensor (probabilities and
   * global phases multiply). If the product would exceed maxHybridStates only
   * the merged slots collapse to a single top state (untouched slots are left
   * alone). Values absent from the table are ignored.
   *
   * @param values The values whose slots should be merged.
   */
  void mergeSlots(ArrayRef<Value> values);

  /// @brief A single HybridState standing in for a slot's disjunction: its
  /// first alternative, keeping only the classical facts every alternative
  /// agrees on.
  [[nodiscard("UnionTable::reducedRepresentative called but ignored")]]
  static HybridState reducedRepresentative(const Slot& slot);

  /**
   * Combines the alternatives of two slots coming from sibling control-flow
   * paths: matching configurations are de-duplicated, the result is
   * renormalized to sum one.
   */
  [[nodiscard("UnionTable::mergeAlternatives called but ignored")]]
  static Slot mergeAlternatives(const Slot& a, const Slot& b);

  /// @brief Order-independent equality of two slots' alternatives.
  [[nodiscard("UnionTable::sameSlot called but ignored")]]
  static bool sameSlot(const Slot& a, const Slot& b);

public:
  /**
   * @param maxNonzeroAmplitudes Per-QuantumState amplitude budget before it
   * collapses to top.
   * @param maxHybridStates Per-slot alternative budget before the whole slot
   * collapses to allTop.
   */
  UnionTable(size_t maxNonzeroAmplitudes, size_t maxHybridStates)
      : maxNonzeroAmplitudes(maxNonzeroAmplitudes),
        maxHybridStates(maxHybridStates) {}

  /**
   * @brief A zero-budget table (every merge overflows to top). The dataflow
   * framework needs a default-constructible lattice payload; the analysis
   * overwrites it with a budgeted table before any real state flows through.
   */
  UnionTable() = default;

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
   * @brief Renames from to to everywhere (qubit or classical). No-op if from is
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
   * @return failure() if a target/control value is absent, the control in/out
   * lengths mismatch, or a classical control is unresolved.
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
   * @return failure() if a target/control value is absent, the two targets
   * coincide, the control in/out lengths mismatch, or a classical control is
   * unresolved.
   */
  [[nodiscard("UnionTable::applyMatrix2Q called but ignored")]] LogicalResult
  applyMatrix2Q(Value in0, Value in1, Value out0, Value out1,
                const Matrix4x4& matrix, ArrayRef<Value> quantumCtrlsIn = {},
                ArrayRef<Value> quantumCtrlsOut = {},
                ArrayRef<Value> posClassicalCtrls = {},
                ArrayRef<Value> negClassicalCtrls = {});

  /**
   * @brief Adds a global phase exp(i*theta), where theta is a classical value.
   *
   * The slot holding theta (and any controls) is coalesced, then each
   * alternative resolves theta from its own constants and applies the phase -
   * uncontrolled into its global phase, controlled as a relative phase.
   *
   * @param theta The classical value holding the rotation angle in radians.
   * @param quantumCtrlsIn The qubits that have to be |1> for the phase.
   * @param quantumCtrlsOut The qubits that quantumCtrlsIn are changed to.
   * @param posClassicalCtrls The classical values that have to be true
   * (nonzero) for the phase.
   * @param negClassicalCtrls The classical values that have to be false (zero)
   * for the phase.
   * @return failure() if theta or a control value is absent, the control in/out
   * lengths mismatch, or theta / a classical control is not a resolved constant
   * where the phase would apply (each indicating a propagation bug).
   */
  [[nodiscard("UnionTable::addGlobalPhase called but ignored")]] LogicalResult
  addGlobalPhase(Value theta, ArrayRef<Value> quantumCtrlsIn = {},
                 ArrayRef<Value> quantumCtrlsOut = {},
                 ArrayRef<Value> posClassicalCtrls = {},
                 ArrayRef<Value> negClassicalCtrls = {});

  /**
   * @brief Folds a classical operation across the distribution.
   *
   * Merges the slots of op 's tracked operands, then folds op per alternative
   * with that alternative's constants, recording any constant results. A result
   * that does not fold stays untracked.
   *
   * @param op The classical operation to propagate.
   */
  void propagateClassical(Operation* op);

  /**
   * @brief Measures in (renamed to out), recording the outcome in
   * classicalResult.
   *
   * Per alternative: an exact bit if in is deterministic there, otherwise that
   * alternative's QuantumState collapses to top, and the result stays unknown.
   *
   * @param in The qubit to be measured.
   * @param out The value to change in to.
   * @param classicalResult The classical value to record the outcome in.
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
   * A conjunction over disjoint factors (each factor must be satisfiable),
   * disjunction over a slot's alternatives (any alternative suffices). Controls
   * absent from the table are treated as possibly satisfiable.
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
   * @returns Whether the whole operation is superfluous (controls can never be
   * satisfied), plus the individual controls that always hold.
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
   * entanglement structure differs or a slot exceeds maxHybridStates.
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
