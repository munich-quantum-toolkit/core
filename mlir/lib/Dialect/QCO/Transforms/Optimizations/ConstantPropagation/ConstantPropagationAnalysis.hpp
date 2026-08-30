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

#include "UnionTable.hpp"
#include "mlir/Dialect/QCO/IR/QCOInterfaces.h"
#include "mlir/Dialect/QCO/IR/QCOOps.h"

#include <llvm/Support/raw_ostream.h>
#include <mlir/Analysis/DataFlow/DenseAnalysis.h>
#include <mlir/Analysis/DataFlowFramework.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>
#include <mlir/Interfaces/ControlFlowInterfaces.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/TypeID.h>

#include <cstddef>
#include <optional>

namespace mlir::qco {

/**
 * @brief The dense-lattice payload for the constant-propagation analysis: one
 * UnionTable per program point.
 *
 * The lattice has an explicit *uninitialized* (bottom) state so that the
 * framework's join-accumulation over control-flow edges works: joining bottom
 * with a value adopts the value; joining two values delegates to
 * UnionTable::join.
 */
class UnionTableLattice : public dataflow::AbstractDenseLattice {
  UnionTable table;
  bool initialized = false;

public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(UnionTableLattice)

  using AbstractDenseLattice::AbstractDenseLattice;

  /// @brief LLVM-RTTI hook. The dataflow solver instantiates exactly one dense
  /// lattice type per analysis, so every AbstractDenseLattice that is handed
  /// (e.g. in join) is a UnionTableLattice.
  static bool classof(const AbstractDenseLattice*) { return true; }

  ChangeResult join(const AbstractDenseLattice& other) override;

  void print(raw_ostream& os) const override;

  [[nodiscard("UnionTableLattice::isInitialized is called but ignored")]] bool
  isInitialized() const {
    return initialized;
  }

  /// @brief The payload. Only meaningful once @isInitialized.
  [[nodiscard("UnionTableLattice::getUnionTable is called but "
              "ignored")]] const UnionTable&
  getUnionTable() const {
    return table;
  }

  /// @brief Replaces the payload; returns whether it changed.
  ChangeResult setUnionTable(UnionTable next);

  /// @brief Joins rhs into the payload (adopting it if still uninitialized).
  ChangeResult joinUnionTable(const UnionTable& rhs);
};

/**
 * Forward dense data-flow analysis that threads a UnionTable through a QCO
 * program, interpreting gates, measurements, resets, global phases, classical
 * folds, control modifiers, and constant/branching `qco.if`.
 *
 * Unsupported constructs (`scf.for`, `qco.index_switch`, and any operation
 * touching qubits that the analysis cannot model) make the pass fail via
 * `emitError`. Precision losses (parametric gates, `qco.inv` / `qco.pow`
 * bodies, non-constant `qco.if`) collapse the affected qubits to top instead.
 *
 * Does not call across a boundary: if the module contains any call, the
 * analysis reports top everywhere. Uncalled helper functions are tolerated -
 * only the entry-point function's qubit arguments are assumed to be |0>; every
 * other function's qubits are treated as unknown.
 */
class ConstantPropagationAnalysis
    : public dataflow::DenseForwardDataFlowAnalysis<UnionTableLattice> {
  size_t maxNonzeroAmplitudes;
  size_t maxHybridStates;

  /// @brief Set in @ref initialize when the module contains any call; every
  /// program point is then top.
  bool bailToTop = false;

  /// @brief A budgeted empty table, or an all-top one when @ref bailToTop.
  [[nodiscard]] UnionTable freshTable() const;

  /// @brief Dispatches a single operation onto table, given the quantum
  /// controls accumulated by any enclosing qco.ctrl.
  LogicalResult applyOperation(UnionTable& table, Operation* op,
                               ArrayRef<Value> quantumControls);

  /// @brief Applies a unitary operation: its 1-/2-qubit matrix if available,
  /// otherwise the targets become top (parametric, >2-qubit, dynamic-matrix, or
  /// an unmodelled qco.inv / qco.pow body).
  static LogicalResult applyUnitary(UnionTable& table, UnitaryOpInterface gate,
                                    ArrayRef<Value> quantumControls);

  /// @brief Interprets a qco.ctrl body, extending the control context.
  LogicalResult applyCtrl(UnionTable& table, CtrlOp ctrl,
                          ArrayRef<Value> quantumControls);

protected:
  void setToEntryState(UnionTableLattice* lattice) override;

public:
  ConstantPropagationAnalysis(DataFlowSolver& solver,
                              size_t maxNonzeroAmplitudes,
                              size_t maxHybridStates);

  LogicalResult initialize(Operation* top) override;

  LogicalResult visitOperation(Operation* op, const UnionTableLattice& before,
                               UnionTableLattice* after) override;

  void visitRegionBranchControlFlowTransfer(RegionBranchOpInterface branch,
                                            std::optional<unsigned> regionFrom,
                                            std::optional<unsigned> regionTo,
                                            const UnionTableLattice& before,
                                            UnionTableLattice* after) override;
};

} // namespace mlir::qco
