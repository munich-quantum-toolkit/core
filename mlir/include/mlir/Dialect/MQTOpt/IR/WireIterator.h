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

#include "mlir/Dialect/MQTOpt/IR/MQTOptDialect.h"

#include <cstddef>
#include <iterator>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/ErrorHandling.h>
#include <mlir/Analysis/SliceAnalysis.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>

namespace mqt::ir::opt {

/**
 * @brief A bidirectional_iterator traversing the def-use chain of a qubit wire.
 *
 * The iterator follows the flow of a qubit through a sequence of quantum
 * operations in a given region. It respects the semantics of the respective
 * quantum operation including control flow constructs (scf::ForOp and
 * scf::IfOp).
 *
 * It treats control flow constructs as a single operation that consumes and
 * yields a corresponding number of qubits, without descending into their nested
 * regions.
 */
class WireIterator {
  /// @returns a view of all input qubits.
  [[nodiscard]] static auto getAllInQubits(UnitaryInterface op) {
    return llvm::concat<mlir::Value>(op.getInQubits(), op.getPosCtrlInQubits(),
                                     op.getNegCtrlInQubits());
  }

  /// @returns a view of all output qubits.
  [[nodiscard]] static auto getAllOutQubits(UnitaryInterface op) {
    return llvm::concat<mlir::Value>(
        op.getOutQubits(), op.getPosCtrlOutQubits(), op.getNegCtrlOutQubits());
  }

  /**
   * @brief Find corresponding output from input value for a unitary (Forward).
   *
   * @note That we don't use the interface method here because
   * it creates temporary std::vectors instead of using views.
   */
  [[nodiscard]] static mlir::Value findOutput(UnitaryInterface op,
                                              mlir::Value in) {
    const auto ins = getAllInQubits(op);
    const auto outs = getAllOutQubits(op);
    const auto it = llvm::find(ins, in);
    assert(it != ins.end() && "input qubit not found in operation");
    const auto index = std::distance(ins.begin(), it);
    return *(std::next(outs.begin(), index));
  }

  /**
   * @brief Find corresponding input from output value for a unitary (Backward).
   *
   * @note That we don't use the interface method here because
   * it creates temporary std::vectors instead of using views.
   */
  [[nodiscard]] static mlir::Value findInput(UnitaryInterface op,
                                             mlir::Value out) {
    const auto ins = getAllInQubits(op);
    const auto outs = getAllOutQubits(op);
    const auto it = llvm::find(outs, out);
    assert(it != outs.end() && "output qubit not found in operation");
    const auto index = std::distance(outs.begin(), it);
    return *(std::next(ins.begin(), index));
  }

  /**
   * @brief Find corresponding result from init argument value (Forward).
   */
  [[nodiscard]] static mlir::Value findResult(mlir::scf::ForOp op,
                                              mlir::Value initArg) {
    const auto initArgs = op.getInitArgs();
    const auto it = llvm::find(initArgs, initArg);
    assert(it != initArgs.end() && "init arg qubit not found in operation");
    const auto index = std::distance(initArgs.begin(), it);
    return op->getResult(index);
  }

  /**
   * @brief Find corresponding init argument from result value (Backward).
   */
  [[nodiscard]] static mlir::Value findInitArg(mlir::scf::ForOp op,
                                               mlir::Value res) {
    return op.getInitArgs()[cast<mlir::OpResult>(res).getResultNumber()];
  }

  /**
   * @brief Find corresponding result value from input qubit value (Forward).
   *
   * @details Recursively traverses the IR "downwards" until the respective
   * yield is found. Requires that each branch takes and returns the same
   * (possibly modified) qubits. Hence, we can just traverse the then-branch.
   */
  [[nodiscard]] static mlir::Value findResult(mlir::scf::IfOp op,
                                              mlir::Value q) {
    /// Use the branch with fewer ops.
    /// Note: LLVM doesn't guarantee that range_size is in O(1).
    /// Might effect performance.
    const auto szThen = llvm::range_size(op.getThenRegion().getOps());
    const auto szElse = llvm::range_size(op.getElseRegion().getOps());
    mlir::Region& region =
        szElse >= szThen ? op.getThenRegion() : op.getElseRegion();

    WireIterator it(q, &region);

    /// Assumptions:
    ///     First, there must be a yield.
    ///     Second, yield is a sentinel.
    /// Then: Advance until the yield before the sentinel.

    it = std::prev(std::ranges::next(it, std::default_sentinel));
    assert(isa<mlir::scf::YieldOp>(*it) && "expected yield op");
    auto yield = cast<mlir::scf::YieldOp>(*it);

    /// Get the corresponding result.

    const auto results = yield.getResults();
    const auto yieldIt = llvm::find(results, it.q);
    assert(yieldIt != results.end() && "yielded qubit not found in operation");
    const auto index = std::distance(results.begin(), yieldIt);
    return op->getResult(index);
  }

  /**
   * @brief Find the first value outside the branch region for a given result
   * value (Backward).
   *
   * @details Recursively traverses the IR "upwards" until a value outside the
   * branch region is found. If the iterator's operation does not change during
   * backward traversal, it indicates that the def-use chain starts within the
   * branch region and does not extend into the parent region.
   */
  [[nodiscard]] static mlir::Value findValue(mlir::scf::IfOp op,
                                             mlir::Value q) {
    const auto num = cast<mlir::OpResult>(q).getResultNumber();
    mlir::Operation* term = op.thenBlock()->getTerminator();
    mlir::scf::YieldOp yield = llvm::cast<mlir::scf::YieldOp>(term);
    mlir::Value v = yield.getResults()[num];
    assert(v != nullptr && "expected yielded value");

    mlir::Operation* prev{};
    WireIterator it(v, &op.getThenRegion());
    while (it.qubit().getParentRegion() != op->getParentRegion()) {
      /// Since the definingOp of q might be a nullptr (BlockArgument), don't
      /// immediately dereference the iterator here.
      mlir::Operation* curr = it.qubit().getDefiningOp();
      if (curr == prev || curr == nullptr) {
        break;
      }
      prev = *it;
      --it;
    }

    return it.qubit();
  }

  /**
   * @brief Return the first user of a value in a given region.
   * @param v The value.
   * @param region The targeted region.
   * @return A pointer to the user, or nullptr if none exists.
   */
  [[nodiscard]] static mlir::Operation* getUserInRegion(mlir::Value v,
                                                        mlir::Region* region) {
    for (mlir::Operation* user : v.getUsers()) {
      if (user->getParentRegion() == region) {
        return user;
      }
    }
    return nullptr;
  }

public:
  using iterator_category = std::bidirectional_iterator_tag;
  using difference_type = std::ptrdiff_t;
  using value_type = mlir::Operation*;

  explicit WireIterator() = default;
  explicit WireIterator(mlir::Value q, mlir::Region* region)
      : currOp(q.getDefiningOp()), q(q), region(region) {}

  [[nodiscard]] mlir::Operation* operator*() const {
    assert(!sentinel && "Dereferencing sentinel iterator");
    assert(currOp && "Dereferencing null operation");
    return currOp;
  }

  [[nodiscard]] mlir::Value qubit() const { return q; }

  WireIterator& operator++() {
    advanceForward();
    return *this;
  }

  WireIterator operator++(int) {
    auto tmp = *this;
    ++*this;
    return tmp;
  }

  WireIterator& operator--() {
    advanceBackward();
    return *this;
  }

  WireIterator operator--(int) {
    auto tmp = *this;
    --*this;
    return tmp;
  }

  bool operator==(const WireIterator& other) const {
    return other.q == q && other.currOp == currOp && other.sentinel == sentinel;
  }

  bool operator==([[maybe_unused]] std::default_sentinel_t s) const {
    return sentinel;
  }

private:
  void advanceForward() {
    /// If we are already at the sentinel, there is nothing to do.
    if (sentinel) {
      return;
    }

    /// Find output from input qubit.
    /// If there is no output qubit, set `sentinel` to true.
    if (q.getDefiningOp() != currOp) {
      mlir::TypeSwitch<mlir::Operation*>(currOp)
          .Case<UnitaryInterface>(
              [&](UnitaryInterface op) { q = findOutput(op, q); })
          .Case<ResetOp>([&](ResetOp op) { q = op.getOutQubit(); })
          .Case<MeasureOp>([&](MeasureOp op) { q = op.getOutQubit(); })
          .Case<mlir::scf::ForOp>(
              [&](mlir::scf::ForOp op) { q = findResult(op, q); })
          .Case<mlir::scf::IfOp>(
              [&](mlir::scf::IfOp op) { q = findResult(op, q); })
          .Case<DeallocQubitOp, mlir::scf::YieldOp>(
              [&](auto) { sentinel = true; })
          .Default([&](mlir::Operation* op) {
            report_fatal_error("unknown op in def-use chain: " +
                               op->getName().getStringRef());
          });
    }

    /// Find the next operation.
    /// If it is a sentinel there are no more ops.
    if (sentinel) {
      return;
    }

    /// If there are no more uses, set `sentinel` to true.
    if (q.use_empty()) {
      sentinel = true;
      return;
    }

    /// Otherwise, search the user in the targeted region.
    currOp = getUserInRegion(q, getRegion());
    if (currOp == nullptr) {
      /// Since !q.use_empty: must be a branching op.
      currOp = q.getUsers().begin()->getParentOp();
      /// For now, just check if it's a scf::IfOp.
      /// Theoretically this could also be an scf::IndexSwitch, etc.
      assert(isa<mlir::scf::IfOp>(currOp));
    }
  }

  void advanceBackward() {
    /// If we are at the sentinel and move backwards, "revive" the
    /// qubit value and operation.
    if (sentinel) {
      sentinel = false;
      return;
    }

    /// Get the operation that produces the qubit value.
    currOp = q.getDefiningOp();

    /// If q is a BlockArgument (no defining op), hold.
    if (currOp == nullptr) {
      return;
    }

    /// Find input from output qubit.
    /// If there is no input qubit, hold.
    mlir::TypeSwitch<mlir::Operation*>(currOp)
        .Case<UnitaryInterface>(
            [&](UnitaryInterface op) { q = findInput(op, q); })
        .Case<ResetOp, MeasureOp>([&](auto op) { q = op.getInQubit(); })
        .Case<DeallocQubitOp>([&](DeallocQubitOp op) { q = op.getQubit(); })
        .Case<mlir::scf::ForOp>(
            [&](mlir::scf::ForOp op) { q = findInitArg(op, q); })
        .Case<mlir::scf::IfOp>(
            [&](mlir::scf::IfOp op) { q = findValue(op, q); })
        .Case<AllocQubitOp, QubitOp>([&](auto) { /* hold (no-op) */ })
        .Default([&](mlir::Operation* op) {
          report_fatal_error("unknown op in def-use chain: " +
                             op->getName().getStringRef());
        });
  }

  /**
   * @brief Return the active region this iterator uses.
   * @return A pointer to the region.
   */
  [[nodiscard]] mlir::Region* getRegion() {
    return region != nullptr ? region : q.getParentRegion();
  }

  mlir::Operation* currOp{};
  mlir::Value q;
  mlir::Region* region{};
  bool sentinel{false};
};

static_assert(std::bidirectional_iterator<WireIterator>);
static_assert(std::sentinel_for<std::default_sentinel_t, WireIterator>,
              "std::default_sentinel_t must be a sentinel for WireIterator.");
} // namespace mqt::ir::opt
