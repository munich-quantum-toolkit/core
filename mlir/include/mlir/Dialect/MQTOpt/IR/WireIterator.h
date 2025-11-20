/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
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
using namespace mlir;

/**
 * @brief A non-recursive bidirectional_iterator traversing the def-use chain of
 * a qubit wire.
 *
 * The iterator follows the flow of a qubit through a sequence of quantum
 * operations in a given region. It respects the semantics of the respective
 * quantum operation including control flow constructs (scf::ForOp and
 * scf::IfOp).
 *
 * It does not visit operations within nested regions. These include the loop
 * body of the scf::ForOp and the THEN and ELSE branches of the scf::IfOp. From
 * the iterator's perspective these act like regular gates. As a consequence,
 * an input qubit is mapped to the respective output qubit. For example, finding
 * which result of an scf::IfOp corresponds to a qubit passed into one of its
 * regions.
 */
class WireIterator {
  /// @returns a view of all input qubits.
  [[nodiscard]] static auto getAllInQubits(UnitaryInterface op) {
    return llvm::concat<Value>(op.getInQubits(), op.getPosCtrlInQubits(),
                               op.getNegCtrlInQubits());
  }

  /// @returns a view of all output qubits.
  [[nodiscard]] static auto getAllOutQubits(UnitaryInterface op) {
    return llvm::concat<Value>(op.getOutQubits(), op.getPosCtrlOutQubits(),
                               op.getNegCtrlOutQubits());
  }

  /**
   * @brief Find corresponding output from input value for a unitary (Forward).
   *
   * @note That we don't use the interface method here because
   * it creates temporary std::vectors instead of using views.
   */
  [[nodiscard]] static Value findOutput(UnitaryInterface op, Value in) {
    const auto ins = getAllInQubits(op);
    const auto it = llvm::find(ins, in);
    assert(it != ins.end() && "input qubit not found in operation");
    const auto index = std::distance(ins.begin(), it);
    return *(std::next(getAllOutQubits(op).begin(), index));
  }

  /**
   * @brief Find corresponding input from output value for a unitary (Backward).
   *
   * @note That we don't use the interface method here because
   * it creates temporary std::vectors instead of using views.
   */
  [[nodiscard]] static Value findInput(UnitaryInterface op, Value out) {
    const auto outs = getAllOutQubits(op);
    const auto it = llvm::find(outs, out);
    assert(it != outs.end() && "output qubit not found in operation");
    const auto index = std::distance(outs.begin(), it);
    return *(std::next(getAllInQubits(op).begin(), index));
  }

  /**
   * @brief Find corresponding result from init argument value (Forward).
   */
  [[nodiscard]] static Value findResult(scf::ForOp op, Value initArg) {
    const auto initArgs = op.getInitArgs();
    const auto it = llvm::find(initArgs, initArg);
    assert(it != initArgs.end() && "init arg qubit not found in operation");
    const auto index = std::distance(initArgs.begin(), it);
    return op->getResult(index);
  }

  /**
   * @brief Find corresponding init argument from result value (Backward).
   */
  [[nodiscard]] static Value findInitArg(scf::ForOp op, Value res) {
    return op.getInitArgs()[cast<OpResult>(res).getResultNumber()];
  }

  /**
   * @brief Find corresponding result value from input qubit value (Forward).
   *
   * @details Recursively traverses the IR "downwards" until the respective
   * yield is found. Assumes that each branch takes and returns the same
   * (possibly modified) qubits. Hence, we can just traverse the then-branch.
   */
  [[nodiscard]] static Value findResult(scf::IfOp op, Value q) {
    WireIterator it(q, &op.getThenRegion());

    /// Assumptions:
    ///     First, there must be a yield.
    ///     Second, yield is a sentinel.
    /// Then: Advance until the yield before the sentinel.

    it = std::prev(std::ranges::next(it, std::default_sentinel));
    assert(isa<scf::YieldOp>(*it) && "expected yield op");
    auto yield = cast<scf::YieldOp>(*it);

    /// Get the corresponding result.

    const auto results = yield.getResults();
    const auto yieldIt = llvm::find(results, it.q);
    assert(yieldIt != results.end() && "yielded qubit not found in operation");
    const auto index = std::distance(results.begin(), yieldIt);
    return op->getResult(index);
  }

  /**
   * @brief Find first out-of-region value for result value (Backward).
   *
   * @details Recursively traverses the IR "upwards" until a out-of-region value
   * is found. If the Operation* of the iterator doesn't change the def-use
   * starts in the branch.
   */
  [[nodiscard]] static Value findValue(scf::IfOp op, Value q) {
    auto yield = llvm::cast<scf::YieldOp>(op.thenBlock()->getTerminator());
    Value v = yield.getResults()[cast<OpResult>(q).getResultNumber()];
    assert(v != nullptr && "expected yielded value");

    Operation* prev{};
    WireIterator it(v, &op.getThenRegion());
    while (*it != prev && it.q.getParentRegion() != op->getParentRegion()) {
      --it;
      prev = *it;
    }
    return it.q;
  }

  /**
   * @brief Return the first user of a value in a given region.
   * @param v The value.
   * @param region The targeted region.
   * @return A pointer to the user, or nullptr if non exists.
   */
  [[nodiscard]] static Operation* getUserInRegion(Value v, Region* region) {
    if (v.hasOneUse()) {
      return *(v.getUsers().begin());
    }

    for (Operation* user : v.getUsers()) {
      if (user->getParentRegion() == region) {
        return user;
      }
    }
    return nullptr;
  }

public:
  using iterator_category = std::bidirectional_iterator_tag;
  using difference_type = std::ptrdiff_t;
  using value_type = Operation*;

  explicit WireIterator() = default;
  explicit WireIterator(Value q, Region* region)
      : currOp(q.getDefiningOp()), q(q), region(region) {}

  Operation* operator*() const {
    assert(!sentinel && "Dereferencing sentinel iterator");
    assert(currOp && "Dereferencing null operation");
    return currOp;
  }

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
    return other.q == q && other.currOp == currOp;
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
      TypeSwitch<Operation*>(currOp)
          .Case<UnitaryInterface>(
              [&](UnitaryInterface op) { q = findOutput(op, q); })
          .Case<AllocQubitOp>([&](AllocQubitOp op) { q = op.getQubit(); })
          .Case<ResetOp>([&](ResetOp op) { q = op.getOutQubit(); })
          .Case<MeasureOp>([&](MeasureOp op) { q = op.getOutQubit(); })
          .Case<scf::ForOp>([&](scf::ForOp op) { q = findResult(op, q); })
          .Case<scf::IfOp>([&](scf::IfOp op) { q = findResult(op, q); })
          .Case<DeallocQubitOp, scf::YieldOp>([&](auto) { sentinel = true; })
          .Default([&](Operation* op) {
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
      /// Theoretically this could also be an scf::scf.index_switch, etc.
      assert(isa<scf::IfOp>(currOp));
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

    /// Find input from output qubit.
    /// If there is no input qubit, hold.
    TypeSwitch<Operation*>(currOp)
        .Case<UnitaryInterface>(
            [&](UnitaryInterface op) { q = findInput(op, q); })
        .Case<ResetOp, MeasureOp>([&](auto op) { q = op.getInQubit(); })
        .Case<DeallocQubitOp>([&](DeallocQubitOp op) { q = op.getQubit(); })
        .Case<scf::ForOp>([&](scf::ForOp op) { q = findInitArg(op, q); })
        .Case<scf::IfOp>([&](scf::IfOp op) { q = findValue(op, q); })
        .Case<AllocQubitOp, QubitOp>([&](auto) { /* hold (no-op) */ })
        .Default([&](Operation* op) {
          report_fatal_error("unknown op in def-use chain: " +
                             op->getName().getStringRef());
        });
  }

  /**
   * @brief Return the active region this iterator uses.
   * @return A pointer to the region.
   */
  [[nodiscard]] Region* getRegion() {
    return region != nullptr ? region : q.getParentRegion();
  }

  Operation* currOp{};
  Value q;
  Region* region{};
  bool sentinel{false};
};

static_assert(std::bidirectional_iterator<WireIterator>);
static_assert(std::sentinel_for<std::default_sentinel_t, WireIterator>,
              "std::default_sentinel_t must be a sentinel for WireIterator.");
} // namespace mqt::ir::opt
