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
public:
  using iterator_category = std::bidirectional_iterator_tag;
  using difference_type = std::ptrdiff_t;
  using value_type = Operation*;

  explicit WireIterator() = default;
  explicit WireIterator(Value q, Region* region)
      : currOp(q.getDefiningOp()), q(q), region(region) {}

  Operation* operator*() const { return currOp; }

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

  /// --- Iterator-To-Iterator Equality ---

  bool operator==(const WireIterator& other) const {
    return other.q == q && other.currOp == currOp;
  }

  /// --- Iterator-To-Sentinel Equality ---

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
          .Case<UnitaryInterface>([&](UnitaryInterface op) {
            const auto inRng =
                llvm::concat<Value>(op.getInQubits(), op.getPosCtrlInQubits(),
                                    op.getNegCtrlInQubits());
            const auto outRng =
                llvm::concat<Value>(op.getOutQubits(), op.getPosCtrlOutQubits(),
                                    op.getNegCtrlOutQubits());
            for (const auto& [in, out] : llvm::zip_equal(inRng, outRng)) {
              if (q == in) {
                q = out;
                return;
              }
            }

            llvm_unreachable("unknown qubit value in def-use chain");
          })
          .Case<AllocQubitOp>([&](AllocQubitOp op) { q = op.getQubit(); })
          .Case<ResetOp>([&](ResetOp op) { q = op.getOutQubit(); })
          .Case<MeasureOp>([&](MeasureOp op) { q = op.getOutQubit(); })
          .Case<DeallocQubitOp>([&](DeallocQubitOp) { sentinel = true; })
          .Case<scf::ForOp>([&](scf::ForOp op) {
            for (const auto& [in, out] :
                 llvm::zip_equal(op.getInitArgs(), op.getResults())) {
              if (q == in) {
                q = out;
                return;
              }
            }

            llvm_unreachable("unknown qubit value in def-use chain");
          })
          .Case<scf::IfOp>([&](scf::IfOp op) {
            /// Find yielded value by using a recursive WireIterator for the
            /// THEN region.
            WireIterator itThen(q, &op.getThenRegion());
            for (; itThen != std::default_sentinel; ++itThen) {
              if (scf::YieldOp yield = dyn_cast<scf::YieldOp>(*itThen)) {
                for (const auto [yielded, res] :
                     llvm::zip(yield.getResults(), op->getResults())) {
                  if (itThen.q == yielded) {
                    q = res;
                    return;
                  }
                }
              }
            }

            /// Otherwise it must be in the ELSE region.
            WireIterator itElse(q, &op.getElseRegion());
            for (; itElse != std::default_sentinel; ++itElse) {
              if (scf::YieldOp yield = dyn_cast<scf::YieldOp>(*itElse)) {
                for (const auto [yielded, res] :
                     llvm::zip(yield.getResults(), op->getResults())) {
                  if (itElse.q == yielded) {
                    q = res;
                    return;
                  }
                }
              }
            }

            llvm_unreachable("unknown qubit value in def-use chain for if");
          })
          .Case<scf::YieldOp>([&](scf::YieldOp) { sentinel = true; })
          .Default([&](Operation* op) {
            llvm::report_fatal_error("unknown op in def-use chain: " +
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

    currOp = q.getDefiningOp();

    /// Find input from output qubit.
    /// If there is no input qubit, hold.
    TypeSwitch<Operation*>(currOp)
        .Case<UnitaryInterface>([&](UnitaryInterface op) {
          const auto inRng =
              llvm::concat<Value>(op.getInQubits(), op.getPosCtrlInQubits(),
                                  op.getNegCtrlInQubits());
          const auto outRng =
              llvm::concat<Value>(op.getOutQubits(), op.getPosCtrlOutQubits(),
                                  op.getNegCtrlOutQubits());
          for (const auto& [in, out] : llvm::zip_equal(inRng, outRng)) {
            if (q == out) {
              q = in;
              return;
            }
          }

          llvm_unreachable("unknown qubit value in def-use chain");
        })
        .Case<ResetOp>([&](ResetOp op) { q = op.getInQubit(); })
        .Case<MeasureOp>([&](MeasureOp op) { q = op.getInQubit(); })
        .Case<DeallocQubitOp>([&](DeallocQubitOp op) { q = op.getQubit(); })
        .Case<scf::ForOp>([&](scf::ForOp op) {
          q = op.getInitArgs()[cast<OpResult>(q).getResultNumber()];
        })
        .Case<scf::IfOp>([&](scf::IfOp op) {
          Region* thenRegion = &op.getThenRegion();
          Operation* term = thenRegion->front().getTerminator();
          scf::YieldOp yield = cast<scf::YieldOp>(term);
          Value yielded =
              yield.getResults()[cast<OpResult>(q).getResultNumber()];

          assert(yielded != nullptr);

          WireIterator thenIt(yielded, thenRegion);
          while (thenIt.q.getParentRegion() != getRegion()) {
            --thenIt;
          }

          q = thenIt.q;
        })
        .Case<QubitOp>([&](QubitOp) { /* no-op */ })
        .Case<AllocQubitOp>([&](AllocQubitOp) { /* no-op */ })
        .Default([&](Operation* op) {
          llvm::report_fatal_error("unknown op in def-use chain: " +
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

  /**
   * @brief Return the first user of a value in a given region.
   * @param v The value.
   * @param region The targeted region.
   * @return A pointer to the user, or nullptr if non exists.
   */
  [[nodiscard]] static Operation* getUserInRegion(Value v, Region* region) {
    for (Operation* user : v.getUsers()) {
      if (user->getParentRegion() == region) {
        return user;
      }
    }
    return nullptr;
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
