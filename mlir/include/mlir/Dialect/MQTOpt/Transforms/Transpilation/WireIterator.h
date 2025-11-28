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

#include "mlir/Dialect/MQTOpt/Transforms/Transpilation/Common.h"

#include "llvm/ADT/STLExtras.h"

#include <cstddef>
#include <iterator>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/ErrorHandling.h>
#include <mlir/Analysis/SliceAnalysis.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>
#include <stdexcept>

namespace mqt::ir::opt {
using namespace mlir;

/**
 * @brief A non-recursive input_iterator traversing the def-use chain of a qubit
 * wire.
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
  static constexpr WireIterator end() { return {}; }

  using difference_type = std::ptrdiff_t;
  using value_type = Operation*;

  constexpr WireIterator() : q(nullptr), region(nullptr) {}

  explicit WireIterator(Value q, Region* region) : q(q), region(region) {
    setNextOp();
  }

  Operation* operator*() const { return currOp; }

  WireIterator& operator++() {
    setNextQubit();
    setNextOp();
    return *this;
  }

  void operator++(int) { ++*this; }

  bool operator==(const WireIterator& other) const { return other.q == q; }

private:
  void setNextOp() {
    if (q == nullptr) {
      return;
    }

    if (q.use_empty()) {
      q = nullptr;
      currOp = nullptr;
      return;
    }

    currOp =
        getUserInRegion(q, region != nullptr ? region : q.getParentRegion());
    if (currOp == nullptr) {
      /// Must be a branching op:
      currOp = q.getUsers().begin()->getParentOp();
      assert(isa<scf::IfOp>(currOp));
    }
  }

  void setNextQubit() {
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
        .Case<ResetOp>([&](ResetOp op) { q = op.getOutQubit(); })
        .Case<MeasureOp>([&](MeasureOp op) { q = op.getOutQubit(); })
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
          /// Find yielded value by using a recursive WireIterator for the THEN
          /// region.
          WireIterator itThen(q, &op.getThenRegion());
          for (; itThen != WireIterator(); ++itThen) {
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
          for (; itElse != WireIterator(); ++itElse) {
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

          llvm_unreachable("must find yielded value.");
        })
        .Case<scf::YieldOp>([&](scf::YieldOp) {
          /// End of region. Invalidate iterator.
          q = nullptr;
          currOp = nullptr;
        })
        .Default([&](Operation*) {
          throw std::runtime_error("unhandled / invalid op in def-use chain");
        });
  }

  Value q;
  Region* region;
  Operation* currOp{};
};

static_assert(std::input_iterator<WireIterator>);
} // namespace mqt::ir::opt
