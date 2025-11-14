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
 * @brief A non-recursive input_iterator traversing the def-use chain of a
 * single qubit (on a wire).
 *
 * It does not visit nested regions (nested ops).
 */
class WireIterator {
public:
  using difference_type = std::ptrdiff_t;
  using value_type = Operation*;

  explicit WireIterator(Value q = nullptr, Region* region = nullptr)
      : q(q), region(region) {
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
          for (const auto& [in, out] :
               llvm::zip_equal(op.getAllInQubits(), op.getAllOutQubits())) {
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
