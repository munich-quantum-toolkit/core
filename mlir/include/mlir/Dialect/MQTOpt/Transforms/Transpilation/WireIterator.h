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
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>

namespace mqt::ir::opt {
using namespace mlir;

class WireIterator {
public:
  using difference_type = std::ptrdiff_t;
  using value_type = Operation*;

  explicit WireIterator(Value q = nullptr) : q(q) { setNextOp(); }

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

    currOp = getUserInRegion(q, q.getParentRegion());
    if (currOp == nullptr) {
      /// Must be a branching op:
      currOp = q.getUsers().begin()->getParentOp();
      assert(isa<scf::IfOp>(currOp));
    }
  }

  void setNextQubit() {
    TypeSwitch<Operation*>(currOp)
        /// MQT
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
        /// SCF
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
        .Case<scf::YieldOp>([&](scf::YieldOp) {
          /// End of region. Invalidate iterator.
          q = nullptr;
          currOp = nullptr;
        })
        .Default([&]([[maybe_unused]] Operation* op) {
          llvm_unreachable("unknown operation in def-use chain");
        });
  }

  Value q;
  Operation* currOp{};
};

static_assert(std::input_iterator<WireIterator>);
} // namespace mqt::ir::opt
