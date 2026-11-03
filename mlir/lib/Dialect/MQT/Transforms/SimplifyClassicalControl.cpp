/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/Dialect/MQT/Transforms/Passes.h"
#include "mqt/Dialect/QCO/IR/QCOOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Value.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopedHashTable.h"

#include <cstdint>

namespace mlir::mqt {

#define GEN_PASS_DEF_SIMPLIFYCLASSICALCONTROL
#include "mqt/Dialect/MQT/Transforms/Passes.h.inc"

namespace {

struct Condition {
  bool truth;
  Block* block;
  Value constant;
};

/// Propagate each QCO condition within the region where its value is known.
class ClassicalControlSimplifier {
  IRRewriter rewriter_;
  llvm::ScopedHashTable<Value, Condition*> conditions_;

public:
  explicit ClassicalControlSimplifier(MLIRContext* context)
      : rewriter_(context) {}

  void run(Region& region) {
    for (Block& block : region) {
      for (Operation& operation : llvm::make_early_inc_range(block)) {
        for (OpOperand& operand : operation.getOpOperands()) {
          if (auto* condition = conditions_.lookup(operand.get())) {
            if (!condition->constant) {
              rewriter_.setInsertionPointToStart(condition->block);
              condition->constant = arith::ConstantIntOp::create(
                  rewriter_, operation.getLoc(),
                  static_cast<int64_t>(condition->truth), 1);
            }
            rewriter_.modifyOpInPlace(
                &operation, [&] { operand.set(condition->constant); });
          }
        }
        for (auto [index, nested] : llvm::enumerate(operation.getRegions())) {
          llvm::ScopedHashTableScope<Value, Condition*> scope(conditions_);
          Condition condition{
              .truth = index == 0,
              .block = nested.empty() ? nullptr : &nested.front(),
              .constant = {},
          };
          if (auto branch = dyn_cast<qco::IfOp>(operation);
              branch && !nested.empty()) {
            conditions_.insert(branch.getCondition(), &condition);
          }
          run(nested);
        }
      }
    }
  }
};

struct SimplifyClassicalControl final
    : impl::SimplifyClassicalControlBase<SimplifyClassicalControl> {
  using SimplifyClassicalControlBase::SimplifyClassicalControlBase;

protected:
  void runOnOperation() override { simplifyClassicalControl(getOperation()); }
};

} // namespace

void simplifyClassicalControl(Operation* operation) {
  ClassicalControlSimplifier simplifier(operation->getContext());
  for (Region& region : operation->getRegions()) {
    simplifier.run(region);
  }
}

} // namespace mlir::mqt
