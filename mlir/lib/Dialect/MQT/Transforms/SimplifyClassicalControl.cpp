/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/Dialect/CBit/IR/CBitAttributes.h"
#include "mqt/Dialect/CBit/IR/CBitOps.h"
#include "mqt/Dialect/MQT/Transforms/Passes.h"
#include "mqt/Dialect/QCO/IR/QCOOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Value.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopedHashTable.h"

#include <cstdint>

namespace mlir::mqt {

#define GEN_PASS_DEF_SIMPLIFYCLASSICALCONTROL
#include "mqt/Dialect/MQT/Transforms/Passes.h.inc"

namespace {

struct RegisterValues {
  DenseMap<int64_t, Value> constants;
  DenseMap<Value, Value> dynamic;
  bool zeroInitialized = false;
};

struct Condition {
  bool truth;
  Block* block;
  Value constant;
};

/// Each operand and operation is visited once. Memory facts stay block-local;
/// branch facts are scoped to the region in which the condition is known.
class ClassicalSimplifier {
  IRRewriter rewriter;
  llvm::ScopedHashTable<Value, Condition*> conditions;

public:
  explicit ClassicalSimplifier(MLIRContext* context) : rewriter(context) {}

  void run(Region& region) {
    for (Block& block : region) {
      DenseMap<Value, RegisterValues> registers;
      for (Operation& operation : llvm::make_early_inc_range(block)) {
        for (OpOperand& operand : operation.getOpOperands()) {
          if (auto* condition = conditions.lookup(operand.get())) {
            if (!condition->constant) {
              rewriter.setInsertionPointToStart(condition->block);
              condition->constant = arith::ConstantIntOp::create(
                  rewriter, operation.getLoc(),
                  static_cast<int64_t>(condition->truth), 1);
            }
            rewriter.modifyOpInPlace(&operation,
                                     [&] { operand.set(condition->constant); });
          }
        }
        if (auto alloc = dyn_cast<cbit::AllocOp>(operation)) {
          registers[alloc.getResult()].zeroInitialized =
              alloc.getInitialization() == cbit::Initialization::Zero;
          continue;
        }
        if (auto store = dyn_cast<cbit::StoreOp>(operation)) {
          auto& values = registers[store.getReg()];
          values.dynamic = decltype(values.dynamic){};
          if (const auto index = getConstantIntValue(store.getIndex())) {
            values.constants[*index] = store.getValue();
          } else {
            values.constants = decltype(values.constants){};
            values.zeroInitialized = false;
            values.dynamic[store.getIndex()] = store.getValue();
          }
          continue;
        }
        if (auto load = dyn_cast<cbit::LoadOp>(operation)) {
          auto& values = registers[load.getReg()];
          const auto index = getConstantIntValue(load.getIndex());
          Value known = index ? values.constants.lookup(*index)
                              : values.dynamic.lookup(load.getIndex());
          rewriter.setInsertionPoint(load);
          if (!known && values.zeroInitialized &&
              (index || values.constants.empty())) {
            known = arith::ConstantIntOp::create(rewriter, load.getLoc(), 0, 1);
          }
          if (index) {
            values.constants[*index] = known ? known : load.getResult();
          } else {
            values.dynamic[load.getIndex()] = known ? known : load.getResult();
          }
          if (known) {
            rewriter.replaceOp(load, known);
          }
          continue;
        }
        if (isa<cbit::ReadOp>(operation)) {
          continue;
        }
        if (operation.getNumRegions() != 0) {
          registers = decltype(registers){};
        } else {
          for (Value operand : operation.getOperands()) {
            if (isa<cbit::RegisterType>(operand.getType())) {
              registers.erase(operand);
            }
          }
        }
        for (auto [index, nested] : llvm::enumerate(operation.getRegions())) {
          llvm::ScopedHashTableScope<Value, Condition*> scope(conditions);
          Condition condition{
              .truth = index == 0,
              .block = nested.empty() ? nullptr : &nested.front(),
              .constant = {},
          };
          if (auto branch = dyn_cast<qco::IfOp>(operation);
              branch && !nested.empty()) {
            conditions.insert(branch.getCondition(), &condition);
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

} /* namespace */

void simplifyClassicalControl(Operation* operation) {
  ClassicalSimplifier simplifier(operation->getContext());
  for (Region& region : operation->getRegions()) {
    simplifier.run(region);
  }
}

} /* namespace mlir::mqt */
