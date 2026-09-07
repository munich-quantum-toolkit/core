/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/CBit/IR/CBitDialect.h"
#include "mlir/Dialect/CBit/IR/CBitOps.h"
#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QCO/Utils/Sorting.h"

#include <gtest/gtest.h>
#include <llvm/ADT/STLExtras.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Support/LLVM.h>

using namespace mlir;

namespace {

class TopologicalSortingTest : public testing::Test {
protected:
  MLIRContext context;

  void SetUp() override {
    context.loadDialect<arith::ArithDialect, cbit::CBitDialect,
                        func::FuncDialect, qco::QCODialect, scf::SCFDialect>();
  }

  OwningOpRef<ModuleOp> parse(StringRef source) {
    return parseSourceString<ModuleOp>(source, &context);
  }

  void sort(ModuleOp moduleOp) {
    auto function = *moduleOp.getOps<func::FuncOp>().begin();
    IRRewriter rewriter(&context);
    qco::reorderTopologically(function.getBody().front(), rewriter);
  }
};

} // namespace

TEST_F(TopologicalSortingTest, RetainsSSAForEffectBearingQubits) {
  auto moduleOp = parse(R"mlir(
    func.func @test(%q: !qco.qubit, %power: f64) -> !qco.qubit {
      %a = qco.x %q : !qco.qubit -> !qco.qubit
      %b = qco.h %a : !qco.qubit -> !qco.qubit
      %r = qco.pow(%power) (%t = %b) {
        %u = qco.s %t : !qco.qubit -> !qco.qubit
        qco.yield %u : !qco.qubit
      } : {!qco.qubit} -> {!qco.qubit}
      return %r : !qco.qubit
    }
  )mlir");
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(verify(*moduleOp)));
  sort(*moduleOp);
  EXPECT_TRUE(succeeded(verify(*moduleOp)));
}

TEST_F(TopologicalSortingTest, RepairsDominanceWithoutReplacingBlockArguments) {
  auto moduleOp = parse(R"mlir(
    func.func @test(%x: i64) -> i64 {
      %a = arith.addi %x, %x : i64
      %b = arith.addi %a, %a : i64
      return %b : i64
    }
  )mlir");
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(verify(*moduleOp)));
  auto function = *moduleOp->getOps<func::FuncOp>().begin();
  auto* block = &function.getBody().front();
  auto argument = block->getArgument(0);
  auto adds = llvm::to_vector(block->getOps<arith::AddIOp>());
  adds[1]->moveBefore(adds[0]);
  sort(*moduleOp);
  EXPECT_TRUE(succeeded(verify(*moduleOp)));
  EXPECT_EQ(&function.getBody().front(), block);
  EXPECT_EQ(function.getArgument(0), argument);
  EXPECT_TRUE(adds[0]->isBeforeInBlock(adds[1]));
}

TEST_F(TopologicalSortingTest, PreservesRepeatedNestedEffectsAndCaptures) {
  auto moduleOp = parse(R"mlir(
    func.func @test(%x: i1) -> i1 {
      %reg = cbit.alloc(#cbit.init<zero>) : !cbit.reg<1>
      %condition = arith.xori %x, %x : i1
      scf.if %condition {
        %before = cbit.read %reg : !cbit.reg<1> -> i1
        %changed = arith.xori %before, %x : i1
        cbit.write %changed, %reg : i1, !cbit.reg<1>
        %again = cbit.read %reg : !cbit.reg<1> -> i1
        %next = arith.xori %again, %x : i1
        cbit.write %next, %reg : i1, !cbit.reg<1>
      }
      %result = cbit.read %reg : !cbit.reg<1> -> i1
      return %result : i1
    }
  )mlir");
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(verify(*moduleOp)));
  sort(*moduleOp);
  ASSERT_TRUE(succeeded(verify(*moduleOp)));
  auto function = *moduleOp->getOps<func::FuncOp>().begin();
  auto alloc = *function.getOps<cbit::AllocOp>().begin();
  auto conditional = *function.getOps<scf::IfOp>().begin();
  auto read = *function.getOps<cbit::ReadOp>().begin();
  EXPECT_TRUE(alloc->isBeforeInBlock(conditional));
  EXPECT_TRUE(conditional->isBeforeInBlock(read));
}

TEST_F(TopologicalSortingTest, PreservesReadyOperationDiscoveryOrder) {
  auto moduleOp = parse(R"mlir(
    func.func @test(%x: i64) -> i64 {
      %a = arith.addi %x, %x : i64
      %b = arith.addi %a, %a : i64
      %c = arith.muli %x, %x : i64
      %d = arith.addi %b, %c : i64
      return %d : i64
    }
  )mlir");
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(verify(*moduleOp)));
  auto function = *moduleOp->getOps<func::FuncOp>().begin();
  auto adds = llvm::to_vector(function.getOps<arith::AddIOp>());
  auto mul = *function.getOps<arith::MulIOp>().begin();
  sort(*moduleOp);
  ASSERT_TRUE(succeeded(verify(*moduleOp)));
  EXPECT_TRUE(adds[0]->isBeforeInBlock(mul));
  EXPECT_TRUE(mul->isBeforeInBlock(adds[1]));
  EXPECT_TRUE(adds[1]->isBeforeInBlock(adds[2]));
}

TEST_F(TopologicalSortingTest, PreservesRepeatedStoresToSameRegisterElement) {
  auto moduleOp = parse(R"mlir(
    func.func @test(%q0: !qco.qubit, %q1: !qco.qubit)
        -> (!qco.qubit, !qco.qubit) {
      %index = arith.constant 0 : index
      %reg = cbit.alloc(#cbit.init<zero>) : !cbit.reg<1>
      %flipped = qco.x %q0 : !qco.qubit -> !qco.qubit
      %out0, %bit0 = qco.measure %flipped : !qco.qubit
      cbit.store %bit0, %reg[%index] : !cbit.reg<1>
      %out1, %bit1 = qco.measure %q1 : !qco.qubit
      cbit.store %bit1, %reg[%index] : !cbit.reg<1>
      return %out0, %out1 : !qco.qubit, !qco.qubit
    }
  )mlir");
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(verify(*moduleOp)));
  auto function = *moduleOp->getOps<func::FuncOp>().begin();
  auto flipped = *function.getOps<qco::XOp>().begin();
  auto stores = llvm::to_vector(function.getOps<cbit::StoreOp>());
  ASSERT_EQ(stores.size(), 2U);
  flipped->moveAfter(stores.back());

  sort(*moduleOp);

  ASSERT_TRUE(succeeded(verify(*moduleOp)));
  EXPECT_TRUE(stores.front()->isBeforeInBlock(stores.back()));
}

TEST_F(TopologicalSortingTest,
       KeepsRegisterWriteBeforeIndexedLoadDuringRepair) {
  auto moduleOp = parse(R"mlir(
    func.func @test() -> i1 {
      %index = arith.constant 0 : index
      %value = arith.constant 0 : i2
      %reg = cbit.alloc(#cbit.init<zero>) : !cbit.reg<2>
      cbit.write %value, %reg : i2, !cbit.reg<2>
      %loaded = cbit.load %reg[%index] : !cbit.reg<2>
      return %loaded : i1
    }
  )mlir");
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(verify(*moduleOp)));
  auto function = *moduleOp->getOps<func::FuncOp>().begin();
  auto write = *function.getOps<cbit::WriteOp>().begin();
  auto load = *function.getOps<cbit::LoadOp>().begin();
  write.getValue().getDefiningOp()->moveAfter(load);

  sort(*moduleOp);

  ASSERT_TRUE(succeeded(verify(*moduleOp)));
  EXPECT_TRUE(write->isBeforeInBlock(load));
}
