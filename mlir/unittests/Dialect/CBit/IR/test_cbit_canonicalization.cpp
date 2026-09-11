/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

/// @file
/// Tests for classical-bit register canonicalization.

#include "mqt/Dialect/CBit/IR/CBitDialect.h"
#include "mqt/Dialect/CBit/IR/CBitOps.h"

#include "gtest/gtest.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/APInt.h"

using namespace mlir;

namespace {

class CBitCanonicalizationTest : public testing::Test {
protected:
  MLIRContext context_;

  void SetUp() override {
    context_.loadDialect<cbit::CBitDialect, arith::ArithDialect,
                         func::FuncDialect>();
  }

  LogicalResult canonicalize(ModuleOp moduleOp) {
    PassManager manager(&context_);
    manager.addPass(createCanonicalizerPass());
    return manager.run(moduleOp);
  }
};

TEST_F(CBitCanonicalizationTest, ForwardsStoredBitAcrossReadOnlySnapshot) {
  auto moduleOp = parseSourceString<ModuleOp>(R"mlir(
    module {
      func.func @test() -> (i1, i2) {
        %c0 = arith.constant 0 : index
        %true = arith.constant true
        %reg = cbit.alloc(#cbit.init<zero>) : !cbit.reg<2>
        cbit.store %true, %reg[%c0] : !cbit.reg<2>
        %snapshot = cbit.read %reg : !cbit.reg<2> -> i2
        %bit = cbit.load %reg[%c0] : !cbit.reg<2>
        return %bit, %snapshot : i1, i2
      }
    }
  )mlir",
                                              &context_);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(verify(*moduleOp)));
  ASSERT_TRUE(succeeded(canonicalize(*moduleOp)));
  ASSERT_TRUE(succeeded(verify(*moduleOp)));

  auto function = moduleOp->lookupSymbol<func::FuncOp>("test");
  auto returned =
      cast<func::ReturnOp>(function.getBody().front().getTerminator());
  APInt value;
  ASSERT_TRUE(matchPattern(returned.getOperand(0), m_ConstantInt(&value)));
  EXPECT_TRUE(value.isOne());
  EXPECT_TRUE(returned.getOperand(1).getDefiningOp<cbit::ReadOp>());
}

TEST_F(CBitCanonicalizationTest, DoesNotForwardAcrossWholeRegisterWrite) {
  auto moduleOp = parseSourceString<ModuleOp>(R"mlir(
    module {
      func.func @test(%value: i2) -> (i1, i2) {
        %c0 = arith.constant 0 : index
        %true = arith.constant true
        %reg = cbit.alloc(#cbit.init<zero>) : !cbit.reg<2>
        cbit.store %true, %reg[%c0] : !cbit.reg<2>
        cbit.write %value, %reg : i2, !cbit.reg<2>
        %snapshot = cbit.read %reg : !cbit.reg<2> -> i2
        %bit = cbit.load %reg[%c0] : !cbit.reg<2>
        return %bit, %snapshot : i1, i2
      }
    }
  )mlir",
                                              &context_);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(verify(*moduleOp)));
  ASSERT_TRUE(succeeded(canonicalize(*moduleOp)));
  ASSERT_TRUE(succeeded(verify(*moduleOp)));

  auto function = moduleOp->lookupSymbol<func::FuncOp>("test");
  auto returned =
      cast<func::ReturnOp>(function.getBody().front().getTerminator());
  EXPECT_TRUE(returned.getOperand(0).getDefiningOp<cbit::LoadOp>());
  EXPECT_TRUE(returned.getOperand(1).getDefiningOp<cbit::ReadOp>());
}

TEST_F(CBitCanonicalizationTest,
       RepeatedLoadsPreserveRegistersIndicesAndInterveningStores) {
  auto moduleOp = parseSourceString<ModuleOp>(R"mlir(
    module {
      func.func @test(%reg: !cbit.reg<2>, %other: !cbit.reg<2>,
                      %index: index, %other_index: index, %value: i1)
          -> (i1, i1, i1, i1, i1) {
        %first = cbit.load %reg[%index] : !cbit.reg<2>
        %repeated = cbit.load %reg[%index] : !cbit.reg<2>
        %other_reg = cbit.load %other[%index] : !cbit.reg<2>
        %other_bit = cbit.load %reg[%other_index] : !cbit.reg<2>
        cbit.store %value, %reg[%index] : !cbit.reg<2>
        %written = cbit.load %reg[%index] : !cbit.reg<2>
        return %first, %repeated, %other_reg, %other_bit, %written
            : i1, i1, i1, i1, i1
      }
    }
  )mlir",
                                              &context_);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(verify(*moduleOp)));
  ASSERT_TRUE(succeeded(canonicalize(*moduleOp)));
  ASSERT_TRUE(succeeded(verify(*moduleOp)));

  auto function = moduleOp->lookupSymbol<func::FuncOp>("test");
  auto returned =
      cast<func::ReturnOp>(function.getBody().front().getTerminator());
  EXPECT_EQ(returned.getOperand(0), returned.getOperand(1));
  auto first = returned.getOperand(0).getDefiningOp<cbit::LoadOp>();
  auto otherReg = returned.getOperand(2).getDefiningOp<cbit::LoadOp>();
  auto otherBit = returned.getOperand(3).getDefiningOp<cbit::LoadOp>();
  ASSERT_TRUE(first);
  ASSERT_TRUE(otherReg);
  ASSERT_TRUE(otherBit);
  EXPECT_EQ(first.getReg(), function.getArgument(0));
  EXPECT_EQ(first.getIndex(), function.getArgument(2));
  EXPECT_EQ(otherReg.getReg(), function.getArgument(1));
  EXPECT_EQ(otherReg.getIndex(), function.getArgument(2));
  EXPECT_EQ(otherBit.getReg(), function.getArgument(0));
  EXPECT_EQ(otherBit.getIndex(), function.getArgument(3));
  EXPECT_EQ(returned.getOperand(4), function.getArgument(4));
}

} // namespace
