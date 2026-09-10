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

} // namespace
