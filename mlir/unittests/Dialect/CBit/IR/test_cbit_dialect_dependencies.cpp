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
/// @brief Tests for CBit canonicalization in a minimal dialect context.

#include "mlir/Dialect/CBit/IR/CBitDialect.h"

#include <gtest/gtest.h>
#include <llvm/ADT/APInt.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Matchers.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/Passes.h>

using namespace mlir;

namespace {

TEST(CBitDialectDependencies, MaterializesZeroWithoutPreloadingArith) {
  MLIRContext context;
  context.loadDialect<cbit::CBitDialect, func::FuncDialect>();
  auto moduleOp = parseSourceString<ModuleOp>(R"mlir(
    module {
      func.func @test(%index: index) -> i1 {
        %reg = cbit.alloc(#cbit.init<zero>) : !cbit.reg<1>
        %bit = cbit.load %reg[%index] : !cbit.reg<1>
        return %bit : i1
      }
    }
  )mlir",
                                              &context);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(verify(*moduleOp)));
  PassManager manager(&context);
  manager.addPass(createCanonicalizerPass());
  ASSERT_TRUE(succeeded(manager.run(*moduleOp)));
  ASSERT_TRUE(succeeded(verify(*moduleOp)));

  auto function = moduleOp->lookupSymbol<func::FuncOp>("test");
  auto returned =
      cast<func::ReturnOp>(function.getBody().front().getTerminator());
  APInt value;
  ASSERT_TRUE(matchPattern(returned.getOperand(0), m_ConstantInt(&value)));
  EXPECT_TRUE(value.isZero());
}

} // namespace
