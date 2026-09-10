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
/// Tests for canonicalization of classical QCO conditional results.

#include "mqt/Dialect/QCO/IR/QCODialect.h"
#include "mqt/Dialect/QCO/IR/QCOOps.h"
#include "mqt/Dialect/QCO/QCOUtils.h"

#include "gtest/gtest.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

namespace {

TEST(QCOControlFlowCanonicalization,
     SharesEarliestClassicalResultAndPreservesLinearSuffix) {
  MLIRContext context;
  context.loadDialect<qco::QCODialect, func::FuncDialect>();
  auto moduleOp = parseSourceString<ModuleOp>(R"mlir(
    module {
      func.func @test(%condition: i1, %a: i32, %b: i32, %same: i32,
                      %q: !qco.qubit) -> (i32, i32, i32, i32, !qco.qubit) {
        %r0, %r1, %r2, %r3, %unused, %q1 = qco.if %condition
            args(%t = %q) -> (i32, i32, i32, i32, i32, !qco.qubit) {
          qco.yield %a, %b, %a, %same, %a, %t
              : i32, i32, i32, i32, i32, !qco.qubit
        } else args(%t = %q) {
          qco.yield %b, %a, %b, %same, %a, %t
              : i32, i32, i32, i32, i32, !qco.qubit
        }
        return %r2, %r1, %r2, %r3, %q1 : i32, i32, i32, i32, !qco.qubit
      }
    }
  )mlir",
                                              &context);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(verify(*moduleOp)));
  ASSERT_TRUE(succeeded(qco::verifyLinearity(*moduleOp)));
  PassManager manager(&context);
  manager.addPass(createCanonicalizerPass());
  ASSERT_TRUE(succeeded(manager.run(*moduleOp)));
  ASSERT_TRUE(succeeded(verify(*moduleOp)));
  ASSERT_TRUE(succeeded(qco::verifyLinearity(*moduleOp)));

  auto function = moduleOp->lookupSymbol<func::FuncOp>("test");
  auto returned =
      cast<func::ReturnOp>(function.getBody().front().getTerminator());
  auto conditional = returned.getOperand(4).getDefiningOp<qco::IfOp>();
  ASSERT_TRUE(conditional);
  auto classicalResults = conditional.getClassicalResults();
  // Keep the first representative to make the surviving result order stable.
  ASSERT_EQ(classicalResults.size(), 2U);
  EXPECT_EQ(returned.getOperand(0), classicalResults[0]);
  EXPECT_EQ(returned.getOperand(1), classicalResults[1]);
  EXPECT_EQ(returned.getOperand(2), classicalResults[0]);
  EXPECT_EQ(returned.getOperand(3), function.getArgument(3));
  auto thenValues = conditional.thenYield().getTargets();
  auto elseValues = conditional.elseYield().getTargets();
  ASSERT_EQ(thenValues.size(), 3U);
  ASSERT_EQ(elseValues.size(), 3U);
  EXPECT_EQ(thenValues[0], function.getArgument(1));
  EXPECT_EQ(thenValues[1], function.getArgument(2));
  EXPECT_EQ(elseValues[0], function.getArgument(2));
  EXPECT_EQ(elseValues[1], function.getArgument(1));
  EXPECT_EQ(thenValues[2], conditional.thenBlock()->getArgument(0));
  EXPECT_EQ(elseValues[2], conditional.elseBlock()->getArgument(0));
}

} // namespace
