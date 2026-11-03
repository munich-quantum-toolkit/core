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
/// Tests for canonicalization of QCO control flow.

#include "mqt/Dialect/QCO/IR/QCODialect.h"
#include "mqt/Dialect/QCO/IR/QCOOps.h"
#include "mqt/Dialect/QCO/QCOUtils.h"
#include "mqt/Dialect/QCO/Utils/LinearityUtils.h"

#include "gtest/gtest.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/STLExtras.h"

#include <utility>

using namespace mlir;

namespace {

TEST(QCOControlFlowCanonicalization, RemovesIdentityBranches) {
  MLIRContext context;
  context.loadDialect<qco::QCODialect, func::FuncDialect>();
  auto moduleOp = parseSourceString<ModuleOp>(R"mlir(
    module {
      func.func @test(%condition: i1, %q0: !qco.qubit,
                      %fixed: tensor<2x!qco.qubit>,
                      %dynamic: tensor<?x!qco.qubit>, %q1: !qco.qubit)
          -> (!qco.qubit, tensor<2x!qco.qubit>,
              tensor<?x!qco.qubit>, !qco.qubit) {
        %out:4 = qco.if %condition
            args(%a = %q0, %b = %fixed, %c = %dynamic, %d = %q1)
            -> (!qco.qubit, tensor<2x!qco.qubit>,
                tensor<?x!qco.qubit>, !qco.qubit) {
          qco.yield %a, %b, %c, %d : !qco.qubit, tensor<2x!qco.qubit>,
                                   tensor<?x!qco.qubit>, !qco.qubit
        } else args(%a = %q0, %b = %fixed, %c = %dynamic, %d = %q1) {
          qco.yield %a, %b, %c, %d : !qco.qubit, tensor<2x!qco.qubit>,
                                   tensor<?x!qco.qubit>, !qco.qubit
        }
        return %out#0, %out#1, %out#2, %out#3
            : !qco.qubit, tensor<2x!qco.qubit>,
              tensor<?x!qco.qubit>, !qco.qubit
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
  EXPECT_TRUE(function.getOps<qco::IfOp>().empty());
  auto returned =
      cast<func::ReturnOp>(function.getBody().front().getTerminator());
  for (auto [actual, input] : llvm::zip_equal(
           returned.getOperands(), function.getArguments().drop_front())) {
    EXPECT_EQ(actual, input);
  }
}

TEST(QCOControlFlowCanonicalization, PreservesBranchOperations) {
  for (const bool thenBranch : {false, true}) {
    for (const bool quantumOperation : {false, true}) {
      SCOPED_TRACE(thenBranch);
      SCOPED_TRACE(quantumOperation);
      MLIRContext context;
      context.loadDialect<qco::QCODialect, func::FuncDialect>();
      auto moduleOp = parseSourceString<ModuleOp>(R"mlir(
        module {
          func.func private @effect()
          func.func @test(%condition: i1, %q: !qco.qubit) -> !qco.qubit {
            %out = qco.if %condition args(%a = %q) -> !qco.qubit {
              qco.yield %a : !qco.qubit
            } else args(%a = %q) {
              qco.yield %a : !qco.qubit
            }
            return %out : !qco.qubit
          }
        }
      )mlir",
                                                  &context);
      ASSERT_TRUE(moduleOp);
      auto function = moduleOp->lookupSymbol<func::FuncOp>("test");
      auto conditional = *function.getOps<qco::IfOp>().begin();
      auto yield =
          thenBranch ? conditional.thenYield() : conditional.elseYield();
      OpBuilder builder(yield);
      if (quantumOperation) {
        auto changed = qco::HOp::create(builder, yield.getLoc(),
                                        yield.getTargets().front());
        yield.setOperand(0, changed);
      } else {
        func::CallOp::create(builder, yield.getLoc(), "effect", TypeRange{},
                             ValueRange{});
      }
      ASSERT_TRUE(succeeded(verify(*moduleOp)));
      ASSERT_TRUE(succeeded(qco::verifyLinearity(*moduleOp)));
      PassManager manager(&context);
      manager.addPass(createCanonicalizerPass());
      ASSERT_TRUE(succeeded(manager.run(*moduleOp)));
      EXPECT_TRUE(succeeded(verify(*moduleOp)));
      EXPECT_TRUE(succeeded(qco::verifyLinearity(*moduleOp)));
      auto returned =
          cast<func::ReturnOp>(function.getBody().front().getTerminator());
      EXPECT_TRUE(returned.getOperand(0).getDefiningOp<qco::IfOp>());
    }
  }
}

TEST(QCOControlFlowCanonicalization, RemovesForwardedIdleLoopQubit) {
  MLIRContext context;
  context.loadDialect<qco::QCODialect, arith::ArithDialect, func::FuncDialect,
                      scf::SCFDialect>();
  auto moduleOp = parseSourceString<ModuleOp>(R"mlir(
    module {
      func.func @test(%count: index) -> i1 {
        %zero = arith.constant 0 : index
        %one = arith.constant 1 : index
        %active = qco.static 0 : !qco.qubit
        %idle = qco.static 1 : !qco.qubit
        %out:2 = scf.for %i = %zero to %count step %one
            iter_args(%a = %active, %b = %idle)
            -> (!qco.qubit, !qco.qubit) {
          %next = qco.x %a : !qco.qubit
          scf.yield %next, %b : !qco.qubit, !qco.qubit
        }
        qco.sink %out#1 : !qco.qubit
        %done, %result = qco.measure %out#0 : !qco.qubit
        qco.sink %done : !qco.qubit
        return %result : i1
      }
    }
  )mlir",
                                              &context);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(verify(*moduleOp)));
  ASSERT_TRUE(succeeded(qco::verifyLinearity(*moduleOp)));
  auto function = moduleOp->lookupSymbol<func::FuncOp>("test");
  auto originalLoop = *function.getOps<scf::ForOp>().begin();
  auto idleResult = originalLoop.getResult(1);
  Operation* sink = *idleResult.getUsers().begin();
  // Exercise the transient state after SCF forwards an invariant result,
  // before it removes the corresponding loop argument.
  idleResult.replaceAllUsesWith(originalLoop.getInitArgs()[1]);
  RewritePatternSet patterns(&context);
  qco::SinkOp::getCanonicalizationPatterns(patterns, &context);
  ASSERT_TRUE(succeeded(applyOpPatternsGreedily(sink, std::move(patterns))));

  PassManager manager(&context);
  manager.addPass(createCanonicalizerPass());
  ASSERT_TRUE(succeeded(manager.run(*moduleOp)));
  ASSERT_TRUE(succeeded(verify(*moduleOp)));
  EXPECT_TRUE(succeeded(qco::verifyLinearity(*moduleOp)));
  auto loops = function.getOps<scf::ForOp>();
  ASSERT_FALSE(loops.empty());
  auto loop = *loops.begin();
  EXPECT_EQ(loop.getRegionIterArgs().size(), 1U);
  moduleOp->walk([](qco::StaticOp qubit) { EXPECT_EQ(qubit.getIndex(), 0); });
}

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
