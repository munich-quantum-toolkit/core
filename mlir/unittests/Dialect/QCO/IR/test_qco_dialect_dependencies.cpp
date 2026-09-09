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
/// @brief Tests for QCO canonicalization in a minimal dialect context.

#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QCO/QCOUtils.h"

#include <gtest/gtest.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
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

TEST(QCODialectDependencies, CanonicalizesInverseWithoutPreloadingArith) {
  MLIRContext context;
  context.loadDialect<qco::QCODialect, func::FuncDialect>();
  auto moduleOp = parseSourceString<ModuleOp>(R"mlir(
    module {
      func.func @test(%q: !qco.qubit, %theta: f64) -> !qco.qubit {
        %r = qco.inv (%a = %q) {
          %s = qco.rx(%theta) %a : !qco.qubit -> !qco.qubit
          qco.yield %s : !qco.qubit
        } : {!qco.qubit} -> {!qco.qubit}
        return %r : !qco.qubit
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
  auto rotation = returned.getOperand(0).getDefiningOp<qco::RXOp>();
  ASSERT_TRUE(rotation);
  EXPECT_EQ(rotation.getQubitIn(), function.getArgument(0));
  auto negation = rotation.getTheta().getDefiningOp<arith::NegFOp>();
  ASSERT_TRUE(negation);
  EXPECT_EQ(negation.getOperand(), function.getArgument(1));
}

} // namespace
