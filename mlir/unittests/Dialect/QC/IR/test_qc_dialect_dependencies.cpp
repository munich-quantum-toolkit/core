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
/// @brief Tests for QC canonicalization in a minimal dialect context.

#include "mlir/Dialect/QC/IR/QCDialect.h"
#include "mlir/Dialect/QC/IR/QCOps.h"

#include <gtest/gtest.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/Passes.h>

using namespace mlir;

namespace {

TEST(QCDialectDependencies, CanonicalizesInverseWithoutPreloadingArith) {
  MLIRContext context;
  context.loadDialect<qc::QCDialect, func::FuncDialect>();
  auto moduleOp = parseSourceString<ModuleOp>(R"mlir(
    module {
      func.func @test(%q: !qc.qubit, %theta: f64) {
        qc.inv (%a = %q) {
          qc.rx(%theta) %a : !qc.qubit
          qc.yield
        } : !qc.qubit
        return
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
  qc::RXOp rotation;
  function.walk([&](qc::RXOp op) { rotation = op; });
  ASSERT_TRUE(rotation);
  EXPECT_EQ(rotation.getTarget(0), function.getArgument(0));
  auto negation = rotation.getTheta().getDefiningOp<arith::NegFOp>();
  ASSERT_TRUE(negation);
  EXPECT_EQ(negation.getOperand(), function.getArgument(1));
}

} // namespace
