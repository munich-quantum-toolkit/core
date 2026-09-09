/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QCO/QCOUtils.h"
#include "mlir/Dialect/QTensor/IR/QTensorDialect.h"
#include "mlir/Dialect/QTensor/IR/QTensorOps.h"

#include <gtest/gtest.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Rewrite/FrozenRewritePatternSet.h>
#include <mlir/Rewrite/PatternApplicator.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

#include <utility>

using namespace mlir;
using namespace mlir::qtensor;

namespace {

class QTensorPairCanonicalizationTest : public testing::Test {
protected:
  MLIRContext context_;

  void SetUp() override {
    context_.loadDialect<QTensorDialect, func::FuncDialect>();
  }

  OwningOpRef<ModuleOp> makePair() {
    return parseSourceString<ModuleOp>(R"mlir(
      module {
        func.func @main(%tensor: tensor<?x!qco.qubit>,
                        %extract_index: index, %insert_index: index)
            -> tensor<?x!qco.qubit> {
          %remaining, %qubit = qtensor.extract %tensor[%extract_index]
              : tensor<?x!qco.qubit>
          %result = qtensor.insert %qubit into %remaining[%insert_index]
              : tensor<?x!qco.qubit>
          return %result : tensor<?x!qco.qubit>
        }
      }
    )mlir",
                                       &context_);
  }

  LogicalResult applyInsertPattern(InsertOp insert) {
    RewritePatternSet patterns(&context_);
    InsertOp::getCanonicalizationPatterns(patterns, &context_);
    FrozenRewritePatternSet frozen(std::move(patterns));
    PatternApplicator applicator(frozen);
    applicator.applyDefaultCostModel();
    PatternRewriter rewriter(&context_);
    rewriter.setInsertionPoint(insert);
    return applicator.matchAndRewrite(insert, rewriter);
  }
};

TEST_F(QTensorPairCanonicalizationTest,
       SameDynamicIndexCancellationPreservesLinearityBeforeDCE) {
  auto moduleOp = makePair();
  ASSERT_TRUE(moduleOp);
  auto function = moduleOp->lookupSymbol<func::FuncOp>("main");
  auto insert = *function.getOps<InsertOp>().begin();
  insert.getIndexMutable().assign(function.getArgument(1));
  ASSERT_TRUE(succeeded(verify(*moduleOp)));
  ASSERT_TRUE(succeeded(qco::verifyLinearity(*moduleOp)));

  ASSERT_TRUE(succeeded(applyInsertPattern(insert)));
  // Both operations must be gone before a later rewrite can observe the IR.
  ASSERT_TRUE(succeeded(verify(*moduleOp)));
  ASSERT_TRUE(succeeded(qco::verifyLinearity(*moduleOp)));
  EXPECT_TRUE(function.getOps<ExtractOp>().empty());
  EXPECT_TRUE(function.getOps<InsertOp>().empty());
  auto returned = cast<func::ReturnOp>(function.getBody().front().back());
  EXPECT_EQ(returned.getOperand(0), function.getArgument(0));
}

TEST_F(QTensorPairCanonicalizationTest,
       DifferentDynamicIndicesKeepExtractAndInsert) {
  auto moduleOp = makePair();
  ASSERT_TRUE(moduleOp);
  auto function = moduleOp->lookupSymbol<func::FuncOp>("main");
  auto extract = *function.getOps<ExtractOp>().begin();
  auto insert = *function.getOps<InsertOp>().begin();
  ASSERT_TRUE(succeeded(verify(*moduleOp)));
  ASSERT_TRUE(succeeded(qco::verifyLinearity(*moduleOp)));

  EXPECT_TRUE(failed(applyInsertPattern(insert)));
  ASSERT_TRUE(succeeded(verify(*moduleOp)));
  ASSERT_TRUE(succeeded(qco::verifyLinearity(*moduleOp)));
  EXPECT_EQ(insert.getScalar(), extract.getResult());
  EXPECT_EQ(insert.getDest(), extract.getOutTensor());
  EXPECT_EQ(extract.getIndex(), function.getArgument(1));
  EXPECT_EQ(insert.getIndex(), function.getArgument(2));
  auto returned = cast<func::ReturnOp>(function.getBody().front().back());
  EXPECT_EQ(returned.getOperand(0), insert.getResult());
}

} // namespace
