/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/Dialect/QCO/IR/QCODialect.h"
#include "mqt/Dialect/QCO/IR/QCOOps.h"
#include "mqt/Dialect/QCO/QCOUtils.h"
#include "mqt/Dialect/QTensor/IR/QTensorDialect.h"
#include "mqt/Dialect/QTensor/IR/QTensorOps.h"

#include "gtest/gtest.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Rewrite/PatternApplicator.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/SmallVector.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>

using namespace mlir;
using namespace mlir::qtensor;

namespace {

class QTensorPairCanonicalizationTest : public testing::Test {
protected:
  MLIRContext context_;

  void SetUp() override {
    context_.loadDialect<QTensorDialect, qco::QCODialect, arith::ArithDialect,
                         func::FuncDialect>();
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

  OwningOpRef<ModuleOp> makeChain() {
    return parseSourceString<ModuleOp>(R"mlir(
      module {
        func.func @main(%tensor: tensor<4x!qco.qubit>, %index: index)
            -> tensor<4x!qco.qubit> {
          %i0 = arith.constant 3 : index
          %i1 = arith.constant 0 : index
          %i2 = arith.constant 2 : index
          %i3 = arith.constant 1 : index
          %r0, %q0 = qtensor.extract %tensor[%i0] : tensor<4x!qco.qubit>
          %h0 = qco.h %q0 : !qco.qubit -> !qco.qubit
          %t0 = qtensor.insert %h0 into %r0[%i0] : tensor<4x!qco.qubit>
          %r1, %q1 = qtensor.extract %t0[%i1] : tensor<4x!qco.qubit>
          %h1 = qco.h %q1 : !qco.qubit -> !qco.qubit
          %t1 = qtensor.insert %h1 into %r1[%i1] : tensor<4x!qco.qubit>
          %r2, %q2 = qtensor.extract %t1[%i2] : tensor<4x!qco.qubit>
          %h2 = qco.h %q2 : !qco.qubit -> !qco.qubit
          %t2 = qtensor.insert %h2 into %r2[%i2] : tensor<4x!qco.qubit>
          %r3, %q3 = qtensor.extract %t2[%i3] : tensor<4x!qco.qubit>
          %h3 = qco.h %q3 : !qco.qubit -> !qco.qubit
          %t3 = qtensor.insert %h3 into %r3[%i3] : tensor<4x!qco.qubit>
          return %t3 : tensor<4x!qco.qubit>
        }
      }
    )mlir",
                                       &context_);
  }

  LogicalResult applyInsertPattern(InsertOp insert,
                                   RewriterBase::Listener* listener = nullptr) {
    RewritePatternSet patterns(&context_);
    InsertOp::getCanonicalizationPatterns(patterns, &context_);
    FrozenRewritePatternSet frozen(std::move(patterns));
    PatternApplicator applicator(frozen);
    applicator.applyDefaultCostModel();
    PatternRewriter rewriter(&context_);
    rewriter.setListener(listener);
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

TEST_F(QTensorPairCanonicalizationTest,
       CommutesDistinctSlotsInOneRewriteFromTheLastPair) {
  auto moduleOp = makeChain();
  ASSERT_TRUE(moduleOp);
  auto function = moduleOp->lookupSymbol<func::FuncOp>("main");
  SmallVector<InsertOp> inserts(function.getOps<InsertOp>());
  ASSERT_EQ(inserts.size(), 4);
  ASSERT_TRUE(succeeded(verify(*moduleOp)));
  ASSERT_TRUE(succeeded(qco::verifyLinearity(*moduleOp)));

  // Each rewrite must traverse only its chain, not recompute whole-block order.
  struct NoOrderRebuild final : RewriterBase::Listener {
    void notifyOperationModified(Operation* operation) override {
      EXPECT_FALSE(operation->getBlock()->isOpOrderValid());
    }
  } listener;
  function.getBody().front().invalidateOpOrder();
  // One rewrite must normalize the whole chain to avoid quadratic commuting.
  ASSERT_TRUE(succeeded(applyInsertPattern(inserts[2], &listener)));
  ASSERT_TRUE(succeeded(verify(*moduleOp)));
  ASSERT_TRUE(succeeded(qco::verifyLinearity(*moduleOp)));
  constexpr std::array<int64_t, 4> indices{3, 0, 2, 1};
  size_t numExtracts = 0;
  size_t numInserts = 0;
  for (auto& operation : function.getBody().front()) {
    if (auto extract = dyn_cast<ExtractOp>(operation)) {
      EXPECT_EQ(numInserts, 0);
      ASSERT_LT(numExtracts, indices.size());
      EXPECT_EQ(getConstantIntValue(extract.getIndex()), indices[numExtracts]);
      ++numExtracts;
    } else if (auto insert = dyn_cast<InsertOp>(operation)) {
      ASSERT_LT(numInserts, indices.size());
      EXPECT_EQ(getConstantIntValue(insert.getIndex()), indices[numInserts]);
      auto h = insert.getScalar().getDefiningOp<qco::HOp>();
      ASSERT_TRUE(h);
      auto extract = h->getOperand(0).getDefiningOp<ExtractOp>();
      ASSERT_TRUE(extract);
      EXPECT_EQ(extract.getIndex(), insert.getIndex());
      ++numInserts;
    }
  }
  EXPECT_EQ(numExtracts, indices.size());
  EXPECT_EQ(numInserts, indices.size());
}

TEST_F(QTensorPairCanonicalizationTest,
       CommutesOnlyAfterAnEarlierWriteToTheSameSlot) {
  auto moduleOp = makeChain();
  ASSERT_TRUE(moduleOp);
  auto function = moduleOp->lookupSymbol<func::FuncOp>("main");
  SmallVector<ExtractOp> extracts(function.getOps<ExtractOp>());
  SmallVector<InsertOp> inserts(function.getOps<InsertOp>());
  extracts[1].getIndexMutable().assign(extracts[0].getIndex());
  inserts[1].getIndexMutable().assign(extracts[0].getIndex());
  ASSERT_TRUE(succeeded(verify(*moduleOp)));
  ASSERT_TRUE(succeeded(qco::verifyLinearity(*moduleOp)));

  ASSERT_TRUE(succeeded(applyInsertPattern(inserts[2])));
  ASSERT_TRUE(succeeded(verify(*moduleOp)));
  ASSERT_TRUE(succeeded(qco::verifyLinearity(*moduleOp)));
  EXPECT_EQ(extracts[1].getTensor(), inserts[0].getResult());
  EXPECT_TRUE(inserts[0]->isBeforeInBlock(extracts[1]));
  EXPECT_TRUE(extracts[3]->isBeforeInBlock(inserts[1]));
}

TEST_F(QTensorPairCanonicalizationTest, DoesNotCommuteAcrossADynamicSlot) {
  auto moduleOp = makeChain();
  ASSERT_TRUE(moduleOp);
  auto function = moduleOp->lookupSymbol<func::FuncOp>("main");
  SmallVector<ExtractOp> extracts(function.getOps<ExtractOp>());
  SmallVector<InsertOp> inserts(function.getOps<InsertOp>());
  extracts[1].getIndexMutable().assign(function.getArgument(1));
  inserts[1].getIndexMutable().assign(function.getArgument(1));
  ASSERT_TRUE(succeeded(verify(*moduleOp)));
  ASSERT_TRUE(succeeded(qco::verifyLinearity(*moduleOp)));

  EXPECT_TRUE(failed(applyInsertPattern(inserts[0])));
  EXPECT_TRUE(failed(applyInsertPattern(inserts[1])));
  ASSERT_TRUE(succeeded(applyInsertPattern(inserts[2])));
  ASSERT_TRUE(succeeded(verify(*moduleOp)));
  ASSERT_TRUE(succeeded(qco::verifyLinearity(*moduleOp)));
  EXPECT_EQ(extracts[1].getTensor(), inserts[0].getResult());
  EXPECT_EQ(extracts[2].getTensor(), inserts[1].getResult());
  EXPECT_TRUE(extracts[3]->isBeforeInBlock(inserts[2]));
}

TEST_F(QTensorPairCanonicalizationTest, ChecksOperationOrderInGraphRegions) {
  for (const bool backwardUse : {false, true}) {
    SCOPED_TRACE(backwardUse);
    auto moduleOp = parseSourceString<ModuleOp>(R"mlir(
      module {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %tensor = qtensor.alloc(%c2) : tensor<2x!qco.qubit>
        %r0, %q0 = qtensor.extract %tensor[%c0] : tensor<2x!qco.qubit>
        %h0 = qco.h %q0 : !qco.qubit -> !qco.qubit
        %t0 = qtensor.insert %h0 into %r0[%c0] : tensor<2x!qco.qubit>
        %r1, %q1 = qtensor.extract %t0[%c1] : tensor<2x!qco.qubit>
        %h1 = qco.h %q1 : !qco.qubit -> !qco.qubit
        %t1 = qtensor.insert %h1 into %r1[%c1] : tensor<2x!qco.qubit>
        qtensor.dealloc %t1 : tensor<2x!qco.qubit>
      }
    )mlir",
                                                &context_);
    ASSERT_TRUE(moduleOp);
    auto insert = *moduleOp->getOps<InsertOp>().begin();
    auto extract = cast<ExtractOp>(*insert.getResult().user_begin());
    if (backwardUse) {
      extract->moveBefore(insert);
    }
    ASSERT_TRUE(succeeded(verify(*moduleOp)));
    ASSERT_TRUE(succeeded(qco::verifyLinearity(*moduleOp)));

    EXPECT_EQ(succeeded(applyInsertPattern(insert)), !backwardUse);
    ASSERT_TRUE(succeeded(verify(*moduleOp)));
    ASSERT_TRUE(succeeded(qco::verifyLinearity(*moduleOp)));
    EXPECT_EQ(extract.getTensor() == insert.getResult(), backwardUse);
  }
}

} // namespace
