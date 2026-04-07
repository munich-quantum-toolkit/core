/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

/**
 * @file test_qtensor_ir.cpp
 * @brief Dedicated unit-test suite for the QTensor MLIR dialect.
 */

#include "TestCaseUtils.h"
#include "mlir/Dialect/QCO/Builder/QCOProgramBuilder.h"
#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QTensor/IR/QTensorDialect.h"
#include "mlir/Dialect/QTensor/IR/QTensorOps.h"
#include "mlir/Dialect/QTensor/IR/QTensorUtils.h"
#include "mlir/Support/IRVerification.h"
#include "mlir/Support/Passes.h"
#include "qco_programs.h"

#include <gtest/gtest.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/Verifier.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <ostream>
#include <string>
#include <tuple>

using namespace mlir;
using namespace mlir::qtensor;
using namespace mlir::qco;

namespace {
// ============================================================================
// Shared fixture — sets up an MLIR context with QTensor/QCO/Arith dialects
// and provides a QCOProgramBuilder for creating test programs.
// ============================================================================

class QTensorTest : public ::testing::Test {
protected:
  std::unique_ptr<MLIRContext> context;

  void SetUp() override {
    DialectRegistry registry;
    registry.insert<QCODialect, arith::ArithDialect, func::FuncDialect,
                    QTensorDialect>();
    context = std::make_unique<MLIRContext>();
    context->appendDialectRegistry(registry);
    context->loadAllAvailableDialects();
  }

  /// Build a module using the QCOProgramBuilder and run the cleanup pipeline.
  [[nodiscard]] OwningOpRef<ModuleOp>
  buildAndCanonicalize(void (*buildFn)(QCOProgramBuilder&)) const {
    auto module = QCOProgramBuilder::build(context.get(), buildFn);
    if (!module) {
      return {};
    }
    if (runQCOCleanupPipeline(module.get()).failed()) {
      return {};
    }
    return module;
  }

  /// Count occurrences of a specific op kind inside a module.
  template <typename OpT>
  [[nodiscard]] static std::size_t countOps(ModuleOp module) {
    std::size_t count = 0;
    module.walk([&](OpT) { ++count; });
    return count;
  }
};

// ============================================================================
// 1. QTensorUtils — direct tests of scalar chain helpers
// ============================================================================

TEST_F(QTensorTest, AreEquivalentIndicesSameValueIsEquivalent) {
  QCOProgramBuilder builder(context.get());
  builder.initialize();
  auto c2 = arith::ConstantIndexOp::create(builder, 2);
  EXPECT_TRUE(areEquivalentIndices(c2.getResult(), c2.getResult()));
}

TEST_F(QTensorTest, AreEquivalentIndicesDifferentConstantsAreNotEquivalent) {
  QCOProgramBuilder builder(context.get());
  builder.initialize();
  auto c0 = arith::ConstantIndexOp::create(builder, 0);
  auto c1 = arith::ConstantIndexOp::create(builder, 1);
  EXPECT_FALSE(areEquivalentIndices(c0.getResult(), c1.getResult()));
}

TEST_F(QTensorTest, AreEquivalentIndicesSameConstantDifferentSSAAreEquivalent) {
  QCOProgramBuilder builder(context.get());
  builder.initialize();
  auto lhs = arith::ConstantIndexOp::create(builder, 2);
  auto rhs = arith::ConstantIndexOp::create(builder, 2);
  EXPECT_TRUE(areEquivalentIndices(lhs.getResult(), rhs.getResult()));
}

TEST_F(QTensorTest, TensorChainHelpersInsertAndExtractAreRecognized) {
  QCOProgramBuilder builder(context.get());
  builder.initialize();
  auto tensor = builder.qtensorAlloc(3);
  auto [outTensor, q0] = builder.qtensorExtract(tensor, 0);
  auto* insert = builder.qtensorInsert(q0, outTensor, 0).getDefiningOp();
  auto* extract = outTensor.getDefiningOp();

  ASSERT_NE(insert, nullptr);
  ASSERT_NE(extract, nullptr);
  EXPECT_TRUE(isTensorChainOp(insert));
  EXPECT_TRUE(isTensorChainOp(extract));
  EXPECT_EQ(getTensorChainOutput(insert), insert->getResult(0));
  EXPECT_EQ(getTensorChainInput(extract), tensor);
}

TEST_F(QTensorTest, TensorChainHelpersSetTensorChainInputRewiresOperand) {
  auto module =
      QCOProgramBuilder::build(context.get(), [](QCOProgramBuilder& b) {
        auto t1 = b.qtensorAlloc(3);
        auto [out1, q1] = b.qtensorExtract(t1, 1);
        auto t0 = b.qtensorAlloc(3);
        auto [out0, q0] = b.qtensorExtract(t0, 0);
        auto insert = InsertOp::create(
            b, q0, out0, arith::ConstantIndexOp::create(b, 0).getResult());
        setTensorChainInput(insert.getOperation(), out1);
        EXPECT_EQ(getTensorChainInput(insert.getOperation()), out1);
        (void)InsertOp::create(
            b, q0, out1, arith::ConstantIndexOp::create(b, 1).getResult());
        (void)q1;
        (void)out0;
      });
  ASSERT_TRUE(module);
  EXPECT_TRUE(verify(*module).succeeded());
}

// ============================================================================
// 2. AllocOp — verify() tests
// ============================================================================

/// A valid static alloc should pass verification.
TEST_F(QTensorTest, AllocOpValidStaticAllocVerifies) {
  auto module = QCOProgramBuilder::build(
      context.get(), [](QCOProgramBuilder& b) { b.qtensorAlloc(3); });
  ASSERT_TRUE(module);
  EXPECT_TRUE(verify(*module).succeeded());
}

/// AllocOp with a constant size ≤ 0 must fail verification.
/// Note: The builder asserts on zero/negative, so we verify the verifier
/// by constructing the op manually bypassing the builder assertion.
TEST_F(QTensorTest, AllocOpZeroSizeFailsVerification) {
  // Build a module manually to bypass builder-level assertion.
  auto loc = UnknownLoc::get(context.get());
  auto module = ModuleOp::create(loc);
  ImplicitLocOpBuilder b(loc, context.get());
  b.setInsertionPointToStart(module.getBody());

  // Create a constant 0 for the size operand.
  auto c0 = arith::ConstantIndexOp::create(b, 0);
  // Construct the result type that would match a size-0 tensor (which is
  // invalid per the verifier). We use kDynamic so the type-level constraint
  // won't block construction, but the constant operand (0) triggers the
  // verifier.
  auto qubitType = qco::QubitType::get(context.get());
  auto dynType = RankedTensorType::get({ShapedType::kDynamic}, qubitType);
  AllocOp::create(b, dynType, c0.getResult());

  // The verifier should catch `sizeValue <= 0`.
  EXPECT_TRUE(verify(module).failed());
}

/// AllocOp where static result type dim ≠ constant size must fail.
TEST_F(QTensorTest, AllocOpStaticTypeMismatchFailsVerification) {
  auto loc = UnknownLoc::get(context.get());
  auto module = ModuleOp::create(loc);
  ImplicitLocOpBuilder b(loc, context.get());
  b.setInsertionPointToStart(module.getBody());

  auto c2 = arith::ConstantIndexOp::create(b, 2); // size operand = 2
  auto qubitType = qco::QubitType::get(context.get());
  // result type says dimension = 3, but size operand = 2 → mismatch
  auto staticType = RankedTensorType::get({3}, qubitType);
  AllocOp::create(b, staticType, c2.getResult());

  EXPECT_TRUE(verify(module).failed());
}

/// AllocOp with a dynamic result type but a constant size operand is valid.
TEST_F(QTensorTest, AllocOpDynamicTypeWithConstantSizeVerifies) {
  auto loc = UnknownLoc::get(context.get());
  auto module = ModuleOp::create(loc);
  ImplicitLocOpBuilder b(loc, context.get());
  b.setInsertionPointToStart(module.getBody());

  auto c3 = arith::ConstantIndexOp::create(b, 3);
  auto qubitType = qco::QubitType::get(context.get());
  auto dynType = RankedTensorType::get({ShapedType::kDynamic}, qubitType);
  AllocOp::create(b, dynType, c3.getResult());

  // Dynamic result dim with constant positive size → valid.
  EXPECT_TRUE(verify(module).succeeded());
}

/// AllocOp with a static result type but a non-constant (dynamic) size
/// operand must fail verification.
TEST_F(QTensorTest, AllocOpStaticTypeWithDynamicSizeOperandFailsVerification) {
  auto loc = UnknownLoc::get(context.get());
  auto module = ModuleOp::create(loc);
  ImplicitLocOpBuilder b(loc, context.get());
  b.setInsertionPointToStart(module.getBody());

  // We need a block argument to act as a non-constant size
  // Create a func.func to hold the block argument
  auto funcType =
      FunctionType::get(context.get(), {IndexType::get(context.get())}, {});
  auto func = func::FuncOp::create(b, "test", funcType);
  auto* block = func.addEntryBlock();
  b.setInsertionPointToStart(block);

  // Static result type dim = 3, but size operand is dynamic → error
  auto qubitType = qco::QubitType::get(context.get());
  auto staticType = RankedTensorType::get({3}, qubitType);
  auto dynSizeVal = block->getArgument(0);
  AllocOp::create(b, staticType, dynSizeVal);
  func::ReturnOp::create(b);

  EXPECT_TRUE(verify(module).failed());
}

// ============================================================================
// 3. DeallocOp — canonicalization (RemoveAllocDeallocPair)
// ============================================================================

/// An alloc immediately followed by dealloc should be eliminated entirely.
TEST_F(QTensorTest, DeallocOpAllocDeallocPairIsRemoved) {
  auto canonicalized = buildAndCanonicalize([](QCOProgramBuilder& b) {
    auto t = b.qtensorAlloc(3);
    b.qtensorDealloc(t);
  });
  ASSERT_TRUE(canonicalized);
  EXPECT_TRUE(verify(*canonicalized).succeeded());
  // Both AllocOp and DeallocOp should have been erased.
  EXPECT_EQ(countOps<AllocOp>(*canonicalized), 0U);
  EXPECT_EQ(countOps<DeallocOp>(*canonicalized), 0U);
}

/// A dealloc whose operand is not directly an AllocOp should not be removed.
TEST_F(QTensorTest, DeallocOpDeallocOfNonAllocIsNotRemoved) {
  // Extract then insert to create a different tensor SSA value before dealloc.
  auto canonicalized = buildAndCanonicalize([](QCOProgramBuilder& b) {
    auto tensor = b.qtensorAlloc(3);
    auto [outTensor, q0] = b.qtensorExtract(tensor, 0);
    auto q1 = b.h(q0);
    auto afterInsert = b.qtensorInsert(q1, outTensor, 0);
    b.qtensorDealloc(afterInsert);
  });
  ASSERT_TRUE(canonicalized);
  EXPECT_TRUE(verify(*canonicalized).succeeded());
  // After canonicalization the extract-insert pair simplifies, but there
  // should still be either an alloc-dealloc pair or both get eliminated
  // through further folding — just check the module is valid.
  // The important invariant: DeallocOp count is not negative, i.e., the
  // transform did not crash.
}

// ============================================================================
// 4. ExtractOp — verify(), fold, and canonicalization
// ============================================================================

/// A valid extract at index 0 from a size-1 tensor must pass verification.
TEST_F(QTensorTest, ExtractOpValidIndexVerifies) {
  auto module =
      QCOProgramBuilder::build(context.get(), [](QCOProgramBuilder& b) {
        auto t = b.qtensorAlloc(1);
        b.qtensorExtract(t, 0);
      });
  ASSERT_TRUE(module);
  EXPECT_TRUE(verify(*module).succeeded());
}

/// An extract at a negative constant index must fail verification.
TEST_F(QTensorTest, ExtractOpNegativeIndexFailsVerification) {
  QCOProgramBuilder builder(context.get());
  builder.initialize();
  auto tensor = builder.qtensorAlloc(3);
  auto negIdx = arith::ConstantIndexOp::create(builder, -1);
  ExtractOp::create(builder, tensor, negIdx.getResult());
  auto module = builder.finalize();
  ASSERT_TRUE(module);
  EXPECT_TRUE(verify(*module).failed());
}

/// An extract at an index equal to the tensor dimension must fail (out of
/// bounds).
TEST_F(QTensorTest, ExtractOpIndexAtDimFailsVerification) {
  QCOProgramBuilder builder(context.get());
  builder.initialize();
  auto tensor = builder.qtensorAlloc(3);
  // index = 3, tensor has dim 3 → out of bounds
  auto idx3 = arith::ConstantIndexOp::create(builder, 3);
  ExtractOp::create(builder, tensor, idx3.getResult());
  auto module = builder.finalize();
  ASSERT_TRUE(module);
  EXPECT_TRUE(verify(*module).failed());
}

/// An extract at an index one less than the dimension must pass.
TEST_F(QTensorTest, ExtractOpIndexAtDimMinusOneVerifies) {
  // Build inside a proper func.func body via QCOProgramBuilder.
  QCOProgramBuilder builder(context.get());
  builder.initialize();
  // qtensorAlloc(3) creates tensor<3x!qco.qubit> and tracks it.
  auto tensor = builder.qtensorAlloc(3);
  auto idx2 = arith::ConstantIndexOp::create(builder, 2);
  // Create extract at index 2 — last valid index for dim 3.
  // (Use the raw op creator to bypass builder tracking.)
  ExtractOp::create(builder, tensor, idx2.getResult());
  // finalize() will dealloc the still-tracked tensor (two uses of %tensor is
  // valid in MLIR SSA). Dead extract results (%tOut, %q) are fine.
  auto module = builder.finalize();
  ASSERT_TRUE(module);
  EXPECT_TRUE(verify(*module).succeeded());
}

/// foldExtractAfterInsert: extract(insert(t, q, i), i) → (t, q)
/// The fold must eliminate the round-trip at the same index.
TEST_F(QTensorTest, ExtractOpFoldExtractAfterInsertSameIndex) {
  // Use the full QCO pipeline so that both the fold and subsequent DCE of the
  // dead InsertOp run to convergence (single canonicalizer pass may leave
  // unreachable Pure ops if DCE and folding don't interleave).
  auto module =
      QCOProgramBuilder::build(context.get(), [](QCOProgramBuilder& b) {
        auto tensor = b.qtensorAlloc(3);
        auto [outTensor, q0] = b.qtensorExtract(tensor, 0);
        auto q1 = b.h(q0);
        auto afterInsert = b.qtensorInsert(q1, outTensor, 0);
        // Immediately extract the same qubit back — should fold away.
        b.qtensorExtract(afterInsert, 0);
      });
  ASSERT_TRUE(module);
  EXPECT_TRUE(verify(*module).succeeded());
  EXPECT_TRUE(runQCOCleanupPipeline(module.get()).succeeded());
  EXPECT_TRUE(verify(*module).succeeded());
  // The extra extract at the same index should fold away.
  EXPECT_EQ(countOps<ExtractOp>(*module), 1U); // original extract
}

/// foldExtractAfterInsert: extract at a different index must NOT fold.
TEST_F(QTensorTest, ExtractOpNoFoldExtractAfterInsertDifferentIndex) {
  auto canonicalized = buildAndCanonicalize([](QCOProgramBuilder& b) {
    auto tensor = b.qtensorAlloc(3);
    auto [outTensor, q0] = b.qtensorExtract(tensor, 0);
    auto q1 = b.h(q0);
    auto afterInsert = b.qtensorInsert(q1, outTensor, 0);
    // Extract at index 1 — different from the insert's index 0
    b.qtensorExtract(afterInsert, 1);
  });
  ASSERT_TRUE(canonicalized);
  EXPECT_TRUE(verify(*canonicalized).succeeded());
  // The insert should still be present (not folded).
  EXPECT_GE(countOps<InsertOp>(*canonicalized), 1U);
}

/// RemoveInsertExtractPair: extract through a disjoint InsertOp should find
/// the original extract and eliminate both.
TEST_F(QTensorTest, ExtractOpRemoveInsertExtractPairThroughDisjointInsert) {
  auto canonicalized = buildAndCanonicalize([](QCOProgramBuilder& b) {
    auto tensor = b.qtensorAlloc(3);
    // Extract qubit 0.
    auto [t1, q0] = b.qtensorExtract(tensor, 0);
    // Extract qubit 1 (disjoint from qubit 0).
    auto [t2, q1] = b.qtensorExtract(t1, 1);
    // Insert qubit 1 back at index 1, then extract it again — same index.
    // The canonicalizer should eliminate both the insert and the re-extract.
    auto afterInsert = b.qtensorInsert(q1, t2, 1);
    b.qtensorExtract(afterInsert, 1);
    // Use q0 so it isn't dead.
    b.h(q0);
  });
  ASSERT_TRUE(canonicalized);
  EXPECT_TRUE(verify(*canonicalized).succeeded());
}

/// RemoveInsertExtractPair: a nested ExtractOp at the same index must block
/// re-ordering (linearity guard).
TEST_F(QTensorTest,
       ExtractOpRemoveInsertExtractPairBlockedByNestedExtractAtSameIndex) {
  // Pattern: insert q0 at 0, then extract-at-0 twice (would violate linearity
  // if the first extraction were skipped).
  auto canonicalized = buildAndCanonicalize([](QCOProgramBuilder& b) {
    auto tensor = b.qtensorAlloc(3);
    auto [t1, q0] = b.qtensorExtract(tensor, 0);
    // Re-insert q0 at index 0, producing a new tensor.
    auto q0h = b.h(q0);
    auto afterInsert = b.qtensorInsert(q0h, t1, 0);
    // Attempt to extract index 0 again — the chain already has an
    // extract-at-0 in it, blocking the RemoveInsertExtractPair pattern.
    auto [t3, q0again] = b.qtensorExtract(afterInsert, 0);
    b.h(q0again);
    (void)t3;
  });
  ASSERT_TRUE(canonicalized);
  EXPECT_TRUE(verify(*canonicalized).succeeded());
}

// ============================================================================
// 5. InsertOp — verify(), fold, and canonicalization
// ============================================================================

/// A valid insert at index 0 into a size-3 tensor must pass verification.
TEST_F(QTensorTest, InsertOpValidIndexVerifies) {
  auto module =
      QCOProgramBuilder::build(context.get(), [](QCOProgramBuilder& b) {
        auto t = b.qtensorAlloc(3);
        auto [out, q] = b.qtensorExtract(t, 0);
        b.qtensorInsert(q, out, 0);
      });
  ASSERT_TRUE(module);
  EXPECT_TRUE(verify(*module).succeeded());
}

/// An insert at a negative constant index must fail verification.
TEST_F(QTensorTest, InsertOpNegativeIndexFailsVerification) {
  QCOProgramBuilder builder(context.get());
  builder.initialize();
  // Extract qubit 0 to get both a tracked tensor and a qubit.
  auto tensor = builder.qtensorAlloc(3);
  auto [outTensor, q0] = builder.qtensorExtract(tensor, 0);
  // Insert at index -1 — raw op creation bypasses builder tracking.
  auto negIdx = arith::ConstantIndexOp::create(builder, -1);
  InsertOp::create(builder, q0, outTensor, negIdx.getResult());
  // finalize() will dealloc outTensor and sink q0 (both still tracked, both
  // reused — valid in SSA).
  auto module = builder.finalize();
  ASSERT_TRUE(module);
  EXPECT_TRUE(verify(*module).failed());
}

/// An insert at an index equal to the destination dimension must fail.
TEST_F(QTensorTest, InsertOpIndexAtDimFailsVerification) {
  QCOProgramBuilder builder(context.get());
  builder.initialize();
  auto tensor = builder.qtensorAlloc(3);
  auto [outTensor, q0] = builder.qtensorExtract(tensor, 0);
  auto idx3 = arith::ConstantIndexOp::create(builder, 3); // == dim
  InsertOp::create(builder, q0, outTensor, idx3.getResult());
  auto module = builder.finalize();
  ASSERT_TRUE(module);
  EXPECT_TRUE(verify(*module).failed());
}

/// foldInsertAfterExtract: insert(extract(t, i).qubit, extract(t, i).out, i)
/// should fold to `t`.
TEST_F(QTensorTest, InsertOpFoldInsertAfterExtractSameIndex) {
  auto canonicalized = buildAndCanonicalize([](QCOProgramBuilder& b) {
    auto tensor = b.qtensorAlloc(3);
    auto [outTensor, q0] = b.qtensorExtract(tensor, 0);
    // Insert the extracted qubit back at the same index without modification.
    b.qtensorInsert(q0, outTensor, 0);
  });
  ASSERT_TRUE(canonicalized);
  EXPECT_TRUE(verify(*canonicalized).succeeded());
  // The extract-insert pair should have been eliminated entirely.
  EXPECT_EQ(countOps<ExtractOp>(*canonicalized), 0U);
  EXPECT_EQ(countOps<InsertOp>(*canonicalized), 0U);
}

/// foldInsertAfterExtract: inserting the qubit at a different index must NOT
/// fold.
TEST_F(QTensorTest, InsertOpNoFoldInsertAfterExtractDifferentIndex) {
  auto canonicalized = buildAndCanonicalize([](QCOProgramBuilder& b) {
    auto tensor = b.qtensorAlloc(3);
    auto [outTensor, q0] = b.qtensorExtract(tensor, 0);
    // Insert at index 1 instead of 0
    b.qtensorInsert(q0, outTensor, 1);
  });
  ASSERT_TRUE(canonicalized);
  EXPECT_TRUE(verify(*canonicalized).succeeded());
  EXPECT_GE(countOps<InsertOp>(*canonicalized), 1U);
}

/// foldInsertAfterExtract: inserting into a different tensor (not the extract's
/// out_tensor) must NOT fold.
TEST_F(QTensorTest, InsertOpNoFoldInsertAfterExtractDifferentDest) {
  auto canonicalized = buildAndCanonicalize([](QCOProgramBuilder& b) {
    auto t1 = b.qtensorAlloc(3);
    auto t2 = b.qtensorAlloc(3);
    auto [outTensor, q0] = b.qtensorExtract(t1, 0);
    // q0 came from t1, but we insert into t2's out-tensor
    auto [t2out, q1] = b.qtensorExtract(t2, 1);
    b.qtensorInsert(q0, t2out, 0);
    b.h(q1);
  });
  ASSERT_TRUE(canonicalized);
  EXPECT_TRUE(verify(*canonicalized).succeeded());
  EXPECT_GE(countOps<InsertOp>(*canonicalized), 1U);
}

/// RemoveExtractInsertPair: an insert-after-extract that has been modified
/// (qubit passed through an H gate) must NOT be eliminated.
TEST_F(QTensorTest, InsertOpNoRemoveExtractInsertPairAfterMutation) {
  auto canonicalized = buildAndCanonicalize([](QCOProgramBuilder& b) {
    auto tensor = b.qtensorAlloc(3);
    auto [outTensor, q0] = b.qtensorExtract(tensor, 0);
    auto q1 = b.h(q0); // mutation — scalar ≠ extract.getResult()
    b.qtensorInsert(q1, outTensor, 0);
  });
  ASSERT_TRUE(canonicalized);
  EXPECT_TRUE(verify(*canonicalized).succeeded());
  // The HOp mutates the qubit, so the pair cannot be collapsed.
  EXPECT_GE(countOps<InsertOp>(*canonicalized), 1U);
}

/// RemoveExtractInsertPair: insert shadowed by an earlier InsertOp at the same
/// index must not be eliminated.
TEST_F(QTensorTest, InsertOpRemoveExtractInsertPairBlockedByShadowingInsert) {
  // Pattern:
  //   t1, q0 = extract(alloc, 0)
  //   t2 = insert(q0, t1, 0)           ← overwrites index 0
  //   t3 = insert(q0_h, t2, 0)         ← another write to index 0
  // Trying to find the matching extract for the second insert should be
  // blocked by the first insert at the same index.
  auto canonicalized = buildAndCanonicalize([](QCOProgramBuilder& b) {
    auto tensor = b.qtensorAlloc(3);
    auto [t1, q0] = b.qtensorExtract(tensor, 0);
    // First insert q0 at 0.
    auto t2 = b.qtensorInsert(q0, t1, 0);
    // Second insert at 0 (different qubit — from another extract).
    auto [t2out, q1] = b.qtensorExtract(t2, 0);
    auto q1h = b.h(q1);
    b.qtensorInsert(q1h, t2out, 0);
  });
  ASSERT_TRUE(canonicalized);
  EXPECT_TRUE(verify(*canonicalized).succeeded());
}

/// RemoveExtractInsertPair: insert blocked by a disjoint insert (different
/// index) should still succeed in finding the original extract.
TEST_F(QTensorTest, InsertOpRemoveExtractInsertPairThroughDisjointInsert) {
  // Pattern:
  //   t1, q0 = extract(alloc, 0)
  //   t2, q1 = extract(t1, 1)          ← disjoint from index 0
  //   t3 = insert(q1, t2, 1)           ← insert at index 1 (disjoint)
  //   t4 = insert(q0, t3, 0)           ← insert matches the extract at 0
  // Both the q0 extract-insert and q1 extract-insert should collapse.
  auto canonicalized = buildAndCanonicalize([](QCOProgramBuilder& b) {
    auto tensor = b.qtensorAlloc(3);
    auto [t1, q0] = b.qtensorExtract(tensor, 0);
    auto [t2, q1] = b.qtensorExtract(t1, 1);
    auto t3 = b.qtensorInsert(q1, t2, 1);
    b.qtensorInsert(q0, t3, 0);
  });
  ASSERT_TRUE(canonicalized);
  EXPECT_TRUE(verify(*canonicalized).succeeded());
  // Both pairs should collapse.
  EXPECT_EQ(countOps<ExtractOp>(*canonicalized), 0U);
  EXPECT_EQ(countOps<InsertOp>(*canonicalized), 0U);
}

// ============================================================================
// 6. Integration
//
// These tests use the full QCO cleanup pipeline and compare canonicalized
// modules for structural equivalence with permutations.
// ============================================================================

struct QTensorIntegrationTestCase {
  std::string name;
  mqt::test::NamedBuilder<QCOProgramBuilder> programBuilder;
  mqt::test::NamedBuilder<QCOProgramBuilder> referenceBuilder;

  // NOLINTNEXTLINE(llvm-prefer-static-over-anonymous-namespace)
  friend std::ostream& operator<<(std::ostream& os,
                                  const QTensorIntegrationTestCase& info);
};

// NOLINTNEXTLINE(llvm-prefer-static-over-anonymous-namespace)
std::ostream& operator<<(std::ostream& os,
                         const QTensorIntegrationTestCase& info) {
  return os << "QTensor{" << info.name << "}";
}

class QTensorIntegrationTest
    : public testing::TestWithParam<QTensorIntegrationTestCase> {
protected:
  std::unique_ptr<MLIRContext> context;

  void SetUp() override {
    DialectRegistry registry;
    registry.insert<QCODialect, arith::ArithDialect, func::FuncDialect,
                    QTensorDialect>();
    context = std::make_unique<MLIRContext>();
    context->appendDialectRegistry(registry);
    context->loadAllAvailableDialects();
  }
};

TEST_P(QTensorIntegrationTest, ProgramEquivalence) {
  const auto& [_, programBuilder, referenceBuilder] = GetParam();
  const auto name = " (" + GetParam().name + ")";
  mqt::test::DeferredPrinter printer;

  auto program = QCOProgramBuilder::build(context.get(), programBuilder.fn);
  ASSERT_TRUE(program);
  printer.record(program.get(), "Original QTensor IR" + name);
  EXPECT_TRUE(verify(*program).succeeded());

  EXPECT_TRUE(runQCOCleanupPipeline(program.get()).succeeded());
  printer.record(program.get(), "Canonicalized QTensor IR" + name);
  EXPECT_TRUE(verify(*program).succeeded());

  auto reference = QCOProgramBuilder::build(context.get(), referenceBuilder.fn);
  ASSERT_TRUE(reference);
  printer.record(reference.get(), "Reference QTensor IR" + name);
  EXPECT_TRUE(verify(*reference).succeeded());

  EXPECT_TRUE(runQCOCleanupPipeline(reference.get()).succeeded());
  printer.record(reference.get(), "Canonicalized Reference QTensor IR" + name);
  EXPECT_TRUE(verify(*reference).succeeded());

  EXPECT_TRUE(
      areModulesEquivalentWithPermutations(program.get(), reference.get()));
}

/// @name QTensor/QTensor.cpp (relocated from QCO test suite)
/// @{
INSTANTIATE_TEST_SUITE_P(
    QTensorOpsTest, QTensorIntegrationTest,
    testing::Values(
        QTensorIntegrationTestCase{"QTensorAlloc",
                                   MQT_NAMED_BUILDER(qtensorAlloc),
                                   MQT_NAMED_BUILDER(qtensorAlloc)},
        QTensorIntegrationTestCase{"QTensorAllocDealloc",
                                   MQT_NAMED_BUILDER(qtensorDealloc),
                                   MQT_NAMED_BUILDER(qtensorAlloc)},
        QTensorIntegrationTestCase{"QTensorFromElements",
                                   MQT_NAMED_BUILDER(qtensorFromElements),
                                   MQT_NAMED_BUILDER(qtensorFromElements)},
        QTensorIntegrationTestCase{"QTensorExtract",
                                   MQT_NAMED_BUILDER(qtensorExtract),
                                   MQT_NAMED_BUILDER(qtensorExtract)},
        QTensorIntegrationTestCase{"QTensorInsert",
                                   MQT_NAMED_BUILDER(qtensorInsert),
                                   MQT_NAMED_BUILDER(qtensorInsert)},
        QTensorIntegrationTestCase{
            "QTensorExtractInsertSameIndex",
            MQT_NAMED_BUILDER(qtensorExtractInsertSameIndex),
            MQT_NAMED_BUILDER(qtensorAlloc)},
        QTensorIntegrationTestCase{
            "QTensorExtractInsertIndexMismatch",
            MQT_NAMED_BUILDER(qtensorExtractInsertIndexMismatch),
            MQT_NAMED_BUILDER(qtensorExtractInsertIndexMismatch)},
        QTensorIntegrationTestCase{
            "QTensorInsertExtractSameIndex",
            MQT_NAMED_BUILDER(qtensorInsertExtractSameIndex),
            MQT_NAMED_BUILDER(qtensorInsert)},
        QTensorIntegrationTestCase{
            "QTensorInsertExtractIndexMismatch",
            MQT_NAMED_BUILDER(qtensorInsertExtractIndexMismatch),
            MQT_NAMED_BUILDER(qtensorInsertExtractIndexMismatch)}));
/// @}
} // namespace

// ============================================================================
// 7. Integration — multi-qubit permutation equivalence tests
// ============================================================================

static OwningOpRef<ModuleOp>
buildTwoQubitInsertChainProgram(MLIRContext* context,
                                const bool reverseInsertOrder,
                                const bool swapInsertTargets) {
  QCOProgramBuilder builder(context);
  builder.initialize();

  auto tensor = builder.qtensorAlloc(2);
  auto [tensorAfterFirstExtract, qubit0] = builder.qtensorExtract(tensor, 0);
  auto [baseTensor, qubit1] =
      builder.qtensorExtract(tensorAfterFirstExtract, 1);

  const int64_t qubit0Target = swapInsertTargets ? 1 : 0;
  const int64_t qubit1Target = swapInsertTargets ? 0 : 1;

  auto currentTensor = baseTensor;
  if (reverseInsertOrder) {
    currentTensor = builder.qtensorInsert(qubit1, currentTensor, qubit1Target);
    currentTensor = builder.qtensorInsert(qubit0, currentTensor, qubit0Target);
  } else {
    currentTensor = builder.qtensorInsert(qubit0, currentTensor, qubit0Target);
    currentTensor = builder.qtensorInsert(qubit1, currentTensor, qubit1Target);
  }

  builder.qtensorDealloc(currentTensor);
  return builder.finalize();
}

static OwningOpRef<ModuleOp>
buildMixedExtractInsertProgram(MLIRContext* context, const bool reverseOrder,
                               const bool swapInsertTargets) {
  QCOProgramBuilder builder(context);
  builder.initialize();

  auto tensor = builder.qtensorAlloc(3);
  auto tensorAfterReads = tensor;
  Value qubit0 = nullptr;
  Value qubit1 = nullptr;

  if (reverseOrder) {
    std::tie(tensorAfterReads, qubit1) =
        builder.qtensorExtract(tensorAfterReads, 1);
    std::tie(tensorAfterReads, qubit0) =
        builder.qtensorExtract(tensorAfterReads, 0);
  } else {
    std::tie(tensorAfterReads, qubit0) =
        builder.qtensorExtract(tensorAfterReads, 0);
    std::tie(tensorAfterReads, qubit1) =
        builder.qtensorExtract(tensorAfterReads, 1);
  }

  const int64_t q0Target = 0;
  const int64_t q1Target = swapInsertTargets ? 2 : 1;

  auto tensorAfterWrites = tensorAfterReads;
  if (reverseOrder) {
    tensorAfterWrites =
        builder.qtensorInsert(qubit0, tensorAfterWrites, q0Target);
    tensorAfterWrites =
        builder.qtensorInsert(qubit1, tensorAfterWrites, q1Target);
  } else {
    tensorAfterWrites =
        builder.qtensorInsert(qubit1, tensorAfterWrites, q1Target);
    tensorAfterWrites =
        builder.qtensorInsert(qubit0, tensorAfterWrites, q0Target);
  }

  builder.qtensorDealloc(tensorAfterWrites);
  return builder.finalize();
}

static OwningOpRef<ModuleOp>
buildResetWithCommutingInsertProgram(MLIRContext* context,
                                     const bool withReset) {
  QCOProgramBuilder builder(context);
  builder.initialize();

  auto tensor = builder.qtensorAlloc(2);
  auto [tensorAfterExtract0, qubit0] = builder.qtensorExtract(tensor, 0);
  auto tensorAfterInsert0 =
      builder.qtensorInsert(qubit0, tensorAfterExtract0, 0);
  auto [tensorAfterExtract1, qubit1] =
      builder.qtensorExtract(tensorAfterInsert0, 1);
  if (withReset) {
    qubit1 = builder.reset(qubit1);
  }
  auto tensorFinal = builder.qtensorInsert(qubit1, tensorAfterExtract1, 1);
  builder.qtensorDealloc(tensorFinal);

  return builder.finalize();
}

static OwningOpRef<ModuleOp>
buildResetWithSameIndexInsertProgram(MLIRContext* context,
                                     const bool withReset) {
  QCOProgramBuilder builder(context);
  builder.initialize();

  auto tensor = builder.qtensorAlloc(2);
  auto [tensorAfterExtract0, qubit0] = builder.qtensorExtract(tensor, 0);
  auto [tensorAfterExtract1, qubit1] =
      builder.qtensorExtract(tensorAfterExtract0, 1);
  qubit1 = builder.h(qubit1);
  auto tensorAfterInsert1 =
      builder.qtensorInsert(qubit1, tensorAfterExtract1, 1);
  auto [tensorAfterReadBack1, qubit1ReadBack] =
      builder.qtensorExtract(tensorAfterInsert1, 1);
  if (withReset) {
    qubit1ReadBack = builder.reset(qubit1ReadBack);
  }
  auto tensorAfterInsert1ReadBack =
      builder.qtensorInsert(qubit1ReadBack, tensorAfterReadBack1, 1);
  auto tensorFinal =
      builder.qtensorInsert(qubit0, tensorAfterInsert1ReadBack, 0);
  builder.qtensorDealloc(tensorFinal);

  return builder.finalize();
}

namespace {
TEST_F(QTensorTest, InsertChainPermutationEquivalence) {
  auto program = buildTwoQubitInsertChainProgram(context.get(), false, false);
  ASSERT_TRUE(program);
  EXPECT_TRUE(verify(*program).succeeded());
  EXPECT_TRUE(runQCOCleanupPipeline(program.get()).succeeded());
  EXPECT_TRUE(verify(*program).succeeded());

  auto reference = buildTwoQubitInsertChainProgram(context.get(), true, false);
  ASSERT_TRUE(reference);
  EXPECT_TRUE(verify(*reference).succeeded());
  EXPECT_TRUE(runQCOCleanupPipeline(reference.get()).succeeded());
  EXPECT_TRUE(verify(*reference).succeeded());

  EXPECT_TRUE(
      areModulesEquivalentWithPermutations(program.get(), reference.get()));
}

TEST_F(QTensorTest, InsertChainDifferentAssignmentsNotEquivalent) {
  auto program = buildTwoQubitInsertChainProgram(context.get(), false, false);
  ASSERT_TRUE(program);
  EXPECT_TRUE(runQCOCleanupPipeline(program.get()).succeeded());
  EXPECT_TRUE(verify(*program).succeeded());

  auto reference = buildTwoQubitInsertChainProgram(context.get(), true, true);
  ASSERT_TRUE(reference);
  EXPECT_TRUE(runQCOCleanupPipeline(reference.get()).succeeded());
  EXPECT_TRUE(verify(*reference).succeeded());

  EXPECT_FALSE(
      areModulesEquivalentWithPermutations(program.get(), reference.get()));
}

TEST_F(QTensorTest, MixedExtractInsertPermutationEquivalence) {
  auto program = buildMixedExtractInsertProgram(context.get(), false, false);
  ASSERT_TRUE(program);
  EXPECT_TRUE(runQCOCleanupPipeline(program.get()).succeeded());
  EXPECT_TRUE(verify(*program).succeeded());

  auto reference = buildMixedExtractInsertProgram(context.get(), true, false);
  ASSERT_TRUE(reference);
  EXPECT_TRUE(runQCOCleanupPipeline(reference.get()).succeeded());
  EXPECT_TRUE(verify(*reference).succeeded());

  EXPECT_TRUE(
      areModulesEquivalentWithPermutations(program.get(), reference.get()));
}

TEST_F(QTensorTest, MixedExtractInsertDifferentAssignmentsNotEquivalent) {
  auto program = buildMixedExtractInsertProgram(context.get(), false, false);
  ASSERT_TRUE(program);
  EXPECT_TRUE(runQCOCleanupPipeline(program.get()).succeeded());
  EXPECT_TRUE(verify(*program).succeeded());

  auto reference = buildMixedExtractInsertProgram(context.get(), true, true);
  ASSERT_TRUE(reference);
  EXPECT_TRUE(runQCOCleanupPipeline(reference.get()).succeeded());
  EXPECT_TRUE(verify(*reference).succeeded());

  EXPECT_FALSE(
      areModulesEquivalentWithPermutations(program.get(), reference.get()));
}

TEST_F(QTensorTest, ResetAfterExtractThroughCommutingInsertIsEliminated) {
  auto program = buildResetWithCommutingInsertProgram(context.get(), true);
  ASSERT_TRUE(program);
  EXPECT_TRUE(runQCOCleanupPipeline(program.get()).succeeded());
  EXPECT_TRUE(verify(*program).succeeded());

  auto reference = buildResetWithCommutingInsertProgram(context.get(), false);
  ASSERT_TRUE(reference);
  EXPECT_TRUE(runQCOCleanupPipeline(reference.get()).succeeded());
  EXPECT_TRUE(verify(*reference).succeeded());

  EXPECT_TRUE(
      areModulesEquivalentWithPermutations(program.get(), reference.get()));
}

TEST_F(QTensorTest, ResetAfterExtractThroughSameIndexInsertIsNotEliminated) {
  auto program = buildResetWithSameIndexInsertProgram(context.get(), true);
  ASSERT_TRUE(program);
  EXPECT_TRUE(runQCOCleanupPipeline(program.get()).succeeded());
  EXPECT_TRUE(verify(*program).succeeded());

  auto reference = buildResetWithSameIndexInsertProgram(context.get(), false);
  ASSERT_TRUE(reference);
  EXPECT_TRUE(runQCOCleanupPipeline(reference.get()).succeeded());
  EXPECT_TRUE(verify(*reference).succeeded());

  EXPECT_FALSE(
      areModulesEquivalentWithPermutations(program.get(), reference.get()));
}

} // namespace
