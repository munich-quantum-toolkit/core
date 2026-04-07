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

class QTensorTest : public ::testing::Test {
protected:
  std::unique_ptr<MLIRContext> context;

  /**
   * @brief Initialize the MLIRContext and register/load the dialects required by the tests.
   *
   * This sets up a fresh MLIRContext, appends a registry containing QCODialect,
   * arith::ArithDialect, func::FuncDialect, and QTensorDialect, and loads all
   * available dialects into the context.
   */
  void SetUp() override {
    DialectRegistry registry;
    registry.insert<QCODialect, arith::ArithDialect, func::FuncDialect,
                    QTensorDialect>();
    context = std::make_unique<MLIRContext>();
    context->appendDialectRegistry(registry);
    context->loadAllAvailableDialects();
  }

  /**
   * @brief Builds an MLIR module with the provided QCOProgramBuilder callback and
   * applies the QCO cleanup pipeline to canonicalize it.
   *
   * @param buildFn Callback that populates a QCOProgramBuilder to construct the
   * module.
   * @return OwningOpRef<ModuleOp> Owning reference to the canonicalized module,
   * or an empty `OwningOpRef` if building or cleanup failed.
   */
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
  /**
   * @brief Count occurrences of a specific operation type within an MLIR module.
   *
   * Counts the number of operations of type `OpT` found anywhere in `module`.
   *
   * @tparam OpT The operation class/type to count.
   * @param module The module to search.
   * @return std::size_t The number of `OpT` operations present in `module`.
   */
  [[nodiscard]] static std::size_t countOps(ModuleOp module) {
    std::size_t count = 0;
    module.walk([&](OpT) { ++count; });
    return count;
  }
};

// ============================================================================
// QTensorUtils
// ============================================================================

TEST_F(QTensorTest, AreEquivalentIndicesSameValueIsEquivalent) {
  QCOProgramBuilder builder(context.get());
  builder.initialize();
  auto c2 = arith::ConstantIndexOp::create(builder, 2);
  EXPECT_TRUE(areEquivalentIndices(c2.getResult(), c2.getResult()));
}

TEST_F(QTensorTest, AreEquivalentIndicesSameConstantsAreEquivalent) {
  QCOProgramBuilder builder(context.get());
  builder.initialize();
  auto lhs = arith::ConstantIndexOp::create(builder, 2);
  auto rhs = arith::ConstantIndexOp::create(builder, 2);
  EXPECT_TRUE(areEquivalentIndices(lhs.getResult(), rhs.getResult()));
}

TEST_F(QTensorTest, AreEquivalentIndicesDifferentConstantsAreNotEquivalent) {
  QCOProgramBuilder builder(context.get());
  builder.initialize();
  auto c0 = arith::ConstantIndexOp::create(builder, 0);
  auto c1 = arith::ConstantIndexOp::create(builder, 1);
  EXPECT_FALSE(areEquivalentIndices(c0.getResult(), c1.getResult()));
}

// ============================================================================
// AllocOp
// ============================================================================

/// AllocOp with a constant size ≤ 0 must fail verification.
TEST_F(QTensorTest, AllocOpZeroSizeFailsVerification) {
  auto loc = UnknownLoc::get(context.get());
  auto module = ModuleOp::create(loc);
  ImplicitLocOpBuilder b(loc, context.get());
  b.setInsertionPointToStart(module.getBody());

  auto qubitType = qco::QubitType::get(context.get());
  auto tensorType = RankedTensorType::get({ShapedType::kDynamic}, qubitType);
  auto c0 = arith::ConstantIndexOp::create(b, 0);
  AllocOp::create(b, tensorType, c0.getResult());

  EXPECT_TRUE(verify(module).failed());
}

/// AllocOp where static result type dim ≠ constant size must fail.
TEST_F(QTensorTest, AllocOpStaticTypeMismatchFailsVerification) {
  auto loc = UnknownLoc::get(context.get());
  auto module = ModuleOp::create(loc);
  ImplicitLocOpBuilder b(loc, context.get());
  b.setInsertionPointToStart(module.getBody());

  auto qubitType = qco::QubitType::get(context.get());
  auto tensorType = RankedTensorType::get({3}, qubitType);
  auto c2 = arith::ConstantIndexOp::create(b, 2);
  AllocOp::create(b, tensorType, c2.getResult());

  EXPECT_TRUE(verify(module).failed());
}

/// AllocOp with a dynamic result type but a constant size operand is valid.
TEST_F(QTensorTest, AllocOpDynamicTypeWithConstantSizeVerifies) {
  auto loc = UnknownLoc::get(context.get());
  auto module = ModuleOp::create(loc);
  ImplicitLocOpBuilder b(loc, context.get());
  b.setInsertionPointToStart(module.getBody());

  auto qubitType = qco::QubitType::get(context.get());
  auto tensorType = RankedTensorType::get({ShapedType::kDynamic}, qubitType);
  auto c3 = arith::ConstantIndexOp::create(b, 3);
  AllocOp::create(b, tensorType, c3.getResult());

  EXPECT_TRUE(verify(module).succeeded());
}

/// AllocOp with a static result type but a dynamic size fails verification.
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

  auto qubitType = qco::QubitType::get(context.get());
  auto tensorType = RankedTensorType::get({3}, qubitType);
  auto size = block->getArgument(0);
  AllocOp::create(b, tensorType, size);
  func::ReturnOp::create(b);

  EXPECT_TRUE(verify(module).failed());
}

// ============================================================================
// DeallocOp
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

// ============================================================================
// ExtractOp
// ============================================================================

/// An extract at a negative constant index fails verification.
TEST_F(QTensorTest, ExtractOpNegativeIndexFailsVerification) {
  QCOProgramBuilder builder(context.get());
  builder.initialize();
  auto tensor = builder.qtensorAlloc(3);
  auto index = arith::ConstantIndexOp::create(builder, -1);
  ExtractOp::create(builder, tensor, index.getResult());
  auto module = builder.finalize();

  ASSERT_TRUE(module);
  EXPECT_TRUE(verify(*module).failed());
}

/// An extract at an index equal to the tensor dimension fails verification.
TEST_F(QTensorTest, ExtractOpIndexAtDimFailsVerification) {
  QCOProgramBuilder builder(context.get());
  builder.initialize();
  auto tensor = builder.qtensorAlloc(3);
  auto index = arith::ConstantIndexOp::create(builder, 3);
  ExtractOp::create(builder, tensor, index.getResult());
  auto module = builder.finalize();

  ASSERT_TRUE(module);
  EXPECT_TRUE(verify(*module).failed());
}

// ============================================================================
// InsertOp
// ============================================================================

/// An insert at a negative constant index fails verification.
TEST_F(QTensorTest, InsertOpNegativeIndexFailsVerification) {
  QCOProgramBuilder builder(context.get());
  builder.initialize();
  auto tensor0 = builder.qtensorAlloc(3);
  auto [tensor1, q0] = builder.qtensorExtract(tensor0, 0);
  auto index = arith::ConstantIndexOp::create(builder, -1);
  InsertOp::create(builder, q0, tensor1, index.getResult());
  auto module = builder.finalize();

  ASSERT_TRUE(module);
  EXPECT_TRUE(verify(*module).failed());
}

/// An insert at an index equal to the destination dimension fails verification.
TEST_F(QTensorTest, InsertOpIndexAtDimFailsVerification) {
  QCOProgramBuilder builder(context.get());
  builder.initialize();
  auto tensor0 = builder.qtensorAlloc(3);
  auto [tensor1, q0] = builder.qtensorExtract(tensor0, 0);
  auto index = arith::ConstantIndexOp::create(builder, 3);
  InsertOp::create(builder, q0, tensor1, index.getResult());
  auto module = builder.finalize();

  ASSERT_TRUE(module);
  EXPECT_TRUE(verify(*module).failed());
}

} // namespace

// ============================================================================
// Canonicalization
/**
 * @brief Builds a small QTensor program that allocates a 2-qubit tensor, extracts both qubits,
 * reinserts them (order and target positions controlled by flags), then deallocates the tensor.
 *
 * The produced module contains a sequence of QTensor operations representing the allocation,
 * two extracts, two inserts (with ordering/target swapping determined by the parameters), and a
 * final deallocation.
 *
 * @param context MLIR context used to create the module and operations.
 * @param reverseInsertOrder If `true`, the second-extracted qubit is inserted before the
 * first-extracted qubit; if `false`, the first-extracted qubit is inserted first.
 * @param swapInsertTargets If `true`, the insertion target indices for the two qubits are
 * swapped (i.e., qubit0 is inserted at index 1 and qubit1 at index 0); if `false`, they
 * are inserted at their original indices (0 and 1 respectively).
 * @return OwningOpRef<ModuleOp> An owning reference to the constructed MLIR module.
 */

static OwningOpRef<ModuleOp>
buildTwoQubitInsertChainProgram(MLIRContext* context,
                                const bool reverseInsertOrder,
                                const bool swapInsertTargets) {
  const int64_t q0Target = swapInsertTargets ? 1 : 0;
  const int64_t q1Target = swapInsertTargets ? 0 : 1;

  QCOProgramBuilder builder(context);
  builder.initialize();

  Value q0 = nullptr;
  Value q1 = nullptr;

  auto tensor = builder.qtensorAlloc(2);
  std::tie(tensor, q0) = builder.qtensorExtract(tensor, 0);
  std::tie(tensor, q1) = builder.qtensorExtract(tensor, 1);

  if (reverseInsertOrder) {
    tensor = builder.qtensorInsert(q1, tensor, q1Target);
    tensor = builder.qtensorInsert(q0, tensor, q0Target);
  } else {
    tensor = builder.qtensorInsert(q0, tensor, q0Target);
    tensor = builder.qtensorInsert(q1, tensor, q1Target);
  }

  builder.qtensorDealloc(tensor);
  return builder.finalize();
}

/**
 * @brief Constructs an MLIR module that allocates a two-element qtensor, extracts
 * and reinserts qubits (optionally resetting the second qubit), and then deallocates it.
 *
 * @param context MLIRContext used to create and own the module and operations.
 * @param withReset If `true`, applies a reset to the second extracted qubit before reinserting it.
 * @return OwningOpRef<ModuleOp> The finalized module containing the built program.
 */
static OwningOpRef<ModuleOp>
buildResetWithCommutingInsertProgram(MLIRContext* context,
                                     const bool withReset) {
  QCOProgramBuilder builder(context);
  builder.initialize();

  Value q0 = nullptr;
  Value q1 = nullptr;

  auto tensor = builder.qtensorAlloc(2);
  std::tie(tensor, q0) = builder.qtensorExtract(tensor, 0);
  tensor = builder.qtensorInsert(q0, tensor, 0);
  std::tie(tensor, q1) = builder.qtensorExtract(tensor, 1);
  if (withReset) {
    q1 = builder.reset(q1);
  }
  tensor = builder.qtensorInsert(q1, tensor, 1);

  builder.qtensorDealloc(tensor);
  return builder.finalize();
}

/**
 * @brief Builds an MLIR module that allocates a 2-qubit QTensor, performs a
 * sequence of extracts and inserts where two inserts target the same index,
 * optionally applies a reset to the second extracted qubit, then deallocates.
 *
 * The constructed program:
 * - allocates a 2-element QTensor,
 * - extracts index 0 (q0) and index 1 (q10),
 * - applies H to q10 and reinserts it at index 1,
 * - extracts index 1 again (q11), optionally resets q11 if `withReset` is true,
 * - reinserts q11 at index 1, reinserts q0 at index 0,
 * - deallocates the tensor and finalizes the module.
 *
 * @param context The MLIRContext used to create IR.
 * @param withReset If true, applies a reset to the second extracted qubit
 *                  before reinserting it at the same index.
 * @return OwningOpRef<ModuleOp> The finalized MLIR module containing the
 *         constructed program.
 */
static OwningOpRef<ModuleOp>
buildResetWithSameIndexInsertProgram(MLIRContext* context,
                                     const bool withReset) {
  QCOProgramBuilder builder(context);
  builder.initialize();

  Value q0 = nullptr;
  Value q10 = nullptr;
  Value q11 = nullptr;

  auto tensor = builder.qtensorAlloc(2);
  std::tie(tensor, q0) = builder.qtensorExtract(tensor, 0);
  std::tie(tensor, q10) = builder.qtensorExtract(tensor, 1);
  q10 = builder.h(q10);
  tensor = builder.qtensorInsert(q10, tensor, 1);
  std::tie(tensor, q11) = builder.qtensorExtract(tensor, 1);
  if (withReset) {
    q11 = builder.reset(q11);
  }
  tensor = builder.qtensorInsert(q11, tensor, 1);
  tensor = builder.qtensorInsert(q0, tensor, 0);

  builder.qtensorDealloc(tensor);
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
  EXPECT_TRUE(verify(*program).succeeded());
  EXPECT_TRUE(runQCOCleanupPipeline(program.get()).succeeded());
  EXPECT_TRUE(verify(*program).succeeded());

  auto reference = buildTwoQubitInsertChainProgram(context.get(), true, true);
  ASSERT_TRUE(reference);
  EXPECT_TRUE(verify(*reference).succeeded());
  EXPECT_TRUE(runQCOCleanupPipeline(reference.get()).succeeded());
  EXPECT_TRUE(verify(*reference).succeeded());

  EXPECT_FALSE(
      areModulesEquivalentWithPermutations(program.get(), reference.get()));
}

TEST_F(QTensorTest, ResetAfterExtractThroughCommutingInsertIsEliminated) {
  auto program = buildResetWithCommutingInsertProgram(context.get(), true);
  ASSERT_TRUE(program);
  EXPECT_TRUE(verify(*program).succeeded());
  EXPECT_TRUE(runQCOCleanupPipeline(program.get()).succeeded());
  EXPECT_TRUE(verify(*program).succeeded());

  auto reference = buildResetWithCommutingInsertProgram(context.get(), false);
  ASSERT_TRUE(reference);
  EXPECT_TRUE(verify(*reference).succeeded());
  EXPECT_TRUE(runQCOCleanupPipeline(reference.get()).succeeded());
  EXPECT_TRUE(verify(*reference).succeeded());

  EXPECT_TRUE(
      areModulesEquivalentWithPermutations(program.get(), reference.get()));
}

/**
 * @brief Verifies that a reset performed after extracting a qubit and reinserting
 * it at the same index is preserved by the QCO cleanup pipeline.
 *
 * Builds a program that performs an extract, reinserts at the same index, then
 * applies a reset; canonicalizes it. Builds a reference program without the
 * reset and canonicalizes it. Asserts that the two canonicalized modules are
 * not equivalent up to qubit permutations.
 */
TEST_F(QTensorTest, ResetAfterExtractThroughSameIndexInsertIsNotEliminated) {
  auto program = buildResetWithSameIndexInsertProgram(context.get(), true);
  ASSERT_TRUE(program);
  EXPECT_TRUE(verify(*program).succeeded());
  EXPECT_TRUE(runQCOCleanupPipeline(program.get()).succeeded());
  EXPECT_TRUE(verify(*program).succeeded());

  auto reference = buildResetWithSameIndexInsertProgram(context.get(), false);
  ASSERT_TRUE(reference);
  EXPECT_TRUE(verify(*reference).succeeded());
  EXPECT_TRUE(runQCOCleanupPipeline(reference.get()).succeeded());
  EXPECT_TRUE(verify(*reference).succeeded());

  EXPECT_FALSE(
      areModulesEquivalentWithPermutations(program.get(), reference.get()));
}

// ============================================================================
// Integration
// ============================================================================

struct QTensorIntegrationTestCase {
  std::string name;
  mqt::test::NamedBuilder<QCOProgramBuilder> programBuilder;
  mqt::test::NamedBuilder<QCOProgramBuilder> referenceBuilder;

  friend std::ostream& operator<<(std::ostream& os,
                                  const QTensorIntegrationTestCase& info);
};

/**
 * @brief Formats a QTensorIntegrationTestCase as `QTensor{<name>}` for streaming.
 *
 * @param os The output stream to write to.
 * @param info The test case whose name will be inserted into the formatted output.
 * @return std::ostream& Reference to the output stream after writing.
 */
std::ostream& operator<<(std::ostream& os,
                         const QTensorIntegrationTestCase& info) {
  return os << "QTensor{" << info.name << "}";
}

class QTensorIntegrationTest
    : public testing::TestWithParam<QTensorIntegrationTestCase> {
protected:
  std::unique_ptr<MLIRContext> context;

  /**
   * @brief Initialize the MLIRContext and register/load the dialects required by the tests.
   *
   * This sets up a fresh MLIRContext, appends a registry containing QCODialect,
   * arith::ArithDialect, func::FuncDialect, and QTensorDialect, and loads all
   * available dialects into the context.
   */
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
