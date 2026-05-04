#include "mlir/Dialect/QCO/Builder/QCOProgramBuilder.h"
#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QCO/Transforms/Passes.h"

#include <gtest/gtest.h>
#include <llvm/Support/Casting.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LLVM.h>

#include <cstddef>

using namespace mlir;
using namespace mlir::qco;

class CancelConsecutiveHermitianGatesTest : public testing::Test {
protected:
  void SetUp() override {
    // Register all necessary dialects
    DialectRegistry registry;
    registry.insert<qco::QCODialect, arith::ArithDialect, func::FuncDialect>();
    context = std::make_unique<MLIRContext>();
    context->appendDialectRegistry(registry);
    context->loadAllAvailableDialects();
  }

  static LogicalResult runPass(OwningOpRef<ModuleOp>& moduleOp) {
    PassManager pm(moduleOp->getContext());
    pm.addPass(createCancelConsecutiveHermitianGates());
    return pm.run(*moduleOp);
  }

  std::unique_ptr<MLIRContext> context;
};

/**
 * @brief Test: Hermitian->Hermitian should be removed (one-qubit).
 */
TEST_F(CancelConsecutiveHermitianGatesTest, OneQubitChainEven) {
  QCOProgramBuilder builder(context.get());
  builder.initialize();

  const auto q00 = builder.allocQubit();

  const auto q01 = builder.x(q00);
  const auto q02 = builder.x(q01);
  EXPECT_TRUE(q02.getDefiningOp()->hasTrait<HermitianTrait>());

  const auto q03 = builder.y(q02);
  const auto q04 = builder.y(q03);
  EXPECT_TRUE(q04.getDefiningOp()->hasTrait<HermitianTrait>());

  const auto q05 = builder.z(q04);
  const auto q06 = builder.z(q05);
  EXPECT_TRUE(q06.getDefiningOp()->hasTrait<HermitianTrait>());

  const auto q07 = builder.h(q06);
  const auto q08 = builder.h(q07);
  EXPECT_TRUE(q08.getDefiningOp()->hasTrait<HermitianTrait>());

  const auto q09 = builder.id(q08);
  const auto q010 = builder.id(q09);
  EXPECT_TRUE(q010.getDefiningOp()->hasTrait<HermitianTrait>());

  builder.sink(q010);

  auto program = builder.finalize();

  runPass(program);

  std::size_t cnt = 0;
  program->walk([&](Operation* op) {
    if (llvm::isa<qco::XOp, qco::YOp, qco::ZOp, qco::HOp, qco::IdOp>(op)) {
      cnt++;
    }
  });

  EXPECT_EQ(cnt, 0);
}

/**
 * @brief Test: Hermitian->Hermitian->Hermitian should remove the first pair
 * (one-qubit).
 */
TEST_F(CancelConsecutiveHermitianGatesTest, OneQubitChainOdd) {
  QCOProgramBuilder builder(context.get());
  builder.initialize();

  const auto q00 = builder.allocQubit();

  const auto q01 = builder.x(q00);
  const auto q02 = builder.x(q01);
  const auto q03 = builder.x(q02);

  builder.sink(q03);

  auto program = builder.finalize();

  runPass(program);

  std::size_t cnt = 0;
  program->walk([&](Operation* op) {
    if (llvm::isa<qco::XOp>(op)) {
      cnt++;
    }
  });

  EXPECT_EQ(cnt, 1);
}

/**
 * @brief Test: Hermitian->Hermitian should be removed on multiple disjoint
 * qubits (one-qubit).
 */
TEST_F(CancelConsecutiveHermitianGatesTest, OneQubitOnDisjointQubits) {
  QCOProgramBuilder builder(context.get());
  builder.initialize();

  const auto q00 = builder.allocQubit();
  const auto q01 = builder.x(q00);
  const auto q02 = builder.x(q01);
  builder.sink(q02);

  const auto q10 = builder.allocQubit();
  const auto q11 = builder.h(q10);
  const auto q12 = builder.h(q11);
  builder.sink(q12);

  auto program = builder.finalize();

  runPass(program);

  std::size_t cnt = 0;
  program->walk([&](Operation* op) {
    if (llvm::isa<qco::XOp, qco::HOp>(op)) {
      cnt++;
    }
  });

  EXPECT_EQ(cnt, 0);
}

/**
 * @brief Test: Don't cancel cancelable gates, if there is another gate in
 * between (one-qubit).
 */
TEST_F(CancelConsecutiveHermitianGatesTest, OneQubitInterrupted) {
  QCOProgramBuilder builder(context.get());
  builder.initialize();

  const auto q00 = builder.allocQubit();

  const auto q01 = builder.x(q00);
  const auto q02 = builder.y(q01);
  const auto q03 = builder.x(q02);
  builder.sink(q03);

  auto program = builder.finalize();

  runPass(program);

  std::size_t cnt = 0;
  program->walk([&](Operation* op) {
    if (llvm::isa<qco::XOp>(op)) {
      cnt++;
    }
  });

  EXPECT_EQ(cnt, 2);
}

/**
 * @brief Test: Hermitian->Hermitian should be removed (two-qubit).
 */
TEST_F(CancelConsecutiveHermitianGatesTest, TwoQubitChainEven) {
  QCOProgramBuilder builder(context.get());
  builder.initialize();

  const auto q00 = builder.allocQubit();
  const auto q10 = builder.allocQubit();

  const auto [q01, q11] = builder.swap(q00, q10);
  EXPECT_TRUE(q01.getDefiningOp()->hasTrait<HermitianTrait>());
  const auto [q02, q12] = builder.swap(q01, q11);

  builder.sink(q02);
  builder.sink(q12);

  auto program = builder.finalize();

  runPass(program);

  std::size_t cnt = 0;
  program->walk([&](Operation* op) {
    if (llvm::isa<qco::SWAPOp>(op)) {
      cnt++;
    }
  });

  EXPECT_EQ(cnt, 0);
}

/**
 * @brief Test: Hermitian->Hermitian->Hermitian should remove the first pair
 * (one-qubit).
 */
TEST_F(CancelConsecutiveHermitianGatesTest, TwoQubitChainOdd) {
  QCOProgramBuilder builder(context.get());
  builder.initialize();

  const auto q00 = builder.allocQubit();
  const auto q10 = builder.allocQubit();

  const auto [q01, q11] = builder.swap(q00, q10);
  const auto [q02, q12] = builder.swap(q01, q11);
  const auto [q03, q13] = builder.swap(q02, q12);

  builder.sink(q03);
  builder.sink(q13);

  auto program = builder.finalize();

  runPass(program);

  std::size_t cnt = 0;
  program->walk([&](Operation* op) {
    if (llvm::isa<qco::SWAPOp>(op)) {
      cnt++;
    }
  });

  EXPECT_EQ(cnt, 1);
}

/**
 * @brief Test: Hermitian->Hermitian should be removed on multiple disjoint
 * qubits (two-qubit).
 */
TEST_F(CancelConsecutiveHermitianGatesTest, TwoQubitOnDisjointQubits) {
  QCOProgramBuilder builder(context.get());
  builder.initialize();

  const auto q00 = builder.allocQubit();
  const auto q10 = builder.allocQubit();
  const auto [q01, q11] = builder.swap(q00, q10);
  const auto [q02, q12] = builder.swap(q01, q11);
  builder.sink(q02);
  builder.sink(q12);

  const auto q20 = builder.allocQubit();
  const auto q30 = builder.allocQubit();
  const auto [q21, q31] = builder.swap(q20, q30);
  const auto [q22, q32] = builder.swap(q21, q31);
  builder.sink(q22);
  builder.sink(q32);

  auto program = builder.finalize();

  runPass(program);

  std::size_t cnt = 0;
  program->walk([&](Operation* op) {
    if (llvm::isa<qco::SWAPOp>(op)) {
      cnt++;
    }
  });

  EXPECT_EQ(cnt, 0);
}

/**
 * @brief Test: Don't cancel cancelable gates, if there is another gate in
 * between (two-qubit).
 */
TEST_F(CancelConsecutiveHermitianGatesTest, TwoQubitInterrupted) {
  QCOProgramBuilder builder(context.get());
  builder.initialize();

  const auto q00 = builder.allocQubit();
  const auto q10 = builder.allocQubit();

  const auto [q01, q11] = builder.swap(q00, q10);
  const auto q02 = builder.h(q01);
  const auto [q03, q12] = builder.swap(q02, q11);

  builder.sink(q03);
  builder.sink(q12);

  auto program = builder.finalize();

  runPass(program);

  std::size_t cnt = 0;
  program->walk([&](Operation* op) {
    if (llvm::isa<qco::SWAPOp>(op)) {
      cnt++;
    }
  });

  EXPECT_EQ(cnt, 2);
}