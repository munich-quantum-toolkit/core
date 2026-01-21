/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QCO/Builder/QCOProgramBuilder.h"
#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QCO/Transforms/Passes.h"

#include <gtest/gtest.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/LogicalResult.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/PassManager.h>
#include <utility>
#include <vector>

using namespace mlir;
using namespace mlir::qco;

class QCOQuaternionMergeTest : public ::testing::Test {
protected:
  MLIRContext context;
  QCOProgramBuilder builder;
  OwningOpRef<ModuleOp> module;

  struct RotationGate {
    llvm::StringLiteral opName;
    std::vector<double> angles;
  };

  QCOQuaternionMergeTest() : builder(&context) {}

  void SetUp() override {
    context.loadDialect<QCODialect>();
    context.loadDialect<func::FuncDialect>();
    context.loadDialect<arith::ArithDialect>();

    // Setup Builder
    builder.initialize();
  }

  // Counts the ammont of operations the current module/circuit contains
  template <typename OpTy> int countOps() {
    int count = 0;
    module->walk([&](OpTy) { ++count; });
    return count;
  }

  // TODO: create docstring
  // testGateMerge takes a list of Rotation gates and uses the builder api to
  // build a small quantum circuit, where a qubit is feed through all rotations
  // in the list.
  LogicalResult testGateMerge(const std::vector<RotationGate>& rotations) {

    auto q = builder.allocQubitRegister(1);

    Value qubit = q[0];

    for (const auto& gate : rotations) {
      if (gate.opName == RXOp::getOperationName()) {
        qubit = builder.rx(gate.angles[0], qubit);
      } else if (gate.opName == RYOp::getOperationName()) {
        qubit = builder.ry(gate.angles[0], qubit);
      } else if (gate.opName == RZOp::getOperationName()) {
        qubit = builder.rz(gate.angles[0], qubit);
      } else if (gate.opName == UOp::getOperationName()) {
        qubit =
            builder.u(gate.angles[0], gate.angles[1], gate.angles[2], qubit);
      }
    }

    module = builder.finalize();
    return runMergePass(module.get());
  }

  LogicalResult runMergePass(ModuleOp module) {
    PassManager pm(module.getContext());
    pm.addPass(qco::createMergeRotationGates());
    return pm.run(module);
  }
};

// ##################################################
// # Two Gate Merging Tests
// ##################################################
//

// RX->RY should merge into a single U gate
TEST_F(QCOQuaternionMergeTest, quaternionMergeRXRYGates) {
  ASSERT_TRUE(testGateMerge({{RXOp::getOperationName(), {1.}},
                             {RYOp::getOperationName(), {1.}}})
                  .succeeded());
  EXPECT_EQ(countOps<UOp>(), 1);
  EXPECT_EQ(countOps<RXOp>(), 0);
  EXPECT_EQ(countOps<RYOp>(), 0);
}

// RX->RZ  should merge into a single U gate
TEST_F(QCOQuaternionMergeTest, quaternionMergeRXRZGates) {
  ASSERT_TRUE(testGateMerge({{RXOp::getOperationName(), {1.}},
                             {RZOp::getOperationName(), {1.}}})
                  .succeeded());
  EXPECT_EQ(countOps<UOp>(), 1);
  EXPECT_EQ(countOps<RXOp>(), 0);
  EXPECT_EQ(countOps<RZOp>(), 0);
}

// RY->RX should merge into a single U gate
TEST_F(QCOQuaternionMergeTest, quaternionMergeRYRXGates) {
  ASSERT_TRUE(testGateMerge({{RYOp::getOperationName(), {1.}},
                             {RXOp::getOperationName(), {1.}}})
                  .succeeded());
  EXPECT_EQ(countOps<UOp>(), 1);
  EXPECT_EQ(countOps<RYOp>(), 0);
  EXPECT_EQ(countOps<RXOp>(), 0);
}

// RY->RZ should merge into a single U gate
TEST_F(QCOQuaternionMergeTest, quaternionMergeRYRZGates) {
  ASSERT_TRUE(testGateMerge({{RYOp::getOperationName(), {1.}},
                             {RZOp::getOperationName(), {1.}}})
                  .succeeded());
  EXPECT_EQ(countOps<UOp>(), 1);
  EXPECT_EQ(countOps<RYOp>(), 0);
  EXPECT_EQ(countOps<RZOp>(), 0);
}

// RZ->RX should merge into a single U gate
TEST_F(QCOQuaternionMergeTest, quaternionMergeRZRXGates) {
  ASSERT_TRUE(testGateMerge({{RZOp::getOperationName(), {1.}},
                             {RXOp::getOperationName(), {1.}}})
                  .succeeded());
  EXPECT_EQ(countOps<UOp>(), 1);
  EXPECT_EQ(countOps<RZOp>(), 0);
  EXPECT_EQ(countOps<RXOp>(), 0);
}

// RZ->RY should merge into a single U gate
TEST_F(QCOQuaternionMergeTest, quaternionMergeRZRYGates) {
  ASSERT_TRUE(testGateMerge({{RZOp::getOperationName(), {1.}},
                             {RYOp::getOperationName(), {1.}}})
                  .succeeded());
  EXPECT_EQ(countOps<UOp>(), 1);
  EXPECT_EQ(countOps<RZOp>(), 0);
  EXPECT_EQ(countOps<RYOp>(), 0);
}

// U->U should merge into a single U gate
TEST_F(QCOQuaternionMergeTest, quaternionMergeUUGates) {
  ASSERT_TRUE(testGateMerge({{UOp::getOperationName(), {1., 2., .3}},
                             {UOp::getOperationName(), {4., 5., 6.}}})
                  .succeeded());
  EXPECT_EQ(countOps<UOp>(), 1);
}

// U->RX should merge into a single U gate
TEST_F(QCOQuaternionMergeTest, quaternionMergeURXGates) {
  ASSERT_TRUE(testGateMerge({{UOp::getOperationName(), {1., 2., .3}},
                             {RXOp::getOperationName(), {1.}}})
                  .succeeded());
  EXPECT_EQ(countOps<UOp>(), 1);
  EXPECT_EQ(countOps<RXOp>(), 0);
}

// U->RY should merge into a single U gate
TEST_F(QCOQuaternionMergeTest, quaternionMergeURYGates) {
  ASSERT_TRUE(testGateMerge({{UOp::getOperationName(), {1., 2., .3}},
                             {RYOp::getOperationName(), {1.}}})
                  .succeeded());
  EXPECT_EQ(countOps<UOp>(), 1);
  EXPECT_EQ(countOps<RYOp>(), 0);
}

// U->RZ should merge into a single U gate
TEST_F(QCOQuaternionMergeTest, quaternionMergeURZGates) {
  ASSERT_TRUE(testGateMerge({{UOp::getOperationName(), {1., 2., .3}},
                             {RZOp::getOperationName(), {1.}}})
                  .succeeded());
  EXPECT_EQ(countOps<UOp>(), 1);
  EXPECT_EQ(countOps<RZOp>(), 0);
}

// RX->U should merge into a single U gate
TEST_F(QCOQuaternionMergeTest, quaternionMergeRXUGates) {
  ASSERT_TRUE(testGateMerge({{RXOp::getOperationName(), {1.}},
                             {UOp::getOperationName(), {1., 2., .3}}})
                  .succeeded());
  EXPECT_EQ(countOps<RXOp>(), 0);
  EXPECT_EQ(countOps<UOp>(), 1);
}

// RY->U should merge into a single U gate
TEST_F(QCOQuaternionMergeTest, quaternionMergeRYUGates) {
  ASSERT_TRUE(testGateMerge({{RYOp::getOperationName(), {1.}},
                             {UOp::getOperationName(), {1., 2., .3}}})
                  .succeeded());
  EXPECT_EQ(countOps<RYOp>(), 0);
  EXPECT_EQ(countOps<UOp>(), 1);
}

// RZ->U should merge into a single U gate
TEST_F(QCOQuaternionMergeTest, quaternionMergeRZUGates) {
  ASSERT_TRUE(testGateMerge({{RZOp::getOperationName(), {1.}},
                             {UOp::getOperationName(), {1., 2., .3}}})
                  .succeeded());
  EXPECT_EQ(countOps<RZOp>(), 0);
  EXPECT_EQ(countOps<UOp>(), 1);
}
  EXPECT_EQ(countOps<UOp>(), 1);
  EXPECT_EQ(countOps<RZOp>(), 0);
}
