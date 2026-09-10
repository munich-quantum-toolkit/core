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

#include "ExactUnitaryTest.h"
#include "gtest/gtest.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/Passes.h"

#include <cstddef>

using namespace mlir;
using namespace mlir::qco;

namespace {

class QCOWireCanonicalizationTest : public testing::Test {
protected:
  MLIRContext context_;

  void SetUp() override {
    context_.loadDialect<QCODialect, arith::ArithDialect, func::FuncDialect>();
  }

  OwningOpRef<ModuleOp> twoQubitFunction() {
    return parseSourceString<ModuleOp>(R"mlir(
      module {
        func.func @main(%q0: !qco.qubit, %q1: !qco.qubit)
            -> (!qco.qubit, !qco.qubit) {
          return %q0, %q1 : !qco.qubit, !qco.qubit
        }
      }
    )mlir",
                                       &context_);
  }

  void canonicalize(ModuleOp moduleOp) {
    ASSERT_TRUE(succeeded(verify(moduleOp)));
    ASSERT_TRUE(succeeded(verifyLinearity(moduleOp)));
    PassManager manager(&context_);
    manager.addPass(createCanonicalizerPass());
    ASSERT_TRUE(succeeded(manager.run(moduleOp)));
    ASSERT_TRUE(succeeded(verify(moduleOp)));
    ASSERT_TRUE(succeeded(verifyLinearity(moduleOp)));
  }

  template <typename GateOp>
  void checkMerge(bool reverseTargets, double secondBeta,
                  size_t expectedGates) {
    auto original = twoQubitFunction();
    ASSERT_TRUE(original);
    auto function = original->lookupSymbol<func::FuncOp>("main");
    auto returned = cast<func::ReturnOp>(function.getBody().front().back());
    OpBuilder builder(returned);
    auto first =
        GateOp::create(builder, function.getLoc(), function.getArgument(0),
                       function.getArgument(1), 0.045, 0.456);
    auto second = GateOp::create(builder, function.getLoc(),
                                 first.getOutputQubit(reverseTargets ? 1 : 0),
                                 first.getOutputQubit(reverseTargets ? 0 : 1),
                                 0.078, secondBeta);
    returned->setOperands(ValueRange{
        second.getOutputQubit(reverseTargets ? 1 : 0),
        second.getOutputQubit(reverseTargets ? 0 : 1),
    });
    ASSERT_TRUE(succeeded(verify(*original)));
    ASSERT_TRUE(succeeded(verifyLinearity(*original)));
    OwningOpRef<ModuleOp> rewritten(original->clone());
    ASSERT_NO_FATAL_FAILURE(canonicalize(*rewritten));

    size_t gateCount = 0;
    rewritten->walk([&](GateOp) { ++gateCount; });
    EXPECT_EQ(gateCount, expectedGates);
    ::mqt::test::expectFullUnitaryEqual(*original, *rewritten, 2);
  }
};

TEST_F(QCOWireCanonicalizationTest, XXPlusYYMergesOrderedWires) {
  checkMerge<XXPlusYYOp>(false, 0.456, 1);
}

TEST_F(QCOWireCanonicalizationTest, XXPlusYYDoesNotMergeReversedWires) {
  checkMerge<XXPlusYYOp>(true, 0.456, 2);
}

TEST_F(QCOWireCanonicalizationTest, XXMinusYYMergesReversedWires) {
  checkMerge<XXMinusYYOp>(true, 0.456, 1);
}

TEST_F(QCOWireCanonicalizationTest, XXPlusYYDoesNotMergeDifferentAxes) {
  checkMerge<XXPlusYYOp>(false, 0.789, 2);
}

TEST_F(QCOWireCanonicalizationTest, XXMinusYYDoesNotMergeDifferentAxes) {
  checkMerge<XXMinusYYOp>(true, 0.789, 2);
}

} // namespace
