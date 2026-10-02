/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/Dialect/MQT/IR/MQTDialect.h"
#include "mqt/Dialect/QC/IR/QCDialect.h"
#include "mqt/Dialect/QC/IR/QCOps.h"

#include "gtest/gtest.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"

using namespace mlir;
using namespace mlir::qc;

namespace {

class QCNumericCanonicalizationTest : public testing::Test {
protected:
  MLIRContext context_;

  void SetUp() override {
    context_.loadDialect<mlir::mqt::MQTDialect, QCDialect, arith::ArithDialect,
                         func::FuncDialect>();
  }

  OwningOpRef<ModuleOp> canonicalize(StringRef source) {
    auto moduleOp = parseSourceString<ModuleOp>(source, &context_);
    EXPECT_TRUE(moduleOp);
    if (!moduleOp) {
      return {};
    }
    EXPECT_TRUE(succeeded(verify(*moduleOp)));
    PassManager manager(&context_);
    manager.addPass(createCanonicalizerPass());
    EXPECT_TRUE(succeeded(manager.run(*moduleOp)));
    EXPECT_TRUE(succeeded(verify(*moduleOp)));
    return moduleOp;
  }
};

TEST_F(QCNumericCanonicalizationTest, LargeEvenPauliPowerBecomesIdentity) {
  auto moduleOp = canonicalize(R"mlir(
    module {
      func.func @test(%q: !qc.qubit) {
        %exponent = arith.constant 10000.0 : f64
        qc.pow(%exponent) (%a = %q) {
          qc.x %a : !qc.qubit
          qc.yield
        } : !qc.qubit
        return
      }
    }
  )mlir");
  ASSERT_TRUE(moduleOp);
  auto function = moduleOp->lookupSymbol<func::FuncOp>("test");
  EXPECT_TRUE(function.getBody().getOps<PowOp>().empty());
  EXPECT_TRUE(function.getBody().getOps<XOp>().empty());
  EXPECT_TRUE(function.getBody().getOps<RXOp>().empty());
  EXPECT_TRUE(function.getBody().getOps<GPhaseOp>().empty());
}

TEST_F(QCNumericCanonicalizationTest, LargeOddPauliPowerPreservesItsTarget) {
  auto moduleOp = canonicalize(R"mlir(
    module {
      func.func @test(%q: !qc.qubit) {
        %exponent = arith.constant 9007199254740991.0 : f64
        qc.pow(%exponent) (%a = %q) {
          qc.x %a : !qc.qubit
          qc.yield
        } : !qc.qubit
        return
      }
    }
  )mlir");
  ASSERT_TRUE(moduleOp);
  auto function = moduleOp->lookupSymbol<func::FuncOp>("test");
  EXPECT_TRUE(function.getBody().getOps<PowOp>().empty());
  EXPECT_TRUE(function.getBody().getOps<GPhaseOp>().empty());
  auto gates = function.getBody().getOps<XOp>();
  ASSERT_TRUE(llvm::hasSingleElement(gates));
  auto gate = *gates.begin();
  EXPECT_EQ(gate.getTarget(0), function.getArgument(0));
}

TEST_F(QCNumericCanonicalizationTest, PowerRetainsDynamicAngle) {
  auto moduleOp = canonicalize(R"mlir(
    module {
      func.func @test(%q: !qc.qubit, %angle: f64) {
        %exponent = arith.constant 2.0 : f64
        qc.pow(%exponent) (%a = %q) {
          qc.rx(%angle) %a : !qc.qubit
          qc.yield
        } : !qc.qubit
        return
      }
    }
  )mlir");
  ASSERT_TRUE(moduleOp);
  auto function = moduleOp->lookupSymbol<func::FuncOp>("test");
  auto powers = function.getBody().getOps<PowOp>();
  ASSERT_TRUE(llvm::hasSingleElement(powers));
  auto power = *powers.begin();
  auto rotations = power.getBody()->getOps<RXOp>();
  ASSERT_TRUE(llvm::hasSingleElement(rotations));
  auto rotation = *rotations.begin();
  EXPECT_EQ(rotation.getTheta(), function.getArgument(1));
}

TEST_F(QCNumericCanonicalizationTest, PowerRetainsUnsafeConstantProducts) {
  auto moduleOp = canonicalize(R"mlir(
    module {
      func.func @overflow(%q: !qc.qubit) {
        %exponent = arith.constant 2.0 : f64
        %angle = arith.constant 1.0e308 : f64
        qc.pow(%exponent) (%a = %q) {
          qc.rx(%angle) %a : !qc.qubit
          qc.yield
        } : !qc.qubit
        return
      }
      func.func @rounding(%q: !qc.qubit) {
        %exponent = arith.constant 3.0 : f64
        %angle = arith.constant 1.0000000000000002e16 : f64
        qc.pow(%exponent) (%a = %q) {
          qc.rx(%angle) %a : !qc.qubit
          qc.yield
        } : !qc.qubit
        return
      }
    }
  )mlir");
  ASSERT_TRUE(moduleOp);
  for (auto function : moduleOp->getOps<func::FuncOp>()) {
    SCOPED_TRACE(function.getSymName().str());
    EXPECT_TRUE(llvm::hasSingleElement(function.getBody().getOps<PowOp>()));
    EXPECT_TRUE(function.getBody().getOps<RXOp>().empty());
  }
}

TEST_F(QCNumericCanonicalizationTest, PowerRetainsOutOfRangeGlobalPhase) {
  auto moduleOp = canonicalize(R"mlir(
    module {
      func.func @test(%q: !qc.qubit) {
        %exponent = arith.constant 2.0 : f64
        %angle = arith.constant 8000.0 : f64
        qc.pow(%exponent) (%a = %q) {
          qc.gphase(%angle)
          qc.yield
        } : !qc.qubit
        return
      }
    }
  )mlir");
  ASSERT_TRUE(moduleOp);
  auto function = moduleOp->lookupSymbol<func::FuncOp>("test");
  auto powers = function.getBody().getOps<PowOp>();
  ASSERT_TRUE(llvm::hasSingleElement(powers));
  auto power = *powers.begin();
  EXPECT_TRUE(llvm::hasSingleElement(power.getBody()->getOps<GPhaseOp>()));
  EXPECT_TRUE(function.getBody().getOps<GPhaseOp>().empty());
}

TEST_F(QCNumericCanonicalizationTest, NestedPowerOverflowKeepsParameterScope) {
  auto moduleOp = canonicalize(R"mlir(
    module {
      func.func @test(%q: !qc.qubit, %angle: f64) {
        %exponent = arith.constant 1.0e308 : f64
        qc.pow(%exponent) (%a = %q) {
          %square = arith.mulf %angle, %angle : f64
          qc.pow(%exponent) (%b = %a) {
            qc.h %b : !qc.qubit
            qc.rx(%square) %b : !qc.qubit
            qc.yield
          } : !qc.qubit
          qc.yield
        } : !qc.qubit
        return
      }
    }
  )mlir");
  ASSERT_TRUE(moduleOp);
  auto function = moduleOp->lookupSymbol<func::FuncOp>("test");
  auto powers = function.getBody().getOps<PowOp>();
  ASSERT_TRUE(llvm::hasSingleElement(powers));
  auto outer = *powers.begin();
  auto innerPowers = outer.getBody()->getOps<PowOp>();
  ASSERT_TRUE(llvm::hasSingleElement(innerPowers));
  auto inner = *innerPowers.begin();
  auto rotations = inner.getBody()->getOps<RXOp>();
  ASSERT_TRUE(llvm::hasSingleElement(rotations));
  auto rotation = *rotations.begin();
  auto square = rotation.getTheta().getDefiningOp<arith::MulFOp>();
  ASSERT_TRUE(square);
  EXPECT_EQ(square->getBlock(), outer.getBody());
  EXPECT_EQ(square.getLhs(), function.getArgument(1));
  EXPECT_EQ(square.getRhs(), function.getArgument(1));
}

TEST_F(QCNumericCanonicalizationTest, NestedPowerRetainsDynamicExponentScope) {
  auto moduleOp = canonicalize(R"mlir(
    module {
      func.func @test(%q: !qc.qubit, %exponent: f64) {
        %two = arith.constant 2.0 : f64
        qc.pow(%two) (%a = %q) {
          %square = arith.mulf %exponent, %exponent : f64
          qc.pow(%square) (%b = %a) {
            qc.h %b : !qc.qubit
            qc.s %b : !qc.qubit
            qc.yield
          } : !qc.qubit
          qc.yield
        } : !qc.qubit
        return
      }
    }
  )mlir");
  ASSERT_TRUE(moduleOp);
  auto function = moduleOp->lookupSymbol<func::FuncOp>("test");
  auto powers = function.getBody().getOps<PowOp>();
  ASSERT_TRUE(llvm::hasSingleElement(powers));
  auto outer = *powers.begin();
  EXPECT_EQ(outer.getExponentValue(), 2.0);
  auto innerPowers = outer.getBody()->getOps<PowOp>();
  ASSERT_TRUE(llvm::hasSingleElement(innerPowers));
  auto inner = *innerPowers.begin();
  auto square = inner.getExponent().getDefiningOp<arith::MulFOp>();
  ASSERT_TRUE(square);
  EXPECT_EQ(square->getBlock(), outer.getBody());
  EXPECT_EQ(square.getLhs(), function.getArgument(1));
  EXPECT_EQ(square.getRhs(), function.getArgument(1));
}

TEST_F(QCNumericCanonicalizationTest, NestedPowerAcrossBranchCutDoesNotMerge) {
  auto moduleOp = canonicalize(R"mlir(
    module {
      func.func @test(%q: !qc.qubit) {
        %half = arith.constant 0.5 : f64
        %two = arith.constant 2.0 : f64
        qc.pow(%half) (%a = %q) {
          qc.pow(%two) (%b = %a) {
            qc.x %b : !qc.qubit
            qc.s %b : !qc.qubit
            qc.yield
          } : !qc.qubit
          qc.yield
        } : !qc.qubit
        return
      }
    }
  )mlir");
  ASSERT_TRUE(moduleOp);
  size_t powerCount = 0;
  moduleOp->walk([&](PowOp) { ++powerCount; });
  EXPECT_EQ(powerCount, 2U);
}

} // namespace
