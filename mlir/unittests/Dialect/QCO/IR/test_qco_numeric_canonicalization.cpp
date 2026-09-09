/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "ExactUnitaryTest.h"
#include "mlir/Dialect/MQT/IR/MQTDialect.h"
#include "mlir/Dialect/MQT/Utils/ConstantFolding.h"
#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QCO/QCOUtils.h"

#include <gtest/gtest.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/StringRef.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/Passes.h>

using namespace mlir;
using namespace mlir::qco;

namespace {

class QCONumericCanonicalizationTest : public testing::Test {
protected:
  MLIRContext context_;

  void SetUp() override {
    context_.loadDialect<mlir::mqt::MQTDialect, QCODialect, arith::ArithDialect,
                         func::FuncDialect>();
  }

  OwningOpRef<ModuleOp> canonicalize(StringRef source) {
    auto moduleOp = parseSourceString<ModuleOp>(source, &context_);
    EXPECT_TRUE(moduleOp);
    if (!moduleOp) {
      return {};
    }
    EXPECT_TRUE(succeeded(verify(*moduleOp)));
    EXPECT_TRUE(succeeded(verifyLinearity(*moduleOp)));
    PassManager manager(&context_);
    manager.addPass(createCanonicalizerPass());
    EXPECT_TRUE(succeeded(manager.run(*moduleOp)));
    EXPECT_TRUE(succeeded(verify(*moduleOp)));
    EXPECT_TRUE(succeeded(verifyLinearity(*moduleOp)));
    return moduleOp;
  }
};

TEST_F(QCONumericCanonicalizationTest, LargeEvenPauliPowerBecomesIdentity) {
  auto moduleOp = canonicalize(R"mlir(
    module {
      func.func @test(%q: !qco.qubit) -> !qco.qubit {
        %exponent = arith.constant 10000.0 : f64
        %o = qco.pow(%exponent) (%a = %q) {
          %x = qco.x %a : !qco.qubit -> !qco.qubit
          qco.yield %x : !qco.qubit
        } : {!qco.qubit} -> {!qco.qubit}
        return %o : !qco.qubit
      }
    }
  )mlir");
  ASSERT_TRUE(moduleOp);
  auto function = moduleOp->lookupSymbol<func::FuncOp>("test");
  EXPECT_TRUE(function.getBody().getOps<PowOp>().empty());
  EXPECT_TRUE(function.getBody().getOps<GPhaseOp>().empty());
  auto returned =
      cast<func::ReturnOp>(function.getBody().front().getTerminator());
  EXPECT_EQ(returned.getOperand(0), function.getArgument(0));
}

TEST_F(QCONumericCanonicalizationTest, LargeOddPauliPowerPreservesUnitary) {
  auto moduleOp = canonicalize(R"mlir(
    module {
      func.func @test(%q: !qco.qubit) -> !qco.qubit {
        %exponent = arith.constant 9007199254740991.0 : f64
        %o = qco.pow(%exponent) (%a = %q) {
          %x = qco.x %a : !qco.qubit -> !qco.qubit
          qco.yield %x : !qco.qubit
        } : {!qco.qubit} -> {!qco.qubit}
        return %o : !qco.qubit
      }
    }
  )mlir");
  auto reference = parseSourceString<ModuleOp>(R"mlir(
    module {
      func.func @test(%q: !qco.qubit) -> !qco.qubit {
        %x = qco.x %q : !qco.qubit -> !qco.qubit
        return %x : !qco.qubit
      }
    }
  )mlir",
                                               &context_);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(reference);
  auto function = moduleOp->lookupSymbol<func::FuncOp>("test");
  EXPECT_TRUE(function.getBody().getOps<PowOp>().empty());
  EXPECT_TRUE(llvm::hasSingleElement(function.getBody().getOps<XOp>()));
  ::mqt::test::expectFullUnitaryEqual(*reference, *moduleOp, 1);
}

TEST_F(QCONumericCanonicalizationTest, PowerRetainsDynamicAngle) {
  auto moduleOp = canonicalize(R"mlir(
    module {
      func.func @test(%q: !qco.qubit, %angle: f64) -> !qco.qubit {
        %exponent = arith.constant 2.0 : f64
        %o = qco.pow(%exponent) (%a = %q) {
          %r = qco.rx(%angle) %a : !qco.qubit -> !qco.qubit
          qco.yield %r : !qco.qubit
        } : {!qco.qubit} -> {!qco.qubit}
        return %o : !qco.qubit
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

TEST_F(QCONumericCanonicalizationTest, PowerRetainsUnsafeConstantProducts) {
  auto moduleOp = canonicalize(R"mlir(
    module {
      func.func @overflow(%q: !qco.qubit) -> !qco.qubit {
        %exponent = arith.constant 2.0 : f64
        %angle = arith.constant 1.0e308 : f64
        %o = qco.pow(%exponent) (%a = %q) {
          %r = qco.rx(%angle) %a : !qco.qubit -> !qco.qubit
          qco.yield %r : !qco.qubit
        } : {!qco.qubit} -> {!qco.qubit}
        return %o : !qco.qubit
      }
      func.func @rounding(%q: !qco.qubit) -> !qco.qubit {
        %exponent = arith.constant 3.0 : f64
        %angle = arith.constant 1.0000000000000002e16 : f64
        %o = qco.pow(%exponent) (%a = %q) {
          %r = qco.rx(%angle) %a : !qco.qubit -> !qco.qubit
          qco.yield %r : !qco.qubit
        } : {!qco.qubit} -> {!qco.qubit}
        return %o : !qco.qubit
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

TEST_F(QCONumericCanonicalizationTest, PowerRetainsOutOfRangeGlobalPhase) {
  auto moduleOp = canonicalize(R"mlir(
    module {
      func.func @test(%q: !qco.qubit) -> !qco.qubit {
        %exponent = arith.constant 2.0 : f64
        %angle = arith.constant 8000.0 : f64
        %o = qco.pow(%exponent) (%a = %q) {
          qco.gphase(%angle)
          qco.yield %a : !qco.qubit
        } : {!qco.qubit} -> {!qco.qubit}
        return %o : !qco.qubit
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

TEST_F(QCONumericCanonicalizationTest, NestedPowerOverflowKeepsParameterScope) {
  auto moduleOp = canonicalize(R"mlir(
    module {
      func.func @test(%q: !qco.qubit, %angle: f64) -> !qco.qubit {
        %exponent = arith.constant 1.0e308 : f64
        %o = qco.pow(%exponent) (%a = %q) {
          %square = arith.mulf %angle, %angle : f64
          %i = qco.pow(%exponent) (%b = %a) {
            %h = qco.h %b : !qco.qubit -> !qco.qubit
            %r = qco.rx(%square) %h : !qco.qubit -> !qco.qubit
            qco.yield %r : !qco.qubit
          } : {!qco.qubit} -> {!qco.qubit}
          qco.yield %i : !qco.qubit
        } : {!qco.qubit} -> {!qco.qubit}
        return %o : !qco.qubit
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

TEST_F(QCONumericCanonicalizationTest, NestedPowerRetainsDynamicExponentScope) {
  auto moduleOp = canonicalize(R"mlir(
    module {
      func.func @test(%q: !qco.qubit, %exponent: f64) -> !qco.qubit {
        %two = arith.constant 2.0 : f64
        %o = qco.pow(%two) (%a = %q) {
          %square = arith.mulf %exponent, %exponent : f64
          %i = qco.pow(%square) (%b = %a) {
            %h = qco.h %b : !qco.qubit -> !qco.qubit
            %s = qco.s %h : !qco.qubit -> !qco.qubit
            qco.yield %s : !qco.qubit
          } : {!qco.qubit} -> {!qco.qubit}
          qco.yield %i : !qco.qubit
        } : {!qco.qubit} -> {!qco.qubit}
        return %o : !qco.qubit
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

TEST_F(QCONumericCanonicalizationTest, RotationMergeRetainsOverflowingAngles) {
  auto moduleOp = canonicalize(R"mlir(
    module {
      func.func @test(%q: !qco.qubit) -> !qco.qubit {
        %angle = arith.constant 1.0e308 : f64
        %a = qco.rx(%angle) %q : !qco.qubit -> !qco.qubit
        %b = qco.rx(%angle) %a : !qco.qubit -> !qco.qubit
        return %b : !qco.qubit
      }
    }
  )mlir");
  ASSERT_TRUE(moduleOp);
  auto function = moduleOp->lookupSymbol<func::FuncOp>("test");
  EXPECT_EQ(llvm::range_size(function.getBody().getOps<RXOp>()), 2U);
}

TEST_F(QCONumericCanonicalizationTest, RotationMergeRetainsLargeRoundingError) {
  constexpr StringLiteral source = R"mlir(
    module {
      func.func @test(%q: !qco.qubit) -> !qco.qubit {
        %large = arith.constant 1.0e16 : f64
        %small = arith.constant 1.0 : f64
        %a = qco.rx(%large) %q : !qco.qubit -> !qco.qubit
        %b = qco.rx(%small) %a : !qco.qubit -> !qco.qubit
        return %b : !qco.qubit
      }
    }
  )mlir";
  auto original = parseSourceString<ModuleOp>(source, &context_);
  auto moduleOp = canonicalize(source);
  ASSERT_TRUE(original);
  ASSERT_TRUE(moduleOp);
  auto function = moduleOp->lookupSymbol<func::FuncOp>("test");
  EXPECT_EQ(llvm::range_size(function.getBody().getOps<RXOp>()), 2U);
  ::mqt::test::expectFullUnitaryEqual(*original, *moduleOp, 1);
}

TEST_F(QCONumericCanonicalizationTest, RotationMergeRetainsDynamicAngles) {
  auto moduleOp = canonicalize(R"mlir(
    module {
      func.func @test(%q: !qco.qubit, %theta: f64, %phi: f64) -> !qco.qubit {
        %a = qco.rx(%theta) %q : !qco.qubit -> !qco.qubit
        %b = qco.rx(%phi) %a : !qco.qubit -> !qco.qubit
        return %b : !qco.qubit
      }
    }
  )mlir");
  ASSERT_TRUE(moduleOp);
  auto function = moduleOp->lookupSymbol<func::FuncOp>("test");
  EXPECT_EQ(llvm::range_size(function.getBody().getOps<RXOp>()), 2U);
  EXPECT_TRUE(function.getBody().getOps<arith::AddFOp>().empty());
}

TEST_F(QCONumericCanonicalizationTest,
       RotationMergeCombinesSafeConstantAngles) {
  constexpr StringLiteral source = R"mlir(
    module {
      func.func @test(%q: !qco.qubit) -> !qco.qubit {
        %theta = arith.constant 0.1 : f64
        %phi = arith.constant 0.2 : f64
        %a = qco.rx(%theta) %q : !qco.qubit -> !qco.qubit
        %b = qco.rx(%phi) %a : !qco.qubit -> !qco.qubit
        return %b : !qco.qubit
      }
    }
  )mlir";
  auto original = parseSourceString<ModuleOp>(source, &context_);
  auto moduleOp = canonicalize(source);
  ASSERT_TRUE(original);
  ASSERT_TRUE(moduleOp);
  auto function = moduleOp->lookupSymbol<func::FuncOp>("test");
  auto rotations = function.getBody().getOps<RXOp>();
  ASSERT_TRUE(llvm::hasSingleElement(rotations));
  auto rotation = *rotations.begin();
  auto angle = mlir::mqt::valueToDouble(rotation.getTheta());
  ASSERT_TRUE(angle);
  EXPECT_NEAR(*angle, 0.3, 1e-15);
  ::mqt::test::expectFullUnitaryEqual(*original, *moduleOp, 1);
}

TEST_F(QCONumericCanonicalizationTest, NestedPowerAcrossBranchCutDoesNotMerge) {
  auto moduleOp = canonicalize(R"mlir(
    module {
      func.func @test(%q: !qco.qubit) -> !qco.qubit {
        %half = arith.constant 0.5 : f64
        %two = arith.constant 2.0 : f64
        %o = qco.pow(%half) (%a = %q) {
          %i = qco.pow(%two) (%b = %a) {
            %x = qco.x %b : !qco.qubit -> !qco.qubit
            %s = qco.s %x : !qco.qubit -> !qco.qubit
            qco.yield %s : !qco.qubit
          } : {!qco.qubit} -> {!qco.qubit}
          qco.yield %i : !qco.qubit
        } : {!qco.qubit} -> {!qco.qubit}
        return %o : !qco.qubit
      }
    }
  )mlir");
  auto reference = parseSourceString<ModuleOp>(R"mlir(
    module {
      func.func @test(%q: !qco.qubit) -> !qco.qubit {
        %angle = arith.constant 0.7853981633974483 : f64
        qco.gphase(%angle)
        return %q : !qco.qubit
      }
    }
  )mlir",
                                               &context_);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(reference);
  size_t powerCount = 0;
  moduleOp->walk([&](PowOp) { ++powerCount; });
  EXPECT_EQ(powerCount, 2U);
  // (S X)^2 = i I, whose principal square root is exp(i pi/4) I.
  ::mqt::test::expectFullUnitaryEqual(*reference, *moduleOp, 1);
}

} // namespace
