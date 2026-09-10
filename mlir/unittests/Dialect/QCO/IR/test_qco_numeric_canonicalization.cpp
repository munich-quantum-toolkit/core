/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/Dialect/MQT/Utils/ConstantFolding.h"
#include "mqt/Dialect/QCO/IR/QCODialect.h"
#include "mqt/Dialect/QCO/IR/QCOOps.h"
#include "mqt/Dialect/QCO/QCOUtils.h"
#include "mqt/Dialect/QCO/Utils/Matrix.h"

#include "ExactUnitaryTest.h"
#include "gtest/gtest.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
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

#include <array>
#include <cmath>
#include <numbers>

using namespace mlir;
using namespace mlir::qco;

namespace {

class QCONumericCanonicalizationTest : public testing::Test {
protected:
  MLIRContext context_;

  void SetUp() override {
    context_.loadDialect<QCODialect, arith::ArithDialect, func::FuncDialect>();
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

TEST_F(QCONumericCanonicalizationTest, RMatrixPreservesLargeAxisAngles) {
  constexpr double theta = 0.3;
  for (double phi : std::array{1.0e16, 1.0e308}) {
    SCOPED_TRACE(phi);
    const double cosine = std::cos(theta / 2.0);
    const double sine = std::sin(theta / 2.0);
    const double x = std::cos(phi);
    const double y = std::sin(phi);
    // exp(-i theta (x X + y Y) / 2), with x^2 + y^2 = 1.
    const auto expected =
        Matrix2x2::fromElements(cosine, qco::Complex{-sine * y, -sine * x},
                                qco::Complex{sine * y, -sine * x}, cosine);
    const auto matrix = ROp::unitaryMatrix(theta, phi);
    EXPECT_TRUE(matrix.isApprox(expected, 1e-15));
    EXPECT_TRUE((matrix.adjoint() * matrix).isIdentity(1e-15));
  }
}

TEST_F(QCONumericCanonicalizationTest, RMergePreservesLargeAxisAngles) {
  constexpr StringLiteral source = R"mlir(
    module {
      func.func @test(%q: !qco.qubit) -> !qco.qubit {
        %theta0 = arith.constant 0.1 : f64
        %theta1 = arith.constant 0.2 : f64
        %phi = arith.constant 1.0e16 : f64
        %a = qco.r(%theta0, %phi) %q : !qco.qubit -> !qco.qubit
        %b = qco.r(%theta1, %phi) %a : !qco.qubit -> !qco.qubit
        return %b : !qco.qubit
      }
    }
  )mlir";
  for (double phi : std::array{1.0e16, 1.0e308}) {
    SCOPED_TRACE(phi);
    auto original = parseSourceString<ModuleOp>(source, &context_);
    ASSERT_TRUE(original);
    auto originalFunction = original->lookupSymbol<func::FuncOp>("test");
    auto firstRotation = *originalFunction.getBody().getOps<ROp>().begin();
    auto axis = firstRotation.getPhi().getDefiningOp<arith::ConstantOp>();
    OpBuilder builder(&context_);
    axis.setValueAttr(builder.getF64FloatAttr(phi));
    ASSERT_TRUE(succeeded(verify(*original)));
    ASSERT_TRUE(succeeded(verifyLinearity(*original)));
    OwningOpRef<ModuleOp> rewritten(original->clone());
    PassManager manager(&context_);
    manager.addPass(createCanonicalizerPass());
    ASSERT_TRUE(succeeded(manager.run(*rewritten)));
    ASSERT_TRUE(succeeded(verify(*rewritten)));
    ASSERT_TRUE(succeeded(verifyLinearity(*rewritten)));

    auto function = rewritten->lookupSymbol<func::FuncOp>("test");
    auto rotations = function.getBody().getOps<ROp>();
    ASSERT_TRUE(llvm::hasSingleElement(rotations));
    auto rotation = *rotations.begin();
    auto angle = mlir::mqt::valueToDouble(rotation.getTheta());
    ASSERT_TRUE(angle);
    EXPECT_NEAR(*angle, 0.3, 1e-15);
    EXPECT_EQ(mlir::mqt::valueToDouble(rotation.getPhi()), phi);
    auto matrix = rotation.getUnitaryMatrix();
    ASSERT_TRUE(matrix);
    const auto expected =
        ROp::unitaryMatrix(0.2, phi) * ROp::unitaryMatrix(0.1, phi);
    EXPECT_TRUE(matrix->isApprox(expected, 1e-15));
    ::mqt::test::expectFullUnitaryEqual(*original, *rewritten, 1);
  }
}

TEST_F(QCONumericCanonicalizationTest, RPowerPreservesLargeAxisAngles) {
  constexpr StringLiteral source = R"mlir(
    module {
      func.func @test(%q: !qco.qubit) -> !qco.qubit {
        %exponent = arith.constant 2.0 : f64
        %theta = arith.constant 0.1 : f64
        %phi = arith.constant 1.0e16 : f64
        %o = qco.pow(%exponent) (%a = %q) {
          %r = qco.r(%theta, %phi) %a : !qco.qubit -> !qco.qubit
          qco.yield %r : !qco.qubit
        } : {!qco.qubit} -> {!qco.qubit}
        return %o : !qco.qubit
      }
    }
  )mlir";
  for (double phi : std::array{1.0e16, 1.0e308}) {
    SCOPED_TRACE(phi);
    auto original = parseSourceString<ModuleOp>(source, &context_);
    ASSERT_TRUE(original);
    auto originalFunction = original->lookupSymbol<func::FuncOp>("test");
    auto power = *originalFunction.getBody().getOps<PowOp>().begin();
    auto bodyRotation = *power.getBody()->getOps<ROp>().begin();
    auto axis = bodyRotation.getPhi().getDefiningOp<arith::ConstantOp>();
    OpBuilder builder(&context_);
    axis.setValueAttr(builder.getF64FloatAttr(phi));
    ASSERT_TRUE(succeeded(verify(*original)));
    ASSERT_TRUE(succeeded(verifyLinearity(*original)));
    OwningOpRef<ModuleOp> rewritten(original->clone());
    PassManager manager(&context_);
    manager.addPass(createCanonicalizerPass());
    ASSERT_TRUE(succeeded(manager.run(*rewritten)));
    ASSERT_TRUE(succeeded(verify(*rewritten)));
    ASSERT_TRUE(succeeded(verifyLinearity(*rewritten)));

    auto function = rewritten->lookupSymbol<func::FuncOp>("test");
    EXPECT_TRUE(function.getBody().getOps<PowOp>().empty());
    auto rotations = function.getBody().getOps<ROp>();
    ASSERT_TRUE(llvm::hasSingleElement(rotations));
    auto rotation = *rotations.begin();
    EXPECT_EQ(mlir::mqt::valueToDouble(rotation.getTheta()), 0.2);
    EXPECT_EQ(mlir::mqt::valueToDouble(rotation.getPhi()), phi);
    auto matrix = rotation.getUnitaryMatrix();
    ASSERT_TRUE(matrix);
    const auto bodyMatrix = ROp::unitaryMatrix(0.1, phi);
    EXPECT_TRUE(matrix->isApprox(bodyMatrix * bodyMatrix, 1e-15));
    ::mqt::test::expectFullUnitaryEqual(*original, *rewritten, 1);
  }
}

TEST_F(QCONumericCanonicalizationTest, UMatricesPreserveLargeEulerAngles) {
  for (auto [phi, lambda] : std::array{
           std::array{1.0e16, 1.0},
           std::array{1.0, 1.0e16},
           std::array{1.0e16, -1.0e16},
           std::array{1.0e308, 1.0e308},
           std::array{-1.0e308, 1.0e308},
       }) {
    SCOPED_TRACE(testing::Message() << "phi=" << phi << ", lambda=" << lambda);
    // U(theta, phi, lambda) = P(phi) RY(theta) P(lambda).
    for (double theta : std::array{0.3, std::numbers::pi / 2.0}) {
      SCOPED_TRACE(theta);
      const auto expected = POp::unitaryMatrix(phi) *
                            RYOp::unitaryMatrix(theta) *
                            POp::unitaryMatrix(lambda);
      const auto matrix = UOp::unitaryMatrix(theta, phi, lambda);
      EXPECT_TRUE(matrix.isApprox(expected, 1e-15));
      EXPECT_TRUE((matrix.adjoint() * matrix).isIdentity(1e-15));
    }
    const auto expected = POp::unitaryMatrix(phi) *
                          RYOp::unitaryMatrix(std::numbers::pi / 2.0) *
                          POp::unitaryMatrix(lambda);
    const auto matrix = U2Op::unitaryMatrix(phi, lambda);
    EXPECT_TRUE(matrix.isApprox(expected, 1e-15));
    EXPECT_TRUE((matrix.adjoint() * matrix).isIdentity(1e-15));
  }
}

TEST_F(QCONumericCanonicalizationTest, UToU2PreservesLargeEulerAngles) {
  constexpr StringLiteral source = R"mlir(
    module {
      func.func @test(%q: !qco.qubit) -> !qco.qubit {
        %theta = arith.constant 1.5707963267948966 : f64
        %phi = arith.constant 1.0e308 : f64
        %lambda = arith.constant 1.0e308 : f64
        %out = qco.u(%theta, %phi, %lambda) %q : !qco.qubit -> !qco.qubit
        return %out : !qco.qubit
      }
    }
  )mlir";
  auto original = parseSourceString<ModuleOp>(source, &context_);
  ASSERT_TRUE(original);
  ASSERT_TRUE(succeeded(verify(*original)));
  ASSERT_TRUE(succeeded(verifyLinearity(*original)));
  auto originalFunction = original->lookupSymbol<func::FuncOp>("test");
  auto sourceGate = *originalFunction.getBody().getOps<UOp>().begin();
  auto expected = sourceGate.getUnitaryMatrix();
  ASSERT_TRUE(expected);
  EXPECT_TRUE((expected->adjoint() * *expected).isIdentity(1e-15));

  auto rewritten = canonicalize(source);
  ASSERT_TRUE(rewritten);
  auto function = rewritten->lookupSymbol<func::FuncOp>("test");
  EXPECT_TRUE(function.getBody().getOps<UOp>().empty());
  auto gates = function.getBody().getOps<U2Op>();
  ASSERT_TRUE(llvm::hasSingleElement(gates));
  auto gate = *gates.begin();
  auto matrix = gate.getUnitaryMatrix();
  ASSERT_TRUE(matrix);
  EXPECT_TRUE(matrix->isApprox(*expected, 1e-15));
}

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
