/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/Conversion/QCToQCO/QCToQCO.h"
#include "mqt/Dialect/MQT/IR/MQTDialect.h"
#include "mqt/Dialect/MQT/Utils/ConstantFolding.h"
#include "mqt/Dialect/QC/IR/QCDialect.h"
#include "mqt/Dialect/QC/IR/QCOps.h"
#include "mqt/Dialect/QCO/IR/QCODialect.h"
#include "mqt/Dialect/QCO/QCOUtils.h"

#include "ExactUnitaryTest.h"

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

#include <numbers>

using namespace mlir;
using namespace mlir::qc;

namespace {

class QCModifierCanonicalizationTest : public testing::Test {
protected:
  MLIRContext context_;

  void SetUp() override {
    context_.loadDialect<mlir::mqt::MQTDialect, QCDialect, qco::QCODialect,
                         arith::ArithDialect, func::FuncDialect>();
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

TEST_F(QCModifierCanonicalizationTest, InverseU2PreservesLargeAngles) {
  constexpr StringLiteral source = R"mlir(
    module {
      func.func @test(%q: !qc.qubit) {
        %phi = arith.constant 1.0e16 : f64
        %lambda = arith.constant 0.0 : f64
        qc.inv(%a = %q) {
          qc.u2(%phi, %lambda) %a : !qc.qubit
          qc.yield
        } : !qc.qubit
        return
      }
    }
  )mlir";
  auto moduleOp = canonicalize(source);
  ASSERT_TRUE(moduleOp);
  auto function = moduleOp->lookupSymbol<func::FuncOp>("test");
  EXPECT_TRUE(function.getBody().getOps<InvOp>().empty());
  auto gates = function.getBody().getOps<UOp>();
  ASSERT_TRUE(llvm::hasSingleElement(gates));
  auto gate = *gates.begin();
  const auto theta = mlir::mqt::valueToDouble(gate.getTheta());
  const auto phi = mlir::mqt::valueToDouble(gate.getPhi());
  const auto lambda = mlir::mqt::valueToDouble(gate.getLambda());
  ASSERT_TRUE(theta);
  ASSERT_TRUE(phi);
  ASSERT_TRUE(lambda);
  EXPECT_DOUBLE_EQ(*theta, -std::numbers::pi / 2.0);
  EXPECT_DOUBLE_EQ(*phi, 0.0);
  EXPECT_DOUBLE_EQ(*lambda, -1.0e16);
}

TEST_F(QCModifierCanonicalizationTest, InverseU2PreservesDynamicAngles) {
  auto moduleOp = canonicalize(R"mlir(
    module {
      func.func @test(%q: !qc.qubit, %phi: f64, %lambda: f64) {
        qc.inv(%a = %q) {
          qc.u2(%phi, %lambda) %a : !qc.qubit
          qc.yield
        } : !qc.qubit
        return
      }
    }
  )mlir");
  ASSERT_TRUE(moduleOp);
  auto function = moduleOp->lookupSymbol<func::FuncOp>("test");
  EXPECT_TRUE(function.getBody().getOps<InvOp>().empty());
  auto gates = function.getBody().getOps<UOp>();
  ASSERT_TRUE(llvm::hasSingleElement(gates));
  auto gate = *gates.begin();
  auto phi = gate.getPhi().getDefiningOp<arith::NegFOp>();
  auto lambda = gate.getLambda().getDefiningOp<arith::NegFOp>();
  ASSERT_TRUE(phi);
  ASSERT_TRUE(lambda);
  EXPECT_EQ(phi.getOperand(), function.getArgument(2));
  EXPECT_EQ(lambda.getOperand(), function.getArgument(1));
  EXPECT_TRUE(function.getBody().getOps<arith::AddFOp>().empty());
  EXPECT_TRUE(function.getBody().getOps<arith::SubFOp>().empty());
}

TEST_F(QCModifierCanonicalizationTest, InverseU2PreservesFullUnitary) {
  auto moduleOp = canonicalize(R"mlir(
    module {
      func.func @test(%q: !qc.qubit) attributes {mqt.entry_point} {
        %phi = arith.constant 2.5745926535897929 : f64
        %lambda = arith.constant -3.3755926535897931 : f64
        qc.inv(%a = %q) {
          qc.u2(%phi, %lambda) %a : !qc.qubit
          qc.yield
        } : !qc.qubit
        return
      }
    }
  )mlir");
  ASSERT_TRUE(moduleOp);
  PassManager converter(&context_);
  converter.addPass(createQCToQCO());
  ASSERT_TRUE(succeeded(converter.run(*moduleOp)));
  ASSERT_TRUE(succeeded(verify(*moduleOp)));
  ASSERT_TRUE(succeeded(qco::verifyLinearity(*moduleOp)));
  auto reference = parseSourceString<ModuleOp>(R"mlir(
    module {
      func.func @test(%q: !qco.qubit) -> !qco.qubit {
        %phi = arith.constant 0.234 : f64
        %lambda = arith.constant 0.567 : f64
        %u = qco.u2(%phi, %lambda) %q : !qco.qubit -> !qco.qubit
        return %u : !qco.qubit
      }
    }
  )mlir",
                                               &context_);
  ASSERT_TRUE(reference);
  ::mqt::test::expectFullUnitaryEqual(*reference, *moduleOp, 1);
}

TEST_F(QCModifierCanonicalizationTest,
       InverseControlledU2PreservesFullUnitary) {
  auto moduleOp = canonicalize(R"mlir(
    module {
      func.func @test(%q0: !qc.qubit, %q1: !qc.qubit, %q2: !qc.qubit)
          attributes {mqt.entry_point} {
        %phi = arith.constant 2.5745926535897929 : f64
        %lambda = arith.constant -3.3755926535897931 : f64
        qc.inv(%a = %q0, %b = %q1, %c = %q2) {
          qc.ctrl(%a, %b) targets(%d = %c) {
            qc.u2(%phi, %lambda) %d : !qc.qubit
            qc.yield
          } : {!qc.qubit, !qc.qubit}, {!qc.qubit}
          qc.yield
        } : !qc.qubit, !qc.qubit, !qc.qubit
        return
      }
    }
  )mlir");
  ASSERT_TRUE(moduleOp);
  PassManager converter(&context_);
  converter.addPass(createQCToQCO());
  ASSERT_TRUE(succeeded(converter.run(*moduleOp)));
  ASSERT_TRUE(succeeded(verify(*moduleOp)));
  ASSERT_TRUE(succeeded(qco::verifyLinearity(*moduleOp)));
  auto reference = parseSourceString<ModuleOp>(R"mlir(
    module {
      func.func @test(%q0: !qco.qubit, %q1: !qco.qubit, %q2: !qco.qubit)
          -> (!qco.qubit, !qco.qubit, !qco.qubit) {
        %phi = arith.constant 0.234 : f64
        %lambda = arith.constant 0.567 : f64
        %o0, %o1, %o2 = qco.ctrl(%q0, %q1) targets(%a = %q2) {
          %u = qco.u2(%phi, %lambda) %a : !qco.qubit -> !qco.qubit
          qco.yield %u : !qco.qubit
        } : ({!qco.qubit, !qco.qubit}, {!qco.qubit}) -> ({!qco.qubit, !qco.qubit}, {!qco.qubit})
        return %o0, %o1, %o2 : !qco.qubit, !qco.qubit, !qco.qubit
      }
    }
  )mlir",
                                               &context_);
  ASSERT_TRUE(reference);
  ::mqt::test::expectFullUnitaryEqual(*reference, *moduleOp, 3);
}

TEST_F(QCModifierCanonicalizationTest, ControlledPhaseDropsUnusedTargets) {
  auto moduleOp = canonicalize(R"mlir(
    module {
      func.func @test(%c0: !qc.qubit, %c1: !qc.qubit, %q: !qc.qubit, %angle: f64) {
        qc.ctrl(%c0, %c1) targets(%t = %q) {
          qc.gphase(%angle)
          qc.yield
        } : {!qc.qubit, !qc.qubit}, {!qc.qubit}
        return
      }
    }
  )mlir");
  ASSERT_TRUE(moduleOp);
  auto function = moduleOp->lookupSymbol<func::FuncOp>("test");
  auto controls = function.getBody().getOps<CtrlOp>();
  ASSERT_TRUE(llvm::hasSingleElement(controls));
  auto ctrl = *controls.begin();
  ASSERT_EQ(ctrl.getNumControls(), 1U);
  ASSERT_EQ(ctrl.getNumTargets(), 1U);
  EXPECT_EQ(ctrl.getControls().front(), function.getArgument(0));
  EXPECT_EQ(ctrl.getTargets().front(), function.getArgument(1));
  auto phases = ctrl.getBody()->getOps<POp>();
  ASSERT_TRUE(llvm::hasSingleElement(phases));
  auto phase = *phases.begin();
  EXPECT_EQ(phase.getTarget(0), ctrl.getBody()->getArgument(0));
  EXPECT_EQ(phase.getTheta(), function.getArgument(3));
}

TEST_F(QCModifierCanonicalizationTest,
       ControlledPhaseHoistsParameterArithmetic) {
  auto moduleOp = canonicalize(R"mlir(
    module {
      func.func @test(%c0: !qc.qubit, %c1: !qc.qubit, %angle: f64) {
        qc.ctrl(%c0, %c1) targets() {
          %square = arith.mulf %angle, %angle : f64
          qc.gphase(%square)
          qc.yield
        } : {!qc.qubit, !qc.qubit}
        return
      }
    }
  )mlir");
  ASSERT_TRUE(moduleOp);
  auto function = moduleOp->lookupSymbol<func::FuncOp>("test");
  auto ctrl = *function.getBody().getOps<CtrlOp>().begin();
  EXPECT_EQ(ctrl.getNumControls(), 1U);
  auto phase = *ctrl.getBody()->getOps<POp>().begin();
  auto square = phase.getTheta().getDefiningOp<arith::MulFOp>();
  ASSERT_TRUE(square);
  EXPECT_EQ(square->getBlock(), &function.getBody().front());
  EXPECT_EQ(square.getLhs(), function.getArgument(2));
  EXPECT_EQ(square.getRhs(), function.getArgument(2));
}

TEST_F(QCModifierCanonicalizationTest, ZeroControlsInlineMultipleGates) {
  auto moduleOp = canonicalize(R"mlir(
    module {
      func.func @test(%q0: !qc.qubit, %q1: !qc.qubit, %angle: f64) {
        "qc.ctrl"(%q0, %q1) <{operandSegmentSizes = array<i32: 0, 2>}> ({
        ^bb0(%a: !qc.qubit, %b: !qc.qubit):
          %square = arith.mulf %angle, %angle : f64
          qc.h %b : !qc.qubit
          qc.rx(%square) %b : !qc.qubit
          qc.yield
        }) : (!qc.qubit, !qc.qubit) -> ()
        return
      }
    }
  )mlir");
  ASSERT_TRUE(moduleOp);
  auto function = moduleOp->lookupSymbol<func::FuncOp>("test");
  EXPECT_TRUE(function.getBody().getOps<CtrlOp>().empty());
  auto h = *function.getBody().getOps<HOp>().begin();
  auto rx = *function.getBody().getOps<RXOp>().begin();
  EXPECT_EQ(h.getTarget(0), function.getArgument(1));
  EXPECT_EQ(rx.getTarget(0), function.getArgument(1));
  EXPECT_TRUE(h->isBeforeInBlock(rx));
  EXPECT_TRUE(rx.getTheta().getDefiningOp<arith::MulFOp>());
}

TEST_F(QCModifierCanonicalizationTest, NestedControlsHoistParameterArithmetic) {
  auto moduleOp = canonicalize(R"mlir(
    module {
      func.func @test(%c0: !qc.qubit, %c1: !qc.qubit, %q: !qc.qubit, %angle: f64) {
        qc.ctrl(%c0) targets(%a = %c1, %b = %q) {
          %square = arith.mulf %angle, %angle : f64
          qc.ctrl(%a) targets(%t = %b) {
            qc.rx(%square) %t : !qc.qubit
            qc.yield
          } : {!qc.qubit}, {!qc.qubit}
          qc.yield
        } : {!qc.qubit}, {!qc.qubit, !qc.qubit}
        return
      }
    }
  )mlir");
  ASSERT_TRUE(moduleOp);
  auto function = moduleOp->lookupSymbol<func::FuncOp>("test");
  auto ctrl = *function.getBody().getOps<CtrlOp>().begin();
  ASSERT_EQ(ctrl.getNumControls(), 2U);
  ASSERT_EQ(ctrl.getNumTargets(), 1U);
  EXPECT_EQ(ctrl.getControls()[0], function.getArgument(0));
  EXPECT_EQ(ctrl.getControls()[1], function.getArgument(1));
  EXPECT_EQ(ctrl.getTargets().front(), function.getArgument(2));
  auto rx = *ctrl.getBody()->getOps<RXOp>().begin();
  auto square = rx.getTheta().getDefiningOp<arith::MulFOp>();
  ASSERT_TRUE(square);
  EXPECT_EQ(square->getBlock(), &function.getBody().front());
  EXPECT_EQ(square.getLhs(), function.getArgument(3));
}

TEST_F(QCModifierCanonicalizationTest, NestedControlsRetainOtherBodyGates) {
  auto moduleOp = canonicalize(R"mlir(
    module {
      func.func @test(%c0: !qc.qubit, %c1: !qc.qubit, %q: !qc.qubit) {
        qc.ctrl(%c0) targets(%a = %c1, %b = %q) {
          qc.h %a : !qc.qubit
          qc.ctrl(%a) targets(%t = %b) {
            qc.x %t : !qc.qubit
            qc.yield
          } : {!qc.qubit}, {!qc.qubit}
          qc.yield
        } : {!qc.qubit}, {!qc.qubit, !qc.qubit}
        return
      }
    }
  )mlir");
  ASSERT_TRUE(moduleOp);
  auto function = moduleOp->lookupSymbol<func::FuncOp>("test");
  auto ctrl = *function.getBody().getOps<CtrlOp>().begin();
  EXPECT_EQ(ctrl.getNumControls(), 1U);
  EXPECT_EQ(ctrl.getNumBodyUnitaries(), 2U);
  EXPECT_TRUE(llvm::hasSingleElement(ctrl.getBody()->getOps<HOp>()));
  EXPECT_TRUE(llvm::hasSingleElement(ctrl.getBody()->getOps<CtrlOp>()));
}

TEST_F(QCModifierCanonicalizationTest, DoubleInverseInlinesMultipleGates) {
  auto moduleOp = canonicalize(R"mlir(
    module {
      func.func @test(%q0: !qc.qubit, %q1: !qc.qubit, %angle: f64) {
        qc.inv (%a = %q0, %b = %q1) {
          %square = arith.mulf %angle, %angle : f64
          qc.inv (%t = %b) {
            qc.h %t : !qc.qubit
            qc.rx(%square) %t : !qc.qubit
            qc.yield
          } : !qc.qubit
          qc.yield
        } : !qc.qubit, !qc.qubit
        return
      }
    }
  )mlir");
  ASSERT_TRUE(moduleOp);
  auto function = moduleOp->lookupSymbol<func::FuncOp>("test");
  EXPECT_TRUE(function.getBody().getOps<InvOp>().empty());
  auto h = *function.getBody().getOps<HOp>().begin();
  auto rx = *function.getBody().getOps<RXOp>().begin();
  EXPECT_EQ(h.getTarget(0), function.getArgument(1));
  EXPECT_EQ(rx.getTarget(0), function.getArgument(1));
  EXPECT_TRUE(h->isBeforeInBlock(rx));
  EXPECT_TRUE(rx.getTheta().getDefiningOp<arith::MulFOp>());
}

TEST_F(QCModifierCanonicalizationTest, SingleInverseRetainsMultipleGates) {
  auto moduleOp = canonicalize(R"mlir(
    module {
      func.func @test(%q: !qc.qubit, %angle: f64) {
        qc.inv (%a = %q) {
          qc.h %a : !qc.qubit
          qc.rx(%angle) %a : !qc.qubit
          qc.yield
        } : !qc.qubit
        return
      }
    }
  )mlir");
  ASSERT_TRUE(moduleOp);
  auto function = moduleOp->lookupSymbol<func::FuncOp>("test");
  auto inverses = function.getBody().getOps<InvOp>();
  ASSERT_TRUE(llvm::hasSingleElement(inverses));
  auto inverse = *inverses.begin();
  EXPECT_EQ(inverse.getNumBodyUnitaries(), 2U);
}

TEST_F(QCModifierCanonicalizationTest, UPowerPreservesSecondModifierTarget) {
  auto moduleOp = canonicalize(R"mlir(
    module {
      func.func @test(%q0: !qc.qubit, %q1: !qc.qubit) {
        %exponent = arith.constant 2.0 : f64
        %theta = arith.constant 0.1 : f64
        %phi = arith.constant 0.2 : f64
        %lambda = arith.constant 0.3 : f64
        qc.pow(%exponent) (%a = %q0, %b = %q1) {
          qc.u(%theta, %phi, %lambda) %b : !qc.qubit
          qc.yield
        } : !qc.qubit, !qc.qubit
        return
      }
    }
  )mlir");
  ASSERT_TRUE(moduleOp);
  auto function = moduleOp->lookupSymbol<func::FuncOp>("test");
  EXPECT_TRUE(function.getBody().getOps<PowOp>().empty());
  auto gates = function.getBody().getOps<UOp>();
  ASSERT_TRUE(llvm::hasSingleElement(gates));
  auto gate = *gates.begin();
  EXPECT_EQ(gate.getTarget(0), function.getArgument(1));
}

} // namespace
