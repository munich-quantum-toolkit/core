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

#include <numbers>

using namespace mlir;
using namespace mlir::qco;

namespace {

class QCOModifierCanonicalizationTest : public testing::Test {
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

TEST_F(QCOModifierCanonicalizationTest, InverseU2PreservesLargeAngles) {
  constexpr StringLiteral source = R"mlir(
    module {
      func.func @test(%q: !qco.qubit) -> !qco.qubit {
        %phi = arith.constant 1.0e16 : f64
        %lambda = arith.constant 0.0 : f64
        %o = qco.inv(%a = %q) {
          %u = qco.u2(%phi, %lambda) %a : !qco.qubit -> !qco.qubit
          qco.yield %u : !qco.qubit
        } : {!qco.qubit} -> {!qco.qubit}
        return %o : !qco.qubit
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
  auto original = parseSourceString<ModuleOp>(source, &context_);
  ASSERT_TRUE(original);
  ::mqt::test::expectFullUnitaryEqual(*original, *moduleOp, 1);
}

TEST_F(QCOModifierCanonicalizationTest, InverseU2PreservesDynamicAngles) {
  auto moduleOp = canonicalize(R"mlir(
    module {
      func.func @test(%q: !qco.qubit, %phi: f64, %lambda: f64) -> !qco.qubit {
        %o = qco.inv(%a = %q) {
          %u = qco.u2(%phi, %lambda) %a : !qco.qubit -> !qco.qubit
          qco.yield %u : !qco.qubit
        } : {!qco.qubit} -> {!qco.qubit}
        return %o : !qco.qubit
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

TEST_F(QCOModifierCanonicalizationTest, InverseU2PreservesFullUnitary) {
  auto moduleOp = canonicalize(R"mlir(
    module {
      func.func @test(%q: !qco.qubit) -> !qco.qubit {
        %phi = arith.constant 2.5745926535897929 : f64
        %lambda = arith.constant -3.3755926535897931 : f64
        %o = qco.inv(%a = %q) {
          %u = qco.u2(%phi, %lambda) %a : !qco.qubit -> !qco.qubit
          qco.yield %u : !qco.qubit
        } : {!qco.qubit} -> {!qco.qubit}
        return %o : !qco.qubit
      }
    }
  )mlir");
  ASSERT_TRUE(moduleOp);
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

TEST_F(QCOModifierCanonicalizationTest,
       InverseControlledU2PreservesFullUnitary) {
  auto moduleOp = canonicalize(R"mlir(
    module {
      func.func @test(%q0: !qco.qubit, %q1: !qco.qubit, %q2: !qco.qubit)
          -> (!qco.qubit, !qco.qubit, !qco.qubit) {
        %phi = arith.constant 2.5745926535897929 : f64
        %lambda = arith.constant -3.3755926535897931 : f64
        %o0, %o1, %o2 = qco.inv(%a = %q0, %b = %q1, %c = %q2) {
          %i0, %i1, %i2 = qco.ctrl(%a, %b) targets(%d = %c) {
            %u = qco.u2(%phi, %lambda) %d : !qco.qubit -> !qco.qubit
            qco.yield %u : !qco.qubit
          } : ({!qco.qubit, !qco.qubit}, {!qco.qubit}) -> ({!qco.qubit, !qco.qubit}, {!qco.qubit})
          qco.yield %i0, %i1, %i2 : !qco.qubit, !qco.qubit, !qco.qubit
        } : {!qco.qubit, !qco.qubit, !qco.qubit} -> {!qco.qubit, !qco.qubit, !qco.qubit}
        return %o0, %o1, %o2 : !qco.qubit, !qco.qubit, !qco.qubit
      }
    }
  )mlir");
  ASSERT_TRUE(moduleOp);
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

TEST_F(QCOModifierCanonicalizationTest, ControlledPhaseDropsUnusedTargets) {
  auto moduleOp = canonicalize(R"mlir(
    module {
      func.func @test(%c0: !qco.qubit, %c1: !qco.qubit, %q: !qco.qubit, %angle: f64)
          -> (!qco.qubit, !qco.qubit, !qco.qubit) {
        %o0, %o1, %o2 = qco.ctrl(%c0, %c1) targets(%t = %q) {
          qco.gphase(%angle)
          qco.yield %t : !qco.qubit
        } : ({!qco.qubit, !qco.qubit}, {!qco.qubit})
          -> ({!qco.qubit, !qco.qubit}, {!qco.qubit})
        return %o0, %o1, %o2 : !qco.qubit, !qco.qubit, !qco.qubit
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
  EXPECT_EQ(ctrl.getControlsIn().front(), function.getArgument(0));
  EXPECT_EQ(ctrl.getTargetsIn().front(), function.getArgument(1));
  auto phases = ctrl.getBody()->getOps<POp>();
  ASSERT_TRUE(llvm::hasSingleElement(phases));
  auto phase = *phases.begin();
  EXPECT_EQ(phase.getInputTarget(0), ctrl.getBody()->getArgument(0));
  EXPECT_EQ(phase.getTheta(), function.getArgument(3));
  auto returned =
      cast<func::ReturnOp>(function.getBody().front().getTerminator());
  EXPECT_EQ(returned.getOperand(0), ctrl.getControlsOut().front());
  EXPECT_EQ(returned.getOperand(1), ctrl.getTargetsOut().front());
  EXPECT_EQ(returned.getOperand(2), function.getArgument(2));
}

TEST_F(QCOModifierCanonicalizationTest,
       ControlledPhaseHoistsParameterArithmetic) {
  auto moduleOp = canonicalize(R"mlir(
    module {
      func.func @test(%c0: !qco.qubit, %c1: !qco.qubit, %angle: f64)
          -> (!qco.qubit, !qco.qubit) {
        %o0, %o1 = qco.ctrl(%c0, %c1) targets() {
          %square = arith.mulf %angle, %angle : f64
          qco.gphase(%square)
          qco.yield
        } : ({!qco.qubit, !qco.qubit}) -> ({!qco.qubit, !qco.qubit})
        return %o0, %o1 : !qco.qubit, !qco.qubit
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

TEST_F(QCOModifierCanonicalizationTest, ZeroControlsInlineMultipleGates) {
  auto moduleOp = canonicalize(R"mlir(
    module {
      func.func @test(%q0: !qco.qubit, %q1: !qco.qubit, %angle: f64)
          -> (!qco.qubit, !qco.qubit) {
        %o0, %o1 = "qco.ctrl"(%q0, %q1) <{
          operandSegmentSizes = array<i32: 0, 2>,
          resultSegmentSizes = array<i32: 0, 2>
        }> ({
        ^bb0(%a: !qco.qubit, %b: !qco.qubit):
          %square = arith.mulf %angle, %angle : f64
          %h = qco.h %b : !qco.qubit -> !qco.qubit
          %r = qco.rx(%square) %h : !qco.qubit -> !qco.qubit
          qco.yield %a, %r : !qco.qubit, !qco.qubit
        }) : (!qco.qubit, !qco.qubit) -> (!qco.qubit, !qco.qubit)
        return %o0, %o1 : !qco.qubit, !qco.qubit
      }
    }
  )mlir");
  ASSERT_TRUE(moduleOp);
  auto function = moduleOp->lookupSymbol<func::FuncOp>("test");
  EXPECT_TRUE(function.getBody().getOps<CtrlOp>().empty());
  auto h = *function.getBody().getOps<HOp>().begin();
  auto rx = *function.getBody().getOps<RXOp>().begin();
  EXPECT_EQ(h.getInputTarget(0), function.getArgument(1));
  EXPECT_EQ(rx.getInputTarget(0), h.getResult());
  EXPECT_TRUE(rx.getTheta().getDefiningOp<arith::MulFOp>());
  auto returned =
      cast<func::ReturnOp>(function.getBody().front().getTerminator());
  EXPECT_EQ(returned.getOperand(0), function.getArgument(0));
  EXPECT_EQ(returned.getOperand(1), rx.getResult());
}

TEST_F(QCOModifierCanonicalizationTest,
       NestedControlsHoistParameterArithmetic) {
  auto moduleOp = canonicalize(R"mlir(
    module {
      func.func @test(%c0: !qco.qubit, %c1: !qco.qubit, %q: !qco.qubit, %angle: f64)
          -> (!qco.qubit, !qco.qubit, !qco.qubit) {
        %o0, %o1, %o2 = qco.ctrl(%c0) targets(%a = %c1, %b = %q) {
          %square = arith.mulf %angle, %angle : f64
          %i0, %i1 = qco.ctrl(%a) targets(%t = %b) {
            %r = qco.rx(%square) %t : !qco.qubit -> !qco.qubit
            qco.yield %r : !qco.qubit
          } : ({!qco.qubit}, {!qco.qubit}) -> ({!qco.qubit}, {!qco.qubit})
          qco.yield %i0, %i1 : !qco.qubit, !qco.qubit
        } : ({!qco.qubit}, {!qco.qubit, !qco.qubit})
          -> ({!qco.qubit}, {!qco.qubit, !qco.qubit})
        return %o0, %o1, %o2 : !qco.qubit, !qco.qubit, !qco.qubit
      }
    }
  )mlir");
  ASSERT_TRUE(moduleOp);
  auto function = moduleOp->lookupSymbol<func::FuncOp>("test");
  auto ctrl = *function.getBody().getOps<CtrlOp>().begin();
  ASSERT_EQ(ctrl.getNumControls(), 2U);
  ASSERT_EQ(ctrl.getNumTargets(), 1U);
  EXPECT_EQ(ctrl.getControlsIn()[0], function.getArgument(0));
  EXPECT_EQ(ctrl.getControlsIn()[1], function.getArgument(1));
  EXPECT_EQ(ctrl.getTargetsIn().front(), function.getArgument(2));
  auto rx = *ctrl.getBody()->getOps<RXOp>().begin();
  auto square = rx.getTheta().getDefiningOp<arith::MulFOp>();
  ASSERT_TRUE(square);
  EXPECT_EQ(square->getBlock(), &function.getBody().front());
  EXPECT_EQ(square.getLhs(), function.getArgument(3));
  auto returned =
      cast<func::ReturnOp>(function.getBody().front().getTerminator());
  EXPECT_TRUE(llvm::equal(returned.getOperands(), ctrl.getResults()));
}

TEST_F(QCOModifierCanonicalizationTest, NestedControlsRetainOtherBodyGates) {
  auto moduleOp = canonicalize(R"mlir(
    module {
      func.func @test(%c0: !qco.qubit, %c1: !qco.qubit, %q: !qco.qubit)
          -> (!qco.qubit, !qco.qubit, !qco.qubit) {
        %o0, %o1, %o2 = qco.ctrl(%c0) targets(%a = %c1, %b = %q) {
          %h = qco.h %a : !qco.qubit -> !qco.qubit
          %i0, %i1 = qco.ctrl(%h) targets(%t = %b) {
            %x = qco.x %t : !qco.qubit -> !qco.qubit
            qco.yield %x : !qco.qubit
          } : ({!qco.qubit}, {!qco.qubit}) -> ({!qco.qubit}, {!qco.qubit})
          qco.yield %i0, %i1 : !qco.qubit, !qco.qubit
        } : ({!qco.qubit}, {!qco.qubit, !qco.qubit})
          -> ({!qco.qubit}, {!qco.qubit, !qco.qubit})
        return %o0, %o1, %o2 : !qco.qubit, !qco.qubit, !qco.qubit
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

TEST_F(QCOModifierCanonicalizationTest, DoubleInversePreservesOutputOrder) {
  auto moduleOp = canonicalize(R"mlir(
    module {
      func.func @test(%q0: !qco.qubit, %q1: !qco.qubit) -> (!qco.qubit, !qco.qubit) {
        %o0, %o1 = qco.inv (%a = %q0, %b = %q1) {
          %i0, %i1 = qco.inv (%c = %b, %d = %a) {
            %s0, %s1 = qco.iswap %c, %d
              : !qco.qubit, !qco.qubit -> !qco.qubit, !qco.qubit
            qco.yield %s0, %s1 : !qco.qubit, !qco.qubit
          } : {!qco.qubit, !qco.qubit} -> {!qco.qubit, !qco.qubit}
          qco.yield %i1, %i0 : !qco.qubit, !qco.qubit
        } : {!qco.qubit, !qco.qubit} -> {!qco.qubit, !qco.qubit}
        return %o0, %o1 : !qco.qubit, !qco.qubit
      }
    }
  )mlir");
  ASSERT_TRUE(moduleOp);
  auto function = moduleOp->lookupSymbol<func::FuncOp>("test");
  EXPECT_TRUE(function.getBody().getOps<InvOp>().empty());
  auto swaps = function.getBody().getOps<iSWAPOp>();
  ASSERT_TRUE(llvm::hasSingleElement(swaps));
  auto iswap = *swaps.begin();
  EXPECT_EQ(iswap.getInputQubits()[0], function.getArgument(1));
  EXPECT_EQ(iswap.getInputQubits()[1], function.getArgument(0));
  auto returned =
      cast<func::ReturnOp>(function.getBody().front().getTerminator());
  EXPECT_EQ(returned.getOperand(0), iswap.getOutputQubits()[1]);
  EXPECT_EQ(returned.getOperand(1), iswap.getOutputQubits()[0]);
}

TEST_F(QCOModifierCanonicalizationTest, DoubleInverseInlinesMultipleGates) {
  auto moduleOp = canonicalize(R"mlir(
    module {
      func.func @test(%q0: !qco.qubit, %q1: !qco.qubit, %angle: f64)
          -> (!qco.qubit, !qco.qubit) {
        %o0, %o1 = qco.inv (%a = %q0, %b = %q1) {
          %square = arith.mulf %angle, %angle : f64
          %i = qco.inv (%t = %b) {
            %h = qco.h %t : !qco.qubit -> !qco.qubit
            %r = qco.rx(%square) %h : !qco.qubit -> !qco.qubit
            qco.yield %r : !qco.qubit
          } : {!qco.qubit} -> {!qco.qubit}
          qco.yield %a, %i : !qco.qubit, !qco.qubit
        } : {!qco.qubit, !qco.qubit} -> {!qco.qubit, !qco.qubit}
        return %o0, %o1 : !qco.qubit, !qco.qubit
      }
    }
  )mlir");
  ASSERT_TRUE(moduleOp);
  auto function = moduleOp->lookupSymbol<func::FuncOp>("test");
  EXPECT_TRUE(function.getBody().getOps<InvOp>().empty());
  auto h = *function.getBody().getOps<HOp>().begin();
  auto rx = *function.getBody().getOps<RXOp>().begin();
  EXPECT_EQ(h.getInputTarget(0), function.getArgument(1));
  EXPECT_EQ(rx.getInputTarget(0), h.getResult());
  EXPECT_TRUE(rx.getTheta().getDefiningOp<arith::MulFOp>());
  auto returned =
      cast<func::ReturnOp>(function.getBody().front().getTerminator());
  EXPECT_EQ(returned.getOperand(0), function.getArgument(0));
  EXPECT_EQ(returned.getOperand(1), rx.getResult());
}

TEST_F(QCOModifierCanonicalizationTest, SingleInverseRetainsMultipleGates) {
  auto moduleOp = canonicalize(R"mlir(
    module {
      func.func @test(%q: !qco.qubit, %angle: f64) -> !qco.qubit {
        %o = qco.inv (%a = %q) {
          %h = qco.h %a : !qco.qubit -> !qco.qubit
          %r = qco.rx(%angle) %h : !qco.qubit -> !qco.qubit
          qco.yield %r : !qco.qubit
        } : {!qco.qubit} -> {!qco.qubit}
        return %o : !qco.qubit
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

TEST_F(QCOModifierCanonicalizationTest, KnownInversePreservesUnusedQubits) {
  auto moduleOp = canonicalize(R"mlir(
    module {
      func.func @test(%q0: !qco.qubit, %q1: !qco.qubit) -> (!qco.qubit, !qco.qubit) {
        %o0, %o1 = qco.inv (%a = %q0, %b = %q1) {
          %t = qco.t %b : !qco.qubit -> !qco.qubit
          qco.yield %a, %t : !qco.qubit, !qco.qubit
        } : {!qco.qubit, !qco.qubit} -> {!qco.qubit, !qco.qubit}
        return %o0, %o1 : !qco.qubit, !qco.qubit
      }
    }
  )mlir");
  ASSERT_TRUE(moduleOp);
  auto function = moduleOp->lookupSymbol<func::FuncOp>("test");
  EXPECT_TRUE(function.getBody().getOps<InvOp>().empty());
  auto gates = function.getBody().getOps<TdgOp>();
  ASSERT_TRUE(llvm::hasSingleElement(gates));
  auto tdg = *gates.begin();
  EXPECT_EQ(tdg.getInputTarget(0), function.getArgument(1));
  auto returned =
      cast<func::ReturnOp>(function.getBody().front().getTerminator());
  EXPECT_EQ(returned.getOperand(0), function.getArgument(0));
  EXPECT_EQ(returned.getOperand(1), tdg.getResult());
}

} // namespace
