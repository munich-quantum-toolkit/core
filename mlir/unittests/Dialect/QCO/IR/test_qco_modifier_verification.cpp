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
#include "mqt/Dialect/QCO/IR/QCOInterfaces.h"
#include "mqt/Dialect/QCO/IR/QCOOps.h"
#include "mqt/Dialect/QCO/QCOUtils.h"

#include "gtest/gtest.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"

#include <string>

using namespace mlir;

TEST(QCOModifierVerificationTest, RequiresUniqueInputsAndPositionalYields) {
  MLIRContext context;
  context
      .loadDialect<qco::QCODialect, arith::ArithDialect, func::FuncDialect>();
  auto module = parseSourceString<ModuleOp>(R"mlir(
    func.func @test(%q0: !qco.qubit, %q1: !qco.qubit, %q2: !qco.qubit)
        -> (!qco.qubit, !qco.qubit, !qco.qubit) {
      %two = arith.constant 2.0 : f64
      %i0, %i1 = qco.inv(%a = %q0, %b = %q1) {
        %x, %y = qco.swap %b, %a : !qco.qubit, !qco.qubit
        qco.yield %y, %x : !qco.qubit, !qco.qubit
      } : {!qco.qubit, !qco.qubit} -> {!qco.qubit, !qco.qubit}
      %p0, %p1 = qco.pow(%two) (%a = %i0, %b = %i1) {
        %x = qco.s %a : !qco.qubit
        qco.yield %x, %b : !qco.qubit, !qco.qubit
      } : {!qco.qubit, !qco.qubit} -> {!qco.qubit, !qco.qubit}
      %c, %t0, %t1 = qco.ctrl(%q2) targets(%a = %p0, %b = %p1) {
        qco.yield %a, %b : !qco.qubit, !qco.qubit
      } : ({!qco.qubit}, {!qco.qubit, !qco.qubit})
      return %t0, %t1, %c : !qco.qubit, !qco.qubit, !qco.qubit
    }
  )mlir",
                                            &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));
  ASSERT_TRUE(succeeded(qco::verifyLinearity(*module)));
  module->walk([&](qco::YieldOp yield) {
    auto* modifier = yield->getParentOp();
    SCOPED_TRACE(modifier->getName().getStringRef().str());
    auto first = yield.getOperand(0);
    auto second = yield.getOperand(1);
    std::string diagnostic;
    ScopedDiagnosticHandler handler(&context, [&](Diagnostic& message) {
      diagnostic = message.str();
      return success();
    });
    for (auto firstYield : {second, first}) {
      yield->setOperands({firstYield, first});
      diagnostic.clear();
      EXPECT_TRUE(failed(verify(modifier)));
      EXPECT_NE(diagnostic.find("positionally"), std::string::npos);
    }
    yield->setOperands({first, second});

    auto inputs = cast<qco::UnitaryOpInterface>(modifier).getInputQubits();
    auto& input = modifier->getOpOperand(inputs.getBeginOperandIndex() + 1);
    auto saved = input.get();
    input.set(inputs.front());
    diagnostic.clear();
    EXPECT_TRUE(failed(verify(modifier)));
    EXPECT_NE(diagnostic.find("duplicate qubit found"), std::string::npos);
    input.set(saved);
  });
  EXPECT_TRUE(succeeded(verify(*module)));
  EXPECT_TRUE(succeeded(qco::verifyLinearity(*module)));
}

TEST(QCOModifierVerificationTest, ChecksLinearityInsideOwnedModifierBodies) {
  MLIRContext context;
  context
      .loadDialect<qco::QCODialect, arith::ArithDialect, func::FuncDialect>();
  auto moduleOp = parseSourceString<ModuleOp>(R"mlir(
    func.func @main() attributes {mqt.entry_point} {
      %two = arith.constant 2.0 : f64
      %q0 = qco.static 0 : !qco.qubit
      %q1 = qco.static 1 : !qco.qubit
      %control, %target = qco.ctrl(%q0) targets(%a = %q1) {
        %inverse = qco.inv(%b = %a) {
          %power = qco.pow(%two) (%c = %b) {
            %x = qco.x %c : !qco.qubit
            qco.yield %x : !qco.qubit
          } : {!qco.qubit} -> {!qco.qubit}
          qco.yield %power : !qco.qubit
        } : {!qco.qubit} -> {!qco.qubit}
        qco.yield %inverse : !qco.qubit
      } : ({!qco.qubit}, {!qco.qubit})
      qco.sink %control : !qco.qubit
      qco.sink %target : !qco.qubit
      return
    }
  )mlir",
                                              &context);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(verify(*moduleOp)));
  ASSERT_TRUE(succeeded(qco::verifyLinearity(*moduleOp)));

  qco::XOp gate;
  moduleOp->walk([&](qco::XOp op) { gate = op; });
  ASSERT_TRUE(gate);
  auto yield = cast<qco::YieldOp>(*gate.getQubitOut().user_begin());
  // Reusing the input leaves the gate result unused inside nested modifiers.
  yield->setOperand(0, gate.getQubitIn());
  std::string diagnostic;
  ScopedDiagnosticHandler handler(&context, [&](Diagnostic& message) {
    diagnostic += message.str();
    return success();
  });
  EXPECT_TRUE(failed(qco::verifyLinearity(*moduleOp)));
  EXPECT_NE(diagnostic.find("exactly one use"), std::string::npos);

  yield->setOperand(0, gate.getQubitOut());
  EXPECT_TRUE(succeeded(verify(*moduleOp)));
  EXPECT_TRUE(succeeded(qco::verifyLinearity(*moduleOp)));
}

TEST(QCOModifierVerificationTest, RequiresMatchingControlResults) {
  MLIRContext context;
  context.loadDialect<qco::QCODialect, func::FuncDialect>();
  ScopedDiagnosticHandler handler(&context,
                                  [](Diagnostic&) { return success(); });
  EXPECT_FALSE(parseSourceString<ModuleOp>(R"mlir(
    func.func @test(%c: !qco.qubit, %q: !qco.qubit) -> !qco.qubit {
      %out = "qco.ctrl"(%c, %q) <{
        operandSegmentSizes = array<i32: 1, 1>,
        resultSegmentSizes = array<i32: 0, 1>
      }> ({
      ^bb0(%arg: !qco.qubit):
        qco.yield %arg : !qco.qubit
      }) : (!qco.qubit, !qco.qubit) -> !qco.qubit
      return %out : !qco.qubit
    }
  )mlir",
                                           &context));
}

TEST(QCOModifierVerificationTest, RejectsCyclicWireProducers) {
  MLIRContext context;
  context.loadDialect<qco::QCODialect, func::FuncDialect>();
  std::string diagnostic;
  ScopedDiagnosticHandler handler(&context, [&](Diagnostic& message) {
    diagnostic = message.str();
    return success();
  });
  EXPECT_FALSE(parseSourceString<ModuleOp>(R"mlir(
    func.func @test(%q0: !qco.qubit, %q1: !qco.qubit)
        -> (!qco.qubit, !qco.qubit) {
      %r0, %r1 = qco.inv(%a = %q0, %b = %q1) {
        %x = qco.x %y : !qco.qubit
        %y = qco.x %x : !qco.qubit
        qco.yield %y, %b : !qco.qubit, !qco.qubit
      } : {!qco.qubit, !qco.qubit} -> {!qco.qubit, !qco.qubit}
      return %r0, %r1 : !qco.qubit, !qco.qubit
    }
  )mlir",
                                           &context));
  EXPECT_NE(diagnostic.find("positionally"), std::string::npos);
}
