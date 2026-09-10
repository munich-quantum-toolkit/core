/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/Compiler/Programs.h"
#include "mqt/Dialect/MQT/Utils/ConstantFolding.h"
#include "mqt/Dialect/QCO/IR/QCOOps.h"
#include "mqt/Support/Verification.h"

#include "gtest/gtest.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Verifier.h"

#include <string>

using namespace mlir;

TEST(ProgramParameters, RejectsHiddenConstantsForBothDialects) {
  for (const std::string dialect : {"qc", "qco"}) {
    std::string source = R"mlir(
      module {
        func.func @main(%input: f64) {
          %q = DIALECT.alloc : !DIALECT.qubit
          %infinity = arith.constant 0x7FF0000000000000 : f64
          %theta = arith.addf %input, %infinity : f64
          GATE
          return
        }
      }
    )mlir";
    const std::string gate =
        dialect == "qc"
            ? "qc.rx(%theta) %q : !qc.qubit\nqc.dealloc %q : !qc.qubit"
            : "%out = qco.rx(%theta) %q : !qco.qubit -> !qco.qubit\nqco.sink "
              "%out : !qco.qubit";
    source.replace(source.find("GATE"), 4, gate);
    for (auto position = source.find("DIALECT"); position != std::string::npos;
         position = source.find("DIALECT")) {
      source.replace(position, 7, dialect);
    }
    if (dialect == "qc") {
      EXPECT_FALSE(QCProgram::fromMLIRString(source));
    } else {
      EXPECT_FALSE(QCOProgram::fromMLIRString(source));
    }
  }
}

TEST(ProgramParameters,
     ChecksUnselectedOperandsAndDoesNotCacheAcrossMutations) {
  auto program = QCOProgram::fromMLIRString(R"mlir(
    module {
      func.func @main(%input: f64, %q: !qco.qubit) -> !qco.qubit {
        %finite = arith.constant 1.0 : f64
        %nan = arith.constant 0x7FF8000000000000 : f64
        %condition = arith.constant true
        %selected = arith.select %condition, %finite, %finite : f64
        %q1 = qco.rx(%input) %q : !qco.qubit -> !qco.qubit
        %q2 = qco.rx(%selected) %q1 : !qco.qubit -> !qco.qubit
        return %q2 : !qco.qubit
      }
    }
  )mlir");
  ASSERT_TRUE(program);
  auto moduleOp = program->module();
  arith::SelectOp select;
  Value nan;
  moduleOp.walk([&](arith::SelectOp operation) { select = operation; });
  moduleOp.walk([&](arith::ConstantOp operation) {
    if (auto value = dyn_cast<FloatAttr>(operation.getValue());
        value && !value.getValue().isFinite()) {
      nan = operation;
    }
  });
  ASSERT_TRUE(select);
  ASSERT_TRUE(nan);
  auto finite = select.getFalseValue();
  select.getFalseValueMutable().assign(nan);
  ASSERT_TRUE(succeeded(verify(moduleOp)));
  ASSERT_EQ(mqt::valueToConstantDouble(select.getResult()), 1.0);
  std::string diagnostic;
  ScopedDiagnosticHandler handler(moduleOp.getContext(),
                                  [&](Diagnostic& emitted) {
                                    diagnostic = emitted.str();
                                    return success();
                                  });
  EXPECT_TRUE(failed(mqt::verifyProgramParameters(moduleOp)));
  EXPECT_NE(diagnostic.find("index 0 must be finite"), std::string::npos);
  EXPECT_FALSE(program->cleanup());
  select.getFalseValueMutable().assign(finite);
  EXPECT_TRUE(succeeded(mqt::verifyProgramParameters(moduleOp)));
  EXPECT_TRUE(program->cleanup());
}

TEST(ProgramParameters, HandlesDeepExpressionsAndFoldedOverflow) {
  auto program = QCOProgram::fromMLIRString(R"mlir(
    module {
      func.func @main(%input: f64, %q: !qco.qubit) -> !qco.qubit {
        %q1 = qco.rx(%input) %q : !qco.qubit -> !qco.qubit
        return %q1 : !qco.qubit
      }
    }
  )mlir");
  ASSERT_TRUE(program);
  auto moduleOp = program->module();
  auto function = moduleOp.lookupSymbol<func::FuncOp>("main");
  auto gate = *function.getOps<qco::RXOp>().begin();
  OpBuilder builder(gate);
  auto input = function.getArgument(0);
  Value parameter = input;
  for (int depth = 0; depth < 10000; ++depth) {
    parameter = arith::AddFOp::create(builder, gate.getLoc(), parameter, input);
  }
  gate.getThetaMutable().set(parameter);
  EXPECT_TRUE(succeeded(mqt::verifyProgramParameters(moduleOp)));
  auto large = arith::ConstantOp::create(builder, gate.getLoc(),
                                         builder.getF64FloatAttr(1.0e308));
  auto overflow = arith::AddFOp::create(builder, gate.getLoc(), large, large);
  gate.getThetaMutable().set(overflow);
  ScopedDiagnosticHandler handler(moduleOp.getContext(),
                                  [](Diagnostic&) { return success(); });
  EXPECT_TRUE(failed(mqt::verifyProgramParameters(moduleOp)));
}
