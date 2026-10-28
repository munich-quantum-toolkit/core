/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/Dialect/CBit/IR/CBitDialect.h"
#include "mqt/Dialect/MQT/IR/MQTDialect.h"
#include "mqt/Dialect/QC/IR/QCDialect.h"
#include "mqt/Target/OpenQASM/TranslateOpenQASMToQC.h"
#include "mqt/Target/OpenQASM/TranslateQCToOpenQASM.h"

#include "gtest/gtest.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LogicalResult.h"

#include <string>

namespace mqt::test {
namespace {

class OpenQASM2EmissionTest : public ::testing::Test {
protected:
  mlir::MLIRContext context;

  void SetUp() override {
    mlir::DialectRegistry registry;
    registry.insert<mlir::arith::ArithDialect, mlir::cbit::CBitDialect,
                    mlir::func::FuncDialect, mlir::memref::MemRefDialect,
                    mlir::mqt::MQTDialect, mlir::qc::QCDialect,
                    mlir::scf::SCFDialect>();
    context.appendDialectRegistry(registry);
  }
};

TEST_F(OpenQASM2EmissionTest, PreservesMeasurementsResetAndWidths) {
  auto program = mlir::qc::translateOpenQASMToQC(R"(
OPENQASM 2.0;
include "qelib1.inc";
qreg q[3]; creg a[2]; creg b[1];
x q[2]; measure q[2] -> a[0]; reset q[2];
measure q[2] -> b[0]; measure q[2] -> a[0];
)",
                                                 &context);
  ASSERT_TRUE(program);
  auto emitted = mlir::qc::translateQCToOpenQASM2(std::move(program));
  ASSERT_TRUE(mlir::succeeded(emitted));
  EXPECT_TRUE(emitted->starts_with("OPENQASM 2.0;"));
  EXPECT_NE(emitted->find("qreg q[3];"), std::string::npos);
  EXPECT_NE(emitted->find("creg a[2];"), std::string::npos);
  EXPECT_NE(emitted->find("measure q[2] -> a[0];"), std::string::npos);
  EXPECT_NE(emitted->find("reset q[2];"), std::string::npos);
  EXPECT_TRUE(mlir::qc::translateOpenQASMToQC(*emitted, &context));
}

TEST_F(OpenQASM2EmissionTest, PreservesOpaqueSignature) {
  auto program = mlir::qc::translateOpenQASMToQC(
      "OPENQASM 2.0; opaque hop(theta) a,b; qreg q[2]; hop(0.125) q[1],q[0];",
      &context);
  ASSERT_TRUE(program);
  auto emitted = mlir::qc::translateQCToOpenQASM2(std::move(program));
  ASSERT_TRUE(mlir::succeeded(emitted));
  EXPECT_NE(emitted->find("opaque hop(p0) q0, q1;"), std::string::npos);
  EXPECT_NE(emitted->find("hop(0.125) q[1], q[0];"), std::string::npos);
  EXPECT_TRUE(mlir::qc::translateOpenQASMToQC(*emitted, &context));
}

TEST_F(OpenQASM2EmissionTest,
       RejectsUnrepresentableControlAndClassicalAssignments) {
  for (const auto* source : {
           "OPENQASM 3; qubit q; bit c; c = true;",
           "OPENQASM 2.0; include \"qelib1.inc\"; qreg q[1]; creg c[2]; "
           "if (c == 0) { measure q[0] -> c[0]; x q[0]; }",
       }) {
    auto program = mlir::qc::translateOpenQASMToQC(source, &context);
    ASSERT_TRUE(program);
    EXPECT_TRUE(
        mlir::failed(mlir::qc::translateQCToOpenQASM2(std::move(program))));
  }
}

TEST_F(OpenQASM2EmissionTest, PreservesConditionalRegisterMeasurement) {
  auto program = mlir::qc::translateOpenQASMToQC(R"(
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2]; creg c[2];
x q[1]; if(c==0) measure q -> c;
if(c==2) reset q;
if(c==2) x q;
)",
                                                 &context);
  ASSERT_TRUE(program);
  auto emitted = mlir::qc::translateQCToOpenQASM2(std::move(program));
  ASSERT_TRUE(mlir::succeeded(emitted));
  EXPECT_NE(emitted->find("if (c == 0) measure q -> c;"), std::string::npos);
  EXPECT_NE(emitted->find("if (c == 2) reset q[0];"), std::string::npos);
  EXPECT_NE(emitted->find("if (c == 2) x q[1];"), std::string::npos);
  EXPECT_TRUE(mlir::qc::translateOpenQASMToQC(*emitted, &context));
}

TEST_F(OpenQASM2EmissionTest, LowersGateExpressionsAndBroadcasts) {
  auto program = mlir::qc::translateOpenQASMToQC(R"(
OPENQASM 2.0;
include "qelib1.inc";
gate expression(a) q {
  U(sin(a)+cos(a),tan(a)-exp(a),ln(a)+sqrt(a)+a^2) q;
  barrier q;
}
qreg q[2]; qreg r[2]; creg c[2];
expression(pi/4) q;
CX q,r;
if(c==0) expression(0.5) r;
measure r -> c;
)",
                                                 &context);
  ASSERT_TRUE(program);
  auto emitted = mlir::qc::translateQCToOpenQASM2(std::move(program));
  ASSERT_TRUE(mlir::succeeded(emitted));
  EXPECT_TRUE(mlir::qc::translateOpenQASMToQC(*emitted, &context));
}

TEST_F(OpenQASM2EmissionTest, PreservesUnsignedWideCondition) {
  auto program =
      mlir::qc::translateOpenQASMToQC("OPENQASM 2.0; qreg q[1]; creg c[65]; "
                                      "if(c==18446744073709551616) reset q;",
                                      &context);
  ASSERT_TRUE(program);
  auto emitted = mlir::qc::translateQCToOpenQASM2(std::move(program));
  ASSERT_TRUE(mlir::succeeded(emitted));
  EXPECT_NE(emitted->find("if (c == 18446744073709551616) reset q[0];"),
            std::string::npos);
  EXPECT_TRUE(mlir::qc::translateOpenQASMToQC(*emitted, &context));
}

TEST_F(OpenQASM2EmissionTest, RejectsStaleCondition) {
  auto program = mlir::parseSourceString<mlir::ModuleOp>(R"(
module {
  func.func @main() -> !cbit.reg<2> attributes {mqt.entry_point} {
    %q = qc.static 0 : !qc.qubit
    %c = cbit.alloc(#cbit.init<zero>) : !cbit.reg<2>
    %value = cbit.read %c : !cbit.reg<2> -> i2
    %zero = arith.constant 0 : i2
    %index = arith.constant 0 : index
    %bit = qc.measure %q : !qc.qubit -> i1
    cbit.store %bit, %c[%index] : !cbit.reg<2>
    %condition = arith.cmpi eq, %value, %zero : i2
    scf.if %condition { qc.reset %q : !qc.qubit }
    return %c : !cbit.reg<2>
  }
}
)",
                                                         &context);
  ASSERT_TRUE(program);
  std::string diagnostics;
  mlir::ScopedDiagnosticHandler handler(&context,
                                        [&](mlir::Diagnostic& diagnostic) {
                                          diagnostics += diagnostic.str();
                                          return mlir::success();
                                        });
  EXPECT_TRUE(
      mlir::failed(mlir::qc::translateQCToOpenQASM2(std::move(program))));
  EXPECT_NE(diagnostics.find("stale register value"), std::string::npos);
}

TEST_F(OpenQASM2EmissionTest, RejectsMeasurementCopies) {
  auto program = mlir::parseSourceString<mlir::ModuleOp>(R"(
module {
  func.func @main() -> !cbit.reg<2> attributes {mqt.entry_point} {
    %q = qc.static 0 : !qc.qubit
    %c = cbit.alloc(#cbit.init<zero>) : !cbit.reg<2>
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %bit = qc.measure %q : !qc.qubit -> i1
    cbit.store %bit, %c[%zero] : !cbit.reg<2>
    cbit.store %bit, %c[%one] : !cbit.reg<2>
    return %c : !cbit.reg<2>
  }
}
)",
                                                         &context);
  ASSERT_TRUE(program);
  EXPECT_TRUE(
      mlir::failed(mlir::qc::translateQCToOpenQASM2(std::move(program))));
}

} // namespace
} // namespace mqt::test
