/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QC/Builder/QCProgramBuilder.h"
#include "mlir/Dialect/QC/IR/QCDialect.h"
#include "mlir/Dialect/QC/Translation/TranslateQASM3ToQC.h"
#include "mlir/Dialect/QC/Translation/TranslateQCToOpenQASM3.h"
#include "mlir/Target/OpenQASM/Frontend.h"

#include <gtest/gtest.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Parser/Parser.h>

#include <array>
#include <string>

using namespace mlir;

namespace {

constexpr llvm::StringLiteral BELL = R"qasm(OPENQASM 3.1;
include "stdgates.inc";
qubit[2] q;
h q[0];
ctrl @ x q[0], q[1];
bit[2] c = measure q;
)qasm";

TEST(OpenQASM3EmissionTest, EmitsStrictPortableBellProgram) {
  MLIRContext context;
  auto moduleOp = qc::translateQASM3ToQC(BELL, &context);
  ASSERT_TRUE(moduleOp);

  auto source = qc::translateQCToOpenQASM3(*moduleOp);
  auto repeatedSource = qc::translateQCToOpenQASM3(*moduleOp);

  ASSERT_TRUE(succeeded(source));
  ASSERT_TRUE(succeeded(repeatedSource));
  EXPECT_EQ(*source, *repeatedSource);
  EXPECT_TRUE(source->starts_with("OPENQASM 3.1;\n"));
  EXPECT_NE(source->find("include \"stdgates.inc\";"), std::string::npos);
  EXPECT_NE(source->find("ctrl @ x"), std::string::npos);
  EXPECT_NE(source->find("output bit[2] c;"), std::string::npos);

  const auto analyzed = oq3::frontend::analyzeOpenQASM(
      *source, {.gatePolicy = oq3::frontend::GatePolicy::Strict});
  EXPECT_TRUE(analyzed);
  EXPECT_TRUE(qc::translateQASM3ToQC(*source, &context));
}

TEST(OpenQASM3EmissionTest, PreservesRepresentativeStructuredControl) {
  constexpr llvm::StringLiteral source = R"qasm(OPENQASM 3.1;
include "stdgates.inc";
qubit q;
bit c;
h q;
c = measure q;
if (c) {
  x q;
}
for int i in [0:2] {
  h q;
}
)qasm";
  MLIRContext context;
  auto moduleOp = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(moduleOp);

  auto emitted = qc::translateQCToOpenQASM3(*moduleOp);

  ASSERT_TRUE(succeeded(emitted));
  EXPECT_NE(emitted->find("if ("), std::string::npos);
  EXPECT_NE(emitted->find("for int "), std::string::npos);
  const auto analyzed = oq3::frontend::analyzeOpenQASM(
      *emitted, {.gatePolicy = oq3::frontend::GatePolicy::Strict});
  EXPECT_TRUE(analyzed);
}

TEST(OpenQASM3EmissionTest, EmitsExpressionBasedWhileWithCarriedState) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
  func.func @main() -> (i64 {qc.openqasm.output_kind = "int"})
      attributes {passthrough = ["entry_point"]} {
    %q = qc.alloc : !qc.qubit
    %zero = arith.constant 0 : i64
    %result = scf.while (%count = %zero) : (i64) -> i64 {
      %two = arith.constant 2 : i64
      %condition = arith.cmpi slt, %count, %two : i64
      scf.condition(%condition) %count : i64
    } do {
    ^bb0(%after: i64):
      qc.y %q : !qc.qubit
      %one = arith.constant 1 : i64
      %next = arith.addi %after, %one : i64
      scf.yield %next : i64
    }
    qc.dealloc %q : !qc.qubit
    return %result : i64
  }
}
)mlir";
  DialectRegistry registry;
  registry.insert<arith::ArithDialect, func::FuncDialect, qc::QCDialect,
                  scf::SCFDialect>();
  MLIRContext context(registry);
  auto moduleOp = parseSourceString<ModuleOp>(source, &context);
  ASSERT_TRUE(moduleOp);

  auto emitted = qc::translateQCToOpenQASM3(*moduleOp);

  ASSERT_TRUE(succeeded(emitted));
  EXPECT_NE(emitted->find("while ("), std::string::npos);
  EXPECT_TRUE(oq3::frontend::analyzeOpenQASM(
      *emitted, {.gatePolicy = oq3::frontend::GatePolicy::Strict}))
      << *emitted;
  EXPECT_TRUE(qc::translateQASM3ToQC(*emitted, &context));
}

TEST(OpenQASM3EmissionTest, DefinesEveryNonStandardGate) {
  DialectRegistry registry;
  registry.insert<arith::ArithDialect, func::FuncDialect, memref::MemRefDialect,
                  qc::QCDialect>();
  MLIRContext context(registry);
  context.loadAllAvailableDialects();
  qc::QCProgramBuilder builder(&context);
  builder.initialize();
  const auto qubits = builder.allocQubitRegister(3);
  const auto q0 = qubits[0];
  const auto q1 = qubits[1];
  const auto q2 = qubits[2];

  builder.sxdg(q0)
      .r(0.1, 0.2, q0)
      .u2(0.2, 0.3, q0)
      .u(0.1, 0.2, 0.3, q0)
      .iswap(q0, q1)
      .dcx(q0, q1)
      .ecr(q0, q1)
      .rxx(0.1, q0, q1)
      .ryy(0.2, q0, q1)
      .rzx(0.3, q0, q1)
      .rzz(0.4, q0, q1)
      .xx_plus_yy(0.5, 0.6, q0, q1)
      .xx_minus_yy(0.7, 0.8, q0, q1)
      .rccx(q0, q1, q2);
  builder.ctrl(q0, q1,
               [&](const Value target) { builder.h(target).x(target); });
  builder.inv(q1, [&](const Value target) {
    builder.gphase(0.125).s(target).t(target);
  });
  builder.pow(0.5, q2,
              [&](const Value target) { builder.x(target).y(target); });
  auto moduleOp = builder.finalize();
  ASSERT_TRUE(moduleOp);

  auto source = qc::translateQCToOpenQASM3(*moduleOp);

  ASSERT_TRUE(succeeded(source));
  constexpr std::array helperNames{
      "_mqt_sxdg",        "_mqt_r(",   "_mqt_u2(",  "_mqt_u(",
      "_mqt_iswap",       "_mqt_dcx",  "_mqt_ecr",  "_mqt_rxx(",
      "_mqt_ryy(",        "_mqt_rzx(", "_mqt_rzz(", "_mqt_xx_plus_yy(",
      "_mqt_xx_minus_yy", "_mqt_rccx",
  };
  for (const auto helper : helperNames) {
    EXPECT_NE(source->find(helper), std::string::npos) << helper;
  }
  EXPECT_NE(source->find("gate _mqt_gate"), std::string::npos);
  EXPECT_TRUE(oq3::frontend::analyzeOpenQASM(
      *source, {.gatePolicy = oq3::frontend::GatePolicy::Strict}));
  EXPECT_TRUE(qc::translateQASM3ToQC(*source, &context));
}

TEST(OpenQASM3EmissionTest, ForwardsCapturedCompositeModifierParameters) {
  constexpr llvm::StringLiteral source = R"qasm(OPENQASM 3.1;
include "stdgates.inc";
gate pair(p0) q {
  rx(p0) q;
  rz(p0) q;
}
qubit q;
float theta = 0.25;
inv @ pair(theta) q;
)qasm";
  MLIRContext context;
  auto moduleOp = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(moduleOp);

  auto emitted = qc::translateQCToOpenQASM3(*moduleOp);

  ASSERT_TRUE(succeeded(emitted));
  EXPECT_NE(emitted->find("gate _mqt_gate0(p0)"), std::string::npos);
  EXPECT_NE(emitted->find("inv @ _mqt_gate0("), std::string::npos);
  EXPECT_TRUE(oq3::frontend::analyzeOpenQASM(
      *emitted, {.gatePolicy = oq3::frontend::GatePolicy::Strict}))
      << *emitted;
  EXPECT_TRUE(qc::translateQASM3ToQC(*emitted, &context));
}

TEST(OpenQASM3EmissionTest, MaterializesSelectAndIndexSwitchResults) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
  func.func @main() -> (i64 {qc.openqasm.output_kind = "int"})
      attributes {passthrough = ["entry_point"]} {
    %q = qc.alloc : !qc.qubit
    %condition = arith.constant true
    %one = arith.constant 1 : i64
    %two = arith.constant 2 : i64
    %three = arith.constant 3 : i64
    %selected = arith.select %condition, %one, %two : i64
    %index = arith.index_cast %selected : i64 to index
    %result = scf.index_switch %index -> i64
    case 1 {
      qc.x %q : !qc.qubit
      scf.yield %one : i64
    }
    case 2 {
      qc.y %q : !qc.qubit
      scf.yield %two : i64
    }
    default {
      qc.z %q : !qc.qubit
      scf.yield %three : i64
    }
    qc.dealloc %q : !qc.qubit
    return %result : i64
  }
}
)mlir";
  DialectRegistry registry;
  registry.insert<arith::ArithDialect, func::FuncDialect, qc::QCDialect,
                  scf::SCFDialect>();
  MLIRContext context(registry);
  auto moduleOp = parseSourceString<ModuleOp>(source, &context);
  ASSERT_TRUE(moduleOp);

  auto emitted = qc::translateQCToOpenQASM3(*moduleOp);

  ASSERT_TRUE(succeeded(emitted));
  EXPECT_NE(emitted->find("if (true)"), std::string::npos);
  EXPECT_NE(emitted->find("else {\n  if ("), std::string::npos);
  EXPECT_TRUE(oq3::frontend::analyzeOpenQASM(
      *emitted, {.gatePolicy = oq3::frontend::GatePolicy::Strict}))
      << *emitted;
  EXPECT_TRUE(qc::translateQASM3ToQC(*emitted, &context));
}

TEST(OpenQASM3EmissionTest, EmitsOpenQASMScalarCastExpressions) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
  func.func @main() -> (f64 {qc.openqasm.output_kind = "float"})
      attributes {passthrough = ["entry_point"]} {
    %integer = arith.constant 7 : i64
    %converted = arith.sitofp %integer : i64 to f64
    return %converted : f64
  }
}
)mlir";
  DialectRegistry registry;
  registry.insert<arith::ArithDialect, func::FuncDialect>();
  MLIRContext context(registry);
  auto moduleOp = parseSourceString<ModuleOp>(source, &context);
  ASSERT_TRUE(moduleOp);

  auto emitted = qc::translateQCToOpenQASM3(*moduleOp);

  ASSERT_TRUE(succeeded(emitted));
  EXPECT_NE(emitted->find("float(7)"), std::string::npos);
}

TEST(OpenQASM3EmissionTest, DistinguishesNamedZeroOutputFromCanonicalStatus) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
  func.func @main() ->
      (i64 {qc.openqasm.output_kind = "int",
            qc.openqasm.output_name = "zero"},
       memref<2xi1> {qc.openqasm.output_kind = "bit_array",
                     qc.openqasm.output_name = "bits"})
      attributes {passthrough = ["entry_point"]} {
    %bits = memref.alloc() : memref<2xi1>
    %zero = arith.constant 0 : i64
    return %zero, %bits : i64, memref<2xi1>
  }
}
)mlir";
  DialectRegistry registry;
  registry
      .insert<arith::ArithDialect, func::FuncDialect, memref::MemRefDialect>();
  MLIRContext context(registry);
  auto moduleOp = parseSourceString<ModuleOp>(source, &context);
  ASSERT_TRUE(moduleOp);

  auto emitted = qc::translateQCToOpenQASM3(*moduleOp);

  ASSERT_TRUE(succeeded(emitted));
  EXPECT_NE(emitted->find("output int zero;"), std::string::npos);
  EXPECT_NE(emitted->find("output bit[2] bits;"), std::string::npos);
  EXPECT_NE(emitted->find("zero = 0;"), std::string::npos);
}

TEST(OpenQASM3EmissionTest, LeavesDestinationEmptyOnUnsupportedSafetyOps) {
  MLIRContext context;
  auto moduleOp = qc::translateQASM3ToQC(BELL, &context);
  ASSERT_TRUE(moduleOp);
  auto function = *moduleOp->getOps<func::FuncOp>().begin();
  OpBuilder builder(function.getBody());
  builder.setInsertionPointToStart(&function.getBody().front());
  const auto location = builder.getUnknownLoc();
  auto condition =
      arith::ConstantOp::create(builder, location, builder.getBoolAttr(true));
  cf::AssertOp::create(builder, location, condition,
                       "unsupported safety check");

  std::string output;
  llvm::raw_string_ostream stream(output);
  EXPECT_TRUE(failed(qc::translateQCToOpenQASM3(*moduleOp, stream)));
  EXPECT_TRUE(output.empty());
}

TEST(OpenQASM3EmissionTest, RejectsDynamicMemoryIndices) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
  func.func @main() -> i64 attributes {passthrough = ["entry_point"]} {
    %bits = memref.alloc() : memref<2xi1>
    %condition = arith.constant true
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %index = arith.select %condition, %zero, %one : index
    %value = memref.load %bits[%index] : memref<2xi1>
    memref.store %value, %bits[%index] : memref<2xi1>
    %status = arith.constant 0 : i64
    memref.dealloc %bits : memref<2xi1>
    return %status : i64
  }
}
)mlir";
  DialectRegistry registry;
  registry
      .insert<arith::ArithDialect, func::FuncDialect, memref::MemRefDialect>();
  MLIRContext context(registry);
  auto moduleOp = parseSourceString<ModuleOp>(source, &context);
  ASSERT_TRUE(moduleOp);

  std::string output;
  llvm::raw_string_ostream stream(output);
  EXPECT_TRUE(failed(qc::translateQCToOpenQASM3(*moduleOp, stream)));
  EXPECT_TRUE(output.empty());
}

} // namespace
