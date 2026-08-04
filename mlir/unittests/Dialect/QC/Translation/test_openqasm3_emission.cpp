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
#include "mlir/Dialect/Utils/Utils.h"
#include "mlir/Support/Passes.h"
#include "mlir/Target/OpenQASM/Frontend.h"

#include <gtest/gtest.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Support/LogicalResult.h>

#include <array>
#include <string>
#include <tuple>

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

TEST(OpenQASM3EmissionTest, RetainsOnlyErasedOutputTypeDistinctions) {
  constexpr llvm::StringLiteral source = R"qasm(OPENQASM 3.1;
include "stdgates.inc";
output bit measured;
output bit[1] vector;
output bool flag;
output int signed_value;
output uint unsigned_value;
output float real;
qubit q;
measured = measure q;
vector[0] = measured;
flag = true;
signed_value = 1;
unsigned_value = 1;
real = 1.0;
)qasm";
  MLIRContext context;
  auto moduleOp = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(moduleOp);
  auto function = utils::getEntryPoint(*moduleOp);
  ASSERT_EQ(function.getNumResults(), 6U);

  const auto getResultAttr = [&](const unsigned index,
                                 const llvm::StringRef name) {
    return function.getResultAttrDict(index).getAs<StringAttr>(name);
  };
  constexpr std::array expectedNames{"measured",     "vector",         "flag",
                                     "signed_value", "unsigned_value", "real"};
  for (const auto [index, expected] : llvm::enumerate(expectedNames)) {
    const auto name = getResultAttr(static_cast<unsigned>(index),
                                    utils::OPENQASM_OUTPUT_NAME_ATTR);
    ASSERT_TRUE(name);
    EXPECT_EQ(name.getValue(), expected);
  }

  const auto scalarBit = getResultAttr(0, utils::OPENQASM_OUTPUT_KIND_ATTR);
  const auto unsignedInteger =
      getResultAttr(4, utils::OPENQASM_OUTPUT_KIND_ATTR);
  ASSERT_TRUE(scalarBit);
  ASSERT_TRUE(unsignedInteger);
  EXPECT_EQ(scalarBit.getValue(), "bit");
  EXPECT_EQ(unsignedInteger.getValue(), "uint");
  for (const auto index : {1U, 2U, 3U, 5U}) {
    EXPECT_FALSE(getResultAttr(index, utils::OPENQASM_OUTPUT_KIND_ATTR));
  }

  auto emitted = qc::translateQCToOpenQASM3(*moduleOp);
  ASSERT_TRUE(succeeded(emitted));
  EXPECT_TRUE(qc::translateQASM3ToQC(*emitted, &context));
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
  ASSERT_TRUE(succeeded(runQCCleanupPipeline(*moduleOp)));

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
  EXPECT_NE(source->find("inv @ sx"), std::string::npos);
  EXPECT_NE(source->find("u2("), std::string::npos);
  EXPECT_NE(source->find("U("), std::string::npos);
  EXPECT_EQ(source->find("_mqt_sxdg"), std::string::npos);
  EXPECT_EQ(source->find("_mqt_u2"), std::string::npos);
  EXPECT_EQ(source->find("_mqt_u("), std::string::npos);
  constexpr std::array helperNames{
      "_mqt_r(",          "_mqt_iswap",       "_mqt_dcx",  "_mqt_ecr",
      "_mqt_rxx(",        "_mqt_ryy(",        "_mqt_rzx(", "_mqt_rzz(",
      "_mqt_xx_plus_yy(", "_mqt_xx_minus_yy", "_mqt_rccx",
  };
  for (const auto* const helper : helperNames) {
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
  EXPECT_NE(emitted->find("switch ("), std::string::npos);
  EXPECT_NE(emitted->find("case 1 {"), std::string::npos);
  EXPECT_NE(emitted->find("default {"), std::string::npos);
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

TEST(OpenQASM3EmissionTest, EmitsSupportedScalarCastTargets) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
  func.func @main() ->
      (i64 {qc.openqasm.output_kind = "int"},
       i64 {qc.openqasm.output_kind = "uint"},
       i1 {qc.openqasm.output_kind = "bool"},
       f64 {qc.openqasm.output_kind = "float"},
       f64 {qc.openqasm.output_kind = "float"},
       i64 {qc.openqasm.output_kind = "int"},
       i64 {qc.openqasm.output_kind = "uint"})
      attributes {passthrough = ["entry_point"]} {
    %truth = arith.constant true
    %integer = arith.constant 7 : i64
    %floating = arith.constant 7.0 : f64
    %signedExtended = arith.extsi %truth : i1 to i64
    %unsignedExtended = arith.extui %truth : i1 to i64
    %truncated = arith.trunci %integer : i64 to i1
    %signedFloat = arith.sitofp %integer : i64 to f64
    %unsignedFloat = arith.uitofp %integer : i64 to f64
    %signedInteger = arith.fptosi %floating : f64 to i64
    %unsignedInteger = arith.fptoui %floating : f64 to i64
    return %signedExtended, %unsignedExtended, %truncated, %signedFloat,
           %unsignedFloat, %signedInteger, %unsignedInteger
        : i64, i64, i1, f64, f64, i64, i64
  }
}
)mlir";
  DialectRegistry registry;
  registry.insert<arith::ArithDialect, func::FuncDialect>();
  MLIRContext context(registry);
  auto moduleOp = parseSourceString<ModuleOp>(source, &context);
  ASSERT_TRUE(moduleOp);

  std::string emitted;
  llvm::raw_string_ostream output(emitted);
  ASSERT_TRUE(succeeded(qc::translateQCToOpenQASM3(*moduleOp, output)));
  output.flush();

  EXPECT_NE(emitted.find("bool(7)"), std::string::npos);
  EXPECT_NE(emitted.find("float(7)"), std::string::npos);
  EXPECT_NE(emitted.find("int(7.0)"), std::string::npos);
  EXPECT_NE(emitted.find("uint(7.0)"), std::string::npos);
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

TEST(OpenQASM3EmissionTest, EmitsPhysicalQubitOperations) {
  DialectRegistry registry;
  registry.insert<arith::ArithDialect, func::FuncDialect, qc::QCDialect>();
  MLIRContext context(registry);
  context.loadAllAvailableDialects();
  qc::QCProgramBuilder builder(&context);
  builder.initialize();
  const auto qubit = builder.staticQubit(7);
  builder.h(qubit).reset(qubit).barrier(qubit);
  std::ignore = builder.measure(qubit);
  auto moduleOp = builder.finalize();
  ASSERT_TRUE(moduleOp);

  auto emitted = qc::translateQCToOpenQASM3(*moduleOp);

  ASSERT_TRUE(succeeded(emitted));
  EXPECT_NE(emitted->find("h $7;"), std::string::npos);
  EXPECT_NE(emitted->find("reset $7;"), std::string::npos);
  EXPECT_NE(emitted->find("barrier $7;"), std::string::npos);
  EXPECT_NE(emitted->find("measure $7;"), std::string::npos);
}

TEST(OpenQASM3EmissionTest, EmitsRepresentativeScalarExpressionFamilies) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
  func.func @main() ->
      (i64 {qc.openqasm.output_kind = "int"},
       i64 {qc.openqasm.output_kind = "uint"},
       i1 {qc.openqasm.output_kind = "bool"},
       f64 {qc.openqasm.output_kind = "float"})
      attributes {passthrough = ["entry_point"]} {
    %one = arith.constant 1 : i64
    %two = arith.constant 2 : i64
    %sum = arith.addi %one, %two : i64
    %signed = arith.divsi %sum, %two : i64
    %unsigned = arith.divui %sum, %two : i64
    %comparison = arith.cmpi sge, %sum, %two : i64
    %angle = arith.constant 0.25 : f64
    %negated = arith.negf %angle : f64
    %result = math.sin %negated : f64
    return %signed, %unsigned, %comparison, %result : i64, i64, i1, f64
  }
}
)mlir";
  DialectRegistry registry;
  registry.insert<arith::ArithDialect, func::FuncDialect, math::MathDialect>();
  MLIRContext context(registry);
  auto moduleOp = parseSourceString<ModuleOp>(source, &context);
  ASSERT_TRUE(moduleOp);

  auto emitted = qc::translateQCToOpenQASM3(*moduleOp);

  ASSERT_TRUE(succeeded(emitted));
  EXPECT_NE(emitted->find("((1 + 2) / 2)"), std::string::npos);
  EXPECT_NE(emitted->find("((1 + 2) >= 2)"), std::string::npos);
  EXPECT_NE(emitted->find("sin((-0.25))"), std::string::npos);
}

TEST(OpenQASM3EmissionTest, EmitsStructuredResultsAndLoopCarriedState) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
  func.func @main() ->
      (i64 {qc.openqasm.output_kind = "int"},
       i64 {qc.openqasm.output_kind = "int"},
       i64 {qc.openqasm.output_kind = "int"})
      attributes {passthrough = ["entry_point"]} {
    %condition = arith.constant true
    %one = arith.constant 1 : i64
    %two = arith.constant 2 : i64
    %choice = scf.if %condition -> i64 {
      scf.yield %one : i64
    } else {
      scf.yield %two : i64
    }
    %lower = arith.constant 0 : index
    %upper = arith.constant 5 : index
    %step = arith.constant 2 : index
    %sum = scf.for %index = %lower to %upper step %step
        iter_args(%state = %choice) -> i64 {
      %next = arith.addi %state, %one : i64
      scf.yield %next : i64
    }
    %emptyLower = arith.constant 1 : index
    %emptyUpper = arith.constant 1 : index
    %emptyStep = arith.constant 1 : index
    %unchanged = scf.for %index = %emptyLower to %emptyUpper step %emptyStep
        iter_args(%state = %sum) -> i64 {
      scf.yield %state : i64
    }
    return %choice, %sum, %unchanged : i64, i64, i64
  }
}
)mlir";
  DialectRegistry registry;
  registry.insert<arith::ArithDialect, func::FuncDialect, scf::SCFDialect>();
  MLIRContext context(registry);
  auto moduleOp = parseSourceString<ModuleOp>(source, &context);
  ASSERT_TRUE(moduleOp);

  auto emitted = qc::translateQCToOpenQASM3(*moduleOp);

  ASSERT_TRUE(succeeded(emitted));
  EXPECT_NE(emitted->find("} else {"), std::string::npos);
  EXPECT_NE(emitted->find("in [0:2:4]"), std::string::npos);
  EXPECT_NE(emitted->find("_mqt_next"), std::string::npos);
  EXPECT_TRUE(oq3::frontend::analyzeOpenQASM(
      *emitted, {.gatePolicy = oq3::frontend::GatePolicy::Strict}))
      << *emitted;
}

TEST(OpenQASM3EmissionTest, EmitsIndexSwitchWithoutExplicitCases) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
  func.func @main() -> (i64 {qc.openqasm.output_kind = "int"})
      attributes {passthrough = ["entry_point"]} {
    %index = arith.constant 0 : index
    %one = arith.constant 1 : i64
    %result = scf.index_switch %index -> i64
    default {
      scf.yield %one : i64
    }
    return %result : i64
  }
}
)mlir";
  DialectRegistry registry;
  registry.insert<arith::ArithDialect, func::FuncDialect, scf::SCFDialect>();
  MLIRContext context(registry);
  auto moduleOp = parseSourceString<ModuleOp>(source, &context);
  ASSERT_TRUE(moduleOp);

  auto emitted = qc::translateQCToOpenQASM3(*moduleOp);

  ASSERT_TRUE(succeeded(emitted));
  EXPECT_EQ(emitted->find("switch ("), std::string::npos);
  EXPECT_NE(emitted->find("= 1;"), std::string::npos);
}

TEST(OpenQASM3EmissionTest, InfersObservableScalarKinds) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
  func.func @main() -> (i1, i1, i1, i64, i64, f64)
      attributes {passthrough = ["entry_point"]} {
    %qubit = qc.alloc : !qc.qubit
    %measured = qc.measure %qubit : !qc.qubit -> i1
    %bits = memref.alloc() : memref<1xi1>
    %zeroIndex = arith.constant 0 : index
    %loaded = memref.load %bits[%zeroIndex] : memref<1xi1>
    %one = arith.constant 1 : i64
    %two = arith.constant 2 : i64
    %comparison = arith.cmpi slt, %one, %two : i64
    %signed = arith.divsi %two, %one : i64
    %unsigned = arith.divui %two, %one : i64
    %floating = arith.constant 0.5 : f64
    memref.dealloc %bits : memref<1xi1>
    qc.dealloc %qubit : !qc.qubit
    return %measured, %loaded, %comparison, %signed, %unsigned, %floating
        : i1, i1, i1, i64, i64, f64
  }
}
)mlir";
  DialectRegistry registry;
  registry.insert<arith::ArithDialect, func::FuncDialect, memref::MemRefDialect,
                  qc::QCDialect>();
  MLIRContext context(registry);
  auto moduleOp = parseSourceString<ModuleOp>(source, &context);
  ASSERT_TRUE(moduleOp);

  auto emitted = qc::translateQCToOpenQASM3(*moduleOp);

  ASSERT_TRUE(succeeded(emitted));
  EXPECT_NE(emitted->find("output bit "), std::string::npos);
  EXPECT_NE(emitted->find("output bool "), std::string::npos);
  EXPECT_NE(emitted->find("output int "), std::string::npos);
  EXPECT_NE(emitted->find("output uint "), std::string::npos);
  EXPECT_NE(emitted->find("output float "), std::string::npos);
}

TEST(OpenQASM3EmissionTest, EmitsAllComparisonPredicates) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
  func.func @main() ->
      (i1 {qc.openqasm.output_kind = "bool"},
       i1 {qc.openqasm.output_kind = "bool"},
       i1 {qc.openqasm.output_kind = "bool"},
       i1 {qc.openqasm.output_kind = "bool"},
       i1 {qc.openqasm.output_kind = "bool"},
       i1 {qc.openqasm.output_kind = "bool"},
       i1 {qc.openqasm.output_kind = "bool"},
       i1 {qc.openqasm.output_kind = "bool"},
       i1 {qc.openqasm.output_kind = "bool"},
       i1 {qc.openqasm.output_kind = "bool"},
       i1 {qc.openqasm.output_kind = "bool"},
       i1 {qc.openqasm.output_kind = "bool"})
      attributes {passthrough = ["entry_point"]} {
    %one = arith.constant 1 : i64
    %two = arith.constant 2 : i64
    %oneFloat = arith.constant 1.0 : f64
    %twoFloat = arith.constant 2.0 : f64
    %eq = arith.cmpi eq, %one, %two : i64
    %ne = arith.cmpi ne, %one, %two : i64
    %slt = arith.cmpi slt, %one, %two : i64
    %sle = arith.cmpi sle, %one, %two : i64
    %sgt = arith.cmpi sgt, %one, %two : i64
    %sge = arith.cmpi sge, %one, %two : i64
    %oeq = arith.cmpf oeq, %oneFloat, %twoFloat : f64
    %oneCmp = arith.cmpf one, %oneFloat, %twoFloat : f64
    %olt = arith.cmpf olt, %oneFloat, %twoFloat : f64
    %ole = arith.cmpf ole, %oneFloat, %twoFloat : f64
    %ogt = arith.cmpf ogt, %oneFloat, %twoFloat : f64
    %oge = arith.cmpf oge, %oneFloat, %twoFloat : f64
    return %eq, %ne, %slt, %sle, %sgt, %sge,
           %oeq, %oneCmp, %olt, %ole, %ogt, %oge
        : i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1
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
  for (const auto* const spelling :
       {" == ", " != ", " < ", " <= ", " > ", " >= "}) {
    EXPECT_NE(emitted->find(spelling), std::string::npos) << spelling;
  }
}

TEST(OpenQASM3EmissionTest, DefinesECRWithOneEntanglingGate) {
  DialectRegistry registry;
  registry.insert<arith::ArithDialect, func::FuncDialect, memref::MemRefDialect,
                  qc::QCDialect>();
  MLIRContext context(registry);
  context.loadAllAvailableDialects();
  qc::QCProgramBuilder builder(&context);
  builder.initialize();
  const auto qubits = builder.allocQubitRegister(2);
  builder.ecr(qubits[0], qubits[1]);
  auto moduleOp = builder.finalize();
  ASSERT_TRUE(moduleOp);

  auto emitted = qc::translateQCToOpenQASM3(*moduleOp);

  ASSERT_TRUE(succeeded(emitted));
  EXPECT_EQ(emitted->find("gate _mqt_rzx"), std::string::npos);
  EXPECT_NE(emitted->find("gate _mqt_ecr"), std::string::npos);
  EXPECT_NE(emitted->find("gphase(-pi / 4);"), std::string::npos);
  EXPECT_EQ(llvm::StringRef(*emitted).count("ctrl @ x"), 1U);
}

TEST(OpenQASM3EmissionTest, DefaultsSignlessIntegerOutputsToInt) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
  func.func @main() -> i64 {
    %value = arith.constant 1 : i64
    return %value : i64
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
  EXPECT_NE(emitted->find("output int _mqt_out0;"), std::string::npos);
}

TEST(OpenQASM3EmissionTest, EmitsInternalStorageAndSelectTypes) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
  func.func @main() ->
      (memref<2xi1> {qc.openqasm.output_kind = "bit_array"}) {
    %bits = memref.alloc() : memref<2xi1>
    %condition = arith.constant true
    %floating = arith.constant 1.0 : f64
    %selectedBit =
        arith.select %condition, %condition, %condition : i1
    %selectedFloat =
        arith.select %condition, %floating, %floating : f64
    return %bits : memref<2xi1>
  }
}
)mlir";
  DialectRegistry registry;
  registry
      .insert<arith::ArithDialect, func::FuncDialect, memref::MemRefDialect>();
  MLIRContext context(registry);
  auto moduleOp = parseSourceString<ModuleOp>(source, &context);
  ASSERT_TRUE(moduleOp);
  auto function = *moduleOp->getOps<func::FuncOp>().begin();
  auto allocation = *function.getOps<memref::AllocOp>().begin();
  allocation->setAttr("mqt.classical_register_name",
                      StringAttr::get(&context, "scratch"));

  auto emitted = qc::translateQCToOpenQASM3(*moduleOp);

  ASSERT_TRUE(succeeded(emitted));
  EXPECT_NE(emitted->find("output bit[2] scratch;"), std::string::npos);
  EXPECT_NE(emitted->find("bool _mqt_v"), std::string::npos);
  EXPECT_NE(emitted->find("float _mqt_v"), std::string::npos);
}

TEST(OpenQASM3EmissionTest, DiagnosesUnsupportedProgramShapesAndTypes) {
  struct Fixture {
    llvm::StringLiteral name;
    llvm::StringLiteral source;
  };
  constexpr std::array fixtures{
      Fixture{.name = "no-entry-function", .source = "module {}"},
      Fixture{.name = "external-entry-function",
              .source = "module { func.func private @main() }"},
      Fixture{.name = "entry-arguments", .source = R"mlir(module {
            func.func @main(%value: i64) {
              return
            }
          })mlir"},
      Fixture{.name = "function-call", .source = R"mlir(module {
            func.func @main() {
              func.call @main() : () -> ()
              return
            }
          })mlir"},
      Fixture{.name = "module-scope-data", .source = R"mlir(module {
            memref.global "private" @bits : memref<1xi1>
            func.func @main() {
              return
            }
          })mlir"},
      Fixture{.name = "multi-block-scf-region", .source = R"mlir(module {
            func.func @main() {
              scf.execute_region {
              ^entry:
                cf.br ^exit
              ^exit:
                scf.yield
              }
              return
            }
          })mlir"},
      Fixture{.name = "conflicting-output-kind", .source = R"mlir(module {
            func.func @main() ->
                (f64 {qc.openqasm.output_kind = "int"}) {
              %value = arith.constant 1.0 : f64
              return %value : f64
            }
          })mlir"},
      Fixture{.name = "rank-two-memory", .source = R"mlir(module {
            func.func @main() {
              %memory = memref.alloc() : memref<2x2xi1>
              memref.dealloc %memory : memref<2x2xi1>
              return
            }
          })mlir"},
      Fixture{.name = "non-bit-memory", .source = R"mlir(module {
            func.func @main() {
              %memory = memref.alloc() : memref<2xi64>
              memref.dealloc %memory : memref<2xi64>
              return
            }
          })mlir"},
      Fixture{.name = "qubit-register-output", .source = R"mlir(module {
            func.func @main() -> memref<2x!qc.qubit> {
              %qubits = memref.alloc() : memref<2x!qc.qubit>
              return %qubits : memref<2x!qc.qubit>
            }
          })mlir"},
      Fixture{.name = "scalar-bit-width", .source = R"mlir(module {
            func.func @main() ->
                (memref<2xi1> {qc.openqasm.output_kind = "bit"}) {
              %bits = memref.alloc() : memref<2xi1>
              return %bits : memref<2xi1>
            }
          })mlir"},
      Fixture{.name = "invalid-bit-output-kind", .source = R"mlir(module {
            func.func @main() ->
                (memref<1xi1> {qc.openqasm.output_kind = "bool"}) {
              %bits = memref.alloc() : memref<1xi1>
              return %bits : memref<1xi1>
            }
          })mlir"},
      Fixture{.name = "returned-memory-view", .source = R"mlir(module {
            func.func @main() -> memref<?xi1> {
              %bits = memref.alloc() : memref<2xi1>
              %view = memref.cast %bits : memref<2xi1> to memref<?xi1>
              return %view : memref<?xi1>
            }
          })mlir"},
      Fixture{.name = "unsupported-select-type", .source = R"mlir(module {
            func.func @main() {
              %condition = arith.constant true
              %one = arith.constant 1 : i32
              %value = arith.select %condition, %one, %one : i32
              return
            }
          })mlir"},
      Fixture{.name = "live-unsupported-expression-type",
              .source = R"mlir(module {
            func.func @main() ->
                (i64 {qc.openqasm.output_kind = "int"}) {
              %narrow = arith.constant 1 : i32
              %sum = arith.addi %narrow, %narrow : i32
              %value = arith.extsi %sum : i32 to i64
              return %value : i64
            }
          })mlir"},
      Fixture{.name = "unsupported-if-result-type", .source = R"mlir(module {
            func.func @main() ->
                (i64 {qc.openqasm.output_kind = "int"}) {
              %condition = arith.constant true
              %one = arith.constant 1 : i32
              %selected = scf.if %condition -> i32 {
                scf.yield %one : i32
              } else {
                scf.yield %one : i32
              }
              %value = arith.extsi %selected : i32 to i64
              return %value : i64
            }
          })mlir"},
      Fixture{.name = "unsupported-if-then-operation", .source = R"mlir(module {
            func.func @main() {
              %source = memref.alloc() : memref<1xi1>
              %target = memref.alloc() : memref<1xi1>
              %condition = arith.constant true
              scf.if %condition {
                memref.copy %source, %target
                    : memref<1xi1> to memref<1xi1>
              }
              return
            }
          })mlir"},
      Fixture{.name = "unsupported-if-else-operation", .source = R"mlir(module {
            func.func @main() {
              %source = memref.alloc() : memref<1xi1>
              %target = memref.alloc() : memref<1xi1>
              %condition = arith.constant true
              scf.if %condition {
              } else {
                memref.copy %source, %target
                    : memref<1xi1> to memref<1xi1>
              }
              return
            }
          })mlir"},
      Fixture{.name = "unsupported-for-body-operation",
              .source = R"mlir(module {
            func.func @main() {
              %source = memref.alloc() : memref<1xi1>
              %target = memref.alloc() : memref<1xi1>
              %zero = arith.constant 0 : index
              %one = arith.constant 1 : index
              scf.for %index = %zero to %one step %one {
                memref.copy %source, %target
                    : memref<1xi1> to memref<1xi1>
              }
              return
            }
          })mlir"},
      Fixture{.name = "unsupported-while-body-operation",
              .source = R"mlir(module {
            func.func @main() {
              %source = memref.alloc() : memref<1xi1>
              %target = memref.alloc() : memref<1xi1>
              %initial = arith.constant 0 : i64
              %result = scf.while (%state = %initial) : (i64) -> i64 {
                %condition = arith.constant true
                scf.condition(%condition) %state : i64
              } do {
              ^bb0(%state: i64):
                memref.copy %source, %target
                    : memref<1xi1> to memref<1xi1>
                scf.yield %state : i64
              }
              return
            }
          })mlir"},
      Fixture{.name = "unsupported-switch-case-operation",
              .source = R"mlir(module {
            func.func @main() {
              %source = memref.alloc() : memref<1xi1>
              %target = memref.alloc() : memref<1xi1>
              %index = arith.constant 0 : index
              scf.index_switch %index
              case 0 {
                memref.copy %source, %target
                    : memref<1xi1> to memref<1xi1>
                scf.yield
              }
              default {
                scf.yield
              }
              return
            }
          })mlir"},
      Fixture{.name = "unsupported-switch-default-operation",
              .source = R"mlir(module {
            func.func @main() {
              %source = memref.alloc() : memref<1xi1>
              %target = memref.alloc() : memref<1xi1>
              %index = arith.constant 0 : index
              scf.index_switch %index
              case 0 {
                scf.yield
              }
              default {
                memref.copy %source, %target
                    : memref<1xi1> to memref<1xi1>
                scf.yield
              }
              return
            }
          })mlir"},
      Fixture{.name = "out-of-bounds-qubit-index", .source = R"mlir(module {
            func.func @main() {
              %qubits = memref.alloc() : memref<1x!qc.qubit>
              %two = arith.constant 2 : index
              %qubit = memref.load %qubits[%two] : memref<1x!qc.qubit>
              qc.x %qubit : !qc.qubit
              return
            }
          })mlir"},
      Fixture{.name = "out-of-bounds-reset", .source = R"mlir(module {
            func.func @main() {
              %qubits = memref.alloc() : memref<1x!qc.qubit>
              %two = arith.constant 2 : index
              %qubit = memref.load %qubits[%two] : memref<1x!qc.qubit>
              qc.reset %qubit : !qc.qubit
              return
            }
          })mlir"},
      Fixture{.name = "out-of-bounds-barrier", .source = R"mlir(module {
            func.func @main() {
              %qubits = memref.alloc() : memref<1x!qc.qubit>
              %two = arith.constant 2 : index
              %qubit = memref.load %qubits[%two] : memref<1x!qc.qubit>
              qc.barrier %qubit : !qc.qubit
              return
            }
          })mlir"},
      Fixture{.name = "out-of-bounds-measurement", .source = R"mlir(module {
            func.func @main() {
              %qubits = memref.alloc() : memref<1x!qc.qubit>
              %two = arith.constant 2 : index
              %qubit = memref.load %qubits[%two] : memref<1x!qc.qubit>
              %result = qc.measure %qubit : !qc.qubit -> i1
              return
            }
          })mlir"},
      Fixture{.name = "packed-integer-bitwise-operation",
              .source = R"mlir(module {
            func.func @main() ->
                (i64 {qc.openqasm.output_kind = "int"}) {
              %one = arith.constant 1 : i64
              %value = arith.andi %one, %one : i64
              return %value : i64
            }
          })mlir"},
      Fixture{.name = "unordered-floating-point-comparison",
              .source = R"mlir(module {
            func.func @main() ->
                (i1 {qc.openqasm.output_kind = "bool"}) {
              %one = arith.constant 1.0 : f64
              %value = arith.cmpf uno, %one, %one : f64
              return %value : i1
            }
          })mlir"},
      Fixture{.name = "non-finite-floating-point-constant",
              .source = R"mlir(module {
            func.func @main() ->
                (f64 {qc.openqasm.output_kind = "float"}) {
              %value = arith.constant 0x7FF0000000000000 : f64
              return %value : f64
            }
          })mlir"},
      Fixture{.name = "dynamic-loop-range", .source = R"mlir(module {
            func.func @main() {
              %condition = arith.constant true
              %zero = arith.constant 0 : index
              %one = arith.constant 1 : index
              %upper = arith.select %condition, %zero, %one : index
              scf.for %index = %zero to %upper step %one {
              }
              return
            }
          })mlir"},
      Fixture{.name = "side-effecting-while-condition",
              .source = R"mlir(module {
            func.func @main() {
              %qubit = qc.alloc : !qc.qubit
              %zero = arith.constant 0 : i64
              %result = scf.while (%state = %zero) : (i64) -> i64 {
                qc.x %qubit : !qc.qubit
                %condition = arith.constant true
                scf.condition(%condition) %state : i64
              } do {
              ^bb0(%state: i64):
                scf.yield %state : i64
              }
              qc.dealloc %qubit : !qc.qubit
              return
            }
          })mlir"},
      Fixture{.name = "unknown-memory-operation", .source = R"mlir(module {
            func.func @main() {
              %source = memref.alloc() : memref<2xi1>
              %target = memref.alloc() : memref<2xi1>
              memref.copy %source, %target
                  : memref<2xi1> to memref<2xi1>
              memref.dealloc %target : memref<2xi1>
              memref.dealloc %source : memref<2xi1>
              return
            }
          })mlir"},
  };

  DialectRegistry registry;
  registry
      .insert<arith::ArithDialect, cf::ControlFlowDialect, func::FuncDialect,
              memref::MemRefDialect, qc::QCDialect, scf::SCFDialect>();
  MLIRContext context(registry);
  for (const auto& fixture : fixtures) {
    SCOPED_TRACE(fixture.name.str());
    auto moduleOp = parseSourceString<ModuleOp>(fixture.source, &context);
    ASSERT_TRUE(moduleOp);
    EXPECT_TRUE(failed(qc::translateQCToOpenQASM3(*moduleOp)));
  }
}

} // namespace
