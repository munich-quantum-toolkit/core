/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/CBit/IR/CBitDialect.h"
#include "mlir/Dialect/QC/Builder/QCProgramBuilder.h"
#include "mlir/Dialect/QC/IR/QCDialect.h"
#include "mlir/Dialect/QC/Translation/TranslateQASM3ToQC.h"
#include "mlir/Dialect/QC/Translation/TranslateQCToOpenQASM3.h"
#include "mlir/Support/Passes.h"
#include "mlir/Target/OpenQASM/Frontend.h"

#include <gtest/gtest.h>
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

namespace mqt::test::openqasm3_emission {
using namespace mlir;

static DialectRegistry emissionDialects() {
  DialectRegistry registry;
  registry.insert<arith::ArithDialect, cbit::CBitDialect,
                  cf::ControlFlowDialect, func::FuncDialect, math::MathDialect,
                  memref::MemRefDialect, qc::QCDialect, scf::SCFDialect>();
  return registry;
}

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
  std::string streamedSource;
  llvm::raw_string_ostream stream(streamedSource);
  EXPECT_TRUE(succeeded(qc::translateQCToOpenQASM3(*moduleOp, stream)));
  stream.flush();
  EXPECT_EQ(streamedSource, *source);
  EXPECT_TRUE(source->starts_with("OPENQASM 3.1;\n"));
  EXPECT_NE(source->find("include \"stdgates.inc\";"), std::string::npos);
  EXPECT_NE(source->find("ctrl @ x"), std::string::npos);
  EXPECT_NE(source->find("output bit[2] c;"), std::string::npos);
  EXPECT_TRUE(oq3::frontend::analyzeOpenQASM(
      *source, {.gatePolicy = oq3::frontend::GatePolicy::Strict}));
  EXPECT_TRUE(qc::translateQASM3ToQC(*source, &context));
}

TEST(OpenQASM3EmissionTest, PreservesMeasurementOrderBeforeDelayedStore) {
  constexpr llvm::StringLiteral source = R"mlir(module {
    func.func @main() -> !cbit.reg<1>
        attributes {mqt.entry_point} {
      %zero = arith.constant 0 : index
      %qubit = qc.alloc : !qc.qubit
      %bits = cbit.alloc(#cbit.init<undefined>) {mqt.register_name = "c"}
          : !cbit.reg<1>
      %measured = qc.measure %qubit : !qc.qubit -> i1
      qc.x %qubit : !qc.qubit
      cbit.store %measured, %bits[%zero] : !cbit.reg<1>
      qc.dealloc %qubit : !qc.qubit
      return %bits : !cbit.reg<1>
    }
  })mlir";
  DialectRegistry registry = emissionDialects();
  MLIRContext context(registry);
  auto moduleOp = parseSourceString<ModuleOp>(source, &context);
  ASSERT_TRUE(moduleOp);

  auto emitted = qc::translateQCToOpenQASM3(*moduleOp);

  ASSERT_TRUE(succeeded(emitted));
  const auto measurement = emitted->find("bit _mqt_b0 = measure _mqt_q0;");
  const auto gate = emitted->find("x _mqt_q0;");
  const auto store = emitted->find("c[0] = _mqt_b0;");
  ASSERT_NE(measurement, std::string::npos) << *emitted;
  ASSERT_NE(gate, std::string::npos) << *emitted;
  ASSERT_NE(store, std::string::npos) << *emitted;
  EXPECT_LT(measurement, gate);
  EXPECT_LT(gate, store);
  EXPECT_TRUE(oq3::frontend::analyzeOpenQASM(
      *emitted, {.gatePolicy = oq3::frontend::GatePolicy::Strict}))
      << *emitted;
}

TEST(OpenQASM3EmissionTest, CanonicalizesFixedAnglesToPortableFloats) {
  constexpr llvm::StringLiteral source = R"qasm(OPENQASM 3.1;
include "stdgates.inc";
angle[8] theta = angle[8](pi / 2);
qubit q;
rx(theta) q;
output bit result;
result = measure q;
)qasm";
  MLIRContext context;
  auto moduleOp = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(moduleOp);

  auto emitted = qc::translateQCToOpenQASM3(*moduleOp);

  ASSERT_TRUE(succeeded(emitted));
  EXPECT_EQ(emitted->find("angle"), std::string::npos);
  EXPECT_EQ(emitted->find("mqt.openqasm"), std::string::npos);
  EXPECT_NE(emitted->find("rx(1.5707963267948966)"), std::string::npos)
      << *emitted;
  EXPECT_TRUE(oq3::frontend::analyzeOpenQASM(
      *emitted, {.gatePolicy = oq3::frontend::GatePolicy::Strict}))
      << *emitted;
}

TEST(OpenQASM3EmissionTest, UsesCanonicalOutputTypesWithoutResultMetadata) {
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
unsigned_value = 2;
real = 3.0;
)qasm";
  MLIRContext context;
  auto moduleOp = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(moduleOp);
  auto function = *moduleOp->getOps<func::FuncOp>().begin();
  ASSERT_EQ(function.getNumResults(), 6U);
  EXPECT_FALSE(function.getAllResultAttrs());

  auto emitted = qc::translateQCToOpenQASM3(*moduleOp);

  ASSERT_TRUE(succeeded(emitted));
  EXPECT_NE(emitted->find("output bit[1] measured;"), std::string::npos);
  EXPECT_NE(emitted->find("output bit[1] vector;"), std::string::npos);
  EXPECT_NE(emitted->find("output bool _mqt_out"), std::string::npos);
  EXPECT_NE(emitted->find("output int _mqt_out"), std::string::npos);
  EXPECT_NE(emitted->find("output float _mqt_out"), std::string::npos);
  EXPECT_EQ(emitted->find("output uint "), std::string::npos);
  EXPECT_TRUE(qc::translateQASM3ToQC(*emitted, &context)) << *emitted;
}

TEST(OpenQASM3EmissionTest, RenamesOutputsThatCollideWithCompatibilityHelpers) {
  constexpr llvm::StringLiteral source = R"qasm(OPENQASM 3.1;
include "stdgates.inc";
output bit r;
qubit q;
r(0.5, 0.25) q;
r = measure q;
)qasm";
  MLIRContext context;
  auto moduleOp = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(moduleOp);

  auto emitted = qc::translateQCToOpenQASM3(*moduleOp);

  ASSERT_TRUE(succeeded(emitted));
  EXPECT_NE(emitted->find("gate r("), std::string::npos);
  EXPECT_NE(emitted->find("output bit[1] _mqt_out0;"), std::string::npos);
  EXPECT_TRUE(oq3::frontend::analyzeOpenQASM(
      *emitted, {.gatePolicy = oq3::frontend::GatePolicy::Strict}))
      << *emitted;
}

TEST(OpenQASM3EmissionTest, RenamesOutputsThatCollideWithStandardGates) {
  constexpr llvm::StringLiteral source = R"mlir(module {
    func.func @main() -> !cbit.reg<1> {
      %bits = cbit.alloc(#cbit.init<zero>) {mqt.register_name = "x"}
          : !cbit.reg<1>
      return %bits : !cbit.reg<1>
    }
  })mlir";
  DialectRegistry registry = emissionDialects();
  MLIRContext context(registry);
  auto moduleOp = parseSourceString<ModuleOp>(source, &context);
  ASSERT_TRUE(moduleOp);

  auto emitted = qc::translateQCToOpenQASM3(*moduleOp);

  ASSERT_TRUE(succeeded(emitted));
  EXPECT_NE(emitted->find("output bit[1] _mqt_out0;"), std::string::npos);
  EXPECT_TRUE(oq3::frontend::analyzeOpenQASM(
      *emitted, {.gatePolicy = oq3::frontend::GatePolicy::Strict}))
      << *emitted;
  EXPECT_TRUE(qc::translateQASM3ToQC(*emitted, &context)) << *emitted;
}

TEST(OpenQASM3EmissionTest, EmitsStatementOnlyStructuredControl) {
  constexpr llvm::StringLiteral source = R"qasm(OPENQASM 3.1;
include "stdgates.inc";
qubit q;
bit condition = measure q;
int selector = 1;
if (condition) {
  for int i in [0:2] {
    x q;
  }
} else {
  y q;
}
while (condition) {
  z q;
}
switch (selector) {
  case 1 {
    h q;
  }
  default {
    sx q;
  }
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
  EXPECT_NE(emitted->find("while ("), std::string::npos);
  EXPECT_TRUE(oq3::frontend::analyzeOpenQASM(
      *emitted, {.gatePolicy = oq3::frontend::GatePolicy::Strict}))
      << *emitted;
}

TEST(OpenQASM3EmissionTest, EmitsNativeIndexSwitch) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
  func.func @main() attributes {mqt.entry_point} {
    %qubit = qc.alloc : !qc.qubit
    %index = arith.constant 1 : index
    scf.index_switch %index
    case 1 {
      qc.x %qubit : !qc.qubit
      scf.yield
    }
    default {
      qc.y %qubit : !qc.qubit
      scf.yield
    }
    qc.dealloc %qubit : !qc.qubit
    return
  }
}
)mlir";
  DialectRegistry registry = emissionDialects();
  MLIRContext context(registry);
  auto moduleOp = parseSourceString<ModuleOp>(source, &context);
  ASSERT_TRUE(moduleOp);

  auto emitted = qc::translateQCToOpenQASM3(*moduleOp);

  ASSERT_TRUE(succeeded(emitted));
  EXPECT_NE(emitted->find("switch (1)"), std::string::npos);
  EXPECT_NE(emitted->find("case 1 {"), std::string::npos);
  EXPECT_NE(emitted->find("default {"), std::string::npos);
  EXPECT_TRUE(oq3::frontend::analyzeOpenQASM(
      *emitted, {.gatePolicy = oq3::frontend::GatePolicy::Strict}))
      << *emitted;
}

TEST(OpenQASM3EmissionTest, EmitsCatalogHelpersUnderTheirNativeNames) {
  DialectRegistry registry = emissionDialects();
  MLIRContext context(registry);
  context.loadAllAvailableDialects();
  qc::QCProgramBuilder builder(&context);
  builder.initialize();
  const auto qubits = builder.allocQubitRegister(3);
  auto q0 = qubits[0];
  auto q1 = qubits[1];
  auto q2 = qubits[2];

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
  builder.ctrl(q0, q1, [&](Value target) { builder.h(target).x(target); });
  builder.pow(0.5, q2, [&](Value target) { builder.z(target); });
  auto moduleOp = builder.finalize();
  ASSERT_TRUE(moduleOp);

  auto emitted = qc::translateQCToOpenQASM3(*moduleOp);

  ASSERT_TRUE(succeeded(emitted));
  EXPECT_NE(emitted->find("inv @ sx"), std::string::npos);
  EXPECT_NE(emitted->find("u2("), std::string::npos);
  EXPECT_NE(emitted->find("U("), std::string::npos);
  constexpr std::array helperNames{
      "r",   "iswap", "dcx",        "ecr",         "rxx",  "ryy",
      "rzx", "rzz",   "xx_plus_yy", "xx_minus_yy", "rccx",
  };
  for (const auto* const helper : helperNames) {
    const auto declaration = "gate " + std::string(helper);
    const auto prefixedDeclaration = "gate _mqt_" + std::string(helper);
    EXPECT_NE(emitted->find(declaration), std::string::npos) << helper;
    EXPECT_EQ(emitted->find(prefixedDeclaration), std::string::npos) << helper;
  }
  EXPECT_NE(emitted->find("gate _mqt_gate"), std::string::npos);
  EXPECT_NE(emitted->find("pow(0.5) @ z"), std::string::npos);
  EXPECT_TRUE(oq3::frontend::analyzeOpenQASM(
      *emitted, {.gatePolicy = oq3::frontend::GatePolicy::Strict}))
      << *emitted;

  auto roundTripped = qc::translateQASM3ToQC(*emitted, &context);
  ASSERT_TRUE(roundTripped);
  for (const auto* const helper : helperNames) {
    bool found = false;
    roundTripped->walk([&](Operation* operation) {
      if (operation->getName().getStringRef() ==
          ("qc." + std::string(helper))) {
        found = true;
      }
    });
    EXPECT_TRUE(found) << helper;
  }
}

TEST(OpenQASM3EmissionTest, ForwardsCompositeModifierParameters) {
  constexpr llvm::StringLiteral source = R"qasm(OPENQASM 3.1;
include "stdgates.inc";
gate pair(p0) q {
  inv @ rx(p0) q;
  z q;
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
}

TEST(OpenQASM3EmissionTest, EmitsSignedBooleanAndFloatingExpressions) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
  func.func @main() -> (i64, i1, f64, f64, i1)
      attributes {mqt.entry_point} {
    %one = arith.constant 1 : i64
    %two = arith.constant 2 : i64
    %sum = arith.addi %one, %two : i64
    %signed = arith.divsi %sum, %two : i64
    %comparison = arith.cmpi sge, %sum, %two : i64
    %angle = arith.constant 0.25 : f64
    %negated = arith.negf %angle : f64
    %sine = math.sin %negated : f64
    %remainder = arith.remf %angle, %sine : f64
    %converted = arith.fptosi %sine : f64 to i64
    %zero = arith.constant 0 : i64
    %nonzero = arith.cmpi ne, %converted, %zero : i64
    return %signed, %comparison, %sine, %remainder, %nonzero
        : i64, i1, f64, f64, i1
  }
}
)mlir";
  DialectRegistry registry = emissionDialects();
  MLIRContext context(registry);
  auto moduleOp = parseSourceString<ModuleOp>(source, &context);
  ASSERT_TRUE(moduleOp);

  auto emitted = qc::translateQCToOpenQASM3(*moduleOp);

  ASSERT_TRUE(succeeded(emitted));
  EXPECT_NE(emitted->find("((1 + 2) / 2)"), std::string::npos);
  EXPECT_NE(emitted->find("((1 + 2) >= 2)"), std::string::npos);
  EXPECT_NE(emitted->find("sin((-0.25))"), std::string::npos);
  EXPECT_NE(emitted->find("mod(0.25, sin((-0.25)))"), std::string::npos);
  EXPECT_NE(emitted->find("int(sin((-0.25)))"), std::string::npos);
  EXPECT_NE(emitted->find("(int(sin((-0.25))) != 0)"), std::string::npos);
}

TEST(OpenQASM3EmissionTest, EmitsFloatingRemainderAsStrictOpenQASMMod) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
  func.func @main() -> f64 {
    %lhs = arith.constant 5.5 : f64
    %rhs = arith.constant 2.0 : f64
    %remainder = arith.remf %lhs, %rhs : f64
    return %remainder : f64
  }
}
)mlir";
  DialectRegistry registry = emissionDialects();
  MLIRContext context(registry);
  auto moduleOp = parseSourceString<ModuleOp>(source, &context);
  ASSERT_TRUE(moduleOp);

  auto emitted = qc::translateQCToOpenQASM3(*moduleOp);

  ASSERT_TRUE(succeeded(emitted));
  EXPECT_NE(emitted->find("mod(5.5, 2.0)"), std::string::npos);
  EXPECT_TRUE(oq3::frontend::analyzeOpenQASM(
      *emitted, {.gatePolicy = oq3::frontend::GatePolicy::Strict}))
      << *emitted;
}

TEST(OpenQASM3EmissionTest, EmitsSignedAndFloatingComparisonFamilies) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
  func.func @main() -> (i1, i1, i1, i1, i1, i1,
                        i1, i1, i1, i1, i1, i1, f64) {
    %one = arith.constant 1 : i64
    %two = arith.constant 2 : i64
    %one_float = arith.sitofp %one : i64 to f64
    %two_float = arith.constant 2.0 : f64
    %ieq = arith.cmpi eq, %one, %two : i64
    %ine = arith.cmpi ne, %one, %two : i64
    %ilt = arith.cmpi slt, %one, %two : i64
    %ile = arith.cmpi sle, %one, %two : i64
    %igt = arith.cmpi sgt, %one, %two : i64
    %ige = arith.cmpi sge, %one, %two : i64
    %feq = arith.cmpf oeq, %one_float, %two_float : f64
    %fne = arith.cmpf one, %one_float, %two_float : f64
    %flt = arith.cmpf olt, %one_float, %two_float : f64
    %fle = arith.cmpf ole, %one_float, %two_float : f64
    %fgt = arith.cmpf ogt, %one_float, %two_float : f64
    %fge = arith.cmpf oge, %one_float, %two_float : f64
    return %ieq, %ine, %ilt, %ile, %igt, %ige,
           %feq, %fne, %flt, %fle, %fgt, %fge, %one_float
        : i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, f64
  }
}
)mlir";
  DialectRegistry registry = emissionDialects();
  MLIRContext context(registry);
  auto moduleOp = parseSourceString<ModuleOp>(source, &context);
  ASSERT_TRUE(moduleOp);

  auto emitted = qc::translateQCToOpenQASM3(*moduleOp);

  ASSERT_TRUE(succeeded(emitted));
  for (const auto* const comparison :
       {"(1 == 2)", "(1 != 2)", "(1 < 2)", "(1 <= 2)", "(1 > 2)", "(1 >= 2)",
        "(float(1) == 2.0)", "(float(1) != 2.0)", "(float(1) < 2.0)",
        "(float(1) <= 2.0)", "(float(1) > 2.0)", "(float(1) >= 2.0)"}) {
    EXPECT_NE(emitted->find(comparison), std::string::npos) << comparison;
  }
}

TEST(OpenQASM3EmissionTest, EmitsCanonicalConstantRangeBoundaries) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
  func.func @main() {
    %qubit = qc.alloc : !qc.qubit
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %two = arith.constant 2 : index
    %five = arith.constant 5 : index
    scf.for %i = %zero to %five step %two {
      qc.x %qubit : !qc.qubit
    }
    scf.for %i = %one to %zero step %one {
      qc.y %qubit : !qc.qubit
    }
    scf.index_switch %zero
    default {
      qc.z %qubit : !qc.qubit
      scf.yield
    }
    qc.dealloc %qubit : !qc.qubit
    return
  }
}
)mlir";
  DialectRegistry registry = emissionDialects();
  MLIRContext context(registry);
  auto moduleOp = parseSourceString<ModuleOp>(source, &context);
  ASSERT_TRUE(moduleOp);

  auto emitted = qc::translateQCToOpenQASM3(*moduleOp);

  ASSERT_TRUE(succeeded(emitted));
  EXPECT_NE(emitted->find("in [0:2:4]"), std::string::npos);
  EXPECT_EQ(emitted->find("y _mqt_q0;"), std::string::npos);
  EXPECT_EQ(emitted->find("switch ("), std::string::npos);
  EXPECT_NE(emitted->find("z _mqt_q0;"), std::string::npos);
}

TEST(OpenQASM3EmissionTest, EmitsPhysicalQubitOperations) {
  DialectRegistry registry = emissionDialects();
  MLIRContext context(registry);
  context.loadAllAvailableDialects();
  qc::QCProgramBuilder builder(&context);
  builder.initialize();
  auto qubit = builder.staticQubit(7);
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

TEST(OpenQASM3EmissionTest, ReusesClassicalRegisterNamesForOutputs) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
  func.func @main() -> (!cbit.reg<1>, !cbit.reg<2>, i1) {
    %single = cbit.alloc(#cbit.init<undefined>) {mqt.register_name = "single"}
        : !cbit.reg<1>
    %bits = cbit.alloc(#cbit.init<undefined>) {mqt.register_name = "bits"}
        : !cbit.reg<2>
    %qubit = qc.alloc : !qc.qubit
    %measured = qc.measure %qubit : !qc.qubit -> i1
    qc.dealloc %qubit : !qc.qubit
    return %single, %bits, %measured : !cbit.reg<1>, !cbit.reg<2>, i1
  }
}
)mlir";
  DialectRegistry registry = emissionDialects();
  MLIRContext context(registry);
  auto moduleOp = parseSourceString<ModuleOp>(source, &context);
  ASSERT_TRUE(moduleOp);

  auto emitted = qc::translateQCToOpenQASM3(*moduleOp);

  ASSERT_TRUE(succeeded(emitted));
  EXPECT_NE(emitted->find("output bit[1] single;"), std::string::npos);
  EXPECT_NE(emitted->find("output bit[2] bits;"), std::string::npos);
  EXPECT_NE(emitted->find("output bit _mqt_out"), std::string::npos);
}

TEST(OpenQASM3EmissionTest, ReusesQubitRegisterNames) {
  DialectRegistry registry = emissionDialects();
  MLIRContext context(registry);
  context.loadAllAvailableDialects();
  qc::QCProgramBuilder builder(&context);
  builder.initialize();
  std::ignore = builder.allocQubitRegister(2, "named_qubits");
  std::ignore = builder.allocQubitRegister(2, "not-valid");
  auto moduleOp = builder.finalize();
  ASSERT_TRUE(moduleOp);

  auto emitted = qc::translateQCToOpenQASM3(*moduleOp);

  ASSERT_TRUE(succeeded(emitted));
  EXPECT_NE(emitted->find("qubit[2] named_qubits;"), std::string::npos);
  EXPECT_EQ(emitted->find("qubit[2] not-valid;"), std::string::npos);
  EXPECT_TRUE(oq3::frontend::analyzeOpenQASM(
      *emitted, {.gatePolicy = oq3::frontend::GatePolicy::Strict}))
      << *emitted;
}

TEST(OpenQASM3EmissionTest, DefinesECRWithOneEntanglingGate) {
  DialectRegistry registry = emissionDialects();
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
  EXPECT_NE(emitted->find("gate ecr"), std::string::npos);
  EXPECT_NE(emitted->find("gphase(-pi / 4);"), std::string::npos);
  EXPECT_EQ(llvm::StringRef(*emitted).count("ctrl @ x"), 1U);
}

TEST(OpenQASM3EmissionTest, LeavesDestinationEmptyOnFailure) {
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

TEST(OpenQASM3EmissionTest, RejectsInvalidModifierBodies) {
  DialectRegistry registry = emissionDialects();
  MLIRContext context(registry);
  context.loadAllAvailableDialects();

  qc::QCProgramBuilder nonUnitaryBuilder(&context);
  nonUnitaryBuilder.initialize();
  auto nonUnitaryQubit = nonUnitaryBuilder.allocQubit();
  nonUnitaryBuilder.inv(nonUnitaryQubit,
                        [&](Value target) { nonUnitaryBuilder.reset(target); });
  auto nonUnitaryModule = nonUnitaryBuilder.finalize();
  ASSERT_TRUE(nonUnitaryModule);
  EXPECT_TRUE(failed(qc::translateQCToOpenQASM3(*nonUnitaryModule)));

  qc::QCProgramBuilder emptyBuilder(&context);
  emptyBuilder.initialize();
  auto emptyQubit = emptyBuilder.allocQubit();
  emptyBuilder.inv(emptyQubit, [](Value) {});
  auto emptyModule = emptyBuilder.finalize();
  ASSERT_TRUE(emptyModule);
  EXPECT_TRUE(failed(qc::translateQCToOpenQASM3(*emptyModule)));

  qc::QCProgramBuilder capturedQubitBuilder(&context);
  capturedQubitBuilder.initialize();
  auto target = capturedQubitBuilder.allocQubit();
  auto captured = capturedQubitBuilder.allocQubit();
  capturedQubitBuilder.inv(target, [&](Value argument) {
    capturedQubitBuilder.x(argument).x(captured);
  });
  auto capturedQubitModule = capturedQubitBuilder.finalize();
  ASSERT_TRUE(capturedQubitModule);
  EXPECT_TRUE(failed(qc::translateQCToOpenQASM3(*capturedQubitModule)));

  qc::QCProgramBuilder zeroTargetBuilder(&context);
  zeroTargetBuilder.initialize();
  zeroTargetBuilder.inv(ValueRange{}, [&](ValueRange) {
    zeroTargetBuilder.gphase(0.25).gphase(0.5);
  });
  auto zeroTargetModule = zeroTargetBuilder.finalize();
  ASSERT_TRUE(zeroTargetModule);
  EXPECT_TRUE(failed(qc::translateQCToOpenQASM3(*zeroTargetModule)));
}

TEST(OpenQASM3EmissionTest, RejectsUnsupportedSubsetConcerns) {
  struct Fixture {
    llvm::StringLiteral name;
    llvm::StringLiteral source;
  };
  constexpr std::array fixtures{
      Fixture{.name = "missing-function", .source = R"mlir(module {
      })mlir"},
      Fixture{.name = "external-function", .source = R"mlir(module {
        func.func private @main()
      })mlir"},
      Fixture{.name = "function-call", .source = R"mlir(module {
        func.func @main() {
          func.call @main() : () -> ()
          return
        }
      })mlir"},
      Fixture{.name = "nested-multiblock-region", .source = R"mlir(module {
        func.func @main() {
          scf.execute_region {
            cf.br ^next
          ^next:
            scf.yield
          }
          return
        }
      })mlir"},
      Fixture{.name = "extra-module-scope-operation", .source = R"mlir(module {
        module @extra {}
        func.func @main() {
          return
        }
      })mlir"},
      Fixture{.name = "unsupported-memory-element", .source = R"mlir(module {
        func.func @main() {
          %memory = memref.alloc() : memref<1xi64>
          return
        }
      })mlir"},
      Fixture{.name = "returned-memory-view", .source = R"mlir(module {
        func.func @main() -> memref<?xi1> {
          %memory = memref.alloc() : memref<1xi1>
          %view = memref.cast %memory : memref<1xi1> to memref<?xi1>
          return %view : memref<?xi1>
        }
      })mlir"},
      Fixture{.name = "returned-qubit-memory", .source = R"mlir(module {
        func.func @main() -> memref<1x!qc.qubit> {
          %memory = memref.alloc() : memref<1x!qc.qubit>
          return %memory : memref<1x!qc.qubit>
        }
      })mlir"},
      Fixture{.name = "unsupported-expression-width", .source = R"mlir(module {
        func.func @main() {
          %one = arith.constant 1 : i32
          %sum = arith.addi %one, %one : i32
          return
        }
      })mlir"},
      Fixture{.name = "sign-extension", .source = R"mlir(module {
        func.func @main() -> i64 {
          %value = arith.constant true
          %extended = arith.extsi %value : i1 to i64
          return %extended : i64
        }
      })mlir"},
      Fixture{.name = "integer-truncation", .source = R"mlir(module {
        func.func @main() -> i1 {
          %value = arith.constant 2 : i64
          %truncated = arith.trunci %value : i64 to i1
          return %truncated : i1
        }
      })mlir"},
      Fixture{.name = "packed-bitwise", .source = R"mlir(module {
        func.func @main() -> i64 {
          %one = arith.constant 1 : i64
          %value = arith.andi %one, %one : i64
          return %value : i64
        }
      })mlir"},
      Fixture{.name = "unordered-float-comparison", .source = R"mlir(module {
        func.func @main() -> i1 {
          %one = arith.constant 1.0 : f64
          %value = arith.cmpf uno, %one, %one : f64
          return %value : i1
        }
      })mlir"},
      Fixture{.name = "non-finite-float", .source = R"mlir(module {
        func.func @main() -> f64 {
          %value = arith.constant 0x7FF0000000000000 : f64
          return %value : f64
        }
      })mlir"},
      Fixture{.name = "while-condition-side-effect", .source = R"mlir(module {
        func.func @main() {
          %memory = memref.alloc() : memref<1xi1>
          %zero = arith.constant 0 : index
          %condition = arith.constant true
          scf.while : () -> () {
            memref.store %condition, %memory[%zero] : memref<1xi1>
            scf.condition(%condition)
          } do {
            scf.yield
          }
          return
        }
      })mlir"},
      Fixture{.name = "out-of-bounds-qubit", .source = R"mlir(module {
        func.func @main() {
          %memory = memref.alloc() : memref<1x!qc.qubit>
          %one = arith.constant 1 : index
          %qubit = memref.load %memory[%one] : memref<1x!qc.qubit>
          qc.x %qubit : !qc.qubit
          return
        }
      })mlir"},
      Fixture{.name = "out-of-bounds-measurement", .source = R"mlir(module {
        func.func @main() -> i1 {
          %memory = memref.alloc() : memref<1x!qc.qubit>
          %one = arith.constant 1 : index
          %qubit = memref.load %memory[%one] : memref<1x!qc.qubit>
          %result = qc.measure %qubit : !qc.qubit -> i1
          return %result : i1
        }
      })mlir"},
      Fixture{.name = "out-of-bounds-store", .source = R"mlir(module {
        func.func @main() {
          %memory = memref.alloc() : memref<1xi1>
          %one = arith.constant 1 : index
          %value = arith.constant false
          memref.store %value, %memory[%one] : memref<1xi1>
          return
        }
      })mlir"},
      Fixture{.name = "select", .source = R"mlir(module {
        func.func @main() -> i64 {
          %condition = arith.constant true
          %one = arith.constant 1 : i64
          %value = arith.select %condition, %one, %one : i64
          return %value : i64
        }
      })mlir"},
      Fixture{.name = "unsigned-arithmetic", .source = R"mlir(module {
        func.func @main() -> i64 {
          %one = arith.constant 1 : i64
          %value = arith.divui %one, %one : i64
          return %value : i64
        }
      })mlir"},
      Fixture{.name = "unsigned-comparison", .source = R"mlir(module {
        func.func @main() -> i1 {
          %one = arith.constant 1 : i64
          %value = arith.cmpi ult, %one, %one : i64
          return %value : i1
        }
      })mlir"},
      Fixture{.name = "unsigned-cast", .source = R"mlir(module {
        func.func @main() -> f64 {
          %one = arith.constant 1 : i64
          %value = arith.uitofp %one : i64 to f64
          return %value : f64
        }
      })mlir"},
      Fixture{.name = "unsupported-output", .source = R"mlir(module {
        func.func @main() -> f32 {
          %value = arith.constant 1.0 : f32
          return %value : f32
        }
      })mlir"},
      Fixture{.name = "if-result", .source = R"mlir(module {
        func.func @main() -> i64 {
          %condition = arith.constant true
          %one = arith.constant 1 : i64
          %value = scf.if %condition -> i64 {
            scf.yield %one : i64
          } else {
            scf.yield %one : i64
          }
          return %value : i64
        }
      })mlir"},
      Fixture{.name = "for-iterated-state", .source = R"mlir(module {
        func.func @main() -> i64 {
          %zero = arith.constant 0 : index
          %one = arith.constant 1 : index
          %initial = arith.constant 0 : i64
          %value = scf.for %i = %zero to %one step %one
              iter_args(%state = %initial) -> i64 {
            scf.yield %state : i64
          }
          return %value : i64
        }
      })mlir"},
      Fixture{.name = "while-state", .source = R"mlir(module {
        func.func @main() {
          %initial = arith.constant 0 : i64
          %value = scf.while (%state = %initial) : (i64) -> i64 {
            %condition = arith.constant false
            scf.condition(%condition) %state : i64
          } do {
          ^bb0(%state: i64):
            scf.yield %state : i64
          }
          return
        }
      })mlir"},
      Fixture{.name = "switch-result", .source = R"mlir(module {
        func.func @main() -> i64 {
          %index = arith.constant 0 : index
          %one = arith.constant 1 : i64
          %value = scf.index_switch %index -> i64
          default {
            scf.yield %one : i64
          }
          return %value : i64
        }
      })mlir"},
      Fixture{.name = "dynamic-index", .source = R"mlir(module {
        func.func @main() -> i1 {
          %bits = memref.alloc() : memref<2xi1>
          %index = arith.constant 0 : i64
          %dynamic = arith.index_cast %index : i64 to index
          %value = memref.load %bits[%dynamic] : memref<2xi1>
          return %value : i1
        }
      })mlir"},
      Fixture{.name = "dynamic-loop-range", .source = R"mlir(module {
        func.func @main() {
          %zero = arith.constant 0 : index
          %integer = arith.constant 1 : i64
          %upper = arith.index_cast %integer : i64 to index
          %one = arith.constant 1 : index
          scf.for %i = %zero to %upper step %one {
          }
          return
        }
      })mlir"},
      Fixture{.name = "rank-two-memory", .source = R"mlir(module {
        func.func @main() {
          %memory = memref.alloc() : memref<2x2xi1>
          return
        }
      })mlir"},
      Fixture{.name = "function-argument", .source = R"mlir(module {
        func.func @main(%value: i64) {
          return
        }
      })mlir"},
  };

  DialectRegistry registry = emissionDialects();
  MLIRContext context(registry);
  for (const auto& fixture : fixtures) {
    SCOPED_TRACE(fixture.name.str());
    auto moduleOp = parseSourceString<ModuleOp>(fixture.source, &context);
    ASSERT_TRUE(moduleOp);
    EXPECT_TRUE(failed(qc::translateQCToOpenQASM3(*moduleOp)));
  }
}

} // namespace

} // namespace mqt::test::openqasm3_emission
