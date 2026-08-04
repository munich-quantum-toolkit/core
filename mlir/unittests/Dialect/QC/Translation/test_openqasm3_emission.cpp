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

using namespace mlir;

static DialectRegistry emissionDialects() {
  DialectRegistry registry;
  registry.insert<arith::ArithDialect, cf::ControlFlowDialect,
                  func::FuncDialect, math::MathDialect, memref::MemRefDialect,
                  qc::QCDialect, scf::SCFDialect>();
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
  EXPECT_TRUE(source->starts_with("OPENQASM 3.1;\n"));
  EXPECT_NE(source->find("include \"stdgates.inc\";"), std::string::npos);
  EXPECT_NE(source->find("ctrl @ x"), std::string::npos);
  EXPECT_NE(source->find("output bit[2] c;"), std::string::npos);
  EXPECT_TRUE(oq3::frontend::analyzeOpenQASM(
      *source, {.gatePolicy = oq3::frontend::GatePolicy::Strict}));
  EXPECT_TRUE(qc::translateQASM3ToQC(*source, &context));
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

TEST(OpenQASM3EmissionTest, EmitsStatementOnlyStructuredControl) {
  constexpr llvm::StringLiteral source = R"qasm(OPENQASM 3.1;
include "stdgates.inc";
qubit q;
bit condition = false;
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
  func.func @main() attributes {passthrough = ["entry_point"]} {
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
}

TEST(OpenQASM3EmissionTest, EmitsSignedBooleanAndFloatingExpressions) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
  func.func @main() -> (i64, i1, f64, i1)
      attributes {passthrough = ["entry_point"]} {
    %one = arith.constant 1 : i64
    %two = arith.constant 2 : i64
    %sum = arith.addi %one, %two : i64
    %signed = arith.divsi %sum, %two : i64
    %comparison = arith.cmpi sge, %sum, %two : i64
    %angle = arith.constant 0.25 : f64
    %negated = arith.negf %angle : f64
    %sine = math.sin %negated : f64
    %converted = arith.fptosi %sine : f64 to i64
    %truncated = arith.trunci %converted : i64 to i1
    return %signed, %comparison, %sine, %truncated : i64, i1, f64, i1
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
  EXPECT_NE(emitted->find("int(sin((-0.25)))"), std::string::npos);
  EXPECT_NE(emitted->find("bool(int(sin((-0.25))))"), std::string::npos);
}

TEST(OpenQASM3EmissionTest, EmitsPhysicalQubitOperations) {
  DialectRegistry registry = emissionDialects();
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

TEST(OpenQASM3EmissionTest, ReusesClassicalRegisterNamesForOutputs) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
  func.func @main() -> (memref<1xi1>, memref<2xi1>, i1) {
    %single = memref.alloc() {mqt.classical_register_name = "single"}
        : memref<1xi1>
    %bits = memref.alloc() {mqt.classical_register_name = "bits"}
        : memref<2xi1>
    %qubit = qc.alloc : !qc.qubit
    %measured = qc.measure %qubit : !qc.qubit -> i1
    qc.dealloc %qubit : !qc.qubit
    return %single, %bits, %measured : memref<1xi1>, memref<2xi1>, i1
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

TEST(OpenQASM3EmissionTest, RejectsUnsupportedSubsetConcerns) {
  struct Fixture {
    llvm::StringLiteral name;
    llvm::StringLiteral source;
  };
  constexpr std::array fixtures{
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
