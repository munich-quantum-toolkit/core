/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/OQ3/IR/OQ3Ops.h"
#include "mlir/Dialect/OQ3/Transforms/Passes.h"
#include "mlir/Dialect/QC/IR/QCDialect.h"
#include "mlir/Dialect/QC/IR/QCOps.h"
#include "mlir/Target/OpenQASM/OpenQASM.h"

#include <gtest/gtest.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/Path.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>

#include <array>
#include <memory>
#include <string>

using namespace mlir;

namespace {

class OpenQASMTargetTest : public testing::Test {
protected:
  void SetUp() override {
    DialectRegistry registry;
    registry.insert<oq3::OQ3Dialect, qc::QCDialect, arith::ArithDialect,
                    func::FuncDialect, memref::MemRefDialect, math::MathDialect,
                    scf::SCFDialect>();
    context = std::make_unique<MLIRContext>(registry);
    context->loadAllAvailableDialects();
  }

  std::string translateFailure(StringRef source) {
    std::string diagnostic;
    ScopedDiagnosticHandler handler(context.get(), [&](Diagnostic& value) {
      llvm::raw_string_ostream stream(diagnostic);
      value.print(stream);
      diagnostic.push_back('\n');
      return success();
    });
    EXPECT_FALSE(oq3::translateOpenQASMToOQ3(source, *context));
    return diagnostic;
  }

  std::unique_ptr<MLIRContext> context;
};

TEST_F(OpenQASMTargetTest, BuildsTypedGateApplicationsAndBroadcasts) {
  auto module = oq3::translateOpenQASMToOQ3(R"qasm(
    OPENQASM 3.1;
    include "stdgates.inc";
    qubit[2] q;
    inv @ pow(2) @ x q;
    barrier q;
  )qasm",
                                            *context);
  ASSERT_TRUE(module);
  EXPECT_TRUE(succeeded(verify(module.get())));

  size_t applications = 0;
  module->walk([&](oq3::ApplyGateOp op) {
    ++applications;
    EXPECT_EQ(op.getCallee(), "x");
    EXPECT_EQ(op.getQubits().size(), 1);
    ASSERT_EQ(op.getModifierKinds().size(), 2);
    EXPECT_EQ(op.getModifierKinds()[0],
              static_cast<int32_t>(oq3::GateModifierKind::inv));
    EXPECT_EQ(op.getModifierKinds()[1],
              static_cast<int32_t>(oq3::GateModifierKind::pow));
    EXPECT_EQ(op.getModifierOperands().size(), 1);
  });
  EXPECT_EQ(applications, 2);
}

TEST_F(OpenQASMTargetTest, DoesNotInjectStandardGatesWithoutInclude) {
  const auto diagnostic = translateFailure(R"qasm(
    OPENQASM 3.1;
    qubit q;
    x q;
  )qasm");
  EXPECT_NE(diagnostic.find("unknown gate 'x'"), std::string::npos);
}

TEST_F(OpenQASMTargetTest, BuildsAndResolvesTypedCustomGates) {
  auto module = oq3::translateOpenQASMToOQ3(R"qasm(
    OPENQASM 3.1;
    include "stdgates.inc";
    gate pair(theta) a, b { rx(theta) a; x b; }
    qubit[2] q;
    pair(0.5) q[0], q[1];
  )qasm",
                                            *context);
  ASSERT_TRUE(module);
  EXPECT_TRUE(succeeded(verify(module.get())));
  EXPECT_TRUE(module->lookupSymbol<oq3::GateOp>("pair"));
}

TEST_F(OpenQASMTargetTest, RejectsGateUseBeforeDefinitionAndRecursion) {
  auto diagnostic = translateFailure(R"qasm(
    OPENQASM 3.1;
    qubit q;
    later q;
    gate later a { U(0, 0, 0) a; }
  )qasm");
  EXPECT_NE(diagnostic.find("unknown gate 'later'"), std::string::npos);

  diagnostic = translateFailure(R"qasm(
    OPENQASM 3.1;
    gate recurse a { recurse a; }
  )qasm");
  EXPECT_NE(diagnostic.find("recursive gate definition is not allowed"),
            std::string::npos);
}

TEST_F(OpenQASMTargetTest, TypedIRPrintsAndParsesWithoutFallbackOperations) {
  auto module = oq3::translateOpenQASMToOQ3(R"qasm(
    OPENQASM 3.1;
    include "stdgates.inc";
    qubit q;
    pow(0.5) @ inv @ rx(1) q;
  )qasm",
                                            *context);
  ASSERT_TRUE(module);
  std::string text;
  llvm::raw_string_ostream(text) << *module;
  auto parsed = parseSourceString<ModuleOp>(text, context.get());
  ASSERT_TRUE(parsed);
  EXPECT_TRUE(succeeded(verify(parsed.get())));
}

TEST_F(OpenQASMTargetTest, LowersCustomGatesAndOrderedModifiersToQC) {
  auto module = oq3::translateOpenQASMToOQ3(R"qasm(
    OPENQASM 3.1;
    include "stdgates.inc";
    gate pair(theta) a, b { rx(theta) a; cx a, b; }
    qubit[3] q;
    negctrl @ inv @ pair(0.5) q[0], q[1], q[2];
  )qasm",
                                            *context);
  ASSERT_TRUE(module);
  PassManager manager(context.get());
  manager.addPass(oq3::createLowerOQ3ToQCPass());
  ASSERT_TRUE(succeeded(manager.run(module.get())));
  EXPECT_TRUE(succeeded(verify(module.get())));

  size_t oq3Operations = 0;
  size_t controls = 0;
  size_t inverses = 0;
  module->walk([&](Operation* operation) {
    if (operation->getName().getDialectNamespace() == "oq3") {
      ++oq3Operations;
    }
    controls += isa<qc::CtrlOp>(operation);
    inverses += isa<qc::InvOp>(operation);
  });
  EXPECT_EQ(oq3Operations, 0);
  EXPECT_GE(controls, 2);
  EXPECT_EQ(inverses, 1);
}

TEST_F(OpenQASMTargetTest, KeepsPowerAsATargetCapabilityDiagnostic) {
  auto module = oq3::translateOpenQASMToOQ3(R"qasm(
    OPENQASM 3.1;
    include "stdgates.inc";
    qubit q;
    pow(0.5) @ x q;
  )qasm",
                                            *context);
  ASSERT_TRUE(module);
  std::string diagnostic;
  ScopedDiagnosticHandler handler(context.get(), [&](Diagnostic& value) {
    llvm::raw_string_ostream(diagnostic) << value;
    return success();
  });
  PassManager manager(context.get());
  manager.addPass(oq3::createLowerOQ3ToQCPass());
  EXPECT_TRUE(failed(manager.run(module.get())));
  EXPECT_NE(diagnostic.find("pow gate modifiers are preserved in OQ3"),
            std::string::npos);
}

TEST_F(OpenQASMTargetTest, ResolvesConfiguredIncludesWithSourceLocations) {
  llvm::SmallString<128> directory;
  ASSERT_FALSE(llvm::sys::fs::createUniqueDirectory("oq3-include", directory));
  llvm::SmallString<128> includePath(directory);
  llvm::sys::path::append(includePath, "local.inc");
  std::error_code error;
  llvm::raw_fd_ostream file(includePath, error);
  ASSERT_FALSE(error);
  file << "gate local a { U(0, 0, 0) a; }\n";
  file.close();

  oq3::OpenQASMTranslationOptions options;
  options.includeDirectories.emplace_back(directory.str());
  auto module = oq3::translateOpenQASMToOQ3(R"qasm(
    OPENQASM 3.1;
    include "local.inc";
    qubit q;
    local q;
  )qasm",
                                            *context, options);
  EXPECT_TRUE(module);
  EXPECT_FALSE(llvm::sys::fs::remove(includePath));
  EXPECT_FALSE(llvm::sys::fs::remove(directory));
}

TEST_F(OpenQASMTargetTest, SupportsOpenQASM2CompatibilitySyntax) {
  auto module = oq3::translateOpenQASMToOQ3(R"qasm(
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[2];
    x q;
    u2(0, 1) q[0];
  )qasm",
                                            *context);
  ASSERT_TRUE(module);
  EXPECT_EQ(module->getOperation()
                ->getAttrOfType<StringAttr>("oq3.version")
                .getValue(),
            "2.0-compat");
}

TEST_F(OpenQASMTargetTest, UsesTheOpenQASM31StandardGateSignatures) {
  auto module = oq3::translateOpenQASMToOQ3(R"qasm(
    OPENQASM 3.1;
    include "stdgates.inc";
    qubit[2] q;
    cu(0, 1, 2, 3) q[0], q[1];
    rx(pi / 2) q[0];
  )qasm",
                                            *context);
  EXPECT_TRUE(module);
}

TEST_F(OpenQASMTargetTest, SupportsBitRegistersAndMeasurementForms) {
  auto module = oq3::translateOpenQASMToOQ3(R"qasm(
    OPENQASM 3.1;
    qubit[2] q;
    bit[2] c;
    c = measure q;
    measure q[0];
  )qasm",
                                            *context);
  ASSERT_TRUE(module);
  size_t measurements = 0;
  module->walk([&](qc::MeasureOp) { ++measurements; });
  EXPECT_EQ(measurements, 3);

  module = oq3::translateOpenQASMToOQ3(R"qasm(
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[2];
    creg c[2];
    measure q -> c;
    if (c == 1) x q[0];
  )qasm",
                                       *context);
  ASSERT_TRUE(module);
  measurements = 0;
  module->walk([&](qc::MeasureOp) { ++measurements; });
  EXPECT_EQ(measurements, 2);
  size_t conditionals = 0;
  module->walk([&](scf::IfOp) { ++conditionals; });
  EXPECT_EQ(conditionals, 1);
}

TEST_F(OpenQASMTargetTest, CarriesMutableMeasurementStateThroughWhile) {
  auto module = oq3::translateOpenQASMToOQ3(R"qasm(
    OPENQASM 3.1;
    qubit q;
    bit c;
    c = measure q;
    while (c == 1) {
      reset q;
      c = measure q;
    }
  )qasm",
                                            *context);
  ASSERT_TRUE(module);
  EXPECT_TRUE(succeeded(verify(module.get())));
  size_t loops = 0;
  module->walk([&](scf::WhileOp) { ++loops; });
  EXPECT_EQ(loops, 1);
}

TEST_F(OpenQASMTargetTest, UsesBuiltinStorageAndArithmeticForScalarState) {
  auto module = oq3::translateOpenQASMToOQ3(R"qasm(
    OPENQASM 3.1;
    const int[16] limit = 3;
    int[16] value = limit;
    float[64] theta = pi / 2;
    bool flag = true;
    value = limit;
    theta = theta + 0.5;
    if ((value == limit) && flag) { barrier; }
  )qasm",
                                            *context);
  ASSERT_TRUE(module);
  EXPECT_TRUE(succeeded(verify(module.get())));
  size_t allocations = 0;
  size_t conditionals = 0;
  module->walk([&](memref::AllocaOp) { ++allocations; });
  module->walk([&](scf::IfOp) { ++conditionals; });
  EXPECT_EQ(allocations, 3);
  EXPECT_EQ(conditionals, 1);
}

TEST_F(OpenQASMTargetTest, PreservesDeclaredInputAndOutputOrderAndTypes) {
  auto module = oq3::translateOpenQASMToOQ3(R"qasm(
    OPENQASM 3.1;
    output bit[2] first_output;
    input bit[3] first_input;
    output bit second_output;
    input bit[4] second_input;
  )qasm",
                                            *context);
  ASSERT_TRUE(module);
  auto main = module->lookupSymbol<func::FuncOp>("main");
  ASSERT_TRUE(main);
  ASSERT_EQ(main.getArgumentTypes().size(), 2);
  ASSERT_EQ(main.getResultTypes().size(), 2);
  EXPECT_EQ(cast<oq3::BitType>(main.getArgumentTypes()[0]).getWidth(), 3);
  EXPECT_EQ(cast<oq3::BitType>(main.getArgumentTypes()[1]).getWidth(), 4);
  EXPECT_EQ(cast<oq3::BitType>(main.getResultTypes()[0]).getWidth(), 2);
  EXPECT_EQ(cast<oq3::BitType>(main.getResultTypes()[1]).getWidth(), 1);
  EXPECT_TRUE(succeeded(verify(module.get())));

  PassManager manager(context.get());
  manager.addPass(oq3::createLowerOQ3ToQCPass());
  ASSERT_TRUE(succeeded(manager.run(module.get())));
  main = module->lookupSymbol<func::FuncOp>("main");
  EXPECT_TRUE(main.getArgumentTypes()[0].isInteger(3));
  EXPECT_TRUE(main.getArgumentTypes()[1].isInteger(4));
  EXPECT_TRUE(main.getResultTypes()[0].isInteger(2));
  EXPECT_TRUE(main.getResultTypes()[1].isInteger(1));
  size_t oq3Operations = 0;
  module->walk([&](Operation* operation) {
    oq3Operations += operation->getName().getDialectNamespace() == "oq3";
  });
  EXPECT_EQ(oq3Operations, 0);
  EXPECT_TRUE(succeeded(verify(module.get())));
}

TEST_F(OpenQASMTargetTest, DefaultsVersionlessProgramsToOpenQASM31) {
  auto module = oq3::translateOpenQASMToOQ3("qubit q;", *context);
  ASSERT_TRUE(module);
  EXPECT_EQ(module->getOperation()
                ->getAttrOfType<StringAttr>("oq3.version")
                .getValue(),
            "3.1");
}

TEST_F(OpenQASMTargetTest, RejectsOpenQASM30) {
  const auto diagnostic = translateFailure("OPENQASM 3.0; qubit q;");
  EXPECT_NE(diagnostic.find("unsupported OpenQASM version '3.0'"),
            std::string::npos);
}

TEST_F(OpenQASMTargetTest, DiagnosesMisplacedAndDuplicateVersionsPrecisely) {
  auto diagnostic = translateFailure("qubit q; OPENQASM 3.1;");
  EXPECT_NE(
      diagnostic.find(
          "version declaration must be the first non-comment source item"),
      std::string::npos);

  diagnostic = translateFailure("OPENQASM 3.1; OPENQASM 3.1; qubit q;");
  EXPECT_NE(diagnostic.find("only one version declaration"), std::string::npos);
}

TEST_F(OpenQASMTargetTest, RejectsConstantZeroRangeStep) {
  const auto diagnostic = translateFailure(R"qasm(
    OPENQASM 3.1;
    for int i in [0:0:4] { barrier; }
  )qasm");
  EXPECT_NE(diagnostic.find("range step cannot be zero"), std::string::npos);
}

TEST_F(OpenQASMTargetTest, DiagnosesUnprovenDynamicStepOnlyAtLowering) {
  OpBuilder builder(context.get());
  auto module = ModuleOp::create(builder.getUnknownLoc());
  auto function =
      func::FuncOp::create(builder, builder.getUnknownLoc(), "main",
                           builder.getFunctionType({builder.getI64Type()}, {}));
  module.getBody()->push_back(function);
  Block* entry = function.addEntryBlock();
  builder.setInsertionPointToStart(entry);
  Value start =
      arith::ConstantIntOp::create(builder, builder.getUnknownLoc(), 0, 64);
  Value stop =
      arith::ConstantIntOp::create(builder, builder.getUnknownLoc(), 4, 64);
  OperationState state(builder.getUnknownLoc(), oq3::ForOp::getOperationName());
  state.addOperands({start, stop, entry->getArgument(0)});
  Region* body = state.addRegion();
  body->push_back(new Block());
  body->front().addArgument(builder.getI64Type(), builder.getUnknownLoc());
  auto loop = cast<oq3::ForOp>(builder.create(state));
  builder.setInsertionPointToStart(&loop.getBody().front());
  oq3::YieldOp::create(builder, builder.getUnknownLoc());
  builder.setInsertionPointToEnd(entry);
  func::ReturnOp::create(builder, builder.getUnknownLoc());
  ASSERT_TRUE(succeeded(verify(module)));

  std::string diagnostic;
  ScopedDiagnosticHandler handler(context.get(), [&](Diagnostic& value) {
    llvm::raw_string_ostream(diagnostic) << value;
    return success();
  });
  PassManager manager(context.get());
  manager.addPass(oq3::createLowerOQ3ToQCPass());
  EXPECT_TRUE(failed(manager.run(module)));
  EXPECT_NE(diagnostic.find("dynamic range step cannot be proven nonzero for "
                            "the selected target"),
            std::string::npos);
}

TEST_F(OpenQASMTargetTest, PreservesDynamicStepsFromSourceUntilLowering) {
  auto module = oq3::translateOpenQASMToOQ3(R"qasm(
    OPENQASM 3.1;
    qubit q;
    bit step;
    step = measure q;
    for int i in [0:step:4] { reset q; }
  )qasm",
                                            *context);
  ASSERT_TRUE(module);
  size_t oq3Loops = 0;
  module->walk([&](oq3::ForOp) { ++oq3Loops; });
  EXPECT_EQ(oq3Loops, 1);

  std::string diagnostic;
  ScopedDiagnosticHandler handler(context.get(), [&](Diagnostic& value) {
    llvm::raw_string_ostream(diagnostic) << value;
    return success();
  });
  PassManager manager(context.get());
  manager.addPass(oq3::createLowerOQ3ToQCPass());
  EXPECT_TRUE(failed(manager.run(module.get())));
  EXPECT_NE(diagnostic.find("dynamic range step cannot be proven nonzero for "
                            "the selected target"),
            std::string::npos);
}

TEST_F(OpenQASMTargetTest, MakesTheVisibleInductionValueAvailableToGateCalls) {
  auto module = oq3::translateOpenQASMToOQ3(R"qasm(
    OPENQASM 3.1;
    include "stdgates.inc";
    qubit q;
    for int i in [0:1:2] { rx(i) q; }
  )qasm",
                                            *context);
  ASSERT_TRUE(module);
  PassManager manager(context.get());
  manager.addPass(oq3::createLowerOQ3ToQCPass());
  EXPECT_TRUE(succeeded(manager.run(module.get())));
  EXPECT_TRUE(succeeded(verify(module.get())));
}

TEST_F(OpenQASMTargetTest, LowersInclusiveRangesWithoutEndpointArithmetic) {
  constexpr std::array<StringRef, 6> programs = {
      "for int i in [0:1:4] { barrier; }",
      "for int i in [4:-1:0] { barrier; }",
      "for int i in [2:1:2] { barrier; }",
      "for int i in [3:1:2] { barrier; }",
      "for int i in [0:2:3] { barrier; }",
      "for int i in [9223372036854775806:1:9223372036854775807] { barrier; }",
  };
  for (const StringRef program : programs) {
    auto module = oq3::translateOpenQASMToOQ3(program, *context);
    ASSERT_TRUE(module) << program.str();
    PassManager manager(context.get());
    manager.addPass(oq3::createLowerOQ3ToQCPass());
    ASSERT_TRUE(succeeded(manager.run(module.get()))) << program.str();
    EXPECT_TRUE(succeeded(verify(module.get()))) << program.str();
    size_t loops = 0;
    module->walk([&](scf::WhileOp loop) {
      ++loops;
      EXPECT_EQ(cast<IntegerType>(loop.getResultTypes().front()).getWidth(),
                65);
    });
    EXPECT_EQ(loops, 1) << program.str();
  }
}

TEST_F(OpenQASMTargetTest, IdentifiesUnsupportedFeatureFamily) {
  const auto diagnostic = translateFailure(R"qasm(
    OPENQASM 3.1;
    def f(int x) -> int { return x; }
  )qasm");
  EXPECT_NE(diagnostic.find("subroutines and externs"), std::string::npos);
}

TEST_F(OpenQASMTargetTest, RejectsZeroWidthSourceTypes) {
  EXPECT_FALSE(oq3::BitType::getChecked(
      [&]() { return emitError(UnknownLoc::get(context.get())); },
      context.get(), 0U));
  EXPECT_FALSE(oq3::AngleType::getChecked(
      [&]() { return emitError(UnknownLoc::get(context.get())); },
      context.get(), 0U));
}

} // namespace
