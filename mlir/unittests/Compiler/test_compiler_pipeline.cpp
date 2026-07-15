/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "TestCaseUtils.h"
#include "ir/QuantumComputation.hpp"
#include "mlir/Compiler/Programs.h"
#include "mlir/Dialect/QC/Builder/QCProgramBuilder.h"
#include "mlir/Dialect/QC/IR/QCDialect.h"
#include "mlir/Dialect/QC/Translation/TranslateQuantumComputationToQC.h"
#include "mlir/Dialect/QCO/Builder/QCOProgramBuilder.h"
#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QIR/Builder/QIRProgramBuilder.h"
#include "mlir/Dialect/QTensor/IR/QTensorDialect.h"
#include "mlir/Support/IRVerification.h"
#include "mlir/Support/Passes.h"
#include "qc_programs.h"
#include "qco_programs.h"
#include "qir_programs.h"
#include "quantum_computation_programs.h"

#include <gtest/gtest.h>
#include <jeff/IR/JeffDialect.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LLVM.h>

#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace mqt::test::compiler {

using namespace mlir;
using namespace mlir::qc;
using namespace mlir::qco;
using namespace mlir::qir;

using QCProgramBuilderFn = NamedMLIRBuilder<QCProgramBuilder>;
using QIRProgramBuilderFn = NamedMLIRBuilder<QIRProgramBuilder>;
using QuantumComputationBuilderFn = NamedBuilder<::qc::QuantumComputation>;

namespace {

struct CompilerPipelineTestCase {
  std::string name;
  QuantumComputationBuilderFn quantumComputationBuilder;
  QCProgramBuilderFn qcProgramBuilder;
  QCProgramBuilderFn qcReferenceBuilder;
  QIRProgramBuilderFn qirReferenceBuilder;
  bool startFromQuantumComputation = true;
  bool convertToQIR = true;

  friend std::ostream& operator<<(std::ostream& os,
                                  const CompilerPipelineTestCase& info);
};

// NOLINTNEXTLINE(llvm-prefer-static-over-anonymous-namespace)
std::ostream& operator<<(std::ostream& os,
                         const CompilerPipelineTestCase& info) {
  os << "CompilerPipeline{" << info.name << ", original=";
  if (info.startFromQuantumComputation) {
    os << displayName(info.quantumComputationBuilder.name);
  } else {
    os << displayName(info.qcProgramBuilder.name);
  }
  os << ", qcReference=" << displayName(info.qcReferenceBuilder.name);
  if (info.convertToQIR) {
    os << ", qirReference=" << displayName(info.qirReferenceBuilder.name);
  }
  return os << "}";
}

class CompilerPipelineTest
    : public testing::TestWithParam<CompilerPipelineTestCase> {
protected:
  std::unique_ptr<MLIRContext> context;

  void SetUp() override {
    DialectRegistry registry;
    registry.insert<QCDialect, QCODialect, qtensor::QTensorDialect,
                    arith::ArithDialect, cf::ControlFlowDialect,
                    func::FuncDialect, memref::MemRefDialect, scf::SCFDialect,
                    LLVM::LLVMDialect, jeff::JeffDialect>();
    context = std::make_unique<MLIRContext>();
    context->appendDialectRegistry(registry);
    context->loadAllAvailableDialects();
  }

  [[nodiscard]] OwningOpRef<ModuleOp>
  buildQCReference(const QCProgramBuilderFn builder) const {
    auto module = mqt::test::buildMLIRProgram(context.get(), builder);
    EXPECT_TRUE(runQCCleanupPipeline(module.get()).succeeded());
    return module;
  }

  [[nodiscard]] OwningOpRef<ModuleOp>
  buildQIRReference(const QIRProgramBuilderFn builder) const {
    auto module = mqt::test::buildMLIRProgram(
        context.get(), builder, QIRProgramBuilder::Profile::Adaptive);
    EXPECT_TRUE(runQIRCleanupPipeline(module.get(), true).succeeded());
    return module;
  }

  [[nodiscard]] OwningOpRef<ModuleOp>
  parseRecordedModule(const std::string& ir) const {
    return parseSourceString<ModuleOp>(ir, context.get());
  }

  void expectEquivalent(const std::string& stage, const std::string& ir,
                        const ModuleOp expected) const {
    auto actual = parseRecordedModule(ir);
    ASSERT_TRUE(actual) << stage << " failed to parse";
    EXPECT_TRUE(verify(*actual).succeeded());
    EXPECT_TRUE(verify(expected).succeeded());
    EXPECT_TRUE(areModulesEquivalentWithPermutations(actual.get(), expected));
  }
};

} // namespace

TEST_P(CompilerPipelineTest, EndToEndPipeline) {
  const auto& testCase = GetParam();
  const auto name = " (" + testCase.name + ")";
  DeferredPrinter printer;

  OwningOpRef<ModuleOp> module;
  if (testCase.startFromQuantumComputation) {
    ASSERT_TRUE(testCase.quantumComputationBuilder);
    ::qc::QuantumComputation comp;
    testCase.quantumComputationBuilder.fn(comp);

    module = translateQuantumComputationToQC(context.get(), comp);
    ASSERT_TRUE(module);
    printer.record(module.get(), "QC Import" + name);
  } else {
    ASSERT_TRUE(testCase.qcProgramBuilder);
    module =
        mqt::test::buildMLIRProgram(context.get(), testCase.qcProgramBuilder);
    ASSERT_TRUE(module);
    printer.record(module.get(), "QC Input" + name);
  }
  EXPECT_TRUE(verify(*module).succeeded());

  std::string source;
  llvm::raw_string_ostream sourceStream(source);
  module->print(sourceStream);
  auto input = QCProgram::fromMLIRString(source);
  ASSERT_TRUE(input);
  auto compiled = runDefaultPipeline(
      CompilerInput{std::move(*input)},
      testCase.convertToQIR ? ProgramFormat::QIRAdaptive : ProgramFormat::QC);
  ASSERT_TRUE(compiled);

  OwningOpRef<ModuleOp> expected;
  if (testCase.convertToQIR) {
    ASSERT_TRUE(testCase.qirReferenceBuilder);
    expected = buildQIRReference(testCase.qirReferenceBuilder);
  } else {
    ASSERT_TRUE(testCase.qcReferenceBuilder);
    expected = buildQCReference(testCase.qcReferenceBuilder);
  }
  ASSERT_TRUE(expected);
  const auto actualIR =
      std::visit([](const auto& value) { return value.str(); }, *compiled);
  expectEquivalent("Final output", actualIR, expected.get());
}

/** @brief Raw QCO stops before the registered default optimization pipeline. */

TEST_F(CompilerPipelineTest, RawAndOptimizedQCOAreDistinctCheckpoints) {
  const std::string qasm = R"(OPENQASM 3.0;
include "stdgates.inc";
qubit q;
rz(1.0) q;
rx(1.0) q;
)";
  auto rawInput = QCProgram::fromQASMString(qasm);
  auto optimizedInput = QCProgram::fromQASMString(qasm);
  ASSERT_TRUE(rawInput);
  ASSERT_TRUE(optimizedInput);

  auto raw = runDefaultPipeline(CompilerInput{std::move(*rawInput)},
                                ProgramFormat::QCO);
  auto optimized = runDefaultPipeline(CompilerInput{std::move(*optimizedInput)},
                                      ProgramFormat::QCOOptimized);
  ASSERT_TRUE(raw);
  ASSERT_TRUE(optimized);
  EXPECT_NE(std::get<QCOProgram>(*raw).str(),
            std::get<QCOProgram>(*optimized).str());
}

TEST_F(CompilerPipelineTest, CustomTextualQCOOptimizationPipeline) {
  const std::string qasm = R"(OPENQASM 3.0;
include "stdgates.inc";
qubit q;
x q;
h q;
)";
  auto input = QCProgram::fromQASMString(qasm);
  ASSERT_TRUE(input);
  auto result =
      runDefaultPipeline(CompilerInput{std::move(*input)},
                         ProgramFormat::QCOOptimized, "hadamard-lifting");
  ASSERT_TRUE(result);
  EXPECT_FALSE(std::get<QCOProgram>(*result).str().empty());
}

/**
 * @brief Test: typed programs transfer ownership between compiler dialects
 */
TEST_F(CompilerPipelineTest, TypedProgramsComposeWithoutImplicitCopies) {
  const std::string qasm = R"(OPENQASM 3.0;
include "stdgates.inc";
qubit q;
h q;
)";

  auto qcResult = QCProgram::fromQASMString(qasm);
  ASSERT_TRUE(qcResult);
  auto qc = std::move(*qcResult);
  EXPECT_TRUE(qc.isValid());
  auto qcoResult = std::move(qc).intoQCO();
  ASSERT_TRUE(qcoResult);
  auto qco = std::move(*qcoResult);
  EXPECT_TRUE(qco.isValid());

  EXPECT_TRUE(qco.cleanup());
  EXPECT_TRUE(qco.mergeSingleQubitRotationGates());
  EXPECT_TRUE(qco.isValid());
  auto roundTripResult = std::move(qco).intoQC();
  ASSERT_TRUE(roundTripResult);
  auto roundTrip = std::move(*roundTripResult);
  EXPECT_TRUE(roundTrip.isValid());
  EXPECT_TRUE(roundTrip.cleanup());
  auto reparsed = parseRecordedModule(roundTrip.str());
  ASSERT_TRUE(reparsed);
  EXPECT_TRUE(mlir::verify(*reparsed).succeeded());
}

/**
 * @brief Test: typed programs import MLIR and OpenQASM from their public APIs
 */
TEST_F(CompilerPipelineTest, TypedProgramImportsAndCopies) {
  const std::string mlir = R"(module {
  %0 = qc.alloc : !qc.qubit
  qc.dealloc %0 : !qc.qubit
})";
  const std::string qasm = R"(OPENQASM 3.0;
include "stdgates.inc";
qubit q;
h q;
)";
  const auto temporaryDirectory = std::filesystem::path(testing::TempDir());
  const auto mlirPath = temporaryDirectory / "typed_program_input.mlir";
  const auto qasmPath = temporaryDirectory / "typed_program_input.qasm";
  std::ofstream(mlirPath) << mlir;
  std::ofstream(qasmPath) << qasm;

  auto qcFromMLIR = QCProgram::fromMLIRString(mlir);
  auto qcFromMLIRFile = QCProgram::fromMLIRFile(mlirPath);
  auto qcFromQASM = QCProgram::fromQASMString(qasm);
  auto qcFromQASMFile = QCProgram::fromQASMFile(qasmPath);
  ::qc::QuantumComputation computation;
  computation.addQubitRegister(1);
  computation.h(0);
  auto qcFromComputation = QCProgram::fromQuantumComputation(computation);

  ASSERT_TRUE(qcFromMLIR);
  ASSERT_TRUE(qcFromMLIRFile);
  ASSERT_TRUE(qcFromQASM);
  ASSERT_TRUE(qcFromQASMFile);
  ASSERT_TRUE(qcFromComputation);
  EXPECT_EQ(qcFromMLIR->str(), qcFromMLIRFile->str());
  EXPECT_EQ(qcFromQASM->str(), qcFromQASMFile->str());
  EXPECT_EQ(qcFromMLIR->str(), qcFromMLIR->copy().str());
  EXPECT_FALSE(qcFromComputation->str().empty());
  EXPECT_FALSE(QCProgram::fromMLIRString("not valid MLIR"));
  EXPECT_FALSE(QCProgram::fromMLIRFile(temporaryDirectory / "missing.mlir"));
  EXPECT_FALSE(QCProgram::fromQASMString("not valid OpenQASM"));
  EXPECT_FALSE(QCProgram::fromQASMFile(temporaryDirectory / "missing.qasm"));
  EXPECT_FALSE(QCOProgram::fromMLIRString("not valid MLIR"));
  EXPECT_FALSE(
      QCOProgram::fromMLIRFile(temporaryDirectory / "missing.qco.mlir"));
  auto qcoFromQC = std::move(*qcFromMLIR).intoQCO();
  ASSERT_TRUE(qcoFromQC);
  EXPECT_FALSE(QCProgram::fromMLIRString(qcoFromQC->str()));
  EXPECT_FALSE(QCOProgram::fromMLIRString(mlir));
}

/**
 * @brief Test: Jeff programs round-trip through their binary APIs
 */
TEST_F(CompilerPipelineTest, JeffProgramsRoundTripThroughBytesAndFiles) {
  const std::string qasm = R"(OPENQASM 3.0;
include "stdgates.inc";
qubit q;
x q;
)";
  const auto path = std::filesystem::path(testing::TempDir()) /
                    "typed_program_round_trip.jeff";

  auto qc = QCProgram::fromQASMString(qasm);
  ASSERT_TRUE(qc);
  auto qco = std::move(*qc).intoQCO();
  ASSERT_TRUE(qco);
  auto jeffResult = std::move(*qco).intoJeff();
  ASSERT_TRUE(jeffResult);
  auto jeff = std::move(*jeffResult);
  const auto bytes = jeff.toBytes();
  ASSERT_FALSE(bytes.empty());
  ASSERT_TRUE(jeff.write(path));

  auto fromBytes = JeffProgram::fromBytes(bytes);
  auto fromFile = JeffProgram::fromFile(path);
  ASSERT_TRUE(fromBytes);
  ASSERT_TRUE(fromFile);
  EXPECT_EQ(fromBytes->str(), fromFile->str());
  EXPECT_EQ(fromBytes->toBytes(), bytes);
  EXPECT_EQ(jeff.copy().toBytes(), bytes);
  EXPECT_TRUE(fromBytes->cleanup());
  EXPECT_FALSE(fromBytes->str().empty());

  auto roundTrip = std::move(*fromFile).intoQCO();
  ASSERT_TRUE(roundTrip);
  auto reparsed = parseRecordedModule(roundTrip->str());
  ASSERT_TRUE(reparsed);
  EXPECT_TRUE(mlir::verify(*reparsed).succeeded());
  const std::vector<std::byte> invalid(1);
  EXPECT_FALSE(JeffProgram::fromBytes(invalid));
  EXPECT_FALSE(jeff.write(path.parent_path() / "missing" / "output.jeff"));
}

/**
 * @brief Test: QCO and QIR typed programs retain their respective semantics
 */
TEST_F(CompilerPipelineTest, QCOAndQIRProgramsImportCopyAndOptimize) {
  const std::string qasm = R"(OPENQASM 3.0;
include "stdgates.inc";
qubit q;
h q;
)";
  const auto qcoPath = std::filesystem::path(testing::TempDir()) /
                       "typed_program_input.qco.mlir";

  auto qc = QCProgram::fromQASMString(qasm);
  ASSERT_TRUE(qc);
  auto qco = std::move(*qc).intoQCO();
  ASSERT_TRUE(qco);
  const auto qcoIR = qco->str();
  std::ofstream(qcoPath) << qcoIR;
  auto qcoFromString = QCOProgram::fromMLIRString(qcoIR);
  auto qcoFromFile = QCOProgram::fromMLIRFile(qcoPath);
  ASSERT_TRUE(qcoFromString);
  ASSERT_TRUE(qcoFromFile);
  EXPECT_EQ(qcoFromString->str(), qcoFromFile->str());
  EXPECT_EQ(qcoFromString->str(), qcoFromString->copy().str());
  EXPECT_TRUE(qcoFromString->liftHadamards());
  EXPECT_TRUE(
      qcoFromString->runPassPipeline("merge-single-qubit-rotation-gates"));
  EXPECT_TRUE(qcoFromString->runPassPipeline("canonicalize,cse"));
  EXPECT_FALSE(qcoFromString->runPassPipeline("not-a-pass"));
  EXPECT_FALSE(qcoFromString->str().empty());

  auto baseInput = QCProgram::fromQASMString(qasm);
  auto adaptiveInput = QCProgram::fromQASMString(qasm);
  ASSERT_TRUE(baseInput);
  ASSERT_TRUE(adaptiveInput);
  auto base = std::move(*baseInput).intoQIR(QIRProfile::Base);
  auto adaptive = std::move(*adaptiveInput).intoQIR(QIRProfile::Adaptive);
  ASSERT_TRUE(base);
  ASSERT_TRUE(adaptive);
  EXPECT_EQ(base->copy().profile(), QIRProfile::Base);
  EXPECT_EQ(adaptive->copy().profile(), QIRProfile::Adaptive);
  auto llvmIR = base->llvmIR();
  ASSERT_TRUE(llvmIR);
  EXPECT_FALSE(llvmIR->empty());
  auto bitcode = base->toBitcode();
  ASSERT_TRUE(bitcode);
  ASSERT_GE(bitcode->size(), 4U);
  EXPECT_EQ((*bitcode)[0], std::byte{'B'});
  EXPECT_EQ((*bitcode)[1], std::byte{'C'});
  const auto bitcodePath =
      std::filesystem::path(testing::TempDir()) / "typed_program_output.bc";
  EXPECT_TRUE(base->writeBitcode(bitcodePath));
  EXPECT_FALSE(
      base->writeBitcode(bitcodePath.parent_path() / "missing" / "output.bc"));
}

/**
 * @brief Test: QCO program APIs configure and execute their associated passes.
 */
TEST_F(CompilerPipelineTest, QCOProgramOptimizationAPIs) {
  const std::string qasm = R"(OPENQASM 3.0;
include "stdgates.inc";
qubit[3] q;
h q[0];
x q[0];
cx q[0], q[2];
)";
  auto qc = QCProgram::fromQASMString(qasm);
  ASSERT_TRUE(qc);
  auto qcoResult = std::move(*qc).intoQCO();
  ASSERT_TRUE(qcoResult);
  auto qco = std::move(*qcoResult);
  const auto beforeFusion = qco.str();

  EXPECT_TRUE(qco.fuseSingleQubitUnitaryRuns("zyz"));
  EXPECT_NE(qco.str(), beforeFusion);
  const std::vector<std::pair<std::size_t, std::size_t>> coupling = {
      {0, 1}, {1, 0}, {1, 2}, {2, 1}};
  EXPECT_TRUE(qco.placeAndRoute(coupling));
  EXPECT_TRUE(qco.runPassPipeline("mqt-qco-default", true, true));

  auto loopModule = mqt::test::buildMLIRProgram(
      context.get(), MQT_NAMED_BUILDER(qco::simpleForLoop));
  ASSERT_TRUE(loopModule);
  std::string loopIR;
  llvm::raw_string_ostream stream(loopIR);
  loopModule->print(stream);
  auto loopProgram = QCOProgram::fromMLIRString(loopIR);
  ASSERT_TRUE(loopProgram);
  EXPECT_NE(loopProgram->str().find("scf.for"), std::string::npos);
  EXPECT_TRUE(loopProgram->unrollQuantumLoops());
  EXPECT_EQ(loopProgram->str().find("scf.for"), std::string::npos);
}

/**
 * @brief Test: default compilation returns the requested typed program format
 */
TEST_F(CompilerPipelineTest, DefaultPipelineSelectsRequestedProgramFormats) {
  const std::string qasm = R"(OPENQASM 3.0;
include "stdgates.inc";
qubit q;
h q;
)";
  const auto compile = [&qasm](const ProgramFormat output) {
    auto input = QCProgram::fromQASMString(qasm);
    EXPECT_TRUE(input);
    return runDefaultPipeline(CompilerInput{std::move(*input)}, output);
  };

  auto qcOutput = compile(ProgramFormat::QC);
  auto qcoOutput = compile(ProgramFormat::QCO);
  auto optimizedQCOOutput = compile(ProgramFormat::QCOOptimized);
  auto jeffOutput = compile(ProgramFormat::Jeff);
  ASSERT_TRUE(qcOutput);
  ASSERT_TRUE(qcoOutput);
  ASSERT_TRUE(optimizedQCOOutput);
  ASSERT_TRUE(jeffOutput);
  EXPECT_TRUE(std::holds_alternative<QCProgram>(*qcOutput));
  EXPECT_TRUE(std::holds_alternative<QCOProgram>(*qcoOutput));
  EXPECT_TRUE(std::holds_alternative<QCOProgram>(*optimizedQCOOutput));
  EXPECT_TRUE(std::holds_alternative<JeffProgram>(*jeffOutput));

  auto profiledInput = QCProgram::fromQASMString(qasm);
  ASSERT_TRUE(profiledInput);
  auto profiled = runDefaultPipeline(CompilerInput{std::move(*profiledInput)},
                                     ProgramFormat::QCOOptimized,
                                     "mqt-qco-default", true, true);
  ASSERT_TRUE(profiled);
  EXPECT_TRUE(std::holds_alternative<QCOProgram>(*profiled));

  auto customPipelineInput = QCProgram::fromQASMString(qasm);
  ASSERT_TRUE(customPipelineInput);
  EXPECT_FALSE(runDefaultPipeline(
      CompilerInput{std::move(*customPipelineInput)}, ProgramFormat::QCO,
      "builtin.module(merge-single-qubit-rotation-gates)"));

  auto base = compile(ProgramFormat::QIRBase);
  ASSERT_TRUE(base);
  ASSERT_TRUE(std::holds_alternative<QIRProgram>(*base));
  EXPECT_EQ(std::get<QIRProgram>(*base).profile(), QIRProfile::Base);
  EXPECT_TRUE(std::get<QIRProgram>(*base).llvmIR());

  auto adaptive = compile(ProgramFormat::QIRAdaptive);
  ASSERT_TRUE(adaptive);
  ASSERT_TRUE(std::holds_alternative<QIRProgram>(*adaptive));
  EXPECT_EQ(std::get<QIRProgram>(*adaptive).profile(), QIRProfile::Adaptive);

  auto imported = QCProgram::fromQASMString(qasm);
  ASSERT_TRUE(imported);
  auto importedResult = runDefaultPipeline(CompilerInput{std::move(*imported)},
                                           ProgramFormat::QCImport);
  ASSERT_TRUE(importedResult);
  EXPECT_TRUE(std::holds_alternative<QCProgram>(*importedResult));

  auto qcoInput = QCProgram::fromQASMString(qasm);
  ASSERT_TRUE(qcoInput);
  auto qco = std::move(*qcoInput).intoQCO();
  ASSERT_TRUE(qco);
  EXPECT_FALSE(
      runDefaultPipeline(CompilerInput{qco->copy()}, ProgramFormat::QCImport));
  EXPECT_FALSE(runDefaultPipeline(CompilerInput{qco->copy()},
                                  ProgramFormat::QCImport,
                                  "merge-single-qubit-rotation-gates"));
  auto fromQCO =
      runDefaultPipeline(CompilerInput{std::move(*qco)}, ProgramFormat::QC);
  ASSERT_TRUE(fromQCO);
  EXPECT_TRUE(std::holds_alternative<QCProgram>(*fromQCO));

  auto jeffInput = QCProgram::fromQASMString(qasm);
  ASSERT_TRUE(jeffInput);
  auto jeffQCO = std::move(*jeffInput).intoQCO();
  ASSERT_TRUE(jeffQCO);
  auto jeff = std::move(*jeffQCO).intoJeff();
  ASSERT_TRUE(jeff);
  auto fromJeff =
      runDefaultPipeline(CompilerInput{std::move(*jeff)}, ProgramFormat::QC);
  ASSERT_TRUE(fromJeff);
  EXPECT_TRUE(std::holds_alternative<QCProgram>(*fromJeff));
}

INSTANTIATE_TEST_SUITE_P(
    QuantumComputationPipelineProgramsTest, CompilerPipelineTest,
    testing::Values(
        CompilerPipelineTestCase{
            "StaticQubits", nullptr, MQT_NAMED_BUILDER(mlir::qc::staticQubits),
            MQT_NAMED_BUILDER(mlir::qc::staticQubits),
            MQT_NAMED_BUILDER(mlir::qir::staticQubits), false},
        CompilerPipelineTestCase{
            "StaticQubitsWithOps", nullptr,
            MQT_NAMED_BUILDER(mlir::qc::staticQubitsWithOps),
            MQT_NAMED_BUILDER(mlir::qc::staticQubitsWithOps),
            MQT_NAMED_BUILDER(mlir::qir::staticQubitsWithOps), false},
        CompilerPipelineTestCase{
            "StaticQubitsWithParametricOps", nullptr,
            MQT_NAMED_BUILDER(mlir::qc::staticQubitsWithParametricOps),
            MQT_NAMED_BUILDER(mlir::qc::staticQubitsWithParametricOps),
            MQT_NAMED_BUILDER(mlir::qir::staticQubitsWithParametricOps), false},
        CompilerPipelineTestCase{
            "StaticQubitsWithTwoTargetOps", nullptr,
            MQT_NAMED_BUILDER(mlir::qc::staticQubitsWithTwoTargetOps),
            MQT_NAMED_BUILDER(mlir::qc::staticQubitsWithTwoTargetOps),
            MQT_NAMED_BUILDER(mlir::qir::staticQubitsWithTwoTargetOps), false},
        CompilerPipelineTestCase{
            "StaticQubitsWithCtrl", nullptr,
            MQT_NAMED_BUILDER(mlir::qc::staticQubitsWithCtrl),
            MQT_NAMED_BUILDER(mlir::qc::staticQubitsWithCtrl),
            MQT_NAMED_BUILDER(mlir::qir::staticQubitsWithCtrl), false},
        CompilerPipelineTestCase{
            "StaticQubitsWithInv", nullptr,
            MQT_NAMED_BUILDER(mlir::qc::staticQubitsWithInv),
            MQT_NAMED_BUILDER(mlir::qc::staticQubitsWithInv),
            MQT_NAMED_BUILDER(mlir::qir::staticQubitsWithInv), false},
        CompilerPipelineTestCase{
            "AllocQubit", MQT_NAMED_BUILDER(::qc::allocQubit), nullptr,
            MQT_NAMED_BUILDER(mlir::qc::alloc1QubitRegister),
            MQT_NAMED_BUILDER(mlir::qir::alloc1QubitRegister<true>)},
        CompilerPipelineTestCase{
            "AllocQubitRegister", MQT_NAMED_BUILDER(::qc::allocQubitRegister),
            nullptr, MQT_NAMED_BUILDER(mlir::qc::allocQubitRegister),
            MQT_NAMED_BUILDER(mlir::qir::allocQubitRegister<true>)},
        CompilerPipelineTestCase{
            "AllocMultipleQubitRegisters",
            MQT_NAMED_BUILDER(::qc::allocMultipleQubitRegisters), nullptr,
            MQT_NAMED_BUILDER(mlir::qc::allocMultipleQubitRegisters),
            MQT_NAMED_BUILDER(mlir::qir::allocMultipleQubitRegisters<true>)},
        CompilerPipelineTestCase{
            "AllocLargeRegister", MQT_NAMED_BUILDER(::qc::allocLargeRegister),
            nullptr, MQT_NAMED_BUILDER(mlir::qc::allocLargeRegister),
            MQT_NAMED_BUILDER(mlir::qir::allocQubitRegister<true>)},
        CompilerPipelineTestCase{
            "SingleMeasurementToSingleBit",
            MQT_NAMED_BUILDER(::qc::singleMeasurementToSingleBit), nullptr,
            MQT_NAMED_BUILDER(mlir::qc::singleMeasurementToSingleBit),
            MQT_NAMED_BUILDER(mlir::qir::singleMeasurementToSingleBit<true>)},
        CompilerPipelineTestCase{
            "RepeatedMeasurementToSameBit",
            MQT_NAMED_BUILDER(::qc::repeatedMeasurementToSameBit), nullptr,
            MQT_NAMED_BUILDER(mlir::qc::repeatedMeasurementToSameBit),
            MQT_NAMED_BUILDER(mlir::qir::repeatedMeasurementToSameBit<true>)},
        CompilerPipelineTestCase{
            "RepeatedMeasurementToDifferentBits",
            MQT_NAMED_BUILDER(::qc::repeatedMeasurementToDifferentBits),
            nullptr,
            MQT_NAMED_BUILDER(mlir::qc::repeatedMeasurementToDifferentBits),
            MQT_NAMED_BUILDER(
                mlir::qir::repeatedMeasurementToDifferentBits<true>)},
        CompilerPipelineTestCase{
            "MultipleClassicalRegistersAndMeasurements",
            MQT_NAMED_BUILDER(::qc::multipleClassicalRegistersAndMeasurements),
            nullptr,
            MQT_NAMED_BUILDER(
                mlir::qc::multipleClassicalRegistersAndMeasurements),
            MQT_NAMED_BUILDER(
                mlir::qir::multipleClassicalRegistersAndMeasurements<true>)},
        CompilerPipelineTestCase{
            "MeasurementWithoutRegisters", nullptr,
            MQT_NAMED_BUILDER(mlir::qc::measurementWithoutRegisters),
            MQT_NAMED_BUILDER(mlir::qc::measurementWithoutRegisters),
            MQT_NAMED_BUILDER(mlir::qir::measurementWithoutRegisters<true>),
            false},
        CompilerPipelineTestCase{
            "ResetQubitAfterSingleOp",
            MQT_NAMED_BUILDER(::qc::resetQubitAfterSingleOp), nullptr,
            MQT_NAMED_BUILDER(mlir::qc::resetQubitAfterSingleOp),
            MQT_NAMED_BUILDER(mlir::qir::resetQubitAfterSingleOp<true>)},
        CompilerPipelineTestCase{
            "ResetMultipleQubitsAfterSingleOp",
            MQT_NAMED_BUILDER(::qc::resetMultipleQubitsAfterSingleOp), nullptr,
            MQT_NAMED_BUILDER(mlir::qc::resetMultipleQubitsAfterSingleOp),
            MQT_NAMED_BUILDER(
                mlir::qir::resetMultipleQubitsAfterSingleOp<true>)},
        CompilerPipelineTestCase{
            "RepeatedResetAfterSingleOp",
            MQT_NAMED_BUILDER(::qc::repeatedResetAfterSingleOp), nullptr,
            MQT_NAMED_BUILDER(mlir::qc::resetQubitAfterSingleOp),
            MQT_NAMED_BUILDER(mlir::qir::resetQubitAfterSingleOp<true>)},
        CompilerPipelineTestCase{
            "GlobalPhase", MQT_NAMED_BUILDER(::qc::globalPhase), nullptr,
            MQT_NAMED_BUILDER(mlir::qc::globalPhase),
            MQT_NAMED_BUILDER(mlir::qir::globalPhase<true>)},
        CompilerPipelineTestCase{
            "Identity", MQT_NAMED_BUILDER(::qc::identity), nullptr,
            MQT_NAMED_BUILDER(mlir::qc::alloc1QubitRegister),
            MQT_NAMED_BUILDER(mlir::qir::alloc1QubitRegister<true>)},
        CompilerPipelineTestCase{
            "SingleControlledIdentity",
            MQT_NAMED_BUILDER(::qc::singleControlledIdentity), nullptr,
            MQT_NAMED_BUILDER(mlir::qc::allocQubitRegister),
            MQT_NAMED_BUILDER(mlir::qir::allocQubitRegister<true>)},
        CompilerPipelineTestCase{
            "MultipleControlledIdentity",
            MQT_NAMED_BUILDER(::qc::multipleControlledIdentity), nullptr,
            MQT_NAMED_BUILDER(mlir::qc::alloc3QubitRegister),
            MQT_NAMED_BUILDER(mlir::qir::alloc3QubitRegister<true>)},
        CompilerPipelineTestCase{"X", MQT_NAMED_BUILDER(::qc::x), nullptr,
                                 MQT_NAMED_BUILDER(mlir::qc::x),
                                 MQT_NAMED_BUILDER(mlir::qir::x<true>)},
        CompilerPipelineTestCase{
            "SingleControlledX", MQT_NAMED_BUILDER(::qc::singleControlledX),
            nullptr, MQT_NAMED_BUILDER(mlir::qc::singleControlledX),
            MQT_NAMED_BUILDER(mlir::qir::singleControlledX<true>)},
        CompilerPipelineTestCase{
            "MultipleControlledX", MQT_NAMED_BUILDER(::qc::multipleControlledX),
            nullptr, MQT_NAMED_BUILDER(mlir::qc::multipleControlledX),
            MQT_NAMED_BUILDER(mlir::qir::multipleControlledX<true>)},
        CompilerPipelineTestCase{"Y", MQT_NAMED_BUILDER(::qc::y), nullptr,
                                 MQT_NAMED_BUILDER(mlir::qc::y),
                                 MQT_NAMED_BUILDER(mlir::qir::y<true>)},
        CompilerPipelineTestCase{
            "SingleControlledY", MQT_NAMED_BUILDER(::qc::singleControlledY),
            nullptr, MQT_NAMED_BUILDER(mlir::qc::singleControlledY),
            MQT_NAMED_BUILDER(mlir::qir::singleControlledY<true>)},
        CompilerPipelineTestCase{
            "MultipleControlledY", MQT_NAMED_BUILDER(::qc::multipleControlledY),
            nullptr, MQT_NAMED_BUILDER(mlir::qc::multipleControlledY),
            MQT_NAMED_BUILDER(mlir::qir::multipleControlledY<true>)},
        CompilerPipelineTestCase{"Z", MQT_NAMED_BUILDER(::qc::z), nullptr,
                                 MQT_NAMED_BUILDER(mlir::qc::z),
                                 MQT_NAMED_BUILDER(mlir::qir::z<true>)},
        CompilerPipelineTestCase{
            "SingleControlledZ", MQT_NAMED_BUILDER(::qc::singleControlledZ),
            nullptr, MQT_NAMED_BUILDER(mlir::qc::singleControlledZ),
            MQT_NAMED_BUILDER(mlir::qir::singleControlledZ<true>)},
        CompilerPipelineTestCase{
            "MultipleControlledZ", MQT_NAMED_BUILDER(::qc::multipleControlledZ),
            nullptr, MQT_NAMED_BUILDER(mlir::qc::multipleControlledZ),
            MQT_NAMED_BUILDER(mlir::qir::multipleControlledZ<true>)},
        CompilerPipelineTestCase{"H", MQT_NAMED_BUILDER(::qc::h), nullptr,
                                 MQT_NAMED_BUILDER(mlir::qc::h),
                                 MQT_NAMED_BUILDER(mlir::qir::h<true>)},
        CompilerPipelineTestCase{
            "SingleControlledH", MQT_NAMED_BUILDER(::qc::singleControlledH),
            nullptr, MQT_NAMED_BUILDER(mlir::qc::singleControlledH),
            MQT_NAMED_BUILDER(mlir::qir::singleControlledH<true>)},
        CompilerPipelineTestCase{
            "MultipleControlledH", MQT_NAMED_BUILDER(::qc::multipleControlledH),
            nullptr, MQT_NAMED_BUILDER(mlir::qc::multipleControlledH),
            MQT_NAMED_BUILDER(mlir::qir::multipleControlledH<true>)},
        CompilerPipelineTestCase{
            "HWithoutRegister", nullptr,
            MQT_NAMED_BUILDER(mlir::qc::hWithoutRegister),
            MQT_NAMED_BUILDER(mlir::qc::hWithoutRegister),
            MQT_NAMED_BUILDER(mlir::qir::hWithoutRegister<true>), false},
        CompilerPipelineTestCase{"S", MQT_NAMED_BUILDER(::qc::s), nullptr,
                                 MQT_NAMED_BUILDER(mlir::qc::s),
                                 MQT_NAMED_BUILDER(mlir::qir::s<true>)},
        CompilerPipelineTestCase{
            "SingleControlledS", MQT_NAMED_BUILDER(::qc::singleControlledS),
            nullptr, MQT_NAMED_BUILDER(mlir::qc::singleControlledS),
            MQT_NAMED_BUILDER(mlir::qir::singleControlledS<true>)},
        CompilerPipelineTestCase{
            "MultipleControlledS", MQT_NAMED_BUILDER(::qc::multipleControlledS),
            nullptr, MQT_NAMED_BUILDER(mlir::qc::multipleControlledS),
            MQT_NAMED_BUILDER(mlir::qir::multipleControlledS<true>)},
        CompilerPipelineTestCase{"Sdg", MQT_NAMED_BUILDER(::qc::sdg), nullptr,
                                 MQT_NAMED_BUILDER(mlir::qc::sdg),
                                 MQT_NAMED_BUILDER(mlir::qir::sdg<true>)},
        CompilerPipelineTestCase{
            "SingleControlledSdg", MQT_NAMED_BUILDER(::qc::singleControlledSdg),
            nullptr, MQT_NAMED_BUILDER(mlir::qc::singleControlledSdg),
            MQT_NAMED_BUILDER(mlir::qir::singleControlledSdg<true>)},
        CompilerPipelineTestCase{
            "MultipleControlledSdg",
            MQT_NAMED_BUILDER(::qc::multipleControlledSdg), nullptr,
            MQT_NAMED_BUILDER(mlir::qc::multipleControlledSdg),
            MQT_NAMED_BUILDER(mlir::qir::multipleControlledSdg<true>)},
        CompilerPipelineTestCase{"T", MQT_NAMED_BUILDER(::qc::t_), nullptr,
                                 MQT_NAMED_BUILDER(mlir::qc::t_),
                                 MQT_NAMED_BUILDER(mlir::qir::t_<true>)},
        CompilerPipelineTestCase{
            "SingleControlledT", MQT_NAMED_BUILDER(::qc::singleControlledT),
            nullptr, MQT_NAMED_BUILDER(mlir::qc::singleControlledT),
            MQT_NAMED_BUILDER(mlir::qir::singleControlledT<true>)},
        CompilerPipelineTestCase{
            "MultipleControlledT", MQT_NAMED_BUILDER(::qc::multipleControlledT),
            nullptr, MQT_NAMED_BUILDER(mlir::qc::multipleControlledT),
            MQT_NAMED_BUILDER(mlir::qir::multipleControlledT<true>)},
        CompilerPipelineTestCase{"Tdg", MQT_NAMED_BUILDER(::qc::tdg), nullptr,
                                 MQT_NAMED_BUILDER(mlir::qc::tdg),
                                 MQT_NAMED_BUILDER(mlir::qir::tdg<true>)},
        CompilerPipelineTestCase{
            "SingleControlledTdg", MQT_NAMED_BUILDER(::qc::singleControlledTdg),
            nullptr, MQT_NAMED_BUILDER(mlir::qc::singleControlledTdg),
            MQT_NAMED_BUILDER(mlir::qir::singleControlledTdg<true>)},
        CompilerPipelineTestCase{
            "MultipleControlledTdg",
            MQT_NAMED_BUILDER(::qc::multipleControlledTdg), nullptr,
            MQT_NAMED_BUILDER(mlir::qc::multipleControlledTdg),
            MQT_NAMED_BUILDER(mlir::qir::multipleControlledTdg<true>)},
        CompilerPipelineTestCase{"SX", MQT_NAMED_BUILDER(::qc::sx), nullptr,
                                 MQT_NAMED_BUILDER(mlir::qc::sx),
                                 MQT_NAMED_BUILDER(mlir::qir::sx<true>)},
        CompilerPipelineTestCase{
            "SingleControlledSX", MQT_NAMED_BUILDER(::qc::singleControlledSx),
            nullptr, MQT_NAMED_BUILDER(mlir::qc::singleControlledSx),
            MQT_NAMED_BUILDER(mlir::qir::singleControlledSx<true>)},
        CompilerPipelineTestCase{
            "MultipleControlledSX",
            MQT_NAMED_BUILDER(::qc::multipleControlledSx), nullptr,
            MQT_NAMED_BUILDER(mlir::qc::multipleControlledSx),
            MQT_NAMED_BUILDER(mlir::qir::multipleControlledSx<true>)},
        CompilerPipelineTestCase{"SXdg", MQT_NAMED_BUILDER(::qc::sxdg), nullptr,
                                 MQT_NAMED_BUILDER(mlir::qc::sxdg),
                                 MQT_NAMED_BUILDER(mlir::qir::sxdg<true>)},
        CompilerPipelineTestCase{
            "SingleControlledSXdg",
            MQT_NAMED_BUILDER(::qc::singleControlledSxdg), nullptr,
            MQT_NAMED_BUILDER(mlir::qc::singleControlledSxdg),
            MQT_NAMED_BUILDER(mlir::qir::singleControlledSxdg<true>)},
        CompilerPipelineTestCase{
            "MultipleControlledSXdg",
            MQT_NAMED_BUILDER(::qc::multipleControlledSxdg), nullptr,
            MQT_NAMED_BUILDER(mlir::qc::multipleControlledSxdg),
            MQT_NAMED_BUILDER(mlir::qir::multipleControlledSxdg<true>)},
        CompilerPipelineTestCase{"RX", MQT_NAMED_BUILDER(::qc::rx), nullptr,
                                 MQT_NAMED_BUILDER(mlir::qc::rx),
                                 MQT_NAMED_BUILDER(mlir::qir::rx<true>)},
        CompilerPipelineTestCase{
            "SingleControlledRX", MQT_NAMED_BUILDER(::qc::singleControlledRx),
            nullptr, MQT_NAMED_BUILDER(mlir::qc::singleControlledRx),
            MQT_NAMED_BUILDER(mlir::qir::singleControlledRx<true>)},
        CompilerPipelineTestCase{
            "MultipleControlledRX",
            MQT_NAMED_BUILDER(::qc::multipleControlledRx), nullptr,
            MQT_NAMED_BUILDER(mlir::qc::multipleControlledRx),
            MQT_NAMED_BUILDER(mlir::qir::multipleControlledRx<true>)},
        CompilerPipelineTestCase{"RY", MQT_NAMED_BUILDER(::qc::ry), nullptr,
                                 MQT_NAMED_BUILDER(mlir::qc::ry),
                                 MQT_NAMED_BUILDER(mlir::qir::ry<true>)},
        CompilerPipelineTestCase{
            "SingleControlledRY", MQT_NAMED_BUILDER(::qc::singleControlledRy),
            nullptr, MQT_NAMED_BUILDER(mlir::qc::singleControlledRy),
            MQT_NAMED_BUILDER(mlir::qir::singleControlledRy<true>)},
        CompilerPipelineTestCase{
            "MultipleControlledRY",
            MQT_NAMED_BUILDER(::qc::multipleControlledRy), nullptr,
            MQT_NAMED_BUILDER(mlir::qc::multipleControlledRy),
            MQT_NAMED_BUILDER(mlir::qir::multipleControlledRy<true>)},
        CompilerPipelineTestCase{"RZ", MQT_NAMED_BUILDER(::qc::rz), nullptr,
                                 MQT_NAMED_BUILDER(mlir::qc::rz),
                                 MQT_NAMED_BUILDER(mlir::qir::rz<true>)},
        CompilerPipelineTestCase{
            "SingleControlledRZ", MQT_NAMED_BUILDER(::qc::singleControlledRz),
            nullptr, MQT_NAMED_BUILDER(mlir::qc::singleControlledRz),
            MQT_NAMED_BUILDER(mlir::qir::singleControlledRz<true>)},
        CompilerPipelineTestCase{
            "MultipleControlledRZ",
            MQT_NAMED_BUILDER(::qc::multipleControlledRz), nullptr,
            MQT_NAMED_BUILDER(mlir::qc::multipleControlledRz),
            MQT_NAMED_BUILDER(mlir::qir::multipleControlledRz<true>)},
        CompilerPipelineTestCase{"P", MQT_NAMED_BUILDER(::qc::p), nullptr,
                                 MQT_NAMED_BUILDER(mlir::qc::p),
                                 MQT_NAMED_BUILDER(mlir::qir::p<true>)},
        CompilerPipelineTestCase{
            "SingleControlledP",
            MQT_NAMED_BUILDER(::qc::singleControlledP), nullptr,
            MQT_NAMED_BUILDER(mlir::qc::singleControlledP),
            MQT_NAMED_BUILDER(mlir::qir::singleControlledP<true>)},
        CompilerPipelineTestCase{
            "MultipleControlledP", MQT_NAMED_BUILDER(::qc::multipleControlledP),
            nullptr, MQT_NAMED_BUILDER(mlir::qc::multipleControlledP),
            MQT_NAMED_BUILDER(mlir::qir::multipleControlledP<true>)},
        CompilerPipelineTestCase{"R", MQT_NAMED_BUILDER(::qc::r), nullptr,
                                 MQT_NAMED_BUILDER(mlir::qc::r),
                                 MQT_NAMED_BUILDER(mlir::qir::r<true>)},
        CompilerPipelineTestCase{
            "SingleControlledR",
            MQT_NAMED_BUILDER(::qc::singleControlledR), nullptr,
            MQT_NAMED_BUILDER(mlir::qc::singleControlledR),
            MQT_NAMED_BUILDER(mlir::qir::singleControlledR<true>)},
        CompilerPipelineTestCase{
            "MultipleControlledR", MQT_NAMED_BUILDER(::qc::multipleControlledR),
            nullptr, MQT_NAMED_BUILDER(mlir::qc::multipleControlledR),
            MQT_NAMED_BUILDER(mlir::qir::multipleControlledR<true>)},
        CompilerPipelineTestCase{"U2", MQT_NAMED_BUILDER(::qc::u2), nullptr,
                                 MQT_NAMED_BUILDER(mlir::qc::u2),
                                 MQT_NAMED_BUILDER(mlir::qir::u2<true>)},
        CompilerPipelineTestCase{
            "SingleControlledU2", MQT_NAMED_BUILDER(::qc::singleControlledU2),
            nullptr, MQT_NAMED_BUILDER(mlir::qc::singleControlledU2),
            MQT_NAMED_BUILDER(mlir::qir::singleControlledU2<true>)},
        CompilerPipelineTestCase{
            "MultipleControlledU2",
            MQT_NAMED_BUILDER(::qc::multipleControlledU2), nullptr,
            MQT_NAMED_BUILDER(mlir::qc::multipleControlledU2),
            MQT_NAMED_BUILDER(mlir::qir::multipleControlledU2<true>)},
        CompilerPipelineTestCase{"U", MQT_NAMED_BUILDER(::qc::u), nullptr,
                                 MQT_NAMED_BUILDER(mlir::qc::u),
                                 MQT_NAMED_BUILDER(mlir::qir::u<true>)},
        CompilerPipelineTestCase{
            "SingleControlledU",
            MQT_NAMED_BUILDER(::qc::singleControlledU), nullptr,
            MQT_NAMED_BUILDER(mlir::qc::singleControlledU),
            MQT_NAMED_BUILDER(mlir::qir::singleControlledU<true>)},
        CompilerPipelineTestCase{
            "MultipleControlledU", MQT_NAMED_BUILDER(::qc::multipleControlledU),
            nullptr, MQT_NAMED_BUILDER(mlir::qc::multipleControlledU),
            MQT_NAMED_BUILDER(mlir::qir::multipleControlledU<true>)},
        CompilerPipelineTestCase{"SWAP", MQT_NAMED_BUILDER(::qc::swap), nullptr,
                                 MQT_NAMED_BUILDER(mlir::qc::swap),
                                 MQT_NAMED_BUILDER(mlir::qir::swap<true>)},
        CompilerPipelineTestCase{
            "SingleControlledSWAP",
            MQT_NAMED_BUILDER(::qc::singleControlledSwap), nullptr,
            MQT_NAMED_BUILDER(mlir::qc::singleControlledSwap),
            MQT_NAMED_BUILDER(mlir::qir::singleControlledSwap<true>)},
        CompilerPipelineTestCase{
            "MultipleControlledSWAP",
            MQT_NAMED_BUILDER(::qc::multipleControlledSwap), nullptr,
            MQT_NAMED_BUILDER(mlir::qc::multipleControlledSwap),
            MQT_NAMED_BUILDER(mlir::qir::multipleControlledSwap<true>)},
        CompilerPipelineTestCase{"iSWAP", MQT_NAMED_BUILDER(::qc::iswap),
                                 nullptr, MQT_NAMED_BUILDER(mlir::qc::iswap),
                                 MQT_NAMED_BUILDER(mlir::qir::iswap<true>)},
        CompilerPipelineTestCase{
            "SingleControllediSWAP",
            MQT_NAMED_BUILDER(::qc::singleControlledIswap), nullptr,
            MQT_NAMED_BUILDER(mlir::qc::singleControlledIswap),
            MQT_NAMED_BUILDER(mlir::qir::singleControlledIswap<true>)},
        CompilerPipelineTestCase{
            "MultipleControllediSWAP",
            MQT_NAMED_BUILDER(::qc::multipleControlledIswap), nullptr,
            MQT_NAMED_BUILDER(mlir::qc::multipleControlledIswap),
            MQT_NAMED_BUILDER(mlir::qir::multipleControlledIswap<true>)},
        CompilerPipelineTestCase{
            "InverseISWAP", MQT_NAMED_BUILDER(::qc::inverseIswap), nullptr,
            MQT_NAMED_BUILDER(mlir::qc::inverseIswap), nullptr, true, false},
        CompilerPipelineTestCase{
            "InverseMultiControlledISWAP",
            MQT_NAMED_BUILDER(::qc::inverseMultipleControlledIswap), nullptr,
            MQT_NAMED_BUILDER(mlir::qc::inverseMultipleControlledIswap),
            nullptr, true, false},
        CompilerPipelineTestCase{"DCX", MQT_NAMED_BUILDER(::qc::dcx), nullptr,
                                 MQT_NAMED_BUILDER(mlir::qc::dcx),
                                 MQT_NAMED_BUILDER(mlir::qir::dcx<true>)},
        CompilerPipelineTestCase{
            "SingleControlledDCX", MQT_NAMED_BUILDER(::qc::singleControlledDcx),
            nullptr, MQT_NAMED_BUILDER(mlir::qc::singleControlledDcx),
            MQT_NAMED_BUILDER(mlir::qir::singleControlledDcx<true>)},
        CompilerPipelineTestCase{
            "MultipleControlledDCX",
            MQT_NAMED_BUILDER(::qc::multipleControlledDcx), nullptr,
            MQT_NAMED_BUILDER(mlir::qc::multipleControlledDcx),
            MQT_NAMED_BUILDER(mlir::qir::multipleControlledDcx<true>)},
        CompilerPipelineTestCase{"ECR", MQT_NAMED_BUILDER(::qc::ecr), nullptr,
                                 MQT_NAMED_BUILDER(mlir::qc::ecr),
                                 MQT_NAMED_BUILDER(mlir::qir::ecr<true>)},
        CompilerPipelineTestCase{
            "SingleControlledECR", MQT_NAMED_BUILDER(::qc::singleControlledEcr),
            nullptr, MQT_NAMED_BUILDER(mlir::qc::singleControlledEcr),
            MQT_NAMED_BUILDER(mlir::qir::singleControlledEcr<true>)},
        CompilerPipelineTestCase{
            "MultipleControlledECR",
            MQT_NAMED_BUILDER(::qc::multipleControlledEcr), nullptr,
            MQT_NAMED_BUILDER(mlir::qc::multipleControlledEcr),
            MQT_NAMED_BUILDER(mlir::qir::multipleControlledEcr<true>)},
        CompilerPipelineTestCase{"RXX", MQT_NAMED_BUILDER(::qc::rxx), nullptr,
                                 MQT_NAMED_BUILDER(mlir::qc::rxx),
                                 MQT_NAMED_BUILDER(mlir::qir::rxx<true>)},
        CompilerPipelineTestCase{
            "SingleControlledRXX", MQT_NAMED_BUILDER(::qc::singleControlledRxx),
            nullptr, MQT_NAMED_BUILDER(mlir::qc::singleControlledRxx),
            MQT_NAMED_BUILDER(mlir::qir::singleControlledRxx<true>)},
        CompilerPipelineTestCase{
            "MultipleControlledRXX",
            MQT_NAMED_BUILDER(::qc::multipleControlledRxx), nullptr,
            MQT_NAMED_BUILDER(mlir::qc::multipleControlledRxx),
            MQT_NAMED_BUILDER(mlir::qir::multipleControlledRxx<true>)},
        CompilerPipelineTestCase{
            "TripleControlledRXX", MQT_NAMED_BUILDER(::qc::tripleControlledRxx),
            nullptr, MQT_NAMED_BUILDER(mlir::qc::tripleControlledRxx),
            MQT_NAMED_BUILDER(mlir::qir::tripleControlledRxx<true>)},
        CompilerPipelineTestCase{"RYY", MQT_NAMED_BUILDER(::qc::ryy), nullptr,
                                 MQT_NAMED_BUILDER(mlir::qc::ryy),
                                 MQT_NAMED_BUILDER(mlir::qir::ryy<true>)},
        CompilerPipelineTestCase{
            "SingleControlledRYY", MQT_NAMED_BUILDER(::qc::singleControlledRyy),
            nullptr, MQT_NAMED_BUILDER(mlir::qc::singleControlledRyy),
            MQT_NAMED_BUILDER(mlir::qir::singleControlledRyy<true>)},
        CompilerPipelineTestCase{
            "MultipleControlledRYY",
            MQT_NAMED_BUILDER(::qc::multipleControlledRyy), nullptr,
            MQT_NAMED_BUILDER(mlir::qc::multipleControlledRyy),
            MQT_NAMED_BUILDER(mlir::qir::multipleControlledRyy<true>)},
        CompilerPipelineTestCase{"RZX", MQT_NAMED_BUILDER(::qc::rzx), nullptr,
                                 MQT_NAMED_BUILDER(mlir::qc::rzx),
                                 MQT_NAMED_BUILDER(mlir::qir::rzx<true>)},
        CompilerPipelineTestCase{
            "SingleControlledRZX", MQT_NAMED_BUILDER(::qc::singleControlledRzx),
            nullptr, MQT_NAMED_BUILDER(mlir::qc::singleControlledRzx),
            MQT_NAMED_BUILDER(mlir::qir::singleControlledRzx<true>)},
        CompilerPipelineTestCase{
            "MultipleControlledRZX",
            MQT_NAMED_BUILDER(::qc::multipleControlledRzx), nullptr,
            MQT_NAMED_BUILDER(mlir::qc::multipleControlledRzx),
            MQT_NAMED_BUILDER(mlir::qir::multipleControlledRzx<true>)},
        CompilerPipelineTestCase{"RZZ", MQT_NAMED_BUILDER(::qc::rzz), nullptr,
                                 MQT_NAMED_BUILDER(mlir::qc::rzz),
                                 MQT_NAMED_BUILDER(mlir::qir::rzz<true>)},
        CompilerPipelineTestCase{
            "SingleControlledRZZ", MQT_NAMED_BUILDER(::qc::singleControlledRzz),
            nullptr, MQT_NAMED_BUILDER(mlir::qc::singleControlledRzz),
            MQT_NAMED_BUILDER(mlir::qir::singleControlledRzz<true>)},
        CompilerPipelineTestCase{
            "MultipleControlledRZZ",
            MQT_NAMED_BUILDER(::qc::multipleControlledRzz), nullptr,
            MQT_NAMED_BUILDER(mlir::qc::multipleControlledRzz),
            MQT_NAMED_BUILDER(mlir::qir::multipleControlledRzz<true>)},
        CompilerPipelineTestCase{"XXPlusYY", MQT_NAMED_BUILDER(::qc::xxPlusYY),
                                 nullptr, MQT_NAMED_BUILDER(mlir::qc::xxPlusYY),
                                 MQT_NAMED_BUILDER(mlir::qir::xxPlusYY<true>)},
        CompilerPipelineTestCase{
            "SingleControlledXXPlusYY",
            MQT_NAMED_BUILDER(::qc::singleControlledXxPlusYY), nullptr,
            MQT_NAMED_BUILDER(mlir::qc::singleControlledXxPlusYY),
            MQT_NAMED_BUILDER(mlir::qir::singleControlledXxPlusYY<true>)},
        CompilerPipelineTestCase{
            "MultipleControlledXXPlusYY",
            MQT_NAMED_BUILDER(::qc::multipleControlledXxPlusYY), nullptr,
            MQT_NAMED_BUILDER(mlir::qc::multipleControlledXxPlusYY),
            MQT_NAMED_BUILDER(mlir::qir::multipleControlledXxPlusYY<true>)},
        CompilerPipelineTestCase{"XXMinusYY",
                                 MQT_NAMED_BUILDER(::qc::xxMinusYY), nullptr,
                                 MQT_NAMED_BUILDER(mlir::qc::xxMinusYY),
                                 MQT_NAMED_BUILDER(mlir::qir::xxMinusYY<true>)},
        CompilerPipelineTestCase{
            "SingleControlledXXMinusYY",
            MQT_NAMED_BUILDER(::qc::singleControlledXxMinusYY), nullptr,
            MQT_NAMED_BUILDER(mlir::qc::singleControlledXxMinusYY),
            MQT_NAMED_BUILDER(mlir::qir::singleControlledXxMinusYY<true>)},
        CompilerPipelineTestCase{
            "MultipleControlledXXMinusYY",
            MQT_NAMED_BUILDER(::qc::multipleControlledXxMinusYY), nullptr,
            MQT_NAMED_BUILDER(mlir::qc::multipleControlledXxMinusYY),
            MQT_NAMED_BUILDER(mlir::qir::multipleControlledXxMinusYY<true>)},
        CompilerPipelineTestCase{"CtrlTwo", MQT_NAMED_BUILDER(::qc::ctrlTwo),
                                 nullptr, MQT_NAMED_BUILDER(mlir::qc::ctrlTwo),
                                 MQT_NAMED_BUILDER(mlir::qir::ctrlTwo<true>)}));

} // namespace mqt::test::compiler
