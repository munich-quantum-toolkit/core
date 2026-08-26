/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "Support/IRVerification.h"
#include "TestCaseUtils.h"
#include "mlir/Compiler/Programs.h"
#include "mlir/Compiler/Target.h"
#include "mlir/Dialect/CBit/IR/CBitDialect.h"
#include "mlir/Dialect/QC/Builder/QCProgramBuilder.h"
#include "mlir/Dialect/QC/IR/QCDialect.h"
#include "mlir/Dialect/QCO/Builder/QCOProgramBuilder.h"
#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QCO/IR/QCOInterfaces.h"
#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QIR/Builder/QIRProgramBuilder.h"
#include "mlir/Dialect/QIR/Utils/QIRUtils.h"
#include "mlir/Dialect/QTensor/IR/QTensorDialect.h"
#include "mlir/Dialect/QTensor/IR/QTensorOps.h"
#include "mlir/Support/Passes.h"
#include "qasm_programs.h"
#include "qc_programs.h"
#include "qco_programs.h"
#include "qir_programs.h"

#include <gtest/gtest.h>
#include <jeff/IR/JeffDialect.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Dialect/UB/IR/UBOps.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LLVM.h>

#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iosfwd>
#include <iterator>
#include <memory>
#include <optional>
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

namespace {

struct CompilerPipelineTestCase {
  std::string name;
  QCProgramBuilderFn qcProgramBuilder;
  QCProgramBuilderFn qcReferenceBuilder;
  QIRProgramBuilderFn qirReferenceBuilder;
  bool convertToQIR = true;
  std::string qcoPipeline = "mqt-qco-default";

  friend std::ostream& operator<<(std::ostream& os,
                                  const CompilerPipelineTestCase& info);
};

// NOLINTNEXTLINE(llvm-prefer-static-over-anonymous-namespace)
std::ostream& operator<<(std::ostream& os,
                         const CompilerPipelineTestCase& info) {
  os << "CompilerPipeline{" << info.name
     << ", original=" << displayName(info.qcProgramBuilder.name);
  os << ", qcReference=" << displayName(info.qcReferenceBuilder.name);
  if (info.convertToQIR) {
    os << ", qirReference=" << displayName(info.qirReferenceBuilder.name);
  }
  if (info.qcoPipeline != "mqt-qco-default") {
    os << ", qcoPipeline=" << info.qcoPipeline;
  }
  return os << "}";
}

class CompilerPipelineTest
    : public testing::TestWithParam<CompilerPipelineTestCase> {
protected:
  std::unique_ptr<MLIRContext> context;

  void SetUp() override {
    DialectRegistry registry;
    registry.insert<cbit::CBitDialect, QCDialect, QCODialect,
                    qtensor::QTensorDialect, arith::ArithDialect,
                    cf::ControlFlowDialect, func::FuncDialect,
                    memref::MemRefDialect, scf::SCFDialect, LLVM::LLVMDialect,
                    jeff::JeffDialect>();
    context = std::make_unique<MLIRContext>();
    context->appendDialectRegistry(registry);
    context->loadAllAvailableDialects();
  }

  [[nodiscard]] OwningOpRef<ModuleOp>
  buildQCReference(const QCProgramBuilderFn builder) const {
    auto module = ::mqt::test::buildMLIRProgram(context.get(), builder);
    EXPECT_TRUE(runQCCleanupPipeline(module.get()).succeeded());
    return module;
  }

  [[nodiscard]] OwningOpRef<ModuleOp>
  buildQIRReference(const QIRProgramBuilderFn builder) const {
    auto module = ::mqt::test::buildMLIRProgram(
        context.get(), builder, QIRProgramBuilder::Profile::Adaptive);
    EXPECT_TRUE(runQIRCleanupPipeline(module.get(), true).succeeded());
    return module;
  }

  [[nodiscard]] OwningOpRef<ModuleOp>
  parseRecordedModule(const std::string& ir) const {
    return parseSourceString<ModuleOp>(ir, context.get());
  }

  static void ignoreSingleQIRResultLabel(ModuleOp module) {
    constexpr llvm::StringLiteral prefix = "qir.result_label_";
    size_t numLabels = 0;
    module.walk([&](LLVM::GlobalOp op) {
      numLabels += op.getSymName().starts_with(prefix);
    });
    if (numLabels != 1) {
      return;
    }
    module.walk([&](Operation* op) {
      if (const auto name = op->getAttrOfType<StringAttr>("sym_name");
          name && name.getValue().starts_with(prefix)) {
        op->removeAttr("sym_name");
        op->removeAttr("value");
      }
      if (const auto name = op->getAttrOfType<FlatSymbolRefAttr>("global_name");
          name && name.getValue().starts_with(prefix)) {
        op->removeAttr("global_name");
      }
    });
  }

  void expectEquivalent(const std::string& stage, const std::string& ir,
                        ModuleOp expected) const {
    auto actual = parseRecordedModule(ir);
    ASSERT_TRUE(actual) << stage << " failed to parse";
    EXPECT_TRUE(verify(*actual).succeeded());
    EXPECT_TRUE(verify(expected).succeeded());
    // Dedicated translation and QIR-lowering tests cover exact source labels.
    // The shared program fixtures use synthesized cN labels, so exclude labels
    // from their structural program comparison.
    ignoreSingleQIRResultLabel(actual.get());
    ignoreSingleQIRResultLabel(expected);
    EXPECT_TRUE(areModulesEquivalentWithPermutations(actual.get(), expected));
  }
};

} // namespace

[[nodiscard]] static CompilerTarget
makeSparseUCZTarget(const bool includeMeasure) {
  using Operation = CompilerTarget::Operation;
  using Site = CompilerTarget::Site;

  std::vector operations{llvm::cantFail(Operation::create("u", 1, 3)),
                         llvm::cantFail(Operation::create("cz", 2, 0))};
  if (includeMeasure) {
    operations.emplace_back(llvm::cantFail(Operation::create("measure", 1, 0)));
  }
  std::vector sites{llvm::cantFail(Site::create(5)),
                    llvm::cantFail(Site::create(9)),
                    llvm::cantFail(Site::create(17))};
  return llvm::cantFail(CompilerTarget::create(
      "sparse-line", std::move(sites),
      std::vector<CompilerTarget::Coupling>{{5, 9}, {9, 17}},
      std::move(operations)));
}

TEST_P(CompilerPipelineTest, EndToEndPipeline) {
  const auto& testCase = GetParam();
  const auto name = " (" + testCase.name + ")";
  DeferredPrinter printer;

  ASSERT_TRUE(testCase.qcProgramBuilder);
  auto module =
      ::mqt::test::buildMLIRProgram(context.get(), testCase.qcProgramBuilder);
  ASSERT_TRUE(module);
  printer.record(module.get(), "QC Input" + name);
  EXPECT_TRUE(verify(*module).succeeded());

  std::string source;
  llvm::raw_string_ostream sourceStream(source);
  module->print(sourceStream);
  auto input = QCProgram::fromMLIRString(source);
  ASSERT_TRUE(input);
  auto compiled = runDefaultPipeline(
      CompilerInput{std::move(*input)},
      testCase.convertToQIR ? ProgramFormat::QIRAdaptive : ProgramFormat::QC,
      nullptr, testCase.qcoPipeline);
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

TEST(CompilerProgramOwnershipTest, ValidatesAndOwnsExistingQCModules) {
  DialectRegistry registry;
  registry.insert<cbit::CBitDialect, QCDialect, arith::ArithDialect,
                  func::FuncDialect, memref::MemRefDialect>();
  auto context = std::make_shared<MLIRContext>(registry);
  context->loadAllAvailableDialects();

  QCProgramBuilder builder(context.get());
  builder.initialize();
  const auto qubit = builder.allocQubit();
  builder.h(qubit);
  auto moduleOp = builder.finalize();
  const auto borrowed = *moduleOp;

  auto program = QCProgram::fromModule(context, std::move(moduleOp));

  ASSERT_TRUE(program);
  EXPECT_EQ(program->module(), borrowed);
  EXPECT_TRUE(program->isValid());

  EXPECT_FALSE(QCProgram::fromModule(context, {}));

  QCProgramBuilder contextlessBuilder(context.get());
  contextlessBuilder.initialize();
  contextlessBuilder.h(contextlessBuilder.allocQubit());
  auto contextlessModule = contextlessBuilder.finalize();
  EXPECT_FALSE(QCProgram::fromModule({}, std::move(contextlessModule)));

  QCProgramBuilder emptyBuilder(context.get());
  emptyBuilder.initialize();
  auto emptyModule = emptyBuilder.finalize();
  EXPECT_FALSE(QCProgram::fromModule(context, std::move(emptyModule)));

  QCProgramBuilder mismatchedBuilder(context.get());
  mismatchedBuilder.initialize();
  mismatchedBuilder.h(mismatchedBuilder.allocQubit());
  auto mismatchedModule = mismatchedBuilder.finalize();
  auto otherContext = std::make_shared<MLIRContext>(registry);
  EXPECT_FALSE(
      QCProgram::fromModule(otherContext, std::move(mismatchedModule)));
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
  auto result = runDefaultPipeline(CompilerInput{std::move(*input)},
                                   ProgramFormat::QCOOptimized, nullptr,
                                   "hadamard-lifting");
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

namespace {

class OpenQASMCompilerPipelineTest
    : public testing::TestWithParam<qasm::OpenQASMProgram> {};

struct EntryInfo {
  std::vector<std::string> resultTypes;
  std::vector<std::string> outputRecordings;
};

} // namespace

[[nodiscard]] static std::string
openQASMProgramName(const testing::TestParamInfo<qasm::OpenQASMProgram>& info) {
  std::string name = info.param.name.str();
  for (auto& character : name) {
    if (std::isalnum(static_cast<unsigned char>(character)) == 0) {
      character = '_';
    }
  }
  return name;
}

[[nodiscard]] static std::string printType(const Type type) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  type.print(stream);
  return text;
}

[[nodiscard]] static std::optional<EntryInfo>
inspectEntry(const llvm::StringRef ir) {
  DialectRegistry registry;
  registry.insert<cbit::CBitDialect, QCDialect, QCODialect,
                  qtensor::QTensorDialect, arith::ArithDialect,
                  cf::ControlFlowDialect, func::FuncDialect, math::MathDialect,
                  memref::MemRefDialect, scf::SCFDialect, tensor::TensorDialect,
                  ub::UBDialect, LLVM::LLVMDialect, jeff::JeffDialect>();
  MLIRContext context(registry);
  context.loadAllAvailableDialects();
  auto moduleOp = parseSourceString<ModuleOp>(ir, &context);
  if (!moduleOp) {
    return std::nullopt;
  }

  EntryInfo info;
  if (auto main = moduleOp->lookupSymbol<func::FuncOp>("main")) {
    for (const auto type : main.getFunctionType().getResults()) {
      info.resultTypes.push_back(printType(type));
    }
    return info;
  }

  auto main = moduleOp->lookupSymbol<LLVM::LLVMFuncOp>("main");
  if (!main) {
    return std::nullopt;
  }
  const auto result = main.getFunctionType().getReturnType();
  if (!isa<LLVM::LLVMVoidType>(result)) {
    info.resultTypes.push_back(printType(result));
  }
  main.walk([&](LLVM::CallOp call) {
    const auto callee = call.getCallee();
    if (callee &&
        (*callee == QIR_RECORD_OUTPUT || *callee == QIR_ARRAY_RECORD_OUTPUT ||
         *callee == QIR_RESULT_ARRAY_RECORD_OUTPUT)) {
      info.outputRecordings.emplace_back(*callee);
    }
  });
  return info;
}

[[nodiscard]] static testing::AssertionResult
throughOptimizedQCO(const qasm::OpenQASMProgram& source,
                    std::optional<QCProgram>& restored,
                    std::vector<std::string>& resultTypes) {
  auto qc = QCProgram::fromQASMString(source.source.str());
  if (!qc) {
    return testing::AssertionFailure()
           << source.name.str() << ": OpenQASM to QC";
  }
  const auto qcEntry = inspectEntry(qc->str());
  if (!qcEntry) {
    return testing::AssertionFailure()
           << source.name.str() << ": inspect QC entry";
  }
  resultTypes = qcEntry->resultTypes;
  auto qco = std::move(*qc).intoQCO();
  if (!qco || !qco->cleanup() || !qco->runPassPipeline("mqt-qco-default") ||
      !qco->cleanup()) {
    return testing::AssertionFailure()
           << source.name.str() << ": QC/QCO optimization";
  }
  restored = std::move(*qco).intoQC();
  if (!restored || !restored->cleanup()) {
    return testing::AssertionFailure()
           << source.name.str() << ": optimized QCO to QC";
  }
  const auto restoredEntry = inspectEntry(restored->str());
  if (!restoredEntry || restoredEntry->resultTypes != resultTypes) {
    return testing::AssertionFailure()
           << source.name.str() << ": reconstructed QC changed entry results";
  }
  return testing::AssertionSuccess();
}

[[nodiscard]] static testing::AssertionResult
roundTripThroughOptimizedJeff(const qasm::OpenQASMProgram& source,
                              std::optional<QCProgram>& restored,
                              std::vector<std::string>& resultTypes) {
  auto qc = QCProgram::fromQASMString(source.source.str());
  if (!qc) {
    return testing::AssertionFailure()
           << source.name.str() << ": OpenQASM to QC";
  }
  const auto qcEntry = inspectEntry(qc->str());
  if (!qcEntry) {
    return testing::AssertionFailure()
           << source.name.str() << ": inspect QC entry";
  }
  resultTypes = qcEntry->resultTypes;

  const auto matchesEntry =
      [&](const Program& program, const llvm::StringRef stage,
          const bool allowClassicalRegisterStorageConversion = false) {
        const auto entry = inspectEntry(program.str());
        if (!entry) {
          return testing::AssertionFailure()
                 << source.name.str() << ": inspect " << stage.str()
                 << " entry";
        }
        auto observedTypes = entry->resultTypes;
        auto expectedTypes = resultTypes;
        if (allowClassicalRegisterStorageConversion) {
          const auto normalizeClassicalRegister = [](std::string& type) {
            const auto text = StringRef(type);
            if (text.starts_with("!cbit.reg<") && text.ends_with(">")) {
              type = "tensor<" +
                     text.drop_front(StringRef("!cbit.reg<").size())
                         .drop_back()
                         .str() +
                     "xi1>";
            }
          };
          llvm::for_each(observedTypes, normalizeClassicalRegister);
          llvm::for_each(expectedTypes, normalizeClassicalRegister);
        }
        if (observedTypes != expectedTypes) {
          return testing::AssertionFailure()
                 << source.name.str() << ": " << stage.str()
                 << " changed entry result types";
        }
        return testing::AssertionSuccess();
      };

  auto qco = std::move(*qc).intoQCO();
  if (!qco || !qco->cleanup() || !qco->runPassPipeline("mqt-qco-default") ||
      !qco->cleanup()) {
    return testing::AssertionFailure()
           << source.name.str() << ": QC/QCO optimization";
  }
  if (auto result = matchesEntry(*qco, "optimized QCO"); !result) {
    return result;
  }
  const auto optimizedQCO = qco->str();
  auto jeff = std::move(*qco).intoJeff();
  if (!jeff || !jeff->cleanup()) {
    return testing::AssertionFailure() << source.name.str() << ": QCO to jeff\n"
                                       << optimizedQCO;
  }
  if (auto result = matchesEntry(*jeff, "jeff", true); !result) {
    return result;
  }
  const auto bytes = jeff->toBytes();
  if (bytes.empty()) {
    return testing::AssertionFailure()
           << source.name.str() << ": jeff serialization";
  }
  auto restoredJeff = JeffProgram::fromBytes(bytes);
  if (!restoredJeff || !restoredJeff->cleanup()) {
    return testing::AssertionFailure()
           << source.name.str() << ": jeff deserialization";
  }
  if (auto result = matchesEntry(*restoredJeff, "restored jeff", true);
      !result) {
    return result;
  }
  auto restoredQCO = std::move(*restoredJeff).intoQCO();
  if (!restoredQCO || !restoredQCO->cleanup()) {
    return testing::AssertionFailure()
           << source.name.str() << ": restored jeff to QCO";
  }
  if (auto result = matchesEntry(*restoredQCO, "restored QCO"); !result) {
    return result;
  }
  restored = std::move(*restoredQCO).intoQC();
  if (!restored || !restored->cleanup()) {
    return testing::AssertionFailure()
           << source.name.str() << ": restored QCO to QC";
  }
  return matchesEntry(*restored, "restored QC");
}

namespace {

TEST(OpenQASMCompilerOutputTest, LowersAffineQuantumLoopsToJeff) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.0;
include "stdgates.inc";
qubit[4] q;
bit[4] c;
for int i in [0:3] {
  h q[i];
}
for int i in [0:2] {
  cx q[i], q[i + 1];
}
c = measure q;
)qasm";

  auto qc = QCProgram::fromQASMString(source.str());
  ASSERT_TRUE(qc);
  const auto imported = qc->str();
  EXPECT_EQ(imported.find("i128"), std::string::npos);
  EXPECT_EQ(imported.find("arith.select"), std::string::npos);
  EXPECT_EQ(imported.find("cf.assert"), std::string::npos);

  auto qco = std::move(*qc).intoQCO();
  ASSERT_TRUE(qco);
  ASSERT_TRUE(qco->cleanup());
  ASSERT_TRUE(qco->runPassPipeline("mqt-qco-default"));
  ASSERT_TRUE(qco->cleanup());
  const auto optimized = qco->str();
  auto jeff = std::move(*qco).intoJeff();
  ASSERT_TRUE(jeff) << optimized;
  EXPECT_TRUE(jeff->cleanup());
}

TEST(OpenQASMCompilerOutputTest,
     CanonicalizesMixedScalarAndRegisterResultsThroughQCO) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
output int count;
count = 1;
output bit[2] bits;
bits[0] = true;
bits[1] = false;
output float ratio;
ratio = 2.0;
)qasm";
  const qasm::OpenQASMProgram program{.name = "mixed-output-results",
                                      .source = source};

  std::optional<QCProgram> restoredQC;
  std::vector<std::string> resultTypes;
  ASSERT_TRUE(throughOptimizedQCO(program, restoredQC, resultTypes));
  EXPECT_EQ(resultTypes,
            (std::vector<std::string>{"i64", "!cbit.reg<2>", "f64"}));
  ASSERT_TRUE(restoredQC);
  auto emitted = restoredQC->toOpenQASM3();
  ASSERT_TRUE(emitted);
  EXPECT_NE(emitted->source().find("output int _mqt_out0;"), std::string::npos);
  EXPECT_NE(emitted->source().find("output bit[2] bits;"), std::string::npos);
  EXPECT_NE(emitted->source().find("output float _mqt_out1;"),
            std::string::npos);
  EXPECT_TRUE(QCProgram::fromQASMString(emitted->source()));
}

TEST(OpenQASMCompilerOutputTest, GlobalPhasesTraverseQCQCOJeffAndQIRScopes) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.0;
include "stdgates.inc";
gate phased q {
  gphase(0.371);
  x q;
}
qubit[2] q;
ctrl @ phased q[0], q[1];
bit flag = measure q[0];
if (flag) {
  gphase(0.25);
  h q[1];
} else {
  gphase(-0.5);
  z q[1];
}
)qasm";

  auto qc = QCProgram::fromQASMString(source.str());
  ASSERT_TRUE(qc);
  ASSERT_TRUE(qc->cleanup());
  auto qco = std::move(*qc).intoQCO();
  ASSERT_TRUE(qco);
  ASSERT_TRUE(qco->cleanup());

  auto jeffInput = qco->copy();
  auto jeff = std::move(jeffInput).intoJeff();
  ASSERT_TRUE(jeff);
  ASSERT_TRUE(jeff->cleanup());

  auto restoredQC = std::move(*qco).intoQC();
  ASSERT_TRUE(restoredQC);
  ASSERT_TRUE(restoredQC->cleanup());
  auto qir = std::move(*restoredQC).intoQIR(QIRProfile::Adaptive);
  ASSERT_TRUE(qir);
  ASSERT_TRUE(qir->cleanup());
  EXPECT_TRUE(qir->llvmIR().has_value());
}

TEST_F(CompilerPipelineTest, EmitsQIR21ProfileModuleFlags) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.0;
qubit q;
bit result;
h q;
result = measure q;
)qasm";

  auto input = QCProgram::fromQASMString(source.str());
  ASSERT_TRUE(input);
  for (const auto profile : {QIRProfile::Base, QIRProfile::Adaptive}) {
    auto qir = std::move(input->copy()).intoQIR(profile);
    ASSERT_TRUE(qir);
    const auto llvmIR = qir->llvmIR();
    ASSERT_TRUE(llvmIR);
    EXPECT_NE(llvmIR->find("define i64 @main()"), std::string::npos);
    EXPECT_NE(llvmIR->find("!\"qir_major_version\", i32 2"), std::string::npos);
    EXPECT_NE(llvmIR->find("!\"qir_minor_version\", i32 1"), std::string::npos);
    if (profile == QIRProfile::Adaptive) {
      EXPECT_NE(llvmIR->find("!\"dynamic_qubit_management\", i1 true"),
                std::string::npos);
      EXPECT_NE(llvmIR->find("!\"dynamic_result_management\", i1 true"),
                std::string::npos);
      EXPECT_NE(llvmIR->find("!\"backwards_branching\", i2 0"),
                std::string::npos);
      EXPECT_NE(llvmIR->find("!\"arrays\", i1 true"), std::string::npos);
    } else {
      EXPECT_NE(llvmIR->find("!\"dynamic_qubit_management\", i1 false"),
                std::string::npos);
      EXPECT_NE(llvmIR->find("!\"dynamic_result_management\", i1 false"),
                std::string::npos);
      EXPECT_EQ(llvmIR->find("!\"backwards_branching\""), std::string::npos);
      EXPECT_EQ(llvmIR->find("!\"arrays\""), std::string::npos);
    }
  }
}

enum class OutputRecordingShape : std::uint8_t { AdaptiveArrays, BaseArrays };

} // namespace

static void expectQIRArtifacts(const QIRProgram& program,
                               const llvm::StringRef name,
                               const ArrayRef<std::string> sourceResultTypes,
                               const OutputRecordingShape outputShape) {
  const auto entry = inspectEntry(program.str());
  ASSERT_TRUE(entry) << name.str() << ": QIR entry inspection";
  ASSERT_EQ(entry->resultTypes.size(), 1) << name.str() << ": QIR main result";
  EXPECT_EQ(entry->resultTypes.front(), "i64")
      << name.str() << ": QIR main status type";
  if (!sourceResultTypes.empty()) {
    EXPECT_FALSE(entry->outputRecordings.empty())
        << name.str() << ": QIR output recording";
  }
  if (name == "broadcast-custom-gate") {
    std::vector<std::string> expected;
    if (outputShape == OutputRecordingShape::AdaptiveArrays) {
      expected.assign(2, QIR_RESULT_ARRAY_RECORD_OUTPUT);
    } else {
      expected = {QIR_ARRAY_RECORD_OUTPUT, QIR_RECORD_OUTPUT,
                  QIR_RECORD_OUTPUT,       QIR_RECORD_OUTPUT,
                  QIR_ARRAY_RECORD_OUTPUT, QIR_RECORD_OUTPUT};
    }
    EXPECT_EQ(entry->outputRecordings, expected)
        << name.str() << ": QIR multi-output recording order";
  }
  auto llvmIR = program.llvmIR();
  ASSERT_TRUE(llvmIR) << name.str() << ": LLVM IR translation";
  EXPECT_FALSE(llvmIR->empty()) << name.str() << ": LLVM IR is empty";
  auto bitcode = program.toBitcode();
  ASSERT_TRUE(bitcode) << name.str() << ": bitcode translation";
  ASSERT_GE(bitcode->size(), 4) << name.str() << ": bitcode header";
  EXPECT_EQ(std::to_integer<std::uint8_t>((*bitcode)[0]), 0x42U);
  EXPECT_EQ(std::to_integer<std::uint8_t>((*bitcode)[1]), 0x43U);
  EXPECT_EQ(std::to_integer<std::uint8_t>((*bitcode)[2]), 0xC0U);
  EXPECT_EQ(std::to_integer<std::uint8_t>((*bitcode)[3]), 0xDEU);
}

namespace {

TEST_P(OpenQASMCompilerPipelineTest, TraversesTheExplicitStandardPipeline) {
  const auto& source = GetParam();
  std::optional<QCProgram> restoredQC;
  std::vector<std::string> resultTypes;
  ASSERT_TRUE(throughOptimizedQCO(source, restoredQC, resultTypes));
  auto qir = std::move(*restoredQC).intoQIR(QIRProfile::Adaptive);
  ASSERT_TRUE(qir) << source.name.str() << ": QC to Adaptive QIR";
  expectQIRArtifacts(*qir, source.name, resultTypes,
                     OutputRecordingShape::AdaptiveArrays);
}

TEST_P(OpenQASMCompilerPipelineTest, TraversesTheDefaultAdaptivePipeline) {
  const auto& source = GetParam();
  auto input = QCProgram::fromQASMString(source.source.str());
  ASSERT_TRUE(input) << source.name.str() << ": OpenQASM to QC";
  const auto inputEntry = inspectEntry(input->str());
  ASSERT_TRUE(inputEntry) << source.name.str() << ": inspect QC entry";
  auto output = runDefaultPipeline(CompilerInput{std::move(*input)},
                                   ProgramFormat::QIRAdaptive);
  ASSERT_TRUE(output) << source.name.str() << ": default Adaptive pipeline";
  auto* qir = std::get_if<QIRProgram>(&*output);
  ASSERT_NE(qir, nullptr) << source.name.str() << ": default output format";
  expectQIRArtifacts(*qir, source.name, inputEntry->resultTypes,
                     OutputRecordingShape::AdaptiveArrays);
}

class OpenQASMBasePipelineTest
    : public testing::TestWithParam<qasm::OpenQASMProgram> {};

class OpenQASMJeffPipelineTest
    : public testing::TestWithParam<qasm::OpenQASMProgram> {};

TEST_P(OpenQASMJeffPipelineTest, TraversesTheExplicitJeffRoundTrip) {
  const auto& source = GetParam();
  std::optional<QCProgram> restoredQC;
  std::vector<std::string> resultTypes;
  ASSERT_TRUE(roundTripThroughOptimizedJeff(source, restoredQC, resultTypes));
  auto qir = std::move(*restoredQC).intoQIR(QIRProfile::Adaptive);
  ASSERT_TRUE(qir) << source.name.str() << ": QC to Adaptive QIR";
  expectQIRArtifacts(*qir, source.name, resultTypes,
                     OutputRecordingShape::AdaptiveArrays);
}

class OpenQASMJeffBoundaryTest
    : public testing::TestWithParam<qasm::OpenQASMProgram> {};

TEST_P(OpenQASMJeffBoundaryTest, FailsAtQCOToJeff) {
  const auto& source = GetParam();
  auto qc = QCProgram::fromQASMString(source.source.str());
  ASSERT_TRUE(qc) << source.name.str() << ": OpenQASM to QC";
  auto qco = std::move(*qc).intoQCO();
  ASSERT_TRUE(qco) << source.name.str() << ": QC to QCO";
  ASSERT_TRUE(qco->cleanup()) << source.name.str() << ": QCO cleanup";
  ASSERT_TRUE(qco->runPassPipeline("mqt-qco-default"))
      << source.name.str() << ": QCO optimization";
  ASSERT_TRUE(qco->cleanup()) << source.name.str() << ": optimized QCO cleanup";
  EXPECT_FALSE(std::move(*qco).intoJeff())
      << source.name.str() << ": unexpectedly converted to jeff";
}

TEST_P(OpenQASMBasePipelineTest, ReachesBaseAndAdaptiveQIR) {
  const auto& source = GetParam();
  std::optional<QCProgram> restoredQC;
  std::vector<std::string> resultTypes;
  ASSERT_TRUE(throughOptimizedQCO(source, restoredQC, resultTypes));
  for (const auto profile : {QIRProfile::Base, QIRProfile::Adaptive}) {
    auto input = restoredQC->copy();
    auto qir = std::move(input).intoQIR(profile);
    ASSERT_TRUE(qir) << source.name.str() << ": QC to QIR";
    expectQIRArtifacts(*qir, source.name, resultTypes,
                       profile == QIRProfile::Base
                           ? OutputRecordingShape::BaseArrays
                           : OutputRecordingShape::AdaptiveArrays);
  }
}

INSTANTIATE_TEST_SUITE_P(OpenQASMPrograms, OpenQASMCompilerPipelineTest,
                         testing::ValuesIn(qasm::standardPipelinePrograms()),
                         openQASMProgramName);

INSTANTIATE_TEST_SUITE_P(OpenQASMPrograms, OpenQASMBasePipelineTest,
                         testing::ValuesIn(qasm::baseProfilePrograms()),
                         openQASMProgramName);

INSTANTIATE_TEST_SUITE_P(OpenQASMPrograms, OpenQASMJeffPipelineTest,
                         testing::ValuesIn(qasm::jeffCompatiblePrograms()),
                         openQASMProgramName);

INSTANTIATE_TEST_SUITE_P(OpenQASMPrograms, OpenQASMJeffBoundaryTest,
                         testing::ValuesIn(qasm::jeffIncompatiblePrograms()),
                         openQASMProgramName);

} // namespace

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

  ASSERT_TRUE(qcFromMLIR);
  ASSERT_TRUE(qcFromMLIRFile);
  ASSERT_TRUE(qcFromQASM);
  ASSERT_TRUE(qcFromQASMFile);
  EXPECT_EQ(qcFromMLIR->str(), qcFromMLIRFile->str());
  EXPECT_EQ(qcFromQASM->str(), qcFromQASMFile->str());
  EXPECT_EQ(qcFromMLIR->str(), qcFromMLIR->copy().str());
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
 * @brief Test: typed programs emit OpenQASM directly and through the pipeline.
 */
TEST_F(CompilerPipelineTest, TypedProgramsEmitOpenQASM) {
  const std::string qasm = R"(OPENQASM 3.1;
include "stdgates.inc";
qubit[2] q;
h q[0];
ctrl @ x q[0], q[1];
bit[2] c = measure q;
)";
  auto directQC = QCProgram::fromQASMString(qasm);
  ASSERT_TRUE(directQC);
  const auto importedIR = directQC->str();
  auto direct = directQC->toOpenQASM3();
  ASSERT_TRUE(direct);
  EXPECT_TRUE(directQC->isValid());
  EXPECT_EQ(directQC->str(), importedIR);
  EXPECT_TRUE(direct->source().starts_with("OPENQASM 3.1;\n"));
  EXPECT_EQ(direct->str(), direct->source());
  EXPECT_NE(direct->source().find("output bit[2] c;"), std::string::npos);
  EXPECT_FALSE(direct->write(std::filesystem::path(testing::TempDir()) /
                             "missing" / "typed_program_output.qasm"));

  const auto path =
      std::filesystem::path(testing::TempDir()) / "typed_program_output.qasm";
  ASSERT_TRUE(direct->write(path));
  std::ifstream input(path);
  const std::string written((std::istreambuf_iterator<char>(input)),
                            std::istreambuf_iterator<char>());
  EXPECT_EQ(written, direct->source());
  EXPECT_TRUE(QCProgram::fromQASMFile(path));

  auto imported =
      runDefaultPipeline(CompilerInput(*direct), ProgramFormat::QCImport);
  ASSERT_TRUE(imported);
  EXPECT_TRUE(std::holds_alternative<QCProgram>(*imported));

  auto compiled =
      runDefaultPipeline(CompilerInput(OpenQASMProgram(direct->source())),
                         ProgramFormat::QIRAdaptive);
  ASSERT_TRUE(compiled);
  EXPECT_TRUE(std::holds_alternative<QIRProgram>(*compiled));

  auto pipelineQC = QCProgram::fromQASMString(qasm);
  ASSERT_TRUE(pipelineQC);
  auto result = runDefaultPipeline(CompilerInput(std::move(*pipelineQC)),
                                   ProgramFormat::OpenQASM3);
  ASSERT_TRUE(result);
  ASSERT_TRUE(std::holds_alternative<OpenQASMProgram>(*result));
  const auto& optimized = std::get<OpenQASMProgram>(*result);
  EXPECT_TRUE(optimized.source().starts_with("OPENQASM 3.1;\n"));
  auto reparsed = QCProgram::fromQASMString(optimized.source());
  ASSERT_TRUE(reparsed);
  auto adaptiveQIR = std::move(*reparsed).intoQIR(QIRProfile::Adaptive);
  EXPECT_TRUE(adaptiveQIR);
}

TEST_F(CompilerPipelineTest, TypedOpenQASMExportReportsUnsupportedQC) {
  constexpr llvm::StringLiteral source = R"mlir(module {
    func.func @main(%value: i64) {
      %qubit = qc.alloc : !qc.qubit
      qc.dealloc %qubit : !qc.qubit
      return
    }
  })mlir";
  auto program = QCProgram::fromMLIRString(source);
  ASSERT_TRUE(program);
  EXPECT_FALSE(program->toOpenQASM3());
}

/**
 * @brief Test: typed programs expose idempotent global-phase normalization.
 */
TEST_F(CompilerPipelineTest, TypedProgramsNormalizeGlobalPhases) {
  const std::string qcSource = R"mlir(module {
    func.func @test(%q: !qc.qubit) {
      %a = arith.constant 0.25 : f64
      qc.gphase(%a)
      qc.x %q : !qc.qubit
      %b = arith.constant 0.5 : f64
      qc.gphase(%b)
      return
    }
  })mlir";
  const std::string qcoSource = R"mlir(module {
    func.func @test(%q: !qco.qubit) -> !qco.qubit {
      %a = arith.constant 0.25 : f64
      qco.gphase(%a)
      %q1 = qco.x %q : !qco.qubit -> !qco.qubit
      %b = arith.constant 0.5 : f64
      qco.gphase(%b)
      return %q1 : !qco.qubit
    }
  })mlir";

  auto qc = QCProgram::fromMLIRString(qcSource);
  auto qco = QCOProgram::fromMLIRString(qcoSource);
  ASSERT_TRUE(qc);
  ASSERT_TRUE(qco);
  ASSERT_TRUE(qc->normalizeGlobalPhases());
  ASSERT_TRUE(qco->normalizeGlobalPhases());
  EXPECT_EQ(StringRef(qc->str()).count("qc.gphase"), 1);
  EXPECT_EQ(StringRef(qco->str()).count("qco.gphase"), 1);

  const auto once = qco->str();
  ASSERT_TRUE(qco->normalizeGlobalPhases());
  EXPECT_EQ(qco->str(), once);

  auto textual = QCOProgram::fromMLIRString(qcoSource);
  ASSERT_TRUE(textual);
  ASSERT_TRUE(textual->runPassPipeline("normalize-global-phases"));
  EXPECT_EQ(StringRef(textual->str()).count("qco.gphase"), 1);
}

/**
 * @brief Test: jeff programs round-trip through their binary APIs
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
  EXPECT_TRUE(qco.runPassPipeline("mqt-qco-default", true, true));

  auto loopModule = ::mqt::test::buildMLIRProgram(
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
 * @brief Test: target compilation decomposes, maps, synthesizes, and verifies.
 */
TEST_F(CompilerPipelineTest, QCOProgramCompilesForTarget) {
  auto qc = QCProgram::fromQASMString(qasm::multipleControlledX);
  ASSERT_TRUE(qc);
  auto qco = std::move(*qc).intoQCO();
  ASSERT_TRUE(qco);

  const auto target = makeSparseUCZTarget(true);
  ASSERT_TRUE(qco->compileForTarget(target));

  auto compiled = parseRecordedModule(qco->str());
  ASSERT_TRUE(compiled);
  EXPECT_TRUE(verify(*compiled).succeeded());

  size_t numStatic = 0;
  size_t numDynamic = 0;
  size_t numSwaps = 0;
  size_t numHigherArity = 0;
  compiled->walk([&](Operation* operation) {
    if (auto staticOp = dyn_cast<qco::StaticOp>(operation)) {
      ++numStatic;
      EXPECT_TRUE(llvm::is_contained(
          target.siteIds(),
          static_cast<CompilerTarget::SiteId>(staticOp.getIndex())));
    }
    numDynamic += isa<qco::AllocOp, qtensor::AllocOp>(operation);
    numSwaps += isa<qco::SWAPOp>(operation);
    if (auto unitary = dyn_cast<qco::UnitaryOpInterface>(operation);
        unitary && unitary.getNumQubits() > 2) {
      ++numHigherArity;
    }
  });
  EXPECT_EQ(numStatic, 3);
  EXPECT_EQ(numDynamic, 0);
  EXPECT_EQ(numSwaps, 0);
  EXPECT_EQ(numHigherArity, 0);

  auto unsupportedQC = QCProgram::fromQASMString(qasm::multipleControlledX);
  ASSERT_TRUE(unsupportedQC);
  auto unsupportedQCO = std::move(*unsupportedQC).intoQCO();
  ASSERT_TRUE(unsupportedQCO);
  EXPECT_FALSE(unsupportedQCO->compileForTarget(makeSparseUCZTarget(false)));
}

/**
 * @brief Test: all-to-all target compilation uses compact placement.
 */
TEST_F(CompilerPipelineTest, QCOProgramUsesCompactAllToAllPlacement) {
  const std::string qasm = R"(OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
bit[2] c;
h q[0];
cx q[0], q[1];
c = measure q;
)";
  auto qc = QCProgram::fromQASMString(qasm);
  ASSERT_TRUE(qc);
  auto qco = std::move(*qc).intoQCO();
  ASSERT_TRUE(qco);

  std::vector sites{llvm::cantFail(CompilerTarget::Site::create(2472)),
                    llvm::cantFail(CompilerTarget::Site::create(18449))};
  const auto target = llvm::cantFail(CompilerTarget::create(std::move(sites)));
  ASSERT_TRUE(qco->compileForTarget(target));

  auto compiled = parseRecordedModule(qco->str());
  ASSERT_TRUE(compiled);
  EXPECT_TRUE(verify(*compiled).succeeded());

  llvm::SmallVector<int64_t> staticSites;
  size_t numDynamic = 0;
  size_t numSwaps = 0;
  compiled->walk([&](Operation* operation) {
    if (auto staticOp = dyn_cast<qco::StaticOp>(operation)) {
      staticSites.emplace_back(staticOp.getIndex());
    }
    numDynamic += isa<qco::AllocOp, qtensor::AllocOp>(operation);
    numSwaps += isa<qco::SWAPOp>(operation);
  });
  EXPECT_EQ(staticSites, (llvm::SmallVector<int64_t>{2472, 18449}));
  EXPECT_EQ(numDynamic, 0);
  EXPECT_EQ(numSwaps, 0);
}

/**
 * @brief Test: target compilation retains unobserved quantum operations.
 */
TEST_F(CompilerPipelineTest, QCOProgramPreservesUnobservedQuantumOperations) {
  constexpr llvm::StringLiteral source = R"(OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
h q[0];
reset q[0];
h q[1];
)";
  const auto target = llvm::cantFail(CompilerTarget::create(3));

  auto qc = QCProgram::fromQASMString(source);
  ASSERT_TRUE(qc);
  auto program = std::move(*qc).intoQCO();
  ASSERT_TRUE(program);
  ASSERT_TRUE(program->compileForTarget(target));
  auto module = parseRecordedModule(program->str());
  ASSERT_TRUE(module);
  EXPECT_TRUE(verify(*module).succeeded());

  size_t unitaryOperations = 0;
  size_t resets = 0;
  size_t staticQubits = 0;
  module->walk([&](Operation* operation) {
    unitaryOperations += isa<UnitaryOpInterface>(operation);
    resets += isa<ResetOp>(operation);
    staticQubits += isa<StaticOp>(operation);
  });
  EXPECT_EQ(unitaryOperations, 2U);
  EXPECT_EQ(resets, 1U);
  EXPECT_EQ(staticQubits, 2U);
}

/**
 * @brief Test: the default pipeline accepts an optional compiler target.
 */
TEST_F(CompilerPipelineTest, DefaultPipelineCompilesForTarget) {
  auto input = QCProgram::fromQASMString(qasm::multipleControlledX);
  ASSERT_TRUE(input);
  const auto target = makeSparseUCZTarget(true);

  auto result = runDefaultPipeline(CompilerInput{std::move(*input)},
                                   ProgramFormat::QCOOptimized, &target);
  ASSERT_TRUE(result);
  ASSERT_TRUE(std::holds_alternative<QCOProgram>(*result));
  const auto& qco = std::get<QCOProgram>(*result);
  EXPECT_NE(qco.str().find("qco.static"), std::string::npos);
  EXPECT_EQ(qco.str().find("qco.swap"), std::string::npos);

  auto qirInput = QCProgram::fromQASMString(qasm::multipleControlledX);
  ASSERT_TRUE(qirInput);
  auto qirResult = runDefaultPipeline(CompilerInput{std::move(*qirInput)},
                                      ProgramFormat::QIRBase, &target);
  ASSERT_TRUE(qirResult);
  ASSERT_TRUE(std::holds_alternative<QIRProgram>(*qirResult));
  const auto& qir = std::get<QIRProgram>(*qirResult);
  auto qirModule = parseRecordedModule(qir.str());
  ASSERT_TRUE(qirModule);
  std::vector<int64_t> qirSiteIds;
  qirModule->walk([&](LLVM::IntToPtrOp intToPtr) {
    auto constant = intToPtr.getArg().getDefiningOp<LLVM::ConstantOp>();
    if (!constant) {
      return;
    }
    if (const auto value = dyn_cast<IntegerAttr>(constant.getValue())) {
      qirSiteIds.emplace_back(value.getInt());
    }
  });
  for (const auto siteId : target.siteIds()) {
    EXPECT_TRUE(llvm::is_contained(qirSiteIds, siteId));
  }
  EXPECT_TRUE(qir.llvmIR());
}

/**
 * @brief Test: QCO programs expose the raw and composite qubit-reuse flows.
 */
TEST_F(CompilerPipelineTest, QCOProgramQubitReuseAPIs) {
  const auto countAllocations = [](const QCOProgram& program) {
    const auto ir = program.str();
    return StringRef(ir).count("qco.alloc");
  };
  const auto buildQCO = [this](const QCProgramBuilderFn& builder) {
    auto module = ::mqt::test::buildMLIRProgram(context.get(), builder);
    std::string source;
    llvm::raw_string_ostream stream(source);
    module->print(stream);
    auto qc = QCProgram::fromMLIRString(source);
    if (!qc) {
      return std::optional<QCOProgram>{};
    }
    return std::move(*qc).intoQCO();
  };

  auto rawQCO = buildQCO(MQT_NAMED_BUILDER(mlir::qc::hGateOnMultipleQubits));
  ASSERT_TRUE(rawQCO);
  ASSERT_EQ(countAllocations(*rawQCO), 2U);
  ASSERT_TRUE(rawQCO->reuseQubits());
  EXPECT_EQ(countAllocations(*rawQCO), 1U);
  EXPECT_NE(rawQCO->str().find("qco.reset"), std::string::npos);

  auto compositeQCO = buildQCO(
      MQT_NAMED_BUILDER(mlir::qc::singleControlledXOnIndividualQubits));
  ASSERT_TRUE(compositeQCO);
  ASSERT_EQ(countAllocations(*compositeQCO), 2U);
  ASSERT_TRUE(compositeQCO->runQubitReusePipeline());
  EXPECT_EQ(countAllocations(*compositeQCO), 1U);
  EXPECT_NE(compositeQCO->str().find("qco.reset"), std::string::npos);
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
                                     ProgramFormat::QCOOptimized, nullptr,
                                     "mqt-qco-default", true, true);
  ASSERT_TRUE(profiled);
  EXPECT_TRUE(std::holds_alternative<QCOProgram>(*profiled));

  auto customPipelineInput = QCProgram::fromQASMString(qasm);
  ASSERT_TRUE(customPipelineInput);
  EXPECT_FALSE(runDefaultPipeline(
      CompilerInput{std::move(*customPipelineInput)}, ProgramFormat::QCO,
      nullptr, "builtin.module(merge-single-qubit-rotation-gates)"));

  const auto target = llvm::cantFail(CompilerTarget::create(1));
  auto targetedImport = QCProgram::fromQASMString(qasm);
  auto targetedRawQCO = QCProgram::fromQASMString(qasm);
  auto targetedJeff = QCProgram::fromQASMString(qasm);
  auto targetedCustom = QCProgram::fromQASMString(qasm);
  ASSERT_TRUE(targetedImport);
  ASSERT_TRUE(targetedRawQCO);
  ASSERT_TRUE(targetedJeff);
  ASSERT_TRUE(targetedCustom);
  EXPECT_FALSE(runDefaultPipeline(CompilerInput{std::move(*targetedImport)},
                                  ProgramFormat::QCImport, &target));
  EXPECT_FALSE(runDefaultPipeline(CompilerInput{std::move(*targetedRawQCO)},
                                  ProgramFormat::QCO, &target));
  EXPECT_FALSE(runDefaultPipeline(CompilerInput{std::move(*targetedJeff)},
                                  ProgramFormat::Jeff, &target));
  EXPECT_FALSE(runDefaultPipeline(CompilerInput{std::move(*targetedCustom)},
                                  ProgramFormat::QCOOptimized, &target,
                                  "hadamard-lifting"));

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
                                  ProgramFormat::QCImport, nullptr,
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

/**
 * @brief Test: QCOProgram::decomposeMultiControlled runs the pass on MCX.
 *
 * @details Correctness of the decomposition is tested in a dedicated suite.
 */
TEST_F(CompilerPipelineTest, DecomposeMultiControlledPass) {
  auto module = mlir::qc::QCProgramBuilder::build(
      context.get(), mlir::qc::multipleControlledX);
  ASSERT_TRUE(module);

  std::string source;
  llvm::raw_string_ostream stream(source);
  module->print(stream);
  auto input = QCProgram::fromMLIRString(source);
  ASSERT_TRUE(input);
  auto qco = std::move(*input).intoQCO();
  ASSERT_TRUE(qco);
  ASSERT_TRUE(qco->cleanup());
  const auto before = qco->copy();
  ASSERT_TRUE(qco->decomposeMultiControlled(3));
  EXPECT_NE(qco->str(), before.str());
}

TEST_F(CompilerPipelineTest, DecomposeMultiControlledPassMcz) {
  auto module = mlir::qc::QCProgramBuilder::build(
      context.get(), mlir::qc::multipleControlledZ);
  ASSERT_TRUE(module);

  std::string source;
  llvm::raw_string_ostream stream(source);
  module->print(stream);
  auto input = QCProgram::fromMLIRString(source);
  ASSERT_TRUE(input);
  auto qco = std::move(*input).intoQCO();
  ASSERT_TRUE(qco);
  ASSERT_TRUE(qco->cleanup());
  const auto before = qco->copy();
  ASSERT_TRUE(qco->runPassPipeline("decompose-multi-controlled{min-qubits=3}"));
  EXPECT_NE(qco->str(), before.str());
}

TEST_F(CompilerPipelineTest,
       RejectsDecomposeMultiControlledMinQubitsBelowThree) {
  EXPECT_FALSE(isDecomposeMultiControlledConfigValid(2U));
  EXPECT_TRUE(isDecomposeMultiControlledConfigValid(3U));

  auto module = mlir::qc::QCProgramBuilder::build(
      context.get(), mlir::qc::multipleControlledX);
  ASSERT_TRUE(module);
  std::string source;
  llvm::raw_string_ostream stream(source);
  module->print(stream);
  auto input = QCProgram::fromMLIRString(source);
  ASSERT_TRUE(input);
  auto qco = std::move(*input).intoQCO();
  ASSERT_TRUE(qco);
  EXPECT_FALSE(qco->decomposeMultiControlled(2));
}

TEST_F(CompilerPipelineTest, PopulateDecomposeMultiControlledPipeline) {
  auto module =
      QCOProgramBuilder::build(context.get(), [](QCOProgramBuilder& builder) {
        builder.mcx({builder.staticQubit(0), builder.staticQubit(1),
                     builder.staticQubit(2)},
                    builder.staticQubit(3));
        return SmallVector<Value>{};
      });
  ASSERT_TRUE(module);

  std::string before;
  llvm::raw_string_ostream beforeStream(before);
  module->print(beforeStream);

  PassManager pm(module->getContext());
  populateDecomposeMultiControlledPipeline(pm, 3);
  ASSERT_TRUE(pm.run(module.get()).succeeded());

  std::string after;
  llvm::raw_string_ostream afterStream(after);
  module->print(afterStream);
  EXPECT_NE(after, before);
}

INSTANTIATE_TEST_SUITE_P(
    NativeQCPrograms, CompilerPipelineTest,
    testing::Values(
        CompilerPipelineTestCase{"StaticQubits",
                                 MQT_NAMED_BUILDER(mlir::qc::staticQubits),
                                 MQT_NAMED_BUILDER(mlir::qc::staticQubits),
                                 MQT_NAMED_BUILDER(mlir::qir::staticQubits)},
        CompilerPipelineTestCase{
            "StaticQubitsWithOps",
            MQT_NAMED_BUILDER(mlir::qc::staticQubitsWithOps),
            MQT_NAMED_BUILDER(mlir::qc::staticQubitsWithOps),
            MQT_NAMED_BUILDER(mlir::qir::staticQubitsWithOps)},
        CompilerPipelineTestCase{
            "StaticQubitsWithParametricOps",
            MQT_NAMED_BUILDER(mlir::qc::staticQubitsWithParametricOps),
            MQT_NAMED_BUILDER(mlir::qc::staticQubitsWithParametricOps),
            MQT_NAMED_BUILDER(mlir::qir::staticQubitsWithParametricOps)},
        CompilerPipelineTestCase{
            "StaticQubitsWithTwoTargetOps",
            MQT_NAMED_BUILDER(mlir::qc::staticQubitsWithTwoTargetOps),
            MQT_NAMED_BUILDER(mlir::qc::staticQubitsWithTwoTargetOps),
            MQT_NAMED_BUILDER(mlir::qir::staticQubitsWithTwoTargetOps)},
        CompilerPipelineTestCase{
            "StaticQubitsWithCtrl",
            MQT_NAMED_BUILDER(mlir::qc::staticQubitsWithCtrl),
            MQT_NAMED_BUILDER(mlir::qc::staticQubitsWithCtrl),
            MQT_NAMED_BUILDER(mlir::qir::staticQubitsWithCtrl)},
        CompilerPipelineTestCase{
            "StaticQubitsWithInv",
            MQT_NAMED_BUILDER(mlir::qc::staticQubitsWithInv),
            MQT_NAMED_BUILDER(mlir::qc::staticQubitsWithInv),
            MQT_NAMED_BUILDER(mlir::qir::staticQubitsWithInv)},
        CompilerPipelineTestCase{
            "PartialMeasurementToRegister",
            MQT_NAMED_BUILDER(mlir::qc::partialMeasurementToRegister),
            MQT_NAMED_BUILDER(mlir::qc::partialMeasurementToRegister),
            MQT_NAMED_BUILDER(mlir::qir::partialMeasurementToRegister)},
        CompilerPipelineTestCase{
            "DynamicallyIndexedMeasurement",
            MQT_NAMED_BUILDER(mlir::qc::dynamicallyIndexedMeasurement),
            MQT_NAMED_BUILDER(mlir::qc::dynamicallyIndexedMeasurement),
            MQT_NAMED_BUILDER(mlir::qir::dynamicallyIndexedMeasurement)},
        CompilerPipelineTestCase{
            "MeasurementWithoutRegisters",
            MQT_NAMED_BUILDER(mlir::qc::measurementWithoutRegisters),
            MQT_NAMED_BUILDER(mlir::qc::measurementWithoutRegisters),
            MQT_NAMED_BUILDER(mlir::qir::measurementWithoutRegisters)},
        CompilerPipelineTestCase{
            "HWithoutRegister", MQT_NAMED_BUILDER(mlir::qc::hWithoutRegister),
            MQT_NAMED_BUILDER(mlir::qc::hWithoutRegister),
            MQT_NAMED_BUILDER(mlir::qir::hWithoutRegister)},
        CompilerPipelineTestCase{
            "InverseIswap", MQT_NAMED_BUILDER(mlir::qc::inverseIswap),
            MQT_NAMED_BUILDER(mlir::qc::inverseIswap), nullptr, false},
        CompilerPipelineTestCase{
            "QubitReuse", MQT_NAMED_BUILDER(mlir::qc::hGateOnMultipleQubits),
            nullptr, MQT_NAMED_BUILDER(mlir::qir::hGatesAndResetsOnOneQubit),
            true, "reuse-qubits,mqt-qco-default"},
        CompilerPipelineTestCase{
            "QubitReuseWithLifting",
            MQT_NAMED_BUILDER(mlir::qc::singleControlledXOnIndividualQubits),
            nullptr, MQT_NAMED_BUILDER(mlir::qir::reusedCX), true,
            "mqt-qubit-reuse,mqt-qco-default"},
        CompilerPipelineTestCase{
            "QubitReuseWithoutLifting",
            MQT_NAMED_BUILDER(mlir::qc::singleControlledXOnIndividualQubits),
            nullptr,
            MQT_NAMED_BUILDER(mlir::qir::singleControlledXOnIndividualQubits),
            true, "reuse-qubits,mqt-qco-default"}));

/**
 * @brief Test: gate counting respects modifiers and skips barriers.
 */
TEST_F(CompilerPipelineTest, QCProgramCountGates) {
  const std::string qasm = R"(OPENQASM 3.0;
include "stdgates.inc";
qubit[3] q;
h q[0];
cx q[0], q[1];
barrier q[0];
swap q[0], q[1];
ccx q[0], q[1], q[2];
ctrl @ swap q[0], q[1], q[2];
inv @ cx q[0], q[1];
barrier q[0], q[1];
)";
  auto qc = QCProgram::fromQASMString(qasm);
  ASSERT_TRUE(qc);
  EXPECT_EQ(qc->numGates(), 6);
  EXPECT_EQ(qc->numSingleQubitGates(), 1);
  EXPECT_EQ(qc->numTwoQubitGates(), 3);
}

/**
 * @brief Test: gate counting includes each structured control-flow region once.
 */
TEST_F(CompilerPipelineTest, QCProgramCountGatesInStructuredControlFlow) {
  const std::string qasm = R"(OPENQASM 3.0;
include "stdgates.inc";
qubit[3] q;
bit condition = measure q[0];
int selector = 1;
if (condition) {
  for int i in [0:2] {
    x q[i];
  }
} else {
  cx q[0], q[1];
}
while (condition) {
  ctrl @ x q[0], q[1];
}
switch (selector) {
  case 1 {
    swap q[0], q[1];
  }
  default {
    z q[2];
  }
}
)";
  auto qc = QCProgram::fromQASMString(qasm);
  ASSERT_TRUE(qc);
  EXPECT_EQ(qc->numGates(), 5);
  EXPECT_EQ(qc->numSingleQubitGates(), 2);
  EXPECT_EQ(qc->numTwoQubitGates(), 3);
}

} // namespace mqt::test::compiler
