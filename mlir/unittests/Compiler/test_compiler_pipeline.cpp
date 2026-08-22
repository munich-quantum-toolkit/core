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
#include "mlir/Compiler/TargetCompilation.h"
#include "mlir/Dialect/CBit/IR/CBitDialect.h"
#include "mlir/Dialect/MQT/IR/MQTDialect.h"
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
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LLVM.h>

#include <array>
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
    registry.insert<cbit::CBitDialect, mlir::mqt::MQTDialect, QCDialect,
                    QCODialect, qtensor::QTensorDialect, arith::ArithDialect,
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

[[nodiscard]] static CompilerTarget
makeTargetWithProfile(const size_t numQubits, const ProgramFormat format,
                      std::vector<ProgramFeature> features) {
  using ExecutionProfile = CompilerTarget::ExecutionProfile;
  std::vector profiles{
      llvm::cantFail(ExecutionProfile::create(format, std::move(features)))};
  return llvm::cantFail(CompilerTarget::create(
      numQubits, std::nullopt, std::nullopt, std::nullopt,
      std::optional<std::vector<ExecutionProfile>>(std::move(profiles))));
}

[[nodiscard]] static bool compileForTargetWithDiagnostics(
    QCOProgram& program, const CompilerTarget& target, std::string& diagnostics,
    const ProgramFormat format = ProgramFormat::QCOOptimized) {
  const ScopedDiagnosticHandler handler(program.module().getContext(),
                                        [&](Diagnostic& diagnostic) {
                                          diagnostics += diagnostic.str();
                                          diagnostics += '\n';
                                          return success();
                                        });
  return program.compileForTarget(target, format);
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

TEST(CompilerTargetCompilationPipelineTest, LoadsMQTDialectDependency) {
  DialectRegistry registry;
  registry.insert<cbit::CBitDialect, QCODialect, qtensor::QTensorDialect,
                  arith::ArithDialect, func::FuncDialect, scf::SCFDialect>();
  MLIRContext context(registry);
  context.loadAllAvailableDialects();
  ASSERT_EQ(context.getLoadedDialect<mlir::mqt::MQTDialect>(), nullptr);

  constexpr StringLiteral source = R"mlir(
    module {
      func.func @main() attributes {mqt.entry_point} {
        %q0 = qco.alloc : !qco.qubit
        qco.sink %q0 : !qco.qubit
        return
      }
    }
  )mlir";
  auto moduleOp = parseSourceString<ModuleOp>(source.str(), &context);
  ASSERT_TRUE(moduleOp);

  PassManager manager(&context);
  populateTargetCompilationPipeline(manager,
                                    llvm::cantFail(CompilerTarget::create(1)));
  EXPECT_TRUE(manager.run(*moduleOp).succeeded());
  EXPECT_NE(context.getLoadedDialect<mlir::mqt::MQTDialect>(), nullptr);
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

/** @brief Test: runtime feedback uses the selected payload profile. */
TEST_F(CompilerPipelineTest,
       TargetCompilationChecksMeasurementFeedbackTransactionally) {
  constexpr StringLiteral source = R"mlir(
    module {
      func.func @main() attributes {mqt.entry_point} {
        %q0 = qco.alloc : !qco.qubit
        %q1 = qco.alloc : !qco.qubit
        %q2, %first = qco.measure %q0 : !qco.qubit
        %q3, %second = qco.measure %q1 : !qco.qubit
        %condition = arith.andi %first, %second : i1
        %q4 = qco.if %condition args(%arg0 = %q2) -> (!qco.qubit) {
          %q5 = qco.x %arg0 : !qco.qubit -> !qco.qubit
          qco.yield %q5 : !qco.qubit
        } else args(%arg0 = %q2) {
          qco.yield %arg0 : !qco.qubit
        }
        qco.sink %q4 : !qco.qubit
        qco.sink %q3 : !qco.qubit
        return
      }
    }
  )mlir";

  auto unsupported = QCOProgram::fromMLIRString(source.str());
  ASSERT_TRUE(unsupported);
  const auto before = unsupported->str();
  std::string diagnostics;
  const auto unsupportedTarget =
      makeTargetWithProfile(2, ProgramFormat::QIRAdaptive, {});
  EXPECT_FALSE(compileForTargetWithDiagnostics(*unsupported, unsupportedTarget,
                                               diagnostics,
                                               ProgramFormat::QIRAdaptive));
  EXPECT_EQ(unsupported->str(), before);
  EXPECT_TRUE(
      StringRef(diagnostics).contains("does not support program feature"))
      << diagnostics;

  auto supported = QCOProgram::fromMLIRString(source.str());
  ASSERT_TRUE(supported);
  const auto target = makeTargetWithProfile(
      2, ProgramFormat::QIRAdaptive,
      {ProgramFeature::MidCircuitMeasurement,
       ProgramFeature::MeasuredQubitReuse, ProgramFeature::MeasurementResultUse,
       ProgramFeature::BooleanComputation, ProgramFeature::ForwardBranching});
  ASSERT_TRUE(supported->compileForTarget(target, ProgramFormat::QIRAdaptive));
  EXPECT_NE(supported->str().find("qco.if"), std::string::npos);
  EXPECT_NE(supported->str().find("qco.static"), std::string::npos);
}

/** @brief Test: lifecycle semantics are explicit profile requirements. */
TEST_F(CompilerPipelineTest,
       TargetCompilationRequiresMeasurementLifecycleFeatures) {
  struct TestCase {
    StringRef name;
    StringRef source;
    size_t numQubits;
    std::vector<ProgramFeature> prerequisites;
    ProgramFeature missingFeature;
    StringRef diagnostic;
  };
  const std::vector<TestCase> testCases{
      {"mid-circuit measurement",
       R"mlir(
         module {
           func.func @main() attributes {mqt.entry_point} {
             %q0 = qco.alloc : !qco.qubit
             %q1 = qco.alloc : !qco.qubit
             %q2, %measurement = qco.measure %q0 : !qco.qubit
             %q3 = qco.x %q1 : !qco.qubit -> !qco.qubit
             qco.sink %q2 : !qco.qubit
             qco.sink %q3 : !qco.qubit
             return
           }
         }
       )mlir",
       2,
       {},
       ProgramFeature::MidCircuitMeasurement,
       "mid-circuit-measurement"},
      {"measured-qubit reuse",
       R"mlir(
         module {
           func.func @main() attributes {mqt.entry_point} {
             %q0 = qco.alloc : !qco.qubit
             %q1, %measurement = qco.measure %q0 : !qco.qubit
             %q2 = qco.x %q1 : !qco.qubit -> !qco.qubit
             qco.sink %q2 : !qco.qubit
             return
           }
         }
       )mlir",
       1,
       {ProgramFeature::MidCircuitMeasurement},
       ProgramFeature::MeasuredQubitReuse,
       "measured-qubit-reuse"},
      {"measurement-result computation",
       R"mlir(
         module {
           func.func @main(%flag: i1) -> i1 attributes {mqt.entry_point} {
             %q0 = qco.alloc : !qco.qubit
             %q1, %measurement = qco.measure %q0 : !qco.qubit
             %result = arith.xori %measurement, %flag : i1
             qco.sink %q1 : !qco.qubit
             return %result : i1
           }
         }
       )mlir",
       1,
       {ProgramFeature::MidCircuitMeasurement,
        ProgramFeature::BooleanComputation},
       ProgramFeature::MeasurementResultUse,
       "measurement-result-use"},
  };

  for (const auto& testCase : testCases) {
    SCOPED_TRACE(testCase.name.str());
    auto emptyProfile = QCOProgram::fromMLIRString(testCase.source.str());
    ASSERT_TRUE(emptyProfile);
    const auto before = emptyProfile->str();
    std::string emptyDiagnostics;
    EXPECT_FALSE(compileForTargetWithDiagnostics(
        *emptyProfile,
        makeTargetWithProfile(testCase.numQubits, ProgramFormat::QCOOptimized,
                              {}),
        emptyDiagnostics));
    EXPECT_EQ(emptyProfile->str(), before);
    EXPECT_TRUE(StringRef(emptyDiagnostics)
                    .contains("does not support program feature"))
        << emptyDiagnostics;

    auto missingFeature = QCOProgram::fromMLIRString(testCase.source.str());
    ASSERT_TRUE(missingFeature);
    const auto missingBefore = missingFeature->str();
    std::string missingDiagnostics;
    EXPECT_FALSE(compileForTargetWithDiagnostics(
        *missingFeature,
        makeTargetWithProfile(testCase.numQubits, ProgramFormat::QCOOptimized,
                              testCase.prerequisites),
        missingDiagnostics));
    EXPECT_EQ(missingFeature->str(), missingBefore);
    EXPECT_TRUE(StringRef(missingDiagnostics).contains(testCase.diagnostic))
        << missingDiagnostics;

    auto supported = QCOProgram::fromMLIRString(testCase.source.str());
    ASSERT_TRUE(supported);
    auto features = testCase.prerequisites;
    features.emplace_back(testCase.missingFeature);
    EXPECT_TRUE(supported->compileForTarget(makeTargetWithProfile(
        testCase.numQubits, ProgramFormat::QCOOptimized, std::move(features))));
  }
}

/** @brief Test: output forwarding stays distinct from runtime result use. */
TEST_F(CompilerPipelineTest,
       TargetCompilationClassifiesForwardedMeasurementResults) {
  struct TestCase {
    StringRef name;
    StringRef source;
    std::vector<ProgramFeature> prerequisites;
    bool requiresMeasurementResultUse;
  };
  const std::vector<TestCase> testCases{
      {"direct CBit output store",
       R"mlir(
         module {
           func.func @main() -> !cbit.reg<1> attributes {mqt.entry_point} {
             %c0 = arith.constant 0 : index
             %reg = cbit.alloc(#cbit.init<zero>) : !cbit.reg<1>
             %q0 = qco.alloc : !qco.qubit
             %q1, %measurement = qco.measure %q0 : !qco.qubit
             cbit.store %measurement, %reg[%c0] : !cbit.reg<1>
             qco.sink %q1 : !qco.qubit
             return %reg : !cbit.reg<1>
           }
         }
       )mlir",
       {},
       false},
      {"same-index killing CBit store",
       R"mlir(
         module {
           func.func @main(%flag: i1)
               -> (i1, !cbit.reg<1>) attributes {mqt.entry_point} {
             %c0 = arith.constant 0 : index
             %false = arith.constant false
             %reg = cbit.alloc(#cbit.init<zero>) : !cbit.reg<1>
             %q0 = qco.alloc : !qco.qubit
             %q1, %measurement = qco.measure %q0 : !qco.qubit
             cbit.store %measurement, %reg[%c0] : !cbit.reg<1>
             cbit.store %false, %reg[%c0] : !cbit.reg<1>
             %loaded = cbit.load %reg[%c0] : !cbit.reg<1>
             %result = arith.xori %loaded, %flag : i1
             qco.sink %q1 : !qco.qubit
             return %result, %reg : i1, !cbit.reg<1>
           }
         }
       )mlir",
       {ProgramFeature::BooleanComputation},
       false},
      {"output-only loop carry",
       R"mlir(
         module {
           func.func @main(%upper: index) -> i1 attributes {mqt.entry_point} {
             %c0 = arith.constant 0 : index
             %c1 = arith.constant 1 : index
             %false = arith.constant false
             %q0 = qco.alloc : !qco.qubit
             %q1, %measurement = qco.measure %q0 : !qco.qubit
             %result = scf.for %index = %c0 to %upper step %c1
                 iter_args(%current = %measurement) -> i1 {
               scf.yield %false : i1
             }
             qco.sink %q1 : !qco.qubit
             return %result : i1
           }
         }
       )mlir",
       {ProgramFeature::CountedIteration},
       false},
      {"dominating CBit load before nested store",
       R"mlir(
         module {
           func.func @main(%upper: index, %loadIndex: index, %flag: i1)
               -> (i1, !cbit.reg<1>) attributes {mqt.entry_point} {
             %c0 = arith.constant 0 : index
             %c1 = arith.constant 1 : index
             %reg = cbit.alloc(#cbit.init<zero>) : !cbit.reg<1>
             %loaded = cbit.load %reg[%loadIndex] : !cbit.reg<1>
             %result = arith.xori %loaded, %flag : i1
             %q0 = qco.alloc : !qco.qubit
             %q1, %measurement = qco.measure %q0 : !qco.qubit
             scf.for %index = %c0 to %upper step %c1 {
               cbit.store %measurement, %reg[%c0] : !cbit.reg<1>
             }
             qco.sink %q1 : !qco.qubit
             return %result, %reg : i1, !cbit.reg<1>
           }
         }
       )mlir",
       {ProgramFeature::CountedIteration, ProgramFeature::BooleanComputation},
       false},
      {"nested CBit load before later store",
       R"mlir(
         module {
           func.func @main(%upper: index, %flag: i1)
               -> (i1, !cbit.reg<1>) attributes {mqt.entry_point} {
             %c0 = arith.constant 0 : index
             %c1 = arith.constant 1 : index
             %false = arith.constant false
             %reg = cbit.alloc(#cbit.init<zero>) : !cbit.reg<1>
             %result = scf.for %index = %c0 to %upper step %c1
                 iter_args(%current = %false) -> i1 {
               %loaded = cbit.load %reg[%c0] : !cbit.reg<1>
               %next = arith.xori %loaded, %flag : i1
               scf.yield %next : i1
             }
             %q0 = qco.alloc : !qco.qubit
             %q1, %measurement = qco.measure %q0 : !qco.qubit
             cbit.store %measurement, %reg[%c0] : !cbit.reg<1>
             qco.sink %q1 : !qco.qubit
             return %result, %reg : i1, !cbit.reg<1>
           }
         }
       )mlir",
       {ProgramFeature::CountedIteration, ProgramFeature::BooleanComputation},
       false},
      {"constant-disjoint CBit load",
       R"mlir(
         module {
           func.func @main(%upper: index, %flag: i1)
               -> (i1, !cbit.reg<2>) attributes {mqt.entry_point} {
             %c0 = arith.constant 0 : index
             %c1 = arith.constant 1 : index
             %reg = cbit.alloc(#cbit.init<zero>) : !cbit.reg<2>
             %q0 = qco.alloc : !qco.qubit
             %q1, %measurement = qco.measure %q0 : !qco.qubit
             scf.for %index = %c0 to %upper step %c1 {
               cbit.store %measurement, %reg[%c0] : !cbit.reg<2>
             }
             %loaded = cbit.load %reg[%c1] : !cbit.reg<2>
             %result = arith.xori %loaded, %flag : i1
             qco.sink %q1 : !qco.qubit
             return %result, %reg : i1, !cbit.reg<2>
           }
         }
       )mlir",
       {ProgramFeature::CountedIteration, ProgramFeature::BooleanComputation},
       false},
      {"possibly-aliasing CBit load",
       R"mlir(
         module {
           func.func @main(%upper: index, %loadIndex: index, %flag: i1)
               -> (i1, !cbit.reg<2>) attributes {mqt.entry_point} {
             %c0 = arith.constant 0 : index
             %c1 = arith.constant 1 : index
             %reg = cbit.alloc(#cbit.init<zero>) : !cbit.reg<2>
             %q0 = qco.alloc : !qco.qubit
             %q1, %measurement = qco.measure %q0 : !qco.qubit
             scf.for %index = %c0 to %upper step %c1 {
               cbit.store %measurement, %reg[%c0] : !cbit.reg<2>
             }
             %loaded = cbit.load %reg[%loadIndex] : !cbit.reg<2>
             %result = arith.xori %loaded, %flag : i1
             qco.sink %q1 : !qco.qubit
             return %result, %reg : i1, !cbit.reg<2>
           }
         }
       )mlir",
       {ProgramFeature::CountedIteration, ProgramFeature::BooleanComputation,
        ProgramFeature::MidCircuitMeasurement},
       true},
      {"loop-carried CBit memory",
       R"mlir(
         module {
           func.func @main(%upper: index, %flag: i1)
               -> (i1, !cbit.reg<1>) attributes {mqt.entry_point} {
             %c0 = arith.constant 0 : index
             %c1 = arith.constant 1 : index
             %false = arith.constant false
             %reg = cbit.alloc(#cbit.init<zero>) : !cbit.reg<1>
             %q0 = qco.alloc : !qco.qubit
             %q1, %measurement = qco.measure %q0 : !qco.qubit
             %result = scf.for %index = %c0 to %upper step %c1
                 iter_args(%current = %false) -> i1 {
               %loaded = cbit.load %reg[%c0] : !cbit.reg<1>
               %next = arith.xori %loaded, %flag : i1
               cbit.store %measurement, %reg[%c0] : !cbit.reg<1>
               scf.yield %next : i1
             }
             qco.sink %q1 : !qco.qubit
             return %result, %reg : i1, !cbit.reg<1>
           }
         }
       )mlir",
       {ProgramFeature::CountedIteration, ProgramFeature::BooleanComputation,
        ProgramFeature::MidCircuitMeasurement},
       true},
      {"nested-loop-carried CBit memory",
       R"mlir(
         module {
           func.func @main(%outerUpper: index, %innerUpper: index, %flag: i1)
               -> (i1, !cbit.reg<1>) attributes {mqt.entry_point} {
             %c0 = arith.constant 0 : index
             %c1 = arith.constant 1 : index
             %false = arith.constant false
             %reg = cbit.alloc(#cbit.init<zero>) : !cbit.reg<1>
             %q0 = qco.alloc : !qco.qubit
             %q1, %measurement = qco.measure %q0 : !qco.qubit
             %result = scf.for %outer = %c0 to %outerUpper step %c1
                 iter_args(%current = %false) -> i1 {
               %loaded = cbit.load %reg[%c0] : !cbit.reg<1>
               %next = arith.xori %loaded, %flag : i1
               scf.for %inner = %c0 to %innerUpper step %c1 {
                 cbit.store %measurement, %reg[%c0] : !cbit.reg<1>
               }
               scf.yield %next : i1
             }
             qco.sink %q1 : !qco.qubit
             return %result, %reg : i1, !cbit.reg<1>
           }
         }
       )mlir",
       {ProgramFeature::CountedIteration, ProgramFeature::BooleanComputation,
        ProgramFeature::MidCircuitMeasurement},
       true},
      {"CBit register SSA alias",
       R"mlir(
         module {
           func.func @main(%choose: i1, %flag: i1)
               -> (i1, !cbit.reg<1>) attributes {mqt.entry_point} {
             %c0 = arith.constant 0 : index
             %stored = cbit.alloc(#cbit.init<zero>) : !cbit.reg<1>
             %empty = cbit.alloc(#cbit.init<zero>) : !cbit.reg<1>
             %q0 = qco.alloc : !qco.qubit
             %q1, %measurement = qco.measure %q0 : !qco.qubit
             cbit.store %measurement, %stored[%c0] : !cbit.reg<1>
             %selected = arith.select %choose, %stored, %empty : !cbit.reg<1>
             %loaded = cbit.load %selected[%c0] : !cbit.reg<1>
             %result = arith.xori %loaded, %flag : i1
             qco.sink %q1 : !qco.qubit
             return %result, %stored : i1, !cbit.reg<1>
           }
         }
       )mlir",
       {ProgramFeature::BooleanComputation,
        ProgramFeature::MidCircuitMeasurement},
       true},
      {"CBit register loop-result alias",
       R"mlir(
         module {
           func.func @main(%upper: index, %choose: i1, %flag: i1)
               -> (i1, !cbit.reg<1>) attributes {mqt.entry_point} {
             %c0 = arith.constant 0 : index
             %c1 = arith.constant 1 : index
             %stored = cbit.alloc(#cbit.init<zero>) : !cbit.reg<1>
             %empty = cbit.alloc(#cbit.init<zero>) : !cbit.reg<1>
             %q0 = qco.alloc : !qco.qubit
             %q1, %measurement = qco.measure %q0 : !qco.qubit
             cbit.store %measurement, %stored[%c0] : !cbit.reg<1>
             %forwarded = scf.for %index = %c0 to %upper step %c1
                 iter_args(%current = %stored) -> !cbit.reg<1> {
               %next = arith.select %choose, %current, %empty : !cbit.reg<1>
               scf.yield %next : !cbit.reg<1>
             }
             %loaded = cbit.load %forwarded[%c0] : !cbit.reg<1>
             %result = arith.xori %loaded, %flag : i1
             qco.sink %q1 : !qco.qubit
             return %result, %forwarded : i1, !cbit.reg<1>
           }
         }
       )mlir",
       {ProgramFeature::CountedIteration, ProgramFeature::BooleanComputation,
        ProgramFeature::MidCircuitMeasurement},
       true},
  };

  for (const auto& testCase : testCases) {
    SCOPED_TRACE(testCase.name.str());
    auto program = QCOProgram::fromMLIRString(testCase.source.str());
    ASSERT_TRUE(program);
    const auto target = makeTargetWithProfile(1, ProgramFormat::QCOOptimized,
                                              testCase.prerequisites);
    if (!testCase.requiresMeasurementResultUse) {
      EXPECT_TRUE(program->compileForTarget(target));
      continue;
    }

    const auto before = program->str();
    std::string diagnostics;
    EXPECT_FALSE(
        compileForTargetWithDiagnostics(*program, target, diagnostics));
    EXPECT_EQ(program->str(), before);
    EXPECT_TRUE(StringRef(diagnostics).contains("measurement-result-use"))
        << diagnostics;

    auto supported = QCOProgram::fromMLIRString(testCase.source.str());
    ASSERT_TRUE(supported);
    auto features = testCase.prerequisites;
    features.emplace_back(ProgramFeature::MeasurementResultUse);
    EXPECT_TRUE(supported->compileForTarget(makeTargetWithProfile(
        1, ProgramFormat::QCOOptimized, std::move(features))));
  }
}

/** @brief Test: unmodeled operations cannot hide CBit register aliases. */
TEST_F(CompilerPipelineTest,
       TargetCompilationRejectsUnmodeledCBitCarrierTransactionally) {
  constexpr StringLiteral source = R"mlir(
    module {
      func.func @main() -> i1 attributes {mqt.entry_point} {
        %c0 = arith.constant 0 : index
        %true = arith.constant true
        %reg = cbit.alloc(#cbit.init<zero>) : !cbit.reg<1>
        %q0 = qco.alloc : !qco.qubit
        %q1, %measurement = qco.measure %q0 : !qco.qubit
        cbit.store %measurement, %reg[%c0] : !cbit.reg<1>
        %alias = builtin.unrealized_conversion_cast %reg
            : !cbit.reg<1> to !cbit.reg<2>
        %loaded = cbit.load %alias[%c0] : !cbit.reg<2>
        %result = arith.xori %loaded, %true : i1
        qco.sink %q1 : !qco.qubit
        return %result : i1
      }
    }
  )mlir";

  auto program = QCOProgram::fromMLIRString(source.str());
  ASSERT_TRUE(program);
  const auto before = program->str();
  std::string diagnostics;
  EXPECT_FALSE(compileForTargetWithDiagnostics(
      *program,
      makeTargetWithProfile(1, ProgramFormat::QCOOptimized,
                            {ProgramFeature::BooleanComputation}),
      diagnostics));
  EXPECT_EQ(program->str(), before);
  EXPECT_TRUE(StringRef(diagnostics)
                  .contains("classical-bit register carried through unmodeled "
                            "operation 'builtin.unrealized_conversion_cast'"))
      << diagnostics;
}

/** @brief Test: CBit definite-kill proofs have a fixed traversal budget. */
TEST_F(CompilerPipelineTest,
       TargetCompilationBoundsCBitAliasProofsTransactionally) {
  constexpr size_t numAliases = 1024;
  constexpr size_t numInterveningStores = 4097;
  std::string source;
  llvm::raw_string_ostream sourceStream(source);
  sourceStream << R"mlir(
    module {
      func.func @main(%upper: index, %load_index: index,
          %conditions: !cbit.reg<)mlir"
               << numAliases << ">, %unrelated: !cbit.reg<"
               << numInterveningStores << R"mlir(>) -> i1
          attributes {mqt.entry_point} {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %false = arith.constant false
        %reg = cbit.alloc(#cbit.init<zero>) : !cbit.reg<2>
  )mlir";
  for (size_t i = 1; i <= numAliases; ++i) {
    sourceStream << "        %condition_index" << i << " = arith.constant "
                 << i - 1 << " : index\n"
                 << "        %condition" << i
                 << " = cbit.load %conditions[%condition_index" << i
                 << "] : !cbit.reg<" << numAliases << ">\n";
  }
  sourceStream
      << R"mlir(        %loop_alias = scf.for %index = %c0 to %upper step %c1
            iter_args(%current = %reg) -> !cbit.reg<2> {
          %next = arith.select %condition1, %current, %reg : !cbit.reg<2>
          scf.yield %next : !cbit.reg<2>
        }
  )mlir";
  for (size_t i = 1; i <= numAliases; ++i) {
    sourceStream << "        %alias" << i << " = arith.select %condition" << i
                 << ", ";
    if (i == 1) {
      sourceStream << "%loop_alias";
    } else {
      sourceStream << "%alias" << i - 1;
    }
    sourceStream << ", %reg : !cbit.reg<2>\n";
  }
  sourceStream << "        %q0 = qco.alloc : !qco.qubit\n"
               << "        %q1, %measurement = qco.measure %q0 : "
                  "!qco.qubit\n"
               << "        cbit.store %measurement, %alias" << numAliases
               << "[%c0] : !cbit.reg<2>\n";
  // Keep more candidate stores than the observation budget before a definite
  // killing store. A bounded proof must conservatively retain the
  // measurement-result-use requirement instead of reaching that late killer.
  for (size_t i = 0; i < numInterveningStores; ++i) {
    sourceStream << "        %store_index" << i << " = arith.constant " << i
                 << " : index\n"
                 << "        cbit.store %false, %unrelated[%store_index" << i
                 << "] : !cbit.reg<" << numInterveningStores << ">\n";
  }
  sourceStream << "        cbit.store %false, %reg[%c0] : !cbit.reg<2>\n"
               << "        %loaded = cbit.load %alias" << numAliases
               << R"mlir([%load_index] : !cbit.reg<2>
        qco.sink %q1 : !qco.qubit
        return %loaded : i1
      }
    }
  )mlir";

  auto program = QCOProgram::fromMLIRString(source);
  ASSERT_TRUE(program);
  const auto before = program->str();
  std::string diagnostics;
  const bool compiled = compileForTargetWithDiagnostics(
      *program,
      makeTargetWithProfile(1, ProgramFormat::QCOOptimized,
                            {ProgramFeature::MidCircuitMeasurement,
                             ProgramFeature::BooleanComputation,
                             ProgramFeature::CountedIteration}),
      diagnostics);
  EXPECT_FALSE(compiled);
  EXPECT_TRUE(program->str() == before);
  EXPECT_TRUE(StringRef(diagnostics).contains("measurement-result-use"))
      << diagnostics;
}

/** @brief Test: deep CBit observation ordering is analyzed linearly. */
TEST_F(CompilerPipelineTest,
       TargetCompilationAnalyzesDeepCBitObservationOrderingLinearly) {
  constexpr size_t nestingDepth = 512U;
  std::string source;
  llvm::raw_string_ostream sourceStream(source);
  sourceStream << R"mlir(
    module {
      func.func @main(%upper: index, %reg: !cbit.reg<1>,
          %output: !cbit.reg<1>) -> !cbit.reg<1>
          attributes {mqt.entry_point} {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
  )mlir";
  for (size_t depth = 0U; depth < nestingDepth; ++depth) {
    sourceStream << "        scf.for %before" << depth
                 << " = %c0 to %upper step %c1 {\n";
  }
  sourceStream << R"mlir(
        %loaded = cbit.load %reg[%c0] : !cbit.reg<1>
        cbit.store %loaded, %output[%c0] : !cbit.reg<1>
  )mlir";
  for (size_t depth = 0U; depth < nestingDepth; ++depth) {
    sourceStream << "        }\n";
  }
  sourceStream << R"mlir(
        %q0 = qco.alloc : !qco.qubit
        %q1, %measurement = qco.measure %q0 : !qco.qubit
  )mlir";
  for (size_t depth = 0U; depth < nestingDepth; ++depth) {
    sourceStream << "        scf.for %after" << depth
                 << " = %c0 to %upper step %c1 {\n";
  }
  sourceStream << R"mlir(
        cbit.store %measurement, %reg[%c0] : !cbit.reg<1>
  )mlir";
  for (size_t depth = 0U; depth < nestingDepth; ++depth) {
    sourceStream << "        }\n";
  }
  sourceStream << R"mlir(
        qco.sink %q1 : !qco.qubit
        return %output : !cbit.reg<1>
      }
    }
  )mlir";

  auto program = QCOProgram::fromMLIRString(source);
  ASSERT_TRUE(program);
  EXPECT_TRUE(program->compileForTarget(makeTargetWithProfile(
      1, ProgramFormat::QCOOptimized, {ProgramFeature::CountedIteration})));
}

/** @brief Test: CBit feedback branches require definite measurement sources. */
TEST_F(CompilerPipelineTest,
       TargetCompilationChecksCBitBranchConditionProvenance) {
  struct TestCase {
    StringRef name;
    StringRef conditionSetup;
    bool isMeasurementDerived;
  };
  const std::vector<TestCase> testCases{
      {"direct same-index load",
       "%condition = cbit.load %stored[%c0] : !cbit.reg<1>", true},
      {"dynamic-index load",
       "%condition = cbit.load %stored[%loadIndex] : !cbit.reg<1>", false},
      {"possibly-empty register alias",
       R"mlir(
         %selected = arith.select %choose, %stored, %empty : !cbit.reg<1>
         %condition = cbit.load %selected[%c0] : !cbit.reg<1>
       )mlir",
       false},
      {"external-bound loop selector",
       R"mlir(
         %condition = scf.for %index = %c0 to %loadIndex step %c1
             iter_args(%current = %measurement) -> i1 {
           scf.yield %false : i1
         }
       )mlir",
       false},
      {"dynamic-loop external killing store",
       R"mlir(
         scf.for %index = %c0 to %loadIndex step %c1 {
           cbit.store %choose, %stored[%c0] : !cbit.reg<1>
         }
         %condition = cbit.load %stored[%c0] : !cbit.reg<1>
       )mlir",
       false},
  };
  const auto target = makeTargetWithProfile(
      1, ProgramFormat::QIRAdaptive,
      {ProgramFeature::MidCircuitMeasurement,
       ProgramFeature::MeasuredQubitReuse, ProgramFeature::MeasurementResultUse,
       ProgramFeature::BooleanComputation, ProgramFeature::ForwardBranching,
       ProgramFeature::CountedIteration});

  for (const auto& testCase : testCases) {
    SCOPED_TRACE(testCase.name.str());
    const auto source =
        R"mlir(
          module {
            func.func @main(%loadIndex: index, %choose: i1)
                attributes {mqt.entry_point} {
              %c0 = arith.constant 0 : index
              %c1 = arith.constant 1 : index
              %false = arith.constant false
              %stored = cbit.alloc(#cbit.init<zero>) : !cbit.reg<1>
              %empty = cbit.alloc(#cbit.init<zero>) : !cbit.reg<1>
              %q0 = qco.alloc : !qco.qubit
              %q1, %measurement = qco.measure %q0 : !qco.qubit
              cbit.store %measurement, %stored[%c0] : !cbit.reg<1>
        )mlir" +
        testCase.conditionSetup.str() + R"mlir(
              %q2 = qco.if %condition args(%arg = %q1) -> (!qco.qubit) {
                %q3 = qco.x %arg : !qco.qubit -> !qco.qubit
                qco.yield %q3 : !qco.qubit
              } else args(%arg = %q1) {
                qco.yield %arg : !qco.qubit
              }
              qco.sink %q2 : !qco.qubit
              return
            }
          }
        )mlir";
    auto program = QCOProgram::fromMLIRString(source);
    ASSERT_TRUE(program);
    if (testCase.isMeasurementDerived) {
      EXPECT_TRUE(
          program->compileForTarget(target, ProgramFormat::QIRAdaptive));
      continue;
    }

    const auto before = program->str();
    std::string diagnostics;
    EXPECT_FALSE(compileForTargetWithDiagnostics(*program, target, diagnostics,
                                                 ProgramFormat::QIRAdaptive));
    EXPECT_EQ(program->str(), before);
    const auto diagnostic = StringRef(diagnostics);
    EXPECT_TRUE(
        diagnostic.contains("derived from a measurement result") ||
        diagnostic.contains("cannot prove measurement-feedback semantics"))
        << diagnostics;
  }
}

/** @brief Test: stored measurement feedback remains valid inside a loop. */
TEST_F(CompilerPipelineTest,
       TargetCompilationChecksStoredMeasurementFeedbackInsideDynamicLoop) {
  constexpr StringLiteral source = R"mlir(
    module {
      func.func @main(%upper: index) attributes {mqt.entry_point} {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %reg = cbit.alloc(#cbit.init<zero>) : !cbit.reg<1>
        %measured = qco.alloc : !qco.qubit
        %data = qco.alloc : !qco.qubit
        %measured_out, %measurement = qco.measure %measured : !qco.qubit
        cbit.store %measurement, %reg[%c0] : !cbit.reg<1>
        %result = scf.for %index = %c0 to %upper step %c1
            iter_args(%current = %data) -> !qco.qubit {
          %condition = cbit.load %reg[%c0] : !cbit.reg<1>
          %next = qco.if %condition args(%arg = %current) -> (!qco.qubit) {
            %gated = qco.x %arg : !qco.qubit -> !qco.qubit
            qco.yield %gated : !qco.qubit
          } else args(%arg = %current) {
            qco.yield %arg : !qco.qubit
          }
          scf.yield %next : !qco.qubit
        }
        qco.sink %result : !qco.qubit
        qco.sink %measured_out : !qco.qubit
        return
      }
    }
  )mlir";
  const std::vector fullProfile{
      ProgramFeature::MidCircuitMeasurement,
      ProgramFeature::MeasurementResultUse,
      ProgramFeature::ForwardBranching,
      ProgramFeature::CountedIteration,
  };
  struct MissingFeature {
    ProgramFeature feature;
    StringRef diagnostic;
  };
  const std::vector<MissingFeature> missingFeatures{
      {ProgramFeature::MidCircuitMeasurement, "mid-circuit-measurement"},
      {ProgramFeature::MeasurementResultUse, "measurement-result-use"},
      {ProgramFeature::ForwardBranching, "forward-branching"},
      {ProgramFeature::CountedIteration, "counted-iteration"},
  };

  for (const auto& missing : missingFeatures) {
    SCOPED_TRACE(missing.diagnostic.str());
    std::vector<ProgramFeature> incompleteProfile;
    for (const auto feature : fullProfile) {
      if (feature != missing.feature) {
        incompleteProfile.emplace_back(feature);
      }
    }
    auto unsupported = QCOProgram::fromMLIRString(source.str());
    ASSERT_TRUE(unsupported);
    const auto before = unsupported->str();
    std::string diagnostics;
    EXPECT_FALSE(compileForTargetWithDiagnostics(
        *unsupported,
        makeTargetWithProfile(2, ProgramFormat::QIRAdaptive,
                              std::move(incompleteProfile)),
        diagnostics, ProgramFormat::QIRAdaptive));
    EXPECT_EQ(unsupported->str(), before);
    EXPECT_TRUE(StringRef(diagnostics).contains(missing.diagnostic))
        << diagnostics;
  }

  auto supported = QCOProgram::fromMLIRString(source.str());
  ASSERT_TRUE(supported);
  EXPECT_TRUE(supported->compileForTarget(
      makeTargetWithProfile(2, ProgramFormat::QIRAdaptive, fullProfile),
      ProgramFormat::QIRAdaptive));
}

/** @brief Test: measurement-feedback producer traversal is globally bounded. */
TEST_F(CompilerPipelineTest,
       TargetCompilationRejectsOversizedFeedbackProducerGraphTransactionally) {
  constexpr size_t numProducerSteps = 4097;
  std::string source;
  llvm::raw_string_ostream sourceStream(source);
  sourceStream << R"mlir(
    module {
      func.func @main() attributes {mqt.entry_point} {
        %control = qco.alloc : !qco.qubit
        %data = qco.alloc : !qco.qubit
        %measured0, %measurement0 = qco.measure %control : !qco.qubit
  )mlir";
  for (size_t i = 1; i <= numProducerSteps; ++i) {
    sourceStream << "        %measured" << i << ", %measurement" << i
                 << " = qco.measure %measured" << i - 1 << " : !qco.qubit\n";
    sourceStream << "        %value" << i << " = arith.xori ";
    if (i == 1) {
      sourceStream << "%measurement0";
    } else {
      sourceStream << "%value" << i - 1;
    }
    sourceStream << ", %measurement" << i << " : i1\n";
  }
  sourceStream << "        %result = qco.if %value" << numProducerSteps
               << R"mlir( args(%arg = %data) -> (!qco.qubit) {
          %gated = qco.x %arg : !qco.qubit -> !qco.qubit
          qco.yield %gated : !qco.qubit
        } else args(%arg = %data) {
          qco.yield %arg : !qco.qubit
        }
        qco.sink %result : !qco.qubit
  )mlir";
  sourceStream << "        qco.sink %measured" << numProducerSteps
               << R"mlir( : !qco.qubit
        return
      }
    }
  )mlir";

  auto program = QCOProgram::fromMLIRString(source);
  ASSERT_TRUE(program);
  const auto before = program->str();
  std::string diagnostics;
  const auto target = makeTargetWithProfile(
      2, ProgramFormat::QIRAdaptive,
      {ProgramFeature::MidCircuitMeasurement,
       ProgramFeature::MeasuredQubitReuse, ProgramFeature::MeasurementResultUse,
       ProgramFeature::BooleanComputation, ProgramFeature::ForwardBranching});
  EXPECT_FALSE(compileForTargetWithDiagnostics(*program, target, diagnostics,
                                               ProgramFormat::QIRAdaptive));
  EXPECT_EQ(program->str(), before);
  EXPECT_TRUE(StringRef(diagnostics).contains("after 4096 producer steps"))
      << diagnostics;
}

/** @brief Test: feedback provenance crosses a then-only conditional. */
TEST_F(CompilerPipelineTest,
       TargetCompilationTracksCBitFeedbackInsideThenOnlyScfIf) {
  constexpr StringLiteral source = R"mlir(
    module {
      func.func @main() -> !cbit.reg<1> attributes {mqt.entry_point} {
        %c0 = arith.constant 0 : index
        %reg = cbit.alloc(#cbit.init<zero>) : !cbit.reg<1>
        %output = cbit.alloc(#cbit.init<zero>) : !cbit.reg<1>
        %measured = qco.alloc : !qco.qubit
        %measured_out, %measurement = qco.measure %measured : !qco.qubit
        cbit.store %measurement, %reg[%c0] : !cbit.reg<1>
        scf.if %measurement {
          %condition = cbit.load %reg[%c0] : !cbit.reg<1>
          scf.if %condition {
            cbit.store %condition, %output[%c0] : !cbit.reg<1>
          }
        }
        qco.sink %measured_out : !qco.qubit
        return %output : !cbit.reg<1>
      }
    }
  )mlir";

  auto program = QCOProgram::fromMLIRString(source.str());
  ASSERT_TRUE(program);
  EXPECT_TRUE(program->compileForTarget(
      makeTargetWithProfile(1, ProgramFormat::QIRAdaptive,
                            {ProgramFeature::MidCircuitMeasurement,
                             ProgramFeature::MeasurementResultUse,
                             ProgramFeature::ForwardBranching}),
      ProgramFormat::QIRAdaptive));
}

/** @brief Test: measured-qubit provenance crosses structured results. */
TEST_F(CompilerPipelineTest,
       TargetCompilationTracksMeasuredQubitForwardedThroughConditional) {
  constexpr StringLiteral source = R"mlir(
    module {
      func.func @main() attributes {mqt.entry_point} {
        %data = qco.alloc : !qco.qubit
        %control = qco.alloc : !qco.qubit
        %measured_control, %condition = qco.measure %control : !qco.qubit
        %merged = qco.if %condition args(%arg = %data) -> (!qco.qubit) {
          %measured_data, %unused = qco.measure %arg : !qco.qubit
          qco.yield %measured_data : !qco.qubit
        } else args(%arg = %data) {
          qco.yield %arg : !qco.qubit
        }
        %result = qco.x %merged : !qco.qubit -> !qco.qubit
        qco.sink %result : !qco.qubit
        qco.sink %measured_control : !qco.qubit
        return
      }
    }
  )mlir";
  const std::vector prerequisites{
      ProgramFeature::MidCircuitMeasurement,
      ProgramFeature::MeasurementResultUse,
      ProgramFeature::BooleanComputation,
      ProgramFeature::ForwardBranching,
  };

  auto unsupported = QCOProgram::fromMLIRString(source.str());
  ASSERT_TRUE(unsupported);
  const auto before = unsupported->str();
  std::string diagnostics;
  EXPECT_FALSE(compileForTargetWithDiagnostics(
      *unsupported,
      makeTargetWithProfile(2, ProgramFormat::QIRAdaptive, prerequisites),
      diagnostics, ProgramFormat::QIRAdaptive));
  EXPECT_EQ(unsupported->str(), before);
  EXPECT_TRUE(StringRef(diagnostics).contains("measured-qubit-reuse"))
      << diagnostics;

  auto supported = QCOProgram::fromMLIRString(source.str());
  ASSERT_TRUE(supported);
  auto features = prerequisites;
  features.emplace_back(ProgramFeature::MeasuredQubitReuse);
  EXPECT_TRUE(supported->compileForTarget(
      makeTargetWithProfile(2, ProgramFormat::QIRAdaptive, std::move(features)),
      ProgramFormat::QIRAdaptive));
}

/** @brief Test: residual arithmetic needs its corresponding feature. */
TEST_F(CompilerPipelineTest,
       TargetCompilationRequiresStandaloneClassicalComputationFeatures) {
  struct TestCase {
    StringRef name;
    StringRef source;
    ProgramFeature feature;
    StringRef diagnostic;
  };
  const std::vector<TestCase> testCases{
      {"Boolean computation",
       R"mlir(
         module {
           func.func @main(%lhs: i1, %rhs: i1) -> i1
               attributes {mqt.entry_point} {
             %result = arith.xori %lhs, %rhs : i1
             %q0 = qco.alloc : !qco.qubit
             qco.sink %q0 : !qco.qubit
             return %result : i1
           }
         }
       )mlir",
       ProgramFeature::BooleanComputation, "boolean-computation"},
      {"integer computation",
       R"mlir(
         module {
           func.func @main(%lhs: i64, %rhs: i64) -> i64
               attributes {mqt.entry_point} {
             %result = arith.addi %lhs, %rhs : i64
             %q0 = qco.alloc : !qco.qubit
             qco.sink %q0 : !qco.qubit
             return %result : i64
           }
         }
       )mlir",
       ProgramFeature::IntegerComputation, "integer-computation"},
      {"floating-point computation",
       R"mlir(
         module {
           func.func @main(%lhs: f64, %rhs: f64) -> f64
               attributes {mqt.entry_point} {
             %result = arith.addf %lhs, %rhs : f64
             %q0 = qco.alloc : !qco.qubit
             qco.sink %q0 : !qco.qubit
             return %result : f64
           }
         }
       )mlir",
       ProgramFeature::FloatComputation, "float-computation"},
      {"math computation",
       R"mlir(
         module {
           func.func @main(%value: f64) -> f64
               attributes {mqt.entry_point} {
             %result = math.sin %value : f64
             %q0 = qco.alloc : !qco.qubit
             qco.sink %q0 : !qco.qubit
             return %result : f64
           }
         }
       )mlir",
       ProgramFeature::FloatComputation, "float-computation"},
      {"LLVM arithmetic",
       R"mlir(
         module {
           func.func @main(%lhs: i64, %rhs: i64) -> i64
               attributes {mqt.entry_point} {
             %result = llvm.add %lhs, %rhs : i64
             %q0 = qco.alloc : !qco.qubit
             qco.sink %q0 : !qco.qubit
             return %result : i64
           }
         }
       )mlir",
       ProgramFeature::IntegerComputation, "integer-computation"},
  };

  for (const auto& testCase : testCases) {
    SCOPED_TRACE(testCase.name.str());
    auto unsupported = QCOProgram::fromMLIRString(testCase.source.str());
    ASSERT_TRUE(unsupported);
    const auto before = unsupported->str();
    std::string diagnostics;
    EXPECT_FALSE(compileForTargetWithDiagnostics(
        *unsupported, makeTargetWithProfile(1, ProgramFormat::QCOOptimized, {}),
        diagnostics));
    EXPECT_EQ(unsupported->str(), before);
    EXPECT_TRUE(StringRef(diagnostics).contains(testCase.diagnostic))
        << diagnostics;

    auto supported = QCOProgram::fromMLIRString(testCase.source.str());
    ASSERT_TRUE(supported);
    EXPECT_TRUE(supported->compileForTarget(makeTargetWithProfile(
        1, ProgramFormat::QCOOptimized, {testCase.feature})));
  }
}

/** @brief Test: mixed-domain arithmetic requires every atomic feature. */
TEST_F(CompilerPipelineTest,
       TargetCompilationRequiresIntegerAndFloatForMixedArithmetic) {
  constexpr StringLiteral source = R"mlir(
    module {
      func.func @main(%value: i64) -> f64 attributes {mqt.entry_point} {
        %result = arith.sitofp %value : i64 to f64
        %q0 = qco.alloc : !qco.qubit
        qco.sink %q0 : !qco.qubit
        return %result : f64
      }
    }
  )mlir";
  const auto expectRejected = [&](std::vector<ProgramFeature> features,
                                  const StringRef diagnostic) {
    auto program = QCOProgram::fromMLIRString(source.str());
    ASSERT_TRUE(program);
    const auto before = program->str();
    std::string diagnostics;
    EXPECT_FALSE(compileForTargetWithDiagnostics(
        *program,
        makeTargetWithProfile(1, ProgramFormat::QCOOptimized,
                              std::move(features)),
        diagnostics));
    EXPECT_EQ(program->str(), before);
    EXPECT_TRUE(StringRef(diagnostics).contains(diagnostic)) << diagnostics;
  };

  expectRejected({ProgramFeature::FloatComputation}, "integer-computation");
  expectRejected({ProgramFeature::IntegerComputation}, "float-computation");

  auto supported = QCOProgram::fromMLIRString(source.str());
  ASSERT_TRUE(supported);
  EXPECT_TRUE(supported->compileForTarget(makeTargetWithProfile(
      1, ProgramFormat::QCOOptimized,
      {ProgramFeature::IntegerComputation, ProgramFeature::FloatComputation})));
}

/** @brief Test: LLVM aggregates expose every nested computation domain. */
TEST_F(CompilerPipelineTest,
       TargetCompilationRequiresNestedLLVMComputationFeatures) {
  constexpr StringLiteral source = R"mlir(
    module {
      func.func @main(
          %value: !llvm.struct<(i1, array<2 x i64>, f64)>)
          -> !llvm.struct<(i1, array<2 x i64>, f64)>
          attributes {mqt.entry_point} {
        %result = llvm.freeze %value
            : !llvm.struct<(i1, array<2 x i64>, f64)>
        %q0 = qco.alloc : !qco.qubit
        qco.sink %q0 : !qco.qubit
        return %result : !llvm.struct<(i1, array<2 x i64>, f64)>
      }
    }
  )mlir";
  struct MissingFeature {
    std::vector<ProgramFeature> supportedFeatures;
    StringRef diagnostic;
  };
  const std::vector<MissingFeature> missingFeatures{
      {{}, "boolean-computation"},
      {{ProgramFeature::BooleanComputation, ProgramFeature::FloatComputation},
       "integer-computation"},
      {{ProgramFeature::BooleanComputation, ProgramFeature::IntegerComputation},
       "float-computation"},
  };

  for (const auto& missing : missingFeatures) {
    SCOPED_TRACE(missing.diagnostic.str());
    auto unsupported = QCOProgram::fromMLIRString(source.str());
    ASSERT_TRUE(unsupported);
    const auto before = unsupported->str();
    std::string diagnostics;
    EXPECT_FALSE(compileForTargetWithDiagnostics(
        *unsupported,
        makeTargetWithProfile(1, ProgramFormat::QCOOptimized,
                              missing.supportedFeatures),
        diagnostics));
    EXPECT_EQ(unsupported->str(), before);
    EXPECT_TRUE(StringRef(diagnostics).contains(missing.diagnostic))
        << diagnostics;
  }

  auto supported = QCOProgram::fromMLIRString(source.str());
  ASSERT_TRUE(supported);
  EXPECT_TRUE(supported->compileForTarget(makeTargetWithProfile(
      1, ProgramFormat::QCOOptimized,
      {ProgramFeature::BooleanComputation, ProgramFeature::IntegerComputation,
       ProgramFeature::FloatComputation})));
}

/** @brief Test: an unmodeled classical producer fails closed. */
TEST_F(CompilerPipelineTest,
       TargetCompilationRejectsUnmodeledClassicalProducerTransactionally) {
  constexpr StringLiteral source = R"mlir(
    module {
      func.func @main(%buffer: memref<?xi64>, %index: index) -> i64
          attributes {mqt.entry_point} {
        %result = memref.load %buffer[%index] : memref<?xi64>
        %q0 = qco.alloc : !qco.qubit
        qco.sink %q0 : !qco.qubit
        return %result : i64
      }
    }
  )mlir";
  auto program = QCOProgram::fromMLIRString(source.str());
  ASSERT_TRUE(program);
  const auto before = program->str();
  std::string diagnostics;
  EXPECT_FALSE(compileForTargetWithDiagnostics(
      *program,
      makeTargetWithProfile(1, ProgramFormat::QCOOptimized,
                            {ProgramFeature::IntegerComputation}),
      diagnostics));
  EXPECT_EQ(program->str(), before);
  EXPECT_TRUE(
      StringRef(diagnostics)
          .contains("cannot classify runtime classical producer 'memref.load'"))
      << diagnostics;
}

/** @brief Test: tuple results do not hide unmodeled classical producers. */
TEST_F(CompilerPipelineTest,
       TargetCompilationRejectsUnmodeledTupleProducerTransactionally) {
  constexpr StringLiteral unsupportedSource = R"mlir(
    module {
      func.func @main(%flag: i1) -> tuple<i1> attributes {mqt.entry_point} {
        %result = builtin.unrealized_conversion_cast %flag
            : i1 to tuple<i1>
        %q0 = qco.alloc : !qco.qubit
        qco.sink %q0 : !qco.qubit
        return %result : tuple<i1>
      }
    }
  )mlir";
  auto unsupported = QCOProgram::fromMLIRString(unsupportedSource.str());
  ASSERT_TRUE(unsupported);
  const auto before = unsupported->str();
  std::string diagnostics;
  EXPECT_FALSE(compileForTargetWithDiagnostics(
      *unsupported, makeTargetWithProfile(1, ProgramFormat::QCOOptimized, {}),
      diagnostics));
  EXPECT_EQ(unsupported->str(), before);
  EXPECT_TRUE(StringRef(diagnostics)
                  .contains("cannot classify runtime classical producer "
                            "'builtin.unrealized_conversion_cast'"))
      << diagnostics;

  constexpr StringLiteral terminalArgument = R"mlir(
    module {
      func.func @main(%input: tuple<i1>) -> tuple<i1>
          attributes {mqt.entry_point} {
        %q0 = qco.alloc : !qco.qubit
        qco.sink %q0 : !qco.qubit
        return %input : tuple<i1>
      }
    }
  )mlir";
  auto supported = QCOProgram::fromMLIRString(terminalArgument.str());
  ASSERT_TRUE(supported);
  EXPECT_TRUE(supported->compileForTarget(
      makeTargetWithProfile(1, ProgramFormat::QCOOptimized, {})));
}

/** @brief Test: execution features do not leak between payload profiles. */
TEST_F(CompilerPipelineTest, TargetCompilationKeepsFeaturesPayloadScoped) {
  constexpr StringLiteral source = R"mlir(
    module {
      func.func @main() attributes {mqt.entry_point} {
        %q0 = qco.alloc : !qco.qubit
        %q1, %condition = qco.measure %q0 : !qco.qubit
        %q2 = qco.if %condition args(%arg0 = %q1) -> (!qco.qubit) {
          %q3 = qco.x %arg0 : !qco.qubit -> !qco.qubit
          qco.yield %q3 : !qco.qubit
        } else args(%arg0 = %q1) {
          qco.yield %arg0 : !qco.qubit
        }
        qco.sink %q2 : !qco.qubit
        return
      }
    }
  )mlir";

  auto program = QCOProgram::fromMLIRString(source.str());
  ASSERT_TRUE(program);
  const auto before = program->str();
  std::string diagnostics;
  const auto target = makeTargetWithProfile(
      1, ProgramFormat::QIRAdaptive,
      {ProgramFeature::MidCircuitMeasurement,
       ProgramFeature::MeasuredQubitReuse, ProgramFeature::MeasurementResultUse,
       ProgramFeature::BooleanComputation, ProgramFeature::ForwardBranching});

  EXPECT_FALSE(compileForTargetWithDiagnostics(*program, target, diagnostics));
  EXPECT_EQ(program->str(), before);
  EXPECT_TRUE(
      StringRef(diagnostics).contains("does not support program feature"))
      << diagnostics;
}

/** @brief Test: finite counted control is legalized when not native. */
TEST_F(CompilerPipelineTest,
       TargetCompilationUnrollsFiniteLoopWithoutCountedIteration) {
  constexpr StringLiteral source = R"mlir(
    module {
      func.func @main() attributes {mqt.entry_point} {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c3 = arith.constant 3 : index
        %q0 = qco.alloc : !qco.qubit
        %q1 = scf.for %index = %c0 to %c3 step %c1
            iter_args(%arg0 = %q0) -> (!qco.qubit) {
          %q2 = qco.x %arg0 : !qco.qubit -> !qco.qubit
          scf.yield %q2 : !qco.qubit
        }
        qco.sink %q1 : !qco.qubit
        return
      }
    }
  )mlir";

  auto program = QCOProgram::fromMLIRString(source.str());
  ASSERT_TRUE(program);
  ASSERT_TRUE(
      program->compileForTarget(llvm::cantFail(CompilerTarget::create(1))));
  EXPECT_EQ(program->str().find("scf.for"), std::string::npos);
}

/** @brief Test: constant-loop legalization is explicitly bounded. */
TEST_F(CompilerPipelineTest,
       TargetCompilationRejectsOversizedConstantLoopTransactionally) {
  constexpr StringLiteral source = R"mlir(
    module {
      func.func @main() attributes {mqt.entry_point} {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c4097 = arith.constant 4097 : index
        %q0 = qco.alloc : !qco.qubit
        %q1 = scf.for %index = %c0 to %c4097 step %c1
            iter_args(%arg0 = %q0) -> (!qco.qubit) {
          %q2 = qco.x %arg0 : !qco.qubit -> !qco.qubit
          scf.yield %q2 : !qco.qubit
        }
        qco.sink %q1 : !qco.qubit
        return
      }
    }
  )mlir";

  auto program = QCOProgram::fromMLIRString(source.str());
  ASSERT_TRUE(program);
  const auto before = program->str();
  std::string diagnostics;
  EXPECT_FALSE(compileForTargetWithDiagnostics(
      *program, makeTargetWithProfile(1, ProgramFormat::QCOOptimized, {}),
      diagnostics));
  EXPECT_EQ(program->str(), before);
  EXPECT_TRUE(
      StringRef(diagnostics).contains("target legalization refuses to unroll"))
      << diagnostics;
  EXPECT_TRUE(StringRef(diagnostics).contains("4097")) << diagnostics;
}

/** @brief Test: the loop budget counts aggregate cloned operations. */
TEST_F(CompilerPipelineTest,
       TargetCompilationRejectsAggregateLoopExpansionTransactionally) {
  constexpr StringLiteral source = R"mlir(
    module {
      func.func @main() attributes {mqt.entry_point} {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c2049 = arith.constant 2049 : index
        %q0 = qco.alloc : !qco.qubit
        %q1 = scf.for %index = %c0 to %c2049 step %c1
            iter_args(%arg0 = %q0) -> (!qco.qubit) {
          %q2 = qco.x %arg0 : !qco.qubit -> !qco.qubit
          %q3 = qco.h %q2 : !qco.qubit -> !qco.qubit
          scf.yield %q3 : !qco.qubit
        }
        qco.sink %q1 : !qco.qubit
        return
      }
    }
  )mlir";

  auto program = QCOProgram::fromMLIRString(source.str());
  ASSERT_TRUE(program);
  const auto before = program->str();
  std::string diagnostics;
  EXPECT_FALSE(compileForTargetWithDiagnostics(
      *program, makeTargetWithProfile(1, ProgramFormat::QCOOptimized, {}),
      diagnostics));
  EXPECT_EQ(program->str(), before);
  EXPECT_TRUE(
      StringRef(diagnostics)
          .contains(
              "aggregate expansion would clone more than 4096 operations"))
      << diagnostics;
  EXPECT_TRUE(StringRef(diagnostics).contains("2049 iterations"))
      << diagnostics;
}

/** @brief Test: unrolling can expose a zero-trip nested loop. */
TEST_F(CompilerPipelineTest, TargetCompilationErasesNewlyConstantZeroTripLoop) {
  constexpr StringLiteral source = R"mlir(
    module {
      func.func @main() attributes {mqt.entry_point} {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c4097 = arith.constant 4097 : index
        %q0 = qco.alloc : !qco.qubit
        %q1 = scf.for %outer = %c0 to %c1 step %c1
            iter_args(%arg0 = %q0) -> (!qco.qubit) {
          %q2 = scf.for %inner = %c0 to %outer step %c1
              iter_args(%arg1 = %arg0) -> (!qco.qubit) {
            %q3 = scf.for %deep = %c0 to %c4097 step %c1
                iter_args(%arg2 = %arg1) -> (!qco.qubit) {
              %q4 = qco.x %arg2 : !qco.qubit -> !qco.qubit
              scf.yield %q4 : !qco.qubit
            }
            scf.yield %q3 : !qco.qubit
          }
          scf.yield %q2 : !qco.qubit
        }
        qco.sink %q1 : !qco.qubit
        return
      }
    }
  )mlir";

  auto program = QCOProgram::fromMLIRString(source.str());
  ASSERT_TRUE(program);
  EXPECT_TRUE(
      program->compileForTarget(llvm::cantFail(CompilerTarget::create(1))));
  EXPECT_EQ(program->str().find("scf.for"), std::string::npos);
}

/** @brief Test: cloned constant expressions expose nested static loops. */
TEST_F(CompilerPipelineTest, TargetCompilationFoldsClonedNestedLoopBounds) {
  constexpr StringLiteral source = R"mlir(
    module {
      func.func @main() attributes {mqt.entry_point} {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %c3 = arith.constant 3 : index
        %q0 = qco.alloc : !qco.qubit
        %q1 = scf.for %outer = %c0 to %c2 step %c1
            iter_args(%arg0 = %q0) -> (!qco.qubit) {
          %upper = arith.addi %outer, %c3 : index
          %q2 = scf.for %inner = %c0 to %upper step %c1
              iter_args(%arg1 = %arg0) -> (!qco.qubit) {
            %q3 = qco.x %arg1 : !qco.qubit -> !qco.qubit
            scf.yield %q3 : !qco.qubit
          }
          scf.yield %q2 : !qco.qubit
        }
        qco.sink %q1 : !qco.qubit
        return
      }
    }
  )mlir";

  auto program = QCOProgram::fromMLIRString(source.str());
  ASSERT_TRUE(program);
  EXPECT_TRUE(
      program->compileForTarget(llvm::cantFail(CompilerTarget::create(1))));
  EXPECT_EQ(program->str().find("scf.for"), std::string::npos);
}

/** @brief Test: cloned region-local bounds expose nested static loops. */
TEST_F(CompilerPipelineTest,
       TargetCompilationFoldsClonedRegionNestedLoopBounds) {
  constexpr StringLiteral source = R"mlir(
    module {
      func.func @main() attributes {mqt.entry_point} {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %c3 = arith.constant 3 : index
        %control = qco.alloc : !qco.qubit
        %measured_control, %condition = qco.measure %control : !qco.qubit
        %data = qco.alloc : !qco.qubit
        %result = scf.for %outer = %c0 to %c2 step %c1
            iter_args(%current = %data) -> (!qco.qubit) {
          %next = qco.if %condition args(%arg0 = %current) -> (!qco.qubit) {
            %upper = arith.addi %outer, %c3 : index
            %then_result = scf.for %inner = %c0 to %upper step %c1
                iter_args(%arg1 = %arg0) -> (!qco.qubit) {
              %updated = qco.x %arg1 : !qco.qubit -> !qco.qubit
              scf.yield %updated : !qco.qubit
            }
            qco.yield %then_result : !qco.qubit
          } else args(%arg0 = %current) {
            qco.yield %arg0 : !qco.qubit
          }
          scf.yield %next : !qco.qubit
        }
        qco.sink %result : !qco.qubit
        qco.sink %measured_control : !qco.qubit
        return
      }
    }
  )mlir";

  auto program = QCOProgram::fromMLIRString(source.str());
  ASSERT_TRUE(program);
  const auto target = makeTargetWithProfile(
      2, ProgramFormat::QCOOptimized,
      {ProgramFeature::MidCircuitMeasurement,
       ProgramFeature::MeasurementResultUse, ProgramFeature::BooleanComputation,
       ProgramFeature::ForwardBranching});
  EXPECT_TRUE(program->compileForTarget(target));
  EXPECT_EQ(program->str().find("scf.for"), std::string::npos);
}

/** @brief Test: folding an unrolled IV removes statically dead control. */
TEST_F(CompilerPipelineTest,
       TargetCompilationIgnoresNewlyStaticUntakenControlRegion) {
  constexpr StringLiteral source = R"mlir(
    module {
      func.func @main() attributes {mqt.entry_point} {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %c4097 = arith.constant 4097 : index
        %q0 = qco.alloc : !qco.qubit
        %q1 = scf.for %outer = %c0 to %c2 step %c1
            iter_args(%current = %q0) -> (!qco.qubit) {
          %condition = arith.cmpi ne, %outer, %c4097 : index
          %next = qco.if %condition args(%arg0 = %current) -> (!qco.qubit) {
            %updated = qco.x %arg0 : !qco.qubit -> !qco.qubit
            qco.yield %updated : !qco.qubit
          } else args(%arg0 = %current) {
            %dead = scf.for %inner = %c0 to %c4097 step %c1
                iter_args(%arg1 = %arg0) -> (!qco.qubit) {
              %updated = qco.x %arg1 : !qco.qubit -> !qco.qubit
              scf.yield %updated : !qco.qubit
            }
            qco.yield %dead : !qco.qubit
          }
          scf.yield %next : !qco.qubit
        }
        qco.sink %q1 : !qco.qubit
        return
      }
    }
  )mlir";

  auto program = QCOProgram::fromMLIRString(source.str());
  ASSERT_TRUE(program);
  EXPECT_TRUE(
      program->compileForTarget(llvm::cantFail(CompilerTarget::create(1))));
  EXPECT_EQ(program->str().find("scf.for"), std::string::npos);
  EXPECT_EQ(program->str().find("qco.if"), std::string::npos);
}

/** @brief Test: dead loop expansion does not consume the live clone budget. */
TEST_F(CompilerPipelineTest,
       TargetCompilationReclaimsBudgetAfterRemovingDeadControlRegion) {
  constexpr StringLiteral source = R"mlir(
    module {
      func.func @main() attributes {mqt.entry_point} {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %c16 = arith.constant 16 : index
        %c2040 = arith.constant 2040 : index
        %c4097 = arith.constant 4097 : index
        %false = arith.constant false
        %condition, %upper = scf.for %outer = %c0 to %c2 step %c1
            iter_args(%previous = %false, %bound = %c0) -> (i1, index) {
          %next = arith.cmpi ne, %outer, %c4097 : index
          scf.yield %next, %c16 : i1, index
        }
        %q0 = qco.alloc : !qco.qubit
        %q1 = qco.if %condition args(%arg0 = %q0) -> (!qco.qubit) {
          %updated = qco.x %arg0 : !qco.qubit -> !qco.qubit
          qco.yield %updated : !qco.qubit
        } else args(%arg0 = %q0) {
          %dead = scf.for %inner = %c0 to %c2040 step %c1
              iter_args(%arg1 = %arg0) -> (!qco.qubit) {
            %updated = qco.x %arg1 : !qco.qubit -> !qco.qubit
            scf.yield %updated : !qco.qubit
          }
          qco.yield %dead : !qco.qubit
        }
        %q2 = scf.for %index = %c0 to %upper step %c1
            iter_args(%current = %q1) -> (!qco.qubit) {
          %updated = qco.h %current : !qco.qubit -> !qco.qubit
          scf.yield %updated : !qco.qubit
        }
        qco.sink %q2 : !qco.qubit
        return
      }
    }
  )mlir";

  auto program = QCOProgram::fromMLIRString(source.str());
  ASSERT_TRUE(program);
  EXPECT_TRUE(
      program->compileForTarget(llvm::cantFail(CompilerTarget::create(1))));
  EXPECT_EQ(program->str().find("scf.for"), std::string::npos);
  EXPECT_EQ(program->str().find("qco.if"), std::string::npos);
}

/** @brief Test: an unrolled result can expose a later constant loop. */
TEST_F(CompilerPipelineTest, TargetCompilationRevisitsLoopUsingUnrolledResult) {
  constexpr StringLiteral source = R"mlir(
    module {
      func.func @main() attributes {mqt.entry_point} {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c3 = arith.constant 3 : index
        %last = scf.for %index = %c0 to %c3 step %c1
            iter_args(%value = %c0) -> (index) {
          scf.yield %index : index
        }
        %q0 = qco.alloc : !qco.qubit
        %q1 = scf.for %index = %c0 to %last step %c1
            iter_args(%current = %q0) -> (!qco.qubit) {
          %updated = qco.x %current : !qco.qubit -> !qco.qubit
          scf.yield %updated : !qco.qubit
        }
        qco.sink %q1 : !qco.qubit
        return
      }
    }
  )mlir";

  auto program = QCOProgram::fromMLIRString(source.str());
  ASSERT_TRUE(program);
  EXPECT_TRUE(
      program->compileForTarget(llvm::cantFail(CompilerTarget::create(1))));
  EXPECT_EQ(program->str().find("scf.for"), std::string::npos);
}

/** @brief Test: newly static one-trip loops safely forward block arguments. */
TEST_F(CompilerPipelineTest,
       TargetCompilationSafelyInlinesNewlyStaticLoopResults) {
  constexpr StringLiteral source = R"mlir(
    module {
      func.func @main() attributes {mqt.entry_point} {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %single_trip_upper = scf.for %seed = %c0 to %c2 step %c1
            iter_args(%value = %c0) -> (index) {
          scf.yield %seed : index
        }
        %iv_result, %carried_result = scf.for %index = %c0
            to %single_trip_upper step %c1
            iter_args(%iv_value = %c0, %carried = %c2) -> (index, index) {
          scf.yield %index, %carried : index, index
        }
        %upper = arith.addi %iv_result, %carried_result : index
        %q0 = qco.alloc : !qco.qubit
        %q1 = scf.for %index = %c0 to %upper step %c1
            iter_args(%current = %q0) -> (!qco.qubit) {
          %updated = qco.x %current : !qco.qubit -> !qco.qubit
          scf.yield %updated : !qco.qubit
        }
        qco.sink %q1 : !qco.qubit
        return
      }
    }
  )mlir";

  auto program = QCOProgram::fromMLIRString(source.str());
  ASSERT_TRUE(program);
  EXPECT_TRUE(
      program->compileForTarget(llvm::cantFail(CompilerTarget::create(1))));
  EXPECT_EQ(program->str().find("scf.for"), std::string::npos);
}

/** @brief Test: many zero-clone loops are processed without full rescans. */
TEST_F(CompilerPipelineTest,
       TargetCompilationProcessesManySingleTripLoopsLinearly) {
  constexpr size_t numLoops = 512;
  std::string source;
  llvm::raw_string_ostream sourceStream(source);
  sourceStream << R"mlir(
    module {
      func.func @main() attributes {mqt.entry_point} {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %single_trip_upper = scf.for %seed = %c0 to %c2 step %c1
            iter_args(%value = %c0) -> (index) {
          scf.yield %seed : index
        }
        %q0 = qco.alloc : !qco.qubit
  )mlir";
  for (size_t i = 1; i <= numLoops; ++i) {
    sourceStream << "        %q" << i << " = scf.for %index" << i
                 << " = %c0 to %single_trip_upper step %c1\n"
                 << "            iter_args(%arg" << i << " = %q" << i - 1
                 << ") -> (!qco.qubit) {\n"
                 << "          %next" << i << " = qco.x %arg" << i
                 << " : !qco.qubit -> !qco.qubit\n"
                 << "          scf.yield %next" << i << " : !qco.qubit\n"
                 << "        }\n";
  }
  sourceStream << "        qco.sink %q" << numLoops << R"mlir( : !qco.qubit
        return
      }
    }
  )mlir";

  auto program = QCOProgram::fromMLIRString(source);
  ASSERT_TRUE(program);
  EXPECT_TRUE(
      program->compileForTarget(llvm::cantFail(CompilerTarget::create(1))));
  EXPECT_EQ(program->str().find("scf.for"), std::string::npos);
}

/** @brief Test: nested one-trip loops are inlined without recursive clones. */
TEST_F(CompilerPipelineTest,
       TargetCompilationProcessesDeepSingleTripLoopsLinearly) {
  constexpr size_t numLoops = 1024U;
  std::string source;
  llvm::raw_string_ostream sourceStream(source);
  sourceStream << R"mlir(
    module {
      func.func @main() attributes {mqt.entry_point} {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %single_trip_upper = scf.for %seed = %c0 to %c2 step %c1
            iter_args(%value = %c0) -> (index) {
          scf.yield %seed : index
        }
        %q0 = qco.alloc : !qco.qubit
  )mlir";
  for (size_t i = 1U; i <= numLoops; ++i) {
    sourceStream << "        %q" << i << " = scf.for %index" << i
                 << " = %c0 to ";
    if (i == 1U) {
      sourceStream << "%single_trip_upper";
    } else {
      sourceStream << "%upper" << i;
    }
    sourceStream << " step %c1\n"
                 << "            iter_args(%arg" << i << " = ";
    if (i == 1U) {
      sourceStream << "%q0";
    } else {
      sourceStream << "%arg" << i - 1U;
    }
    sourceStream << ") -> (!qco.qubit) {\n";
    if (i < numLoops) {
      sourceStream << "          %upper" << i + 1U << " = arith.addi %index"
                   << i << ", %c1 : index\n";
    }
  }
  sourceStream << "          %deep = qco.x %arg" << numLoops
               << " : !qco.qubit -> !qco.qubit\n"
               << "          scf.yield %deep : !qco.qubit\n"
               << "        }\n";
  for (size_t i = numLoops; i > 1U; --i) {
    sourceStream << "          scf.yield %q" << i << " : !qco.qubit\n"
                 << "        }\n";
  }
  sourceStream << R"mlir(
        qco.sink %q1 : !qco.qubit
        return
      }
    }
  )mlir";

  auto program = QCOProgram::fromMLIRString(source);
  ASSERT_TRUE(program);
  EXPECT_TRUE(
      program->compileForTarget(llvm::cantFail(CompilerTarget::create(1))));
  EXPECT_EQ(program->str().find("scf.for"), std::string::npos);
}

/** @brief Test: loop-IV folding follows changed SSA edges only. */
TEST_F(CompilerPipelineTest,
       TargetCompilationProcessesNestedArithmeticDependenciesLinearly) {
  constexpr size_t numLoops = 1024U;
  std::string source;
  llvm::raw_string_ostream sourceStream(source);
  sourceStream << R"mlir(
    module {
      func.func @main(%dynamic: index) attributes {mqt.entry_point} {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %single_trip_upper = scf.for %seed = %c0 to %c2 step %c1
            iter_args(%value = %c0) -> (index) {
          scf.yield %seed : index
        }
        %q0 = qco.alloc : !qco.qubit
  )mlir";
  for (size_t i = 1U; i <= numLoops; ++i) {
    sourceStream << "        %q" << i << " = scf.for %index" << i
                 << " = %c0 to ";
    if (i == 1U) {
      sourceStream << "%single_trip_upper";
    } else {
      sourceStream << "%upper" << i;
    }
    sourceStream << " step %c1\n"
                 << "            iter_args(%arg" << i << " = ";
    if (i == 1U) {
      sourceStream << "%q0";
    } else {
      sourceStream << "%arg" << i - 1U;
    }
    sourceStream << ") -> (!qco.qubit) {\n"
                 << "          %x" << i << " = arith.addi ";
    if (i == 1U) {
      sourceStream << "%dynamic";
    } else {
      sourceStream << "%x" << i - 1U;
    }
    sourceStream << ", %index" << i << " : index\n";
    if (i < numLoops) {
      sourceStream << "          %upper" << i + 1U << " = arith.addi %index"
                   << i << ", %c1 : index\n";
    }
  }
  sourceStream << "          %condition = arith.cmpi eq, %x" << numLoops
               << ", %dynamic : index\n"
               << "          %selected = qco.if %condition args(%arg = %arg"
               << numLoops << ") -> (!qco.qubit) {\n"
               << "            %updated = qco.x %arg : !qco.qubit -> "
                  "!qco.qubit\n"
               << "            qco.yield %updated : !qco.qubit\n"
               << "          } else args(%arg = %arg" << numLoops << ") {\n"
               << "            qco.yield %arg : !qco.qubit\n"
               << "          }\n"
               << "          scf.yield %selected : !qco.qubit\n"
               << "        }\n";
  for (size_t i = numLoops; i > 1U; --i) {
    sourceStream << "          scf.yield %q" << i << " : !qco.qubit\n"
                 << "        }\n";
  }
  sourceStream << R"mlir(
        qco.sink %q1 : !qco.qubit
        return
      }
    }
  )mlir";

  auto program = QCOProgram::fromMLIRString(source);
  ASSERT_TRUE(program);
  EXPECT_TRUE(
      program->compileForTarget(llvm::cantFail(CompilerTarget::create(1))));
  EXPECT_EQ(program->str().find("scf.for"), std::string::npos);
}

/** @brief Test: unsigned static trip counts cannot bypass the clone limit. */
TEST_F(CompilerPipelineTest,
       TargetCompilationRejectsStaticTripCountAboveInt64Transactionally) {
  constexpr StringLiteral source = R"mlir(
    module {
      func.func @main() attributes {mqt.entry_point} {
        %c0 = arith.constant 0 : i64
        %cmax = arith.constant -1 : i64
        %c1 = arith.constant 1 : i64
        %q0 = qco.alloc : !qco.qubit
        %q1 = scf.for unsigned %index = %c0 to %cmax step %c1
            iter_args(%arg0 = %q0) -> (!qco.qubit) : i64 {
          %q2 = qco.x %arg0 : !qco.qubit -> !qco.qubit
          scf.yield %q2 : !qco.qubit
        }
        qco.sink %q1 : !qco.qubit
        return
      }
    }
  )mlir";

  auto program = QCOProgram::fromMLIRString(source.str());
  ASSERT_TRUE(program);
  const auto before = program->str();
  std::string diagnostics;
  EXPECT_FALSE(compileForTargetWithDiagnostics(
      *program, makeTargetWithProfile(1, ProgramFormat::QCOOptimized, {}),
      diagnostics));
  EXPECT_EQ(program->str(), before);
  EXPECT_TRUE(
      StringRef(diagnostics).contains("target legalization refuses to unroll"))
      << diagnostics;
  EXPECT_TRUE(StringRef(diagnostics).contains("at least 4097 iterations"))
      << diagnostics;
}

/** @brief Test: integer loop unrolling does not narrow bounds to int64_t. */
TEST_F(CompilerPipelineTest,
       TargetCompilationSafelyUnrollsWideAndNarrowIntegerLoops) {
  struct TestCase {
    StringRef comparison;
    StringRef type;
    StringRef upper;
    StringRef step;
  };
  constexpr std::array testCases{
      TestCase{"unsigned ", "i8", "-1", "1"},
      TestCase{"unsigned ", "i128", "2", "1"},
      TestCase{"", "i64", "9223372036854775807", "9223372036854775806"},
  };

  for (const TestCase testCase : testCases) {
    SCOPED_TRACE(testCase.type.str());
    std::string source;
    llvm::raw_string_ostream sourceStream(source);
    sourceStream << R"mlir(
      module {
        func.func @main() attributes {mqt.entry_point} {
          %lower = arith.constant 0 : )mlir"
                 << testCase.type << "\n          %upper = arith.constant "
                 << testCase.upper << " : " << testCase.type
                 << "\n          %step = arith.constant " << testCase.step
                 << " : " << testCase.type << R"mlir(
          %q0 = qco.alloc : !qco.qubit
          %q1 = scf.for )mlir"
                 << testCase.comparison
                 << R"mlir(%index = %lower to %upper step %step
              iter_args(%arg0 = %q0) -> (!qco.qubit) : )mlir"
                 << testCase.type << R"mlir( {
            %q2 = qco.x %arg0 : !qco.qubit -> !qco.qubit
            scf.yield %q2 : !qco.qubit
          }
          qco.sink %q1 : !qco.qubit
          return
        }
      }
    )mlir";

    auto program = QCOProgram::fromMLIRString(source);
    ASSERT_TRUE(program);
    EXPECT_TRUE(
        program->compileForTarget(llvm::cantFail(CompilerTarget::create(1))));
    EXPECT_EQ(program->str().find("scf.for"), std::string::npos);
  }
}

/** @brief Test: upstream signed trip-count overflow fails before cleanup. */
TEST_F(CompilerPipelineTest,
       TargetCompilationRejectsSignedTripCountOverflowTransactionally) {
  constexpr StringLiteral source = R"mlir(
    module {
      func.func @main() attributes {mqt.entry_point} {
        %min = arith.constant -9223372036854775808 : index
        %max = arith.constant 9223372036854775807 : index
        %c1 = arith.constant 1 : index
        %q0 = qco.alloc : !qco.qubit
        %q1 = scf.for %index = %min to %max step %c1
            iter_args(%arg0 = %q0) -> (!qco.qubit) {
          %q2 = qco.x %arg0 : !qco.qubit -> !qco.qubit
          scf.yield %q2 : !qco.qubit
        }
        qco.sink %q1 : !qco.qubit
        return
      }
    }
  )mlir";

  auto program = QCOProgram::fromMLIRString(source.str());
  ASSERT_TRUE(program);
  const auto before = program->str();
  std::string diagnostics;
  EXPECT_FALSE(compileForTargetWithDiagnostics(
      *program,
      makeTargetWithProfile(1, ProgramFormat::QCOOptimized,
                            {ProgramFeature::CountedIteration}),
      diagnostics));
  EXPECT_EQ(program->str(), before);
  EXPECT_TRUE(StringRef(diagnostics).contains("overflows MLIR's analysis"))
      << diagnostics;
}

/** @brief Test: large static loops remain potentially repeating. */
TEST_F(CompilerPipelineTest,
       TargetCompilationChecksNextIterationUseForLargeStaticLoop) {
  constexpr StringLiteral source = R"mlir(
    module {
      func.func @main(%reg: !cbit.reg<1>, %output: !cbit.reg<1>)
          -> !cbit.reg<1>
          attributes {mqt.entry_point} {
        %c0 = arith.constant 0 : index
        %lower = arith.constant 0 : i64
        %upper = arith.constant -1 : i64
        %step = arith.constant 1 : i64
        %q0 = qco.alloc : !qco.qubit
        %q1 = scf.for unsigned %index = %lower to %upper step %step
            iter_args(%arg0 = %q0) -> (!qco.qubit) : i64 {
          %loaded = cbit.load %reg[%c0] : !cbit.reg<1>
          cbit.store %loaded, %output[%c0] : !cbit.reg<1>
          %q2, %measurement = qco.measure %arg0 : !qco.qubit
          cbit.store %measurement, %reg[%c0] : !cbit.reg<1>
          scf.yield %q2 : !qco.qubit
        }
        qco.sink %q1 : !qco.qubit
        return %output : !cbit.reg<1>
      }
    }
  )mlir";

  auto program = QCOProgram::fromMLIRString(source.str());
  ASSERT_TRUE(program);
  const auto before = program->str();
  std::string diagnostics;
  EXPECT_FALSE(compileForTargetWithDiagnostics(
      *program,
      makeTargetWithProfile(1, ProgramFormat::QCOOptimized,
                            {ProgramFeature::MidCircuitMeasurement,
                             ProgramFeature::MeasuredQubitReuse,
                             ProgramFeature::CountedIteration}),
      diagnostics));
  EXPECT_EQ(program->str(), before);
  EXPECT_TRUE(StringRef(diagnostics).contains("measurement-result-use"))
      << diagnostics;

  auto supported = QCOProgram::fromMLIRString(source.str());
  ASSERT_TRUE(supported);
  EXPECT_TRUE(supported->compileForTarget(makeTargetWithProfile(
      1, ProgramFormat::QCOOptimized,
      {ProgramFeature::MidCircuitMeasurement,
       ProgramFeature::MeasuredQubitReuse, ProgramFeature::MeasurementResultUse,
       ProgramFeature::CountedIteration})));
}

/** @brief Test: structural regions canonicalize before target legality. */
TEST_F(CompilerPipelineTest,
       TargetCompilationCanonicalizesExecuteRegionBeforeLegality) {
  constexpr StringLiteral source = R"mlir(
    module {
      func.func @main() attributes {mqt.entry_point} {
        %q0 = qco.alloc : !qco.qubit
        scf.execute_region {
          scf.yield
        }
        qco.sink %q0 : !qco.qubit
        return
      }
    }
  )mlir";

  auto program = QCOProgram::fromMLIRString(source.str());
  ASSERT_TRUE(program);
  ASSERT_TRUE(
      program->compileForTarget(llvm::cantFail(CompilerTarget::create(1))));
  EXPECT_EQ(program->str().find("scf.execute_region"), std::string::npos);
}

/** @brief Test: residual counted control needs a selected-format feature. */
TEST_F(CompilerPipelineTest,
       TargetCompilationRequiresCountedIterationForDynamicLoop) {
  constexpr StringLiteral source = R"mlir(
    module {
      func.func @main(%upper: index) attributes {mqt.entry_point} {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %q0 = qco.alloc : !qco.qubit
        %q1 = scf.for %index = %c0 to %upper step %c1
            iter_args(%arg0 = %q0) -> (!qco.qubit) {
          %q2 = qco.x %arg0 : !qco.qubit -> !qco.qubit
          scf.yield %q2 : !qco.qubit
        }
        qco.sink %q1 : !qco.qubit
        return
      }
    }
  )mlir";

  auto unsupported = QCOProgram::fromMLIRString(source.str());
  ASSERT_TRUE(unsupported);
  const auto before = unsupported->str();
  std::string diagnostics;
  EXPECT_FALSE(compileForTargetWithDiagnostics(
      *unsupported, llvm::cantFail(CompilerTarget::create(1)), diagnostics));
  EXPECT_EQ(unsupported->str(), before);
  EXPECT_TRUE(StringRef(diagnostics).contains("counted-iteration"))
      << diagnostics;

  auto supported = QCOProgram::fromMLIRString(source.str());
  ASSERT_TRUE(supported);
  const auto target = makeTargetWithProfile(1, ProgramFormat::QCOOptimized,
                                            {ProgramFeature::CountedIteration});
  ASSERT_TRUE(supported->compileForTarget(target));
  EXPECT_NE(supported->str().find("scf.for"), std::string::npos);
}

/** @brief Test: target legality is rooted at the program entry point. */
TEST_F(CompilerPipelineTest, TargetCompilationIgnoresUnusedHelperControl) {
  constexpr StringLiteral source = R"mlir(
    module {
      func.func @helper(%condition: i1) -> i64 {
        %zero = arith.constant 0 : i64
        %result = scf.while (%value = %zero) : (i64) -> i64 {
          scf.condition(%condition) %value : i64
        } do {
        ^bb0(%value: i64):
          %one = arith.constant 1 : i64
          %next = arith.addi %value, %one : i64
          scf.yield %next : i64
        }
        %min = arith.constant -9223372036854775808 : index
        %max = arith.constant 9223372036854775807 : index
        %step = arith.constant 1 : index
        %loop_result = scf.for %index = %min to %max step %step
            iter_args(%current = %result) -> i64 {
          scf.yield %current : i64
        }
        return %loop_result : i64
      }
      func.func @main() attributes {mqt.entry_point} {
        %q0 = qco.alloc : !qco.qubit
        qco.sink %q0 : !qco.qubit
        return
      }
    }
  )mlir";

  auto program = QCOProgram::fromMLIRString(source.str());
  ASSERT_TRUE(program);
  ASSERT_TRUE(
      program->compileForTarget(llvm::cantFail(CompilerTarget::create(1))));
  EXPECT_NE(program->str().find("scf.while"), std::string::npos);
}

/** @brief Test: unused helpers do not participate in target conformance. */
TEST_F(CompilerPipelineTest,
       TargetCompilationIgnoresTargetInvalidUnusedHelper) {
  constexpr StringLiteral source = R"mlir(
    module {
      func.func @helper() {
        %q0 = qco.static 99 : !qco.qubit
        %q1 = qco.h %q0 : !qco.qubit -> !qco.qubit
        qco.sink %q1 : !qco.qubit
        return
      }
      func.func @main() attributes {mqt.entry_point} {
        %q0 = qco.alloc : !qco.qubit
        %q1 = qco.x %q0 : !qco.qubit -> !qco.qubit
        qco.sink %q1 : !qco.qubit
        return
      }
    }
  )mlir";

  std::vector operations{
      llvm::cantFail(CompilerTarget::Operation::create("x", 1, 0))};
  const auto target = llvm::cantFail(
      CompilerTarget::create(1, std::nullopt, std::move(operations)));
  auto program = QCOProgram::fromMLIRString(source.str());
  ASSERT_TRUE(program);

  ASSERT_TRUE(program->compileForTarget(target));
  EXPECT_NE(program->str().find("qco.static 99"), std::string::npos);
  EXPECT_NE(program->str().find("qco.h"), std::string::npos);
}

/** @brief Test: calls in the entry-point subtree fail closed. */
TEST_F(CompilerPipelineTest, TargetCompilationRejectsReachableFunctionCall) {
  constexpr StringLiteral source = R"mlir(
    module {
      func.func @helper() {
        return
      }
      func.func @main() attributes {mqt.entry_point} {
        func.call @helper() : () -> ()
        %q0 = qco.alloc : !qco.qubit
        qco.sink %q0 : !qco.qubit
        return
      }
    }
  )mlir";

  auto program = QCOProgram::fromMLIRString(source.str());
  ASSERT_TRUE(program);
  const auto before = program->str();
  std::string diagnostics;
  EXPECT_FALSE(compileForTargetWithDiagnostics(
      *program, llvm::cantFail(CompilerTarget::create(1)), diagnostics));
  EXPECT_EQ(program->str(), before);
  EXPECT_TRUE(
      StringRef(diagnostics).contains("reachable function call 'func.call'"))
      << diagnostics;
}

/** @brief Test: all reachable call-interface operations fail closed. */
TEST_F(CompilerPipelineTest,
       TargetCompilationRejectsReachableIndirectFunctionCall) {
  constexpr StringLiteral source = R"mlir(
    module {
      func.func @main(%callee: () -> ()) attributes {mqt.entry_point} {
        func.call_indirect %callee() : () -> ()
        %q0 = qco.alloc : !qco.qubit
        qco.sink %q0 : !qco.qubit
        return
      }
    }
  )mlir";

  auto program = QCOProgram::fromMLIRString(source.str());
  ASSERT_TRUE(program);
  const auto before = program->str();
  std::string diagnostics;
  EXPECT_FALSE(compileForTargetWithDiagnostics(
      *program, llvm::cantFail(CompilerTarget::create(1)), diagnostics));
  EXPECT_EQ(program->str(), before);
  EXPECT_TRUE(StringRef(diagnostics)
                  .contains("reachable function call 'func.call_indirect'"))
      << diagnostics;
}

/** @brief Test: runtime branch provenance and arithmetic remain explicit. */
TEST_F(CompilerPipelineTest, TargetCompilationChecksBranchConditionSemantics) {
  constexpr StringLiteral externalCondition = R"mlir(
    module {
      func.func @main(%condition: i1) attributes {mqt.entry_point} {
        %q0 = qco.alloc : !qco.qubit
        %q1 = qco.if %condition args(%arg0 = %q0) -> (!qco.qubit) {
          %q2 = qco.x %arg0 : !qco.qubit -> !qco.qubit
          qco.yield %q2 : !qco.qubit
        } else args(%arg0 = %q0) {
          qco.yield %arg0 : !qco.qubit
        }
        qco.sink %q1 : !qco.qubit
        return
      }
    }
  )mlir";
  const std::vector baseline{
      ProgramFeature::MidCircuitMeasurement, ProgramFeature::MeasuredQubitReuse,
      ProgramFeature::MeasurementResultUse,  ProgramFeature::BooleanComputation,
      ProgramFeature::ForwardBranching,
  };
  const auto baselineTarget =
      makeTargetWithProfile(2, ProgramFormat::QIRAdaptive, baseline);

  auto external = QCOProgram::fromMLIRString(externalCondition.str());
  ASSERT_TRUE(external);
  const auto externalBefore = external->str();
  std::string externalDiagnostics;
  EXPECT_FALSE(compileForTargetWithDiagnostics(*external, baselineTarget,
                                               externalDiagnostics,
                                               ProgramFormat::QIRAdaptive));
  EXPECT_EQ(external->str(), externalBefore);
  EXPECT_TRUE(StringRef(externalDiagnostics)
                  .contains("derived from a measurement result"))
      << externalDiagnostics;

  constexpr StringLiteral widerArithmetic = R"mlir(
    module {
      func.func @main() attributes {mqt.entry_point} {
        %q0 = qco.alloc : !qco.qubit
        %q1 = qco.alloc : !qco.qubit
        %q2, %first = qco.measure %q0 : !qco.qubit
        %q3, %second = qco.measure %q1 : !qco.qubit
        %first_i64 = arith.extui %first : i1 to i64
        %second_i64 = arith.extui %second : i1 to i64
        %sum = arith.addi %first_i64, %second_i64 : i64
        %one = arith.constant 1 : i64
        %condition = arith.cmpi eq, %sum, %one : i64
        %q4 = qco.if %condition args(%arg0 = %q2) -> (!qco.qubit) {
          %q5 = qco.x %arg0 : !qco.qubit -> !qco.qubit
          qco.yield %q5 : !qco.qubit
        } else args(%arg0 = %q2) {
          qco.yield %arg0 : !qco.qubit
        }
        qco.sink %q4 : !qco.qubit
        qco.sink %q3 : !qco.qubit
        return
      }
    }
  )mlir";

  auto unsupported = QCOProgram::fromMLIRString(widerArithmetic.str());
  ASSERT_TRUE(unsupported);
  const auto unsupportedBefore = unsupported->str();
  std::string unsupportedDiagnostics;
  EXPECT_FALSE(compileForTargetWithDiagnostics(*unsupported, baselineTarget,
                                               unsupportedDiagnostics,
                                               ProgramFormat::QIRAdaptive));
  EXPECT_EQ(unsupported->str(), unsupportedBefore);
  EXPECT_TRUE(StringRef(unsupportedDiagnostics).contains("integer-computation"))
      << unsupportedDiagnostics;

  auto supported = QCOProgram::fromMLIRString(widerArithmetic.str());
  ASSERT_TRUE(supported);
  auto integerFeatures = baseline;
  integerFeatures.emplace_back(ProgramFeature::IntegerComputation);
  EXPECT_TRUE(supported->compileForTarget(
      makeTargetWithProfile(2, ProgramFormat::QIRAdaptive,
                            std::move(integerFeatures)),
      ProgramFormat::QIRAdaptive));
}

/** @brief Test: a loop induction value retains bound provenance. */
TEST_F(CompilerPipelineTest,
       TargetCompilationAcceptsMeasurementDerivedLoopInductionFeedback) {
  constexpr StringLiteral source = R"mlir(
    module {
      func.func @main() attributes {mqt.entry_point} {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %q0 = qco.alloc : !qco.qubit
        %q1, %measurement = qco.measure %q0 : !qco.qubit
        %measurement_index = arith.index_castui %measurement : i1 to index
        %upper = arith.muli %measurement_index, %c2 : index
        %last = scf.for %index = %c0 to %upper step %c1
            iter_args(%current = %c0) -> index {
          scf.yield %index : index
        }
        %condition = arith.cmpi ne, %last, %c0 : index
        %q2 = qco.if %condition args(%arg0 = %q1) -> (!qco.qubit) {
          %q3 = qco.x %arg0 : !qco.qubit -> !qco.qubit
          qco.yield %q3 : !qco.qubit
        } else args(%arg0 = %q1) {
          qco.yield %arg0 : !qco.qubit
        }
        qco.sink %q2 : !qco.qubit
        return
      }
    }
  )mlir";

  auto program = QCOProgram::fromMLIRString(source.str());
  ASSERT_TRUE(program);
  const auto target = makeTargetWithProfile(
      1, ProgramFormat::QCOOptimized,
      {ProgramFeature::MidCircuitMeasurement,
       ProgramFeature::MeasuredQubitReuse, ProgramFeature::MeasurementResultUse,
       ProgramFeature::BooleanComputation, ProgramFeature::IntegerComputation,
       ProgramFeature::ForwardBranching, ProgramFeature::CountedIteration});
  EXPECT_TRUE(program->compileForTarget(target));
}

/** @brief Test: target compilation validates its program-format boundary. */
TEST_F(CompilerPipelineTest,
       TargetCompilationRejectsUnsupportedAndInvalidProgramFormats) {
  constexpr StringLiteral source = R"mlir(
    module {
      func.func @main() attributes {mqt.entry_point} {
        %q0 = qco.alloc : !qco.qubit
        qco.sink %q0 : !qco.qubit
        return
      }
    }
  )mlir";
  const std::vector formats{
      ProgramFormat::QCImport,
      ProgramFormat::QCO,
      ProgramFormat::Jeff,
      static_cast<ProgramFormat>(255),
  };
  const auto target = llvm::cantFail(CompilerTarget::create(1));

  for (const ProgramFormat format : formats) {
    SCOPED_TRACE(static_cast<unsigned>(format));
    auto program = QCOProgram::fromMLIRString(source.str());
    ASSERT_TRUE(program);
    const auto before = program->str();
    std::string diagnostics;
    EXPECT_FALSE(
        compileForTargetWithDiagnostics(*program, target, diagnostics, format));
    EXPECT_EQ(program->str(), before);
    EXPECT_TRUE(StringRef(diagnostics)
                    .contains("target compilation requires QCOOptimized, QC, "
                              "OpenQASM3, or QIR "
                              "output"))
        << diagnostics;
  }
}

/** @brief Test: dynamic qubit indexing remains an unsupported lowering form. */
TEST_F(CompilerPipelineTest,
       TargetCompilationRejectsDynamicQubitIndexingBeforeMapping) {
  constexpr StringLiteral source = R"mlir(
    module {
      func.func @main(%index: index)
          attributes {mqt.entry_point} {
        %c2 = arith.constant 2 : index
        %tensor0 = qtensor.alloc(%c2) : tensor<2x!qco.qubit>
        %tensor1, %q0 = qtensor.extract %tensor0[%index]
            : tensor<2x!qco.qubit>
        %q1 = qco.x %q0 : !qco.qubit -> !qco.qubit
        %tensor2 = qtensor.insert %q1 into %tensor1[%index]
            : tensor<2x!qco.qubit>
        qtensor.dealloc %tensor2 : tensor<2x!qco.qubit>
        return
      }
    }
  )mlir";

  auto program = QCOProgram::fromMLIRString(source.str());
  ASSERT_TRUE(program);
  const auto before = program->str();
  std::string diagnostics;
  EXPECT_FALSE(compileForTargetWithDiagnostics(
      *program, llvm::cantFail(CompilerTarget::create(2)), diagnostics));
  EXPECT_EQ(program->str(), before);
  EXPECT_TRUE(StringRef(diagnostics)
                  .contains("'qtensor.extract' with a dynamic qubit index"))
      << diagnostics;
}

/** @brief Test: unranked quantum inputs fail target conformance safely. */
TEST_F(CompilerPipelineTest,
       TargetCompilationRejectsUnrankedQuantumTensorInputTransactionally) {
  constexpr StringLiteral source = R"mlir(
    module {
      func.func @main(%input: tensor<*x!qco.qubit>)
          -> tensor<*x!qco.qubit> attributes {mqt.entry_point} {
        %q0 = qco.alloc : !qco.qubit
        %q1 = qco.x %q0 : !qco.qubit -> !qco.qubit
        qco.sink %q1 : !qco.qubit
        return %input : tensor<*x!qco.qubit>
      }
    }
  )mlir";

  auto program = QCOProgram::fromMLIRString(source.str());
  ASSERT_TRUE(program);
  const auto before = program->str();
  std::string diagnostics;
  EXPECT_FALSE(compileForTargetWithDiagnostics(
      *program, llvm::cantFail(CompilerTarget::create(1)), diagnostics));
  EXPECT_EQ(program->str(), before);
  EXPECT_TRUE(StringRef(diagnostics)
                  .contains("target conformance requires quantum function "
                            "inputs to be assigned to qco.static target sites"))
      << diagnostics;
}

/** @brief Test: nested tuple quantum inputs fail target conformance safely. */
TEST_F(CompilerPipelineTest,
       TargetCompilationRejectsNestedTupleQuantumInputTransactionally) {
  constexpr StringLiteral source = R"mlir(
    module {
      func.func @main(%input: tuple<i1, tuple<!qco.qubit>>)
          -> tuple<i1, tuple<!qco.qubit>> attributes {mqt.entry_point} {
        %q0 = qco.alloc : !qco.qubit
        %q1 = qco.x %q0 : !qco.qubit -> !qco.qubit
        qco.sink %q1 : !qco.qubit
        return %input : tuple<i1, tuple<!qco.qubit>>
      }
    }
  )mlir";

  auto program = QCOProgram::fromMLIRString(source.str());
  ASSERT_TRUE(program);
  const auto before = program->str();
  std::string diagnostics;
  EXPECT_FALSE(compileForTargetWithDiagnostics(
      *program, llvm::cantFail(CompilerTarget::create(1)), diagnostics));
  EXPECT_EQ(program->str(), before);
  EXPECT_TRUE(StringRef(diagnostics)
                  .contains("target conformance requires quantum function "
                            "inputs to be assigned to qco.static target sites"))
      << diagnostics;
}

/** @brief Test: nested LLVM quantum inputs fail target conformance safely. */
TEST_F(CompilerPipelineTest,
       TargetCompilationRejectsNestedLLVMQuantumInputTransactionally) {
  constexpr StringLiteral source = R"mlir(
    module {
      func.func @main(
          %input: !llvm.struct<(i1, array<2 x !qco.qubit>)>)
          -> !llvm.struct<(i1, array<2 x !qco.qubit>)>
          attributes {mqt.entry_point} {
        %q0 = qco.alloc : !qco.qubit
        qco.sink %q0 : !qco.qubit
        return %input : !llvm.struct<(i1, array<2 x !qco.qubit>)>
      }
    }
  )mlir";

  auto program = QCOProgram::fromMLIRString(source.str());
  ASSERT_TRUE(program);
  const auto before = program->str();
  std::string diagnostics;
  EXPECT_FALSE(compileForTargetWithDiagnostics(
      *program, llvm::cantFail(CompilerTarget::create(1)), diagnostics));
  EXPECT_EQ(program->str(), before);
  EXPECT_TRUE(StringRef(diagnostics)
                  .contains("target conformance requires quantum function "
                            "inputs to be assigned to qco.static target sites"))
      << diagnostics;
}

/** @brief Test: recursive LLVM quantum carriers are rejected safely. */
TEST_F(CompilerPipelineTest,
       TargetCompilationRejectsRecursiveLLVMQuantumCarrierTransactionally) {
  constexpr StringLiteral source = R"mlir(
    !recursive = !llvm.struct<"recursive", (
        !llvm.struct<"recursive">, i1, !qco.qubit)>
    module {
      func.func @main(%input: !recursive) -> !recursive
          attributes {mqt.entry_point} {
        %result = llvm.freeze %input : !recursive
        %q0 = qco.alloc : !qco.qubit
        qco.sink %q0 : !qco.qubit
        return %result : !recursive
      }
    }
  )mlir";

  auto program = QCOProgram::fromMLIRString(source.str());
  ASSERT_TRUE(program);
  const auto before = program->str();
  std::string diagnostics;
  EXPECT_FALSE(compileForTargetWithDiagnostics(
      *program,
      makeTargetWithProfile(1, ProgramFormat::QCOOptimized,
                            {ProgramFeature::BooleanComputation}),
      diagnostics));
  EXPECT_EQ(program->str(), before);
  EXPECT_TRUE(StringRef(diagnostics)
                  .contains("quantum state carried through unmodeled operation "
                            "'llvm.freeze'"))
      << diagnostics;
}

/** @brief Test: generic type parameters cannot hide quantum state. */
TEST_F(CompilerPipelineTest,
       TargetCompilationRejectsLLVMTargetExtensionQuantumCarrier) {
  constexpr StringLiteral source = R"mlir(
    module {
      func.func @main(%input: i1) -> i1 attributes {mqt.entry_point} {
        %q0 = qco.alloc : !qco.qubit
        qco.sink %q0 : !qco.qubit
        return %input : i1
      }
    }
  )mlir";

  auto program = QCOProgram::fromMLIRString(source.str());
  ASSERT_TRUE(program);
  MLIRContext* const programContext = program->module().getContext();
  const Type carrier = LLVM::LLVMTargetExtType::get(
      programContext, "mqt.test", {qco::QubitType::get(programContext)}, {});
  func::FuncOp function = *program->module().getOps<func::FuncOp>().begin();
  function.getArgument(0).setType(carrier);
  function.setType(FunctionType::get(programContext, {carrier}, {carrier}));
  func::ReturnOp returnOp = *function.getOps<func::ReturnOp>().begin();
  OpBuilder builder(returnOp);
  auto frozen = LLVM::FreezeOp::create(builder, returnOp.getLoc(),
                                       function.getArgument(0));
  returnOp.setOperand(0, frozen.getRes());

  const auto before = program->str();
  std::string diagnostics;
  EXPECT_FALSE(compileForTargetWithDiagnostics(
      *program, llvm::cantFail(CompilerTarget::create(1)), diagnostics));
  EXPECT_EQ(program->str(), before);
  EXPECT_TRUE(StringRef(diagnostics)
                  .contains("quantum state carried through unmodeled operation "
                            "'llvm.freeze'"))
      << diagnostics;
}

/** @brief Test: deeply nested acyclic types use a bounded native stack. */
TEST_F(CompilerPipelineTest,
       TargetCompilationRejectsDeepAcyclicQuantumCarrierIteratively) {
  constexpr StringLiteral source = R"mlir(
    module {
      func.func @main(%input: i1) -> i1 attributes {mqt.entry_point} {
        %q0 = qco.alloc : !qco.qubit
        qco.sink %q0 : !qco.qubit
        return %input : i1
      }
    }
  )mlir";

  auto program = QCOProgram::fromMLIRString(source.str());
  ASSERT_TRUE(program);
  MLIRContext* const programContext = program->module().getContext();
  Type carrier = qco::QubitType::get(programContext);
  for (size_t depth = 0U; depth < 4096U; ++depth) {
    carrier = LLVM::LLVMArrayType::get(carrier, 1U);
  }
  func::FuncOp function = *program->module().getOps<func::FuncOp>().begin();
  function.getArgument(0).setType(carrier);
  function.setType(FunctionType::get(programContext, {carrier}, {carrier}));

  std::string diagnostics;
  EXPECT_FALSE(compileForTargetWithDiagnostics(
      *program, llvm::cantFail(CompilerTarget::create(1)), diagnostics));
  EXPECT_EQ(function.getArgument(0).getType(), carrier);
  EXPECT_FALSE(program->module()->hasAttr("mqt.target_env"));
  EXPECT_TRUE(StringRef(diagnostics)
                  .contains("target conformance requires quantum function "
                            "inputs to be assigned to qco.static target sites"))
      << diagnostics;
}

/** @brief Test: runtime control cannot carry a qubit tensor. */
TEST_F(CompilerPipelineTest,
       TargetCompilationRejectsQuantumTensorCarriedByDynamicLoop) {
  constexpr StringLiteral source = R"mlir(
    module {
      func.func @main(%upper: index) attributes {mqt.entry_point} {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %tensor0 = qtensor.alloc(%c2) : tensor<2x!qco.qubit>
        %tensor1 = scf.for %index = %c0 to %upper step %c1
            iter_args(%arg0 = %tensor0) -> (tensor<2x!qco.qubit>) {
          %tensor2, %q0 = qtensor.extract %arg0[%c0]
              : tensor<2x!qco.qubit>
          %q1 = qco.x %q0 : !qco.qubit -> !qco.qubit
          %tensor3 = qtensor.insert %q1 into %tensor2[%c0]
              : tensor<2x!qco.qubit>
          scf.yield %tensor3 : tensor<2x!qco.qubit>
        }
        qtensor.dealloc %tensor1 : tensor<2x!qco.qubit>
        return
      }
    }
  )mlir";

  auto program = QCOProgram::fromMLIRString(source.str());
  ASSERT_TRUE(program);
  const auto before = program->str();
  std::string diagnostics;
  const auto target = makeTargetWithProfile(2, ProgramFormat::QCOOptimized,
                                            {ProgramFeature::CountedIteration});
  EXPECT_FALSE(compileForTargetWithDiagnostics(*program, target, diagnostics));
  EXPECT_EQ(program->str(), before);
  EXPECT_TRUE(StringRef(diagnostics)
                  .contains("quantum tensor state carried through "
                            "classical-control construct "
                            "'scf.for'"))
      << diagnostics;
}

/** @brief Test: unmodeled quantum carriers fail without reaching mapping. */
TEST_F(CompilerPipelineTest,
       TargetCompilationRejectsUnmodeledQuantumCarrierTransactionally) {
  constexpr StringLiteral source = R"mlir(
    module {
      func.func @main() attributes {mqt.entry_point} {
        %control = qco.alloc : !qco.qubit
        %lhs = qco.alloc : !qco.qubit
        %rhs = qco.alloc : !qco.qubit
        %measured_control, %condition = qco.measure %control : !qco.qubit
        %selected = arith.select %condition, %lhs, %rhs : !qco.qubit
        %result = qco.x %selected : !qco.qubit -> !qco.qubit
        qco.sink %result : !qco.qubit
        qco.sink %measured_control : !qco.qubit
        return
      }
    }
  )mlir";

  auto program = QCOProgram::fromMLIRString(source.str());
  ASSERT_TRUE(program);
  const auto before = program->str();
  std::string diagnostics;
  const auto target = makeTargetWithProfile(
      3, ProgramFormat::QIRAdaptive,
      {ProgramFeature::MidCircuitMeasurement,
       ProgramFeature::MeasuredQubitReuse, ProgramFeature::MeasurementResultUse,
       ProgramFeature::BooleanComputation, ProgramFeature::ForwardBranching});
  EXPECT_FALSE(compileForTargetWithDiagnostics(*program, target, diagnostics,
                                               ProgramFormat::QIRAdaptive));
  EXPECT_EQ(program->str(), before);
  EXPECT_TRUE(
      StringRef(diagnostics)
          .contains("cannot lower quantum state carried through unmodeled "
                    "operation 'arith.select'"))
      << diagnostics;
}

/** @brief Test: unmodeled regions cannot hide captured quantum state. */
TEST_F(CompilerPipelineTest,
       TargetCompilationRejectsUnmodeledQuantumCapturingRegion) {
  constexpr StringLiteral source = R"mlir(
    module {
      func.func @main() attributes {mqt.entry_point} {
        %q0 = qco.alloc : !qco.qubit
        qco.sink %q0 : !qco.qubit
        return
      }
    }
  )mlir";

  auto program = QCOProgram::fromMLIRString(source.str());
  ASSERT_TRUE(program);
  MLIRContext* const programContext = program->module().getContext();
  programContext->allowUnregisteredDialects();
  programContext->getOrLoadDialect<ub::UBDialect>();
  func::FuncOp function = *program->module().getOps<func::FuncOp>().begin();
  qco::AllocOp allocation = *function.getOps<qco::AllocOp>().begin();
  qco::SinkOp originalSink = *function.getOps<qco::SinkOp>().begin();
  originalSink.erase();

  OpBuilder builder(function.getBody().front().getTerminator());
  OperationState state(builder.getUnknownLoc(), "test.capture");
  state.addRegion();
  Operation* const capture = builder.create(state);
  Region& region = capture->getRegion(0);
  region.push_back(new Block());
  builder.setInsertionPointToStart(&region.front());
  qco::SinkOp::create(builder, capture->getLoc(), allocation.getResult());
  ub::UnreachableOp::create(builder, capture->getLoc());

  const auto before = program->str();
  std::string diagnostics;
  EXPECT_FALSE(compileForTargetWithDiagnostics(
      *program, llvm::cantFail(CompilerTarget::create(1)), diagnostics));
  EXPECT_EQ(program->str(), before);
  EXPECT_TRUE(
      StringRef(diagnostics)
          .contains(
              "cannot verify regions of unmodeled operation 'test.capture'"))
      << diagnostics;
}

/** @brief Test: quantum-capture indexing stays linear in control nesting. */
TEST_F(CompilerPipelineTest,
       TargetCompilationIndexesDeepStructuredControlCaptures) {
  constexpr size_t numNestedControls = 512;
  const auto makeSource = [](const bool captureQuantumState) {
    std::string source;
    llvm::raw_string_ostream sourceStream(source);
    sourceStream << R"mlir(
      module {
        func.func @main(%reg: !cbit.reg<1>) -> i1 attributes {mqt.entry_point} {
          %c0 = arith.constant 0 : index
          %q0 = qco.alloc : !qco.qubit
          %q1, %condition = qco.measure %q0 : !qco.qubit
    )mlir";
    for (size_t i = 0; i < numNestedControls; ++i) {
      sourceStream << "      scf.if %condition {\n";
    }
    if (captureQuantumState) {
      sourceStream << R"mlir(
        %q2 = qco.x %q1 : !qco.qubit -> !qco.qubit
        qco.sink %q2 : !qco.qubit
      )mlir";
    } else {
      sourceStream << "      cbit.store %condition, %reg[%c0] : !cbit.reg<1>\n";
    }
    for (size_t i = 0; i < numNestedControls; ++i) {
      sourceStream << "      }\n";
    }
    if (!captureQuantumState) {
      sourceStream << "      qco.sink %q1 : !qco.qubit\n";
    }
    sourceStream << R"mlir(
          return %condition : i1
        }
      }
    )mlir";
    return source;
  };

  const auto target = makeTargetWithProfile(
      1, ProgramFormat::QCOOptimized,
      {ProgramFeature::MidCircuitMeasurement,
       ProgramFeature::MeasuredQubitReuse, ProgramFeature::MeasurementResultUse,
       ProgramFeature::BooleanComputation, ProgramFeature::ForwardBranching});

  auto uncaptured = QCOProgram::fromMLIRString(makeSource(false));
  ASSERT_TRUE(uncaptured);
  EXPECT_TRUE(uncaptured->compileForTarget(target));

  auto captured = QCOProgram::fromMLIRString(makeSource(true));
  ASSERT_TRUE(captured);
  const auto before = captured->str();
  std::string diagnostics;
  EXPECT_FALSE(compileForTargetWithDiagnostics(*captured, target, diagnostics));
  EXPECT_EQ(captured->str(), before);
  EXPECT_TRUE(
      StringRef(diagnostics)
          .contains(
              "quantum state captured by classical-control construct 'scf.if'"))
      << diagnostics;
}

/** @brief Test: runtime control cannot capture linear quantum state. */
TEST_F(CompilerPipelineTest,
       TargetCompilationRejectsQuantumStateCapturedByConditional) {
  constexpr StringLiteral source = R"mlir(
    module {
      func.func @main() attributes {mqt.entry_point} {
        %q0 = qco.alloc : !qco.qubit
        %q1, %condition = qco.measure %q0 : !qco.qubit
        scf.if %condition {
          %q2 = qco.x %q1 : !qco.qubit -> !qco.qubit
          qco.sink %q2 : !qco.qubit
        }
        return
      }
    }
  )mlir";

  auto program = QCOProgram::fromMLIRString(source.str());
  ASSERT_TRUE(program);
  const auto before = program->str();
  std::string diagnostics;
  const auto target = makeTargetWithProfile(
      1, ProgramFormat::QIRAdaptive,
      {ProgramFeature::MidCircuitMeasurement,
       ProgramFeature::MeasuredQubitReuse, ProgramFeature::MeasurementResultUse,
       ProgramFeature::BooleanComputation, ProgramFeature::ForwardBranching});
  EXPECT_FALSE(compileForTargetWithDiagnostics(*program, target, diagnostics,
                                               ProgramFormat::QIRAdaptive));
  EXPECT_EQ(program->str(), before);
  EXPECT_TRUE(
      StringRef(diagnostics)
          .contains(
              "quantum state captured by classical-control construct 'scf.if'"))
      << diagnostics;
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

} // namespace mqt::test::compiler
