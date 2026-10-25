/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/Dialect/QIR/Builder/QIRProgramBuilder.h"
#include "mqt/Dialect/QIR/Transforms/Passes.h"
#include "mqt/Dialect/QIR/Utils/QIRUtils.h"
#include "mqt/Support/Passes.h"

#include "Support/IRVerification.h"
#include "TestCaseUtils.h"
#include "qir_programs.h"

#include "gtest/gtest.h"

#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BlockSupport.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Casting.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <memory>
#include <ostream>
#include <string>
#include <tuple>
#include <variant>

using namespace mlir;
using namespace qir;

static LLVM::ModuleFlagAttr findModuleFlag(ModuleOp moduleOp,
                                           const StringRef name) {
  LLVM::ModuleFlagAttr result;
  moduleOp->walk([&](LLVM::ModuleFlagsOp flagsOp) {
    for (const auto flag :
         flagsOp.getFlags().getAsRange<LLVM::ModuleFlagAttr>()) {
      if (flag.getKey().getValue() == name) {
        result = flag;
      }
    }
  });
  return result;
}

static LogicalResult attachQIRMetadata(ModuleOp module,
                                       bool useAdaptive = false) {
  PassManager manager(module.getContext());
  manager.addPass(qir::createQIRSetAttributesAndMetadata({useAdaptive}));
  return manager.run(module);
}

namespace {

struct QIRTestCase {
  std::string name;
  ::mqt::test::NamedMLIRBuilder<QIRProgramBuilder> programBuilder;
  ::mqt::test::NamedMLIRBuilder<QIRProgramBuilder> referenceBuilder;

  friend std::ostream& operator<<(std::ostream& os, const QIRTestCase& info);
};

// NOLINTNEXTLINE(llvm-prefer-static-over-anonymous-namespace)
std::ostream& operator<<(std::ostream& os, const QIRTestCase& info) {
  return os << "QIR{" << info.name << ", original="
            << ::mqt::test::displayName(info.programBuilder.name)
            << ", reference="
            << ::mqt::test::displayName(info.referenceBuilder.name) << "}";
}

class QIRTest : public testing::TestWithParam<QIRTestCase> {
protected:
  std::unique_ptr<MLIRContext> context;

  // GoogleTest requires this override name.
  // NOLINTNEXTLINE(readability-identifier-naming)
  void SetUp() override {
    DialectRegistry registry;
    registry.insert<LLVM::LLVMDialect>();
    context = std::make_unique<MLIRContext>();
    context->appendDialectRegistry(registry);
    context->loadAllAvailableDialects();
  }
};

} // namespace

TEST_P(QIRTest, ProgramEquivalence) {
  const auto& [_, programBuilder, referenceBuilder] = GetParam();
  const auto name = " (" + GetParam().name + ")";
  ::mqt::test::DeferredPrinter printer;

  auto program = ::mqt::test::buildMLIRProgram(
      context.get(), programBuilder, QIRProgramBuilder::Profile::Adaptive);
  ASSERT_TRUE(program);
  printer.record(program.get(), "Original QIR IR" + name);
  EXPECT_TRUE(verify(*program).succeeded());

  EXPECT_TRUE(runQIRCleanupPipeline(program.get(), true).succeeded());
  printer.record(program.get(), "Canonicalized QIR IR" + name);
  EXPECT_TRUE(verify(*program).succeeded());

  auto reference = ::mqt::test::buildMLIRProgram(
      context.get(), referenceBuilder, QIRProgramBuilder::Profile::Adaptive);
  ASSERT_TRUE(reference);
  printer.record(reference.get(), "Reference QIR IR" + name);
  EXPECT_TRUE(verify(*reference).succeeded());

  EXPECT_TRUE(runQIRCleanupPipeline(reference.get(), true).succeeded());
  printer.record(reference.get(), "Canonicalized Reference QIR IR" + name);
  EXPECT_TRUE(verify(*reference).succeeded());

  // Builder references may order module symbols differently.
  EXPECT_TRUE(
      areModulesEquivalentWithPermutations(program.get(), reference.get()));
}

TEST(QIRBuilderTest, BuildsWithFreshContext) {
  MLIRContext context;
  auto moduleOp = QIRProgramBuilder::build(&context, [](QIRProgramBuilder& b) {
    b.x(b.allocQubit());
    return b.intConstant(0);
  });
  ASSERT_TRUE(moduleOp);
  EXPECT_TRUE(succeeded(verify(*moduleOp)));
}

TEST_F(QIRTest, BuilderTransfersModuleOwnership) {
  auto moduleOp = [&] {
    QIRProgramBuilder source(context.get());
    source.initialize();
    auto qubit = source.allocQubit();
    QIRProgramBuilder destination(std::move(source));
    destination.x(qubit);
    return destination.finalize();
  }();
  ASSERT_TRUE(moduleOp);
  EXPECT_TRUE(succeeded(verify(*moduleOp)));
}

TEST_F(QIRTest, BuilderRejectsConditionalQubitAllocation) {
  EXPECT_DEATH(
      {
        QIRProgramBuilder builder(context.get());
        builder.initialize();
        builder.scfIf(true, [&] { builder.x(builder.allocQubit()); });
      },
      "Allocate dynamic qubits before building control flow");
}

TEST_F(QIRTest, BuilderRejectsRepeatedQubitAllocation) {
  EXPECT_DEATH(
      {
        QIRProgramBuilder builder(context.get());
        builder.initialize();
        builder.scfWhile([&] {
          auto qubits = builder.allocQubitRegister(1);
          return builder.measure(qubits[0], 0);
        });
      },
      "Allocate dynamic qubits before building control flow");
}

TEST_F(QIRTest, BuilderReturnsEmptyModuleOnVerificationFailure) {
  QIRProgramBuilder builder(context.get());
  builder.initialize();
  ScopedDiagnosticHandler diagnostics(context.get(),
                                      [](Diagnostic&) { return success(); });
  EXPECT_FALSE(builder.finalize(builder.doubleConstant(0.0)));
}

TEST_F(QIRTest, BuilderRejectsMixedStaticAndDynamicQubitAllocationModes) {
  EXPECT_DEATH(
      {
        QIRProgramBuilder builder(context.get());
        builder.initialize();
        mixedStaticThenDynamicQubit(builder);
      },
      "Cannot mix static and dynamic qubit allocation modes");

  EXPECT_DEATH(
      {
        QIRProgramBuilder builder(context.get());
        builder.initialize();
        mixedDynamicRegisterThenStaticQubit(builder);
      },
      "Cannot mix static and dynamic qubit allocation modes");
}

TEST_F(QIRTest, AdaptiveBuilderRecordsAndReleasesResourcesInAllocationOrder) {
  QIRProgramBuilder builder(context.get());
  builder.initialize();
  std::array<SmallVector<Value>, 4> allocated;
  for (int64_t i = 0; i < 12; ++i) {
    auto qubit = builder.allocQubit();
    allocated[0].push_back(qubit);
    auto qubits = builder.allocQubitRegister(1);
    allocated[1].push_back(qubits.value);
    auto index = builder.intConstant(0);
    builder.x(builder.loadQubit(qubits.value, index));
    builder.x(builder.loadQubit(qubits.value, index));
    auto result = builder.measure(qubit, i);
    auto allocation = result.getDefiningOp<LLVM::CallOp>();
    ASSERT_TRUE(allocation);
    EXPECT_EQ(allocation.getCallee(), QIR_RESULT_ALLOC);
    allocated[2].push_back(result);
    auto results = builder.allocClassicalBitRegister(1);
    builder.measure(qubit, results, 0);
    allocated[3].push_back(results.array);
  }
  auto moduleOp = builder.finalize();
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(verify(*moduleOp)));
  std::array<SmallVector<Value>, 4> released;
  SmallVector<Value> recorded;
  moduleOp->walk([&](LLVM::CallOp call) {
    if (call.getCallee() == QIR_QUBIT_RELEASE) {
      released[0].push_back(call.getOperand(0));
    } else if (call.getCallee() == QIR_QUBIT_ARRAY_RELEASE) {
      released[1].push_back(call.getOperand(1));
    } else if (call.getCallee() == QIR_RESULT_RELEASE) {
      released[2].push_back(call.getOperand(0));
    } else if (call.getCallee() == QIR_RESULT_ARRAY_RELEASE) {
      released[3].push_back(call.getOperand(1));
    } else if (call.getCallee() == QIR_RESULT_ARRAY_RECORD_OUTPUT) {
      recorded.push_back(call.getOperand(1));
    }
  });
  EXPECT_EQ(released, allocated);
  EXPECT_EQ(recorded, allocated[3]);
}

TEST_F(QIRTest, AdaptiveBuilderUsesAnArrayForWideClassicalRegisters) {
  QIRProgramBuilder builder(context.get());
  builder.initialize();
  auto reg = builder.allocClassicalBitRegister(1000000);
  EXPECT_TRUE(reg.results.empty());
  EXPECT_TRUE(reg.array);
  EXPECT_EQ(std::get<int64_t>(reg.size), 1000000);
  auto moduleOp = builder.finalize();
  ASSERT_TRUE(moduleOp);
  EXPECT_TRUE(succeeded(verify(*moduleOp)));
}

TEST_F(QIRTest, ResultLabelsAvoidConflictingSymbols) {
  auto moduleOp =
      QIRProgramBuilder::build(context.get(), [](QIRProgramBuilder& builder) {
        return builder.intConstant(0);
      });
  ASSERT_TRUE(moduleOp);
  OpBuilder builder(context.get());
  builder.setInsertionPointToEnd(moduleOp->getBody());
  auto function = LLVM::LLVMFuncOp::create(
      builder, moduleOp->getLoc(), "qir.result_label_collision",
      LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(context.get()), {}));
  auto address = createResultLabel(builder, *moduleOp, "collision");
  auto global = moduleOp->lookupSymbol<LLVM::GlobalOp>(address.getGlobalName());
  ASSERT_TRUE(global);
  EXPECT_NE(global.getSymName(), function.getSymName());
  EXPECT_EQ(global.getValueAttr(),
            builder.getStringAttr(StringRef("collision\0", 10)));
  auto original = createResultLabel(builder, *moduleOp, "first");
  auto conflicting =
      moduleOp->lookupSymbol<LLVM::GlobalOp>(original.getGlobalName());
  conflicting.setValueAttr(builder.getStringAttr(StringRef("other\0", 6)));
  auto corrected = createResultLabel(builder, *moduleOp, "first");
  EXPECT_NE(corrected.getGlobalName(), original.getGlobalName());
  EXPECT_EQ(moduleOp->lookupSymbol<LLVM::GlobalOp>(corrected.getGlobalName())
                .getValueAttr(),
            builder.getStringAttr(StringRef("first\0", 6)));
  for (const bool isConstant : {false, true}) {
    const StringRef label = isConstant ? "address_space" : "mutable";
    auto initializer = builder.getStringAttr(label.str() + '\0');
    builder.setInsertionPointToEnd(moduleOp->getBody());
    auto collision = LLVM::GlobalOp::create(
        builder, moduleOp->getLoc(),
        LLVM::LLVMArrayType::get(builder.getI8Type(), label.size() + 1),
        isConstant, LLVM::Linkage::Internal,
        ("qir.result_label_" + label).str(), initializer);
    collision.setAddrSpace(isConstant ? 1U : 0U);
    auto resolved = createResultLabel(builder, *moduleOp, label);
    auto labelGlobal =
        moduleOp->lookupSymbol<LLVM::GlobalOp>(resolved.getGlobalName());
    ASSERT_TRUE(labelGlobal);
    EXPECT_NE(labelGlobal.getSymName(), collision.getSymName());
    EXPECT_TRUE(labelGlobal.getConstant());
    EXPECT_EQ(labelGlobal.getAddrSpace(), 0U);
    EXPECT_EQ(labelGlobal.getValueAttr(), initializer);
  }
  EXPECT_TRUE(succeeded(verify(*moduleOp)));
}

TEST_F(QIRTest, AdaptiveBuilderDoesNotReleaseExplicitStaticResults) {
  QIRProgramBuilder builder(context.get());
  builder.initialize();
  builder.staticResult(0);
  auto module = builder.finalize();
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));
  EXPECT_FALSE(module->lookupSymbol<LLVM::LLVMFuncOp>(QIR_RESULT_RELEASE));
}

TEST_F(QIRTest, BuilderRejectsMixedResultAllocationModes) {
  EXPECT_DEATH(
      {
        QIRProgramBuilder builder(context.get());
        builder.initialize();
        builder.staticResult(0);
        builder.allocClassicalBitRegister(1);
      },
      "Cannot mix static and dynamic result allocation modes");
  EXPECT_DEATH(
      {
        QIRProgramBuilder builder(context.get());
        builder.initialize();
        builder.allocClassicalBitRegister(1);
        builder.staticResult(0);
      },
      "Cannot mix static and dynamic result allocation modes");
}

TEST_F(QIRTest, BuilderRejectsOutOfBoundsClassicalRegisterIndices) {
  EXPECT_DEATH(
      {
        QIRProgramBuilder builder(context.get());
        builder.initialize();
        auto q = builder.allocQubit();
        const auto c = builder.allocClassicalBitRegister(1);
        builder.measure(q, c, -1);
      },
      "Register index must be non-negative");

  EXPECT_DEATH(
      {
        QIRProgramBuilder builder(context.get());
        builder.initialize();
        auto q = builder.allocQubit();
        const auto c = builder.allocClassicalBitRegister(1);
        builder.measure(q, c, 1);
      },
      "Register index is out of bounds");
}

TEST_F(QIRTest, BuilderReturnsCompleteClassicalRegister) {
  ClassicalRegister returnedRegister;
  auto module = QIRProgramBuilder::build(
      context.get(),
      [&](QIRProgramBuilder& builder) {
        returnedRegister = builder.allocClassicalBitRegister(2, false);
        return builder.intConstant(0);
      },
      QIRProgramBuilder::Profile::Base);

  ASSERT_TRUE(module);
  EXPECT_FALSE(returnedRegister.record);
  EXPECT_EQ(returnedRegister.results.size(), 2U);
  EXPECT_FALSE(returnedRegister.array);
}

TEST_F(QIRTest, ReusedDeclarationsPreserveAttributesIdempotently) {
  OpBuilder builder(context.get());
  const auto location = builder.getUnknownLoc();
  auto moduleOp = ModuleOp::create(location);
  builder.setInsertionPointToStart(moduleOp.getBody());
  const auto ptrType = LLVM::LLVMPointerType::get(context.get());
  const Type voidType = LLVM::LLVMVoidType::get(context.get());
  const auto nounwind = builder.getStringAttr("nounwind");
  const auto targetCPU = builder.getStrArrayAttr({"target-cpu", "generic"});

  for (const StringRef name : {
           StringRef(QIR_MEASURE),
           StringRef(QIR_RESET),
           StringRef(QIR_READ_RESULT),
       }) {
    SCOPED_TRACE(name.str());
    const SmallVector<Type> parameters(name == QIR_MEASURE ? 2 : 1, ptrType);
    const auto returnType =
        name == QIR_READ_RESULT ? builder.getI1Type() : voidType;
    const auto functionType =
        LLVM::LLVMFunctionType::get(returnType, parameters);
    auto declaration =
        LLVM::LLVMFuncOp::create(builder, location, name, functionType);
    declaration->setAttr("passthrough",
                         builder.getArrayAttr({nounwind, targetCPU}));

    EXPECT_EQ(
        getOrCreateFunctionDeclaration(builder, moduleOp, name, functionType),
        declaration);
    EXPECT_EQ(
        getOrCreateFunctionDeclaration(builder, moduleOp, name, functionType),
        declaration);

    const auto passthrough =
        declaration->getAttrOfType<ArrayAttr>("passthrough");
    ASSERT_TRUE(passthrough);
    ASSERT_EQ(passthrough.size(), name == QIR_READ_RESULT ? 2U : 3U);
    EXPECT_EQ(passthrough[0], nounwind);
    EXPECT_EQ(passthrough[1], targetCPU);
    EXPECT_EQ(llvm::count(passthrough, builder.getStringAttr("irreversible")),
              name == QIR_READ_RESULT ? 0 : 1);
    if (name == QIR_MEASURE) {
      EXPECT_TRUE(
          declaration.getArgAttr(1, LLVM::LLVMDialect::getWriteOnlyAttrName()));
    } else if (name == QIR_READ_RESULT) {
      EXPECT_TRUE(
          declaration.getArgAttr(0, LLVM::LLVMDialect::getReadonlyAttrName()));
    }
  }
}

TEST_F(QIRTest, MetadataPassRequiresExactlyOneEntryPointAtomically) {
  for (const size_t numEntryPoints : {0U, 2U}) {
    SCOPED_TRACE(testing::Message() << "numEntryPoints=" << numEntryPoints);
    OpBuilder builder(context.get());
    const auto location = builder.getUnknownLoc();
    auto module = ModuleOp::create(location);
    builder.setInsertionPointToStart(module.getBody());
    const auto functionType =
        LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(context.get()), {});
    for (size_t i = 0; i < std::max<size_t>(numEntryPoints, 1); ++i) {
      auto function = LLVM::LLVMFuncOp::create(
          builder, location, "function" + std::to_string(i), functionType);
      if (i < numEntryPoints) {
        function->setAttr("passthrough",
                          builder.getStrArrayAttr({"entry_point"}));
      }
      auto* block = function.addEntryBlock(builder);
      builder.setInsertionPointToEnd(block);
      LLVM::ReturnOp::create(builder, location, ValueRange{});
      builder.setInsertionPointToEnd(module.getBody());
    }
    ASSERT_TRUE(succeeded(verify(module)));

    OwningOpRef<ModuleOp> before = cast<ModuleOp>(module->clone());
    EXPECT_TRUE(failed(attachQIRMetadata(module)));
    EXPECT_TRUE(OperationEquivalence::isEquivalentTo(
        module, before->getOperation(), OperationEquivalence::Flags::None));
  }
}

TEST_F(QIRTest, PreservesUnrelatedMetadataIdempotently) {
  OpBuilder builder(context.get());
  const auto location = builder.getUnknownLoc();
  auto module = ModuleOp::create(location);
  builder.setInsertionPointToStart(module.getBody());
  const auto unrelated =
      LLVM::ModuleFlagAttr::get(context.get(), LLVM::ModFlagBehavior::Warning,
                                builder.getStringAttr("Debug Info Version"),
                                builder.getI32IntegerAttr(3));
  const auto staleQIR = LLVM::ModuleFlagAttr::get(
      context.get(), LLVM::ModFlagBehavior::Error,
      builder.getStringAttr("qir_major_version"), builder.getI32IntegerAttr(1));
  LLVM::ModuleFlagsOp::create(builder, location,
                              builder.getArrayAttr({unrelated, staleQIR}));
  const auto functionType =
      LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(context.get()), {});
  auto main = LLVM::LLVMFuncOp::create(builder, location, "main", functionType);
  const auto nounwind = builder.getStringAttr("nounwind");
  const auto target = builder.getStrArrayAttr({"target-cpu", "generic"});
  const auto staleProfile =
      builder.getStrArrayAttr({"qir_profiles", "stale_profile"});
  main.setPassthroughAttr(builder.getArrayAttr(
      {builder.getStringAttr("entry_point"), nounwind, target, staleProfile}));
  auto* block = main.addEntryBlock(builder);
  builder.setInsertionPointToEnd(block);
  LLVM::ReturnOp::create(builder, location, ValueRange{});
  ASSERT_TRUE(succeeded(verify(module)));

  ASSERT_TRUE(succeeded(attachQIRMetadata(module)));
  const auto preserved = findModuleFlag(module, "Debug Info Version");
  ASSERT_TRUE(preserved);
  EXPECT_EQ(preserved.getBehavior(), LLVM::ModFlagBehavior::Warning);
  EXPECT_EQ(cast<IntegerAttr>(preserved.getValue()).getInt(), 3);
  const auto qirMajor = findModuleFlag(module, "qir_major_version");
  ASSERT_TRUE(qirMajor);
  EXPECT_EQ(cast<IntegerAttr>(qirMajor.getValue()).getInt(), 2);

  const auto passthrough = main.getPassthroughAttr();
  ASSERT_TRUE(passthrough);
  EXPECT_TRUE(llvm::is_contained(passthrough, nounwind));
  EXPECT_TRUE(llvm::is_contained(passthrough, target));
  const auto baseProfile =
      builder.getStrArrayAttr({"qir_profiles", "base_profile"});
  EXPECT_EQ(llvm::count(passthrough, baseProfile), 1U);
  EXPECT_FALSE(llvm::is_contained(passthrough, staleProfile));

  OwningOpRef<ModuleOp> afterFirstRun = cast<ModuleOp>(module->clone());
  ASSERT_TRUE(succeeded(attachQIRMetadata(module)));
  EXPECT_TRUE(OperationEquivalence::isEquivalentTo(
      module, afterFirstRun->getOperation(),
      OperationEquivalence::Flags::None));
}

TEST_F(QIRTest, MetadataDeclaresResourceCapacities) {
  struct CapacityCase {
    SmallVector<int64_t> indices;
    StringRef requiredCapacity;
  };
  const std::array cases{
      CapacityCase{.indices = {}, .requiredCapacity = "0"},
      CapacityCase{.indices = {0}, .requiredCapacity = "1"},
      CapacityCase{.indices = {0, 1, 0}, .requiredCapacity = "2"},
      CapacityCase{.indices = {7, 2, 7}, .requiredCapacity = "8"},
  };

  for (const auto profile : {
           QIRProgramBuilder::Profile::Base,
           QIRProgramBuilder::Profile::Adaptive,
       }) {
    for (const auto& testCase : cases) {
      SCOPED_TRACE(testing::Message()
                   << "profile=" << static_cast<int>(profile)
                   << ", requiredCapacity=" << testCase.requiredCapacity.str());
      auto moduleOp = QIRProgramBuilder::build(
          context.get(),
          [&](QIRProgramBuilder& builder) {
            for (const auto index : testCase.indices) {
              auto qubit = builder.staticQubit(index);
              builder.x(qubit);
              builder.measure(qubit, index);
            }
            return builder.intConstant(0);
          },
          profile);

      ASSERT_TRUE(moduleOp);
      ASSERT_TRUE(succeeded(verify(*moduleOp)));
      auto main = getMainFunction(moduleOp.get());
      ASSERT_TRUE(main);
      const auto passthrough = main.getPassthroughAttr();
      ASSERT_TRUE(passthrough);
      OpBuilder builder(context.get());
      for (const StringRef attribute :
           {"required_num_qubits", "required_num_results"}) {
        EXPECT_TRUE(llvm::is_contained(
            passthrough,
            builder.getStrArrayAttr(
                {attribute,
                 attribute == "required_num_results" &&
                         profile == QIRProgramBuilder::Profile::Adaptive
                     ? StringRef("0")
                     : testCase.requiredCapacity})));
      }
    }
  }
}

TEST_F(QIRTest, MetadataCountsResourceRolesAcrossCalls) {
  struct Case {
    StringRef instruction;
    StringRef qubits;
    StringRef results;
  };
  for (const auto& testCase : {
           Case{
               .instruction =
                   "llvm.call @__quantum__qis__mz__body(%zero, %r9) : "
                   "(!llvm.ptr, !llvm.ptr) -> ()",
               .qubits = "1",
               .results = "10",
           },
           Case{
               .instruction =
                   "llvm.call @__quantum__qis__cnot__body(%q7, %zero) : "
                   "(!llvm.ptr, !llvm.ptr) -> ()",
               .qubits = "8",
               .results = "0",
           },
           Case{
               .instruction =
                   "%bit = llvm.call @__quantum__rt__read_result(%r9) : "
                   "(!llvm.ptr) -> i1",
               .qubits = "0",
               .results = "10",
           },
           Case{
               .instruction =
                   "llvm.call @__quantum__rt__result_record_output(%r9, %zero) "
                   ": "
                   "(!llvm.ptr, !llvm.ptr) -> ()",
               .qubits = "0",
               .results = "10",
           },
           Case{
               .instruction =
                   "llvm.call @__quantum__qis__mz__body(%zero, %zero) : "
                   "(!llvm.ptr, !llvm.ptr) -> ()",
               .qubits = "1",
               .results = "1",
           },
           Case{
               .instruction = "llvm.call @__quantum__qis__mz__body(%q, %r) : "
                              "(!llvm.ptr, !llvm.ptr) -> ()",
               .qubits = "0",
               .results = "0",
           },
           Case{
               .instruction = "llvm.call @helper(%zero, %r9) : "
                              "(!llvm.ptr, !llvm.ptr) -> ()\n"
                              "llvm.call @helper(%q7, %zero) : "
                              "(!llvm.ptr, !llvm.ptr) -> ()",
               .qubits = "8",
               .results = "10",
           },
           Case{
               .instruction =
                   "%returned = llvm.call @supply() : () -> !llvm.ptr\n"
                   "%again = llvm.call @supply() : () -> !llvm.ptr\n"
                   "llvm.call @__quantum__qis__cnot__body(%returned, %again) : "
                   "(!llvm.ptr, !llvm.ptr) -> ()",
               .qubits = "8",
               .results = "0",
           },
           Case{
               .instruction =
                   "%choice = llvm.mlir.constant(0 : i32) : i32\n"
                   "llvm.switch %choice : i32, ^merge(%zero : !llvm.ptr) "
                   "[0: ^merge(%q7 : !llvm.ptr), 1: ^merge(%r9 : !llvm.ptr)]\n"
                   "^merge(%selected: !llvm.ptr):\n"
                   "llvm.call @__quantum__qis__cnot__body(%selected, "
                   "%selected) : "
                   "(!llvm.ptr, !llvm.ptr) -> ()",
               .qubits = "10",
               .results = "0",
           },
       }) {
    SCOPED_TRACE(testCase.instruction.str());
    const std::string ir = R"mlir(module {
      llvm.func @__quantum__qis__mz__body(!llvm.ptr, !llvm.ptr)
      llvm.func @__quantum__qis__cnot__body(!llvm.ptr, !llvm.ptr)
      llvm.func @__quantum__rt__read_result(!llvm.ptr) -> i1
      llvm.func @__quantum__rt__result_record_output(!llvm.ptr, !llvm.ptr)
      llvm.func @helper(%q: !llvm.ptr, %r: !llvm.ptr) {
        llvm.call @__quantum__qis__mz__body(%q, %r) : (!llvm.ptr, !llvm.ptr) -> ()
        llvm.return
      }
      llvm.func @supply() -> !llvm.ptr {
        %seven = llvm.mlir.constant(7 : i64) : i64
        %qubit = llvm.inttoptr %seven : i64 to !llvm.ptr
        llvm.return %qubit : !llvm.ptr
      }
      llvm.func @main(%q: !llvm.ptr, %r: !llvm.ptr)
          attributes {passthrough = ["entry_point"]} {
        %seven = llvm.mlir.constant(7 : i64) : i64
        %nine = llvm.mlir.constant(9 : i64) : i64
        %zero = llvm.mlir.zero : !llvm.ptr
        %q7 = llvm.inttoptr %seven : i64 to !llvm.ptr
        %r9 = llvm.inttoptr %nine : i64 to !llvm.ptr
    )mlir" + testCase.instruction.str() +
                           R"mlir(
        llvm.return
      }
    })mlir";
    auto moduleOp = parseSourceString<ModuleOp>(ir, context.get());
    ASSERT_TRUE(moduleOp);
    ASSERT_TRUE(succeeded(verify(*moduleOp)));
    ASSERT_TRUE(succeeded(attachQIRMetadata(*moduleOp)));
    EXPECT_TRUE(succeeded(verify(*moduleOp)));
    OpBuilder builder(context.get());
    const auto attributes = getMainFunction(*moduleOp).getPassthroughAttr();
    EXPECT_TRUE(llvm::is_contained(
        attributes,
        builder.getStrArrayAttr({"required_num_qubits", testCase.qubits})));
    EXPECT_TRUE(llvm::is_contained(
        attributes,
        builder.getStrArrayAttr({"required_num_results", testCase.results})));
  }
}

TEST_F(QIRTest, MetadataClassifiesLoopDependenciesAcrossFunctions) {
  for (const auto& [condition, expected] : {
           std::pair{
               StringRef("%condition = llvm.icmp \"slt\" %next, %ten : i64"),
               1},
           std::pair{
               StringRef(
                   "%condition = llvm.call @__quantum__rt__read_result(%zero) "
                   ": (!llvm.ptr) -> i1"),
               2},
           std::pair{
               StringRef("%bit = llvm.call @__quantum__rt__read_result(%zero) "
                         ": (!llvm.ptr) -> i1\n"
                         "%wide = llvm.zext %bit : i1 to i64\n"
                         "%condition = llvm.icmp \"eq\" %wide, %ten : i64"),
               2},
           std::pair{StringRef("%condition = llvm.call @unknown() : () -> i1"),
                     3},
           std::pair{
               StringRef("%bit = llvm.call @__quantum__rt__read_result(%zero) "
                         ": (!llvm.ptr) -> i1\n"
                         "%selector = llvm.zext %bit : i1 to i32\n"
                         "llvm.switch %selector : i32, ^count [1: ^exit]\n"
                         "^count:\n"
                         "%condition = llvm.icmp \"slt\" %next, %ten : i64"),
               2},
           std::pair{
               StringRef("%bit = llvm.call @__quantum__rt__read_result(%zero) "
                         ": (!llvm.ptr) -> i1\n"
                         "llvm.cond_br %bit, ^header(%next : i64), ^check\n"
                         "^check:\n"
                         "%condition = llvm.icmp \"slt\" %next, %ten : i64"),
               2},
           std::pair{
               StringRef(
                   "llvm.br ^inner(%start : i64)\n"
                   "^inner(%j: i64):\n"
                   "%inner_next = llvm.add %j, %one : i64\n"
                   "%again = llvm.call @__quantum__rt__read_result(%zero) "
                   ": (!llvm.ptr) -> i1\n"
                   "llvm.cond_br %again, ^inner(%inner_next : i64), "
                   "^after_inner\n"
                   "^after_inner:\n"
                   "%condition = llvm.icmp \"slt\" %next, %ten : i64"),
               3},
           std::pair{StringRef(R"mlir(
                 %choose = llvm.mlir.constant(true) : i1
                 llvm.cond_br %choose, ^left(%next : i64), ^right(%next : i64)
               ^left(%local: i64):
                 %incremented = llvm.add %local, %one : i64
                 llvm.br ^right(%incremented : i64)
               ^right(%counter: i64):
                 %repeat = llvm.icmp "slt" %counter, %ten : i64
                 llvm.cond_br %repeat, ^left(%counter : i64), ^after_cycle
               ^after_cycle:
                 %condition = llvm.icmp "slt" %next, %ten : i64
               )mlir"),
                     3},
       }) {
    SCOPED_TRACE(condition.str());
    const std::string ir = R"mlir(module {
      llvm.func @__quantum__rt__read_result(!llvm.ptr) -> i1
      llvm.func @unknown() -> i1
      llvm.func @loop(%ten: i64) {
        %zero = llvm.mlir.zero : !llvm.ptr
        %start = llvm.mlir.constant(0 : i64) : i64
        %one = llvm.mlir.constant(1 : i64) : i64
        llvm.br ^header(%start : i64)
      ^header(%i: i64):
        llvm.br ^latch
      ^latch:
        %next = llvm.add %i, %one : i64
    )mlir" + condition.str() +
                           R"mlir(
        llvm.cond_br %condition, ^header(%next : i64), ^exit
      ^exit:
        llvm.return
      }
      llvm.func @main() attributes {passthrough = ["entry_point"]} {
        %ten = llvm.mlir.constant(10 : i64) : i64
        llvm.call @loop(%ten) : (i64) -> ()
        llvm.return
      }
    })mlir";
    auto moduleOp = parseSourceString<ModuleOp>(ir, context.get());
    ASSERT_TRUE(moduleOp);
    ASSERT_TRUE(succeeded(verify(*moduleOp)));
    ASSERT_TRUE(succeeded(attachQIRMetadata(*moduleOp, true)));
    EXPECT_TRUE(succeeded(verify(*moduleOp)));
    const auto flag = findModuleFlag(*moduleOp, "backwards_branching");
    ASSERT_TRUE(flag);
    EXPECT_EQ(cast<IntegerAttr>(flag.getValue()).getInt(), expected);
  }
}

TEST_F(QIRTest, MetadataRejectsUnrepresentableStaticResourceCapacity) {
  auto moduleOp = parseSourceString<ModuleOp>(R"mlir(module {
    llvm.func @__quantum__qis__x__body(!llvm.ptr)
    llvm.func @main() attributes {passthrough = ["entry_point"]} {
      %index = llvm.mlir.constant(-1 : i64) : i64
      %qubit = llvm.inttoptr %index : i64 to !llvm.ptr
      llvm.call @__quantum__qis__x__body(%qubit) : (!llvm.ptr) -> ()
      llvm.return
    }
  })mlir",
                                              context.get());
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(verify(*moduleOp)));
  OwningOpRef<ModuleOp> before = moduleOp->clone();

  EXPECT_TRUE(failed(attachQIRMetadata(moduleOp.get())));
  EXPECT_TRUE(OperationEquivalence::isEquivalentTo(
      moduleOp.get(), before->getOperation(),
      OperationEquivalence::Flags::None));
}

TEST_F(QIRTest, MetadataIncludesStaticQubitsLoadedFromRegisters) {
  auto moduleOp = parseSourceString<ModuleOp>(R"mlir(module {
    llvm.func @__quantum__qis__x__body(!llvm.ptr)
    llvm.func @main(%index: i64) attributes {passthrough = ["entry_point"]} {
      %two = llvm.mlir.constant(2 : i64) : i64
      %seven = llvm.mlir.constant(7 : i64) : i64
      %other_id = llvm.mlir.constant(99 : i64) : i64
      %zero = llvm.mlir.zero : !llvm.ptr
      %qubit = llvm.inttoptr %seven : i64 to !llvm.ptr
      %other = llvm.inttoptr %other_id : i64 to !llvm.ptr
      %qubits = llvm.alloca %two x !llvm.ptr : (i64) -> !llvm.ptr
      %unrelated = llvm.alloca %two x !llvm.ptr : (i64) -> !llvm.ptr
      llvm.store %zero, %qubits {qir.qubit_store} : !llvm.ptr, !llvm.ptr
      %second = llvm.getelementptr %qubits[1] : (!llvm.ptr) -> !llvm.ptr, !llvm.ptr
      llvm.store %qubit, %second {qir.qubit_store} : !llvm.ptr, !llvm.ptr
      llvm.store %other, %unrelated : !llvm.ptr, !llvm.ptr
      %slot = llvm.getelementptr %qubits[%index] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.ptr
      %selected = llvm.load %slot : !llvm.ptr -> !llvm.ptr
      llvm.call @__quantum__qis__x__body(%selected) : (!llvm.ptr) -> ()
      llvm.return
    }
  })mlir",
                                              context.get());
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(verify(*moduleOp)));
  ASSERT_TRUE(succeeded(attachQIRMetadata(moduleOp.get(), true)));
  auto main = getMainFunction(moduleOp.get());
  OpBuilder builder(context.get());
  EXPECT_TRUE(llvm::is_contained(
      main.getPassthroughAttr(),
      builder.getStrArrayAttr({"required_num_qubits", "8"})));
  const auto arrays = findModuleFlag(*moduleOp, "arrays");
  ASSERT_TRUE(arrays);
  EXPECT_EQ(cast<IntegerAttr>(arrays.getValue()).getInt(), 1);
  const auto dynamic = findModuleFlag(*moduleOp, "dynamic_qubit_management");
  ASSERT_TRUE(dynamic);
  EXPECT_EQ(cast<IntegerAttr>(dynamic.getValue()).getInt(), 0);
}

TEST_F(QIRTest, AdaptiveBuilderSelectsControlledSpecializationsByArity) {
  auto module = QIRProgramBuilder::build(
      context.get(),
      [](QIRProgramBuilder& builder) {
        auto control0 = builder.allocQubit();
        auto control1 = builder.allocQubit();
        auto control2 = builder.allocQubit();
        auto target = builder.allocQubit();
        builder.crx(0.25, control0, target);
        builder.mcrx(0.5, {control0, control1}, target);
        builder.mcrx(0.75, {control0, control1, control2}, target);
        builder.r(0.5, 0.75, target);
        return builder.intConstant(0);
      },
      QIRProgramBuilder::Profile::Adaptive);

  ASSERT_TRUE(module);
  EXPECT_TRUE(module->lookupSymbol<LLVM::LLVMFuncOp>(QIR_CRX));
  EXPECT_TRUE(module->lookupSymbol<LLVM::LLVMFuncOp>(QIR_CCRX));
  auto controlledRx = module->lookupSymbol<LLVM::LLVMFuncOp>(QIR_RX_CTL);
  ASSERT_TRUE(controlledRx);
  const auto argumentTypes = controlledRx.getFunctionType().getParams();
  ASSERT_EQ(argumentTypes.size(), 2U);
  EXPECT_EQ(argumentTypes[0], LLVM::LLVMPointerType::get(context.get()));
  EXPECT_EQ(argumentTypes[1], LLVM::LLVMPointerType::get(context.get()));
  EXPECT_TRUE(module->lookupSymbol<LLVM::LLVMFuncOp>(QIR_ARRAY_CREATE));
  EXPECT_TRUE(module->lookupSymbol<LLVM::LLVMFuncOp>(QIR_TUPLE_CREATE));
  auto main = getMainFunction(module.get());
  ASSERT_TRUE(main);
  const auto passthrough = main->getAttrOfType<ArrayAttr>("passthrough");
  ASSERT_TRUE(passthrough);
  EXPECT_TRUE(llvm::is_contained(
      passthrough,
      ArrayAttr::get(context.get(),
                     {StringAttr::get(context.get(), "qir_profiles"),
                      StringAttr::get(context.get(), "adaptive_profile")})));

  EXPECT_STREQ(QIR_R, "__quantum__qis__prx__body");
  EXPECT_TRUE(module->lookupSymbol<LLVM::LLVMFuncOp>(QIR_R));
  EXPECT_FALSE(
      module->lookupSymbol<LLVM::LLVMFuncOp>("__quantum__qis__r__body"));
}

TEST_F(QIRTest, BaseBuilderUsesGenericSpecializationForThreeControls) {
  auto module = QIRProgramBuilder::build(
      context.get(),
      [](QIRProgramBuilder& builder) {
        auto control0 = builder.staticQubit(0);
        auto control1 = builder.staticQubit(1);
        auto control2 = builder.staticQubit(2);
        auto target = builder.staticQubit(7);
        builder.mcrx(0.25, {control0, control1, control2}, target);
        return builder.intConstant(0);
      },
      QIRProgramBuilder::Profile::Base);

  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));
  EXPECT_TRUE(module->lookupSymbol<LLVM::LLVMFuncOp>(QIR_RX_CTL));
  auto main = getMainFunction(module.get());
  ASSERT_TRUE(main);
  const auto passthrough = main->getAttrOfType<ArrayAttr>("passthrough");
  ASSERT_TRUE(passthrough);
  EXPECT_TRUE(llvm::is_contained(
      passthrough,
      ArrayAttr::get(context.get(),
                     {StringAttr::get(context.get(), "qir_profiles"),
                      StringAttr::get(context.get(), "base_profile")})));
  EXPECT_TRUE(llvm::is_contained(
      passthrough,
      ArrayAttr::get(context.get(),
                     {StringAttr::get(context.get(), "required_num_qubits"),
                      StringAttr::get(context.get(), "8")})));
}

TEST_F(QIRTest, UsesTranslationCompatibleModuleFlagWidths) {
  const auto build = [&](const QIRProgramBuilder::Profile profile) {
    return QIRProgramBuilder::build(
        context.get(),
        [](QIRProgramBuilder& builder) { return builder.intConstant(0); },
        profile);
  };
  auto base = build(QIRProgramBuilder::Profile::Base);
  ASSERT_TRUE(base);
  const auto baseDynamicQubits =
      findModuleFlag(base.get(), "dynamic_qubit_management");
  ASSERT_TRUE(baseDynamicQubits);
  const auto baseDynamicQubitsValue =
      dyn_cast<IntegerAttr>(baseDynamicQubits.getValue());
  ASSERT_TRUE(baseDynamicQubitsValue);
  EXPECT_EQ(baseDynamicQubitsValue.getType(),
            IntegerType::get(context.get(), 32));
  EXPECT_FALSE(findModuleFlag(base.get(), "backwards_branching"));

  auto adaptive = build(QIRProgramBuilder::Profile::Adaptive);
  ASSERT_TRUE(adaptive);
  const auto backwardsBranching =
      findModuleFlag(adaptive.get(), "backwards_branching");
  ASSERT_TRUE(backwardsBranching);
  const auto backwardsBranchingValue =
      dyn_cast<IntegerAttr>(backwardsBranching.getValue());
  ASSERT_TRUE(backwardsBranchingValue);
  EXPECT_EQ(backwardsBranchingValue.getType(),
            IntegerType::get(context.get(), 32));
  const auto arrays = findModuleFlag(adaptive.get(), "arrays");
  ASSERT_TRUE(arrays);
  const auto arraysValue = dyn_cast<IntegerAttr>(arrays.getValue());
  ASSERT_TRUE(arraysValue);
  EXPECT_EQ(arraysValue.getType(), IntegerType::get(context.get(), 32));
}

TEST_F(QIRTest, ClassicalStorageRequiresArrayAndElementCapabilities) {
  auto moduleOp = parseSourceString<ModuleOp>(R"mlir(module {
    llvm.func @__quantum__qis__mz__body(!llvm.ptr, !llvm.ptr)
    llvm.func @__quantum__rt__read_result(!llvm.ptr) -> i1
    llvm.func @main() -> i64 attributes {passthrough = ["entry_point"]} {
      %two = llvm.mlir.constant(2 : i64) : i64
      %memory = llvm.alloca %two x f64 : (i64) -> !llvm.ptr
      %resource = llvm.mlir.zero : !llvm.ptr
      llvm.call @__quantum__qis__mz__body(%resource, %resource) : (!llvm.ptr, !llvm.ptr) -> ()
      %bit = llvm.call @__quantum__rt__read_result(%resource) : (!llvm.ptr) -> i1
      %index = llvm.zext %bit : i1 to i64
      %element = llvm.getelementptr %memory[%index] : (!llvm.ptr, i64) -> !llvm.ptr, f64
      %value = llvm.mlir.constant(1.0 : f64) : f64
      llvm.store %value, %element : f64, !llvm.ptr
      %zero = llvm.mlir.constant(0 : i64) : i64
      llvm.return %zero : i64
    }
  })mlir",
                                              context.get());
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(attachQIRMetadata(*moduleOp, true)));
  const auto arrays = findModuleFlag(*moduleOp, "arrays");
  ASSERT_TRUE(arrays);
  EXPECT_EQ(cast<IntegerAttr>(arrays.getValue()).getInt(), 1);
  const auto floating = findModuleFlag(*moduleOp, "float_computations");
  ASSERT_TRUE(floating);
  EXPECT_EQ(floating.getValue(),
            Builder(context.get()).getStrArrayAttr({"double"}));
  const auto loops = findModuleFlag(*moduleOp, "backwards_branching");
  ASSERT_TRUE(loops);
  EXPECT_EQ(cast<IntegerAttr>(loops.getValue()).getInt(), 0);
}

TEST_F(QIRTest, OutputLabelPointerDoesNotRequireArrays) {
  auto moduleOp = parseSourceString<ModuleOp>(R"mlir(module {
    llvm.mlir.global internal constant @labels("xresult\00")
    llvm.func @__quantum__rt__result_record_output(!llvm.ptr, !llvm.ptr)
    llvm.func @main() -> i64 attributes {passthrough = ["entry_point"]} {
      %labels = llvm.mlir.addressof @labels : !llvm.ptr
      %label = llvm.getelementptr inbounds %labels[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<8 x i8>
      %result = llvm.mlir.zero : !llvm.ptr
      llvm.call @__quantum__rt__result_record_output(%result, %label) : (!llvm.ptr, !llvm.ptr) -> ()
      %zero = llvm.mlir.constant(0 : i64) : i64
      llvm.return %zero : i64
    }
  })mlir",
                                              context.get());
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(attachQIRMetadata(*moduleOp, true)));
  const auto arrays = findModuleFlag(*moduleOp, "arrays");
  ASSERT_TRUE(arrays);
  EXPECT_EQ(cast<IntegerAttr>(arrays.getValue()).getInt(), 0);
}

TEST_F(QIRTest, DerivesAdaptiveClassicalCapabilities) {
  OpBuilder builder(context.get());
  const auto location = builder.getUnknownLoc();
  auto moduleOp = ModuleOp::create(location);
  builder.setInsertionPointToStart(moduleOp.getBody());

  const auto mainType = LLVM::LLVMFunctionType::get(builder.getI64Type(), {});
  auto main = LLVM::LLVMFuncOp::create(builder, location, "main", mainType);
  main->setAttr("passthrough", builder.getStrArrayAttr({"entry_point"}));
  auto* mainBlock = main.addEntryBlock(builder);
  builder.setInsertionPointToEnd(mainBlock);
  Value exitCode =
      LLVM::ConstantOp::create(builder, location, builder.getI64IntegerAttr(0))
          .getResult();
  LLVM::ReturnOp::create(builder, location, exitCode);

  builder.setInsertionPointToEnd(moduleOp.getBody());
  const auto helperType = LLVM::LLVMFunctionType::get(
      builder.getF64Type(), {builder.getI8Type(), builder.getF32Type()});
  auto helper =
      LLVM::LLVMFuncOp::create(builder, location, "helper", helperType);
  auto* helperEntry = helper.addEntryBlock(builder);
  auto* defaultBlock = helper.addBlock();
  auto* caseBlock = helper.addBlock();
  builder.setInsertionPointToEnd(helperEntry);
  Value left =
      LLVM::ConstantOp::create(builder, location, builder.getI32IntegerAttr(1))
          .getResult();
  Value right =
      LLVM::ConstantOp::create(builder, location, builder.getI32IntegerAttr(2))
          .getResult();
  std::ignore = LLVM::ICmpOp::create(builder, location,
                                     LLVM::ICmpPredicate::slt, left, right);
  Value floating =
      LLVM::FPExtOp::create(builder, location, builder.getF64Type(),
                            helperEntry->getArgument(1))
          .getResult();
  Value doubled =
      LLVM::FAddOp::create(builder, location, floating, floating).getResult();
  const std::array<APInt, 1> caseValues{APInt(8, 0)};
  const std::array<Block*, 1> caseDestinations{caseBlock};
  const std::array<ValueRange, 1> caseOperands{ValueRange{}};
  LLVM::SwitchOp::create(builder, location, helperEntry->getArgument(0),
                         defaultBlock, ValueRange{}, caseValues,
                         BlockRange(caseDestinations), caseOperands);
  builder.setInsertionPointToEnd(defaultBlock);
  LLVM::ReturnOp::create(builder, location, doubled);
  builder.setInsertionPointToEnd(caseBlock);
  LLVM::ReturnOp::create(builder, location, doubled);

  ASSERT_TRUE(attachQIRMetadata(moduleOp, true).succeeded());
  const auto integerFlag = findModuleFlag(moduleOp, "int_computations");
  ASSERT_TRUE(integerFlag);
  EXPECT_EQ(integerFlag.getBehavior(), LLVM::ModFlagBehavior::Append);
  const auto integerTypes = cast<ArrayAttr>(integerFlag.getValue());
  ASSERT_TRUE(integerTypes);
  ASSERT_EQ(integerTypes.size(), 2U);
  EXPECT_EQ(cast<StringAttr>(integerTypes[0]).getValue(), "i32");
  EXPECT_EQ(cast<StringAttr>(integerTypes[1]).getValue(), "i8");
  const auto floatingFlag = findModuleFlag(moduleOp, "float_computations");
  ASSERT_TRUE(floatingFlag);
  EXPECT_EQ(floatingFlag.getBehavior(), LLVM::ModFlagBehavior::Append);
  const auto floatingTypes = cast<ArrayAttr>(floatingFlag.getValue());
  ASSERT_TRUE(floatingTypes);
  ASSERT_EQ(floatingTypes.size(), 2U);
  EXPECT_EQ(cast<StringAttr>(floatingTypes[0]).getValue(), "double");
  EXPECT_EQ(cast<StringAttr>(floatingTypes[1]).getValue(), "float");

  for (const auto* const flag : {
           "ir_functions",
           "multiple_target_branching",
           "multiple_return_points",
       }) {
    const auto moduleFlag = findModuleFlag(moduleOp, flag);
    ASSERT_TRUE(moduleFlag) << flag;
    EXPECT_EQ(moduleFlag.getValue(), builder.getI32IntegerAttr(1)) << flag;
  }

  ASSERT_TRUE(attachQIRMetadata(moduleOp).succeeded());
  EXPECT_FALSE(findModuleFlag(moduleOp, "int_computations"));
  EXPECT_FALSE(findModuleFlag(moduleOp, "float_computations"));
  EXPECT_FALSE(findModuleFlag(moduleOp, "ir_functions"));
  EXPECT_FALSE(findModuleFlag(moduleOp, "multiple_target_branching"));
  EXPECT_FALSE(findModuleFlag(moduleOp, "multiple_return_points"));
}

TEST(QIRModuleFlagsTest, RecordsAdaptiveClassicalCapabilities) {
  MLIRContext mlirContext;
  mlirContext.loadDialect<LLVM::LLVMDialect>();
  registerBuiltinDialectTranslation(mlirContext);
  registerLLVMDialectTranslation(mlirContext);
  auto sourceModule = parseSourceString<ModuleOp>(R"mlir(
    module {
      llvm.module_flags [
        #llvm.mlir.module_flag<append, "int_computations", ["i8"]>,
        #llvm.mlir.module_flag<append, "float_computations", ["double"]>,
        #llvm.mlir.module_flag<error, "ir_functions", 1 : i32>,
        #llvm.mlir.module_flag<error, "multiple_target_branching", 1 : i32>,
        #llvm.mlir.module_flag<error, "multiple_return_points", 1 : i32>
      ]
    }
  )mlir",
                                                  &mlirContext);
  ASSERT_TRUE(sourceModule);
  llvm::LLVMContext llvmContext;
  auto translated = translateModuleToLLVMIR(*sourceModule, llvmContext);
  ASSERT_NE(translated, nullptr);
  auto& moduleOp = *translated;
  normalizeQIRModuleFlags(moduleOp);

  const auto* integerTypes =
      llvm::dyn_cast<llvm::MDNode>(moduleOp.getModuleFlag("int_computations"));
  ASSERT_NE(integerTypes, nullptr);
  ASSERT_EQ(integerTypes->getNumOperands(), 1U);
  EXPECT_EQ(
      llvm::cast<llvm::MDString>(integerTypes->getOperand(0))->getString(),
      "i8");
  const auto* floatingTypes = llvm::dyn_cast<llvm::MDNode>(
      moduleOp.getModuleFlag("float_computations"));
  ASSERT_NE(floatingTypes, nullptr);
  ASSERT_EQ(floatingTypes->getNumOperands(), 1U);
  EXPECT_EQ(
      llvm::cast<llvm::MDString>(floatingTypes->getOperand(0))->getString(),
      "double");
  for (const auto* const flag : {
           "ir_functions",
           "multiple_target_branching",
           "multiple_return_points",
       }) {
    const auto* metadata =
        llvm::dyn_cast<llvm::ConstantAsMetadata>(moduleOp.getModuleFlag(flag));
    ASSERT_NE(metadata, nullptr) << flag;
    const auto* value = llvm::dyn_cast<llvm::ConstantInt>(metadata->getValue());
    ASSERT_NE(value, nullptr) << flag;
    EXPECT_EQ(value->getBitWidth(), 1U) << flag;
    EXPECT_TRUE(value->isOne()) << flag;
  }
}

/// \name QIR/Operations/StandardGates/DcxOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QIRDCXOpTest, QIRTest,
    testing::Values(QIRTestCase{"DCX", MQT_NAMED_BUILDER(dcx<>),
                                MQT_NAMED_BUILDER(dcx<>)},
                    QIRTestCase{"SingleControlledDCX",
                                MQT_NAMED_BUILDER(singleControlledDcx<>),
                                MQT_NAMED_BUILDER(singleControlledDcx<>)},
                    QIRTestCase{"MultipleControlledDCX",
                                MQT_NAMED_BUILDER(multipleControlledDcx<>),
                                MQT_NAMED_BUILDER(multipleControlledDcx<>)}));
/// @}

/// \name QIR/Operations/StandardGates/EcrOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QIRECROpTest, QIRTest,
    testing::Values(QIRTestCase{"ECR", MQT_NAMED_BUILDER(ecr<>),
                                MQT_NAMED_BUILDER(ecr<>)},
                    QIRTestCase{"SingleControlledECR",
                                MQT_NAMED_BUILDER(singleControlledEcr<>),
                                MQT_NAMED_BUILDER(singleControlledEcr<>)},
                    QIRTestCase{"MultipleControlledECR",
                                MQT_NAMED_BUILDER(multipleControlledEcr<>),
                                MQT_NAMED_BUILDER(multipleControlledEcr<>)}));
/// @}

/// \name QIR/Operations/StandardGates/GphaseOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(QIRGPhaseOpTest, QIRTest,
                         testing::Values(QIRTestCase{
                             "GlobalPhase", MQT_NAMED_BUILDER(globalPhase<>),
                             MQT_NAMED_BUILDER(globalPhase<>)}));
/// @}

/// \name QIR/Operations/StandardGates/HOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QIRHOpTest, QIRTest,
    testing::Values(
        QIRTestCase{"H", MQT_NAMED_BUILDER(h<>), MQT_NAMED_BUILDER(h<>)},
        QIRTestCase{"SingleControlledH", MQT_NAMED_BUILDER(singleControlledH<>),
                    MQT_NAMED_BUILDER(singleControlledH<>)},
        QIRTestCase{"MultipleControlledH",
                    MQT_NAMED_BUILDER(multipleControlledH<>),
                    MQT_NAMED_BUILDER(multipleControlledH<>)}));
/// @}

/// \name QIR/Operations/StandardGates/IdOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QIRIDOpTest, QIRTest,
    testing::Values(QIRTestCase{"Identity", MQT_NAMED_BUILDER(identity<>),
                                MQT_NAMED_BUILDER(identity<>)},
                    QIRTestCase{"SingleControlledIdentity",
                                MQT_NAMED_BUILDER(singleControlledIdentity<>),
                                MQT_NAMED_BUILDER(singleControlledIdentity<>)},
                    QIRTestCase{
                        "MultipleControlledIdentity",
                        MQT_NAMED_BUILDER(multipleControlledIdentity<>),
                        MQT_NAMED_BUILDER(multipleControlledIdentity<>)}));
/// @}

/// \name QIR/Operations/StandardGates/IswapOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QIRiSWAPOpTest, QIRTest,
    testing::Values(QIRTestCase{"iSWAP", MQT_NAMED_BUILDER(iswap<>),
                                MQT_NAMED_BUILDER(iswap<>)},
                    QIRTestCase{"SingleControllediSWAP",
                                MQT_NAMED_BUILDER(singleControlledIswap<>),
                                MQT_NAMED_BUILDER(singleControlledIswap<>)},
                    QIRTestCase{"MultipleControllediSWAP",
                                MQT_NAMED_BUILDER(multipleControlledIswap<>),
                                MQT_NAMED_BUILDER(multipleControlledIswap<>)}));
/// @}

/// \name QIR/Operations/StandardGates/POp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QIRPOpTest, QIRTest,
    testing::Values(
        QIRTestCase{"P", MQT_NAMED_BUILDER(p<>), MQT_NAMED_BUILDER(p<>)},
        QIRTestCase{"SingleControlledP", MQT_NAMED_BUILDER(singleControlledP<>),
                    MQT_NAMED_BUILDER(singleControlledP<>)},
        QIRTestCase{"MultipleControlledP",
                    MQT_NAMED_BUILDER(multipleControlledP<>),
                    MQT_NAMED_BUILDER(multipleControlledP<>)}));
/// @}

/// \name QIR/Operations/StandardGates/RCCXOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QIRRCCXOpTest, QIRTest,
    testing::Values(QIRTestCase{"RCCX", MQT_NAMED_BUILDER(rccx<>),
                                MQT_NAMED_BUILDER(rccx<>)},
                    QIRTestCase{"SingleControlledRCCX",
                                MQT_NAMED_BUILDER(singleControlledRccx<>),
                                MQT_NAMED_BUILDER(singleControlledRccx<>)},
                    QIRTestCase{"MultipleControlledRCCX",
                                MQT_NAMED_BUILDER(multipleControlledRccx<>),
                                MQT_NAMED_BUILDER(multipleControlledRccx<>)}));
/// @}

/// \name QIR/Operations/StandardGates/ROp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QIRROpTest, QIRTest,
    testing::Values(
        QIRTestCase{"R", MQT_NAMED_BUILDER(r<>), MQT_NAMED_BUILDER(r<>)},
        QIRTestCase{"SingleControlledR", MQT_NAMED_BUILDER(singleControlledR<>),
                    MQT_NAMED_BUILDER(singleControlledR<>)},
        QIRTestCase{"MultipleControlledR",
                    MQT_NAMED_BUILDER(multipleControlledR<>),
                    MQT_NAMED_BUILDER(multipleControlledR<>)}));
/// @}

/// \name QIR/Operations/StandardGates/RxOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QIRRXOpTest, QIRTest,
    testing::Values(QIRTestCase{"RX", MQT_NAMED_BUILDER(rx<>),
                                MQT_NAMED_BUILDER(rx<>)},
                    QIRTestCase{"SingleControlledRX",
                                MQT_NAMED_BUILDER(singleControlledRx<>),
                                MQT_NAMED_BUILDER(singleControlledRx<>)},
                    QIRTestCase{"MultipleControlledRX",
                                MQT_NAMED_BUILDER(multipleControlledRx<>),
                                MQT_NAMED_BUILDER(multipleControlledRx<>)}));
/// @}

/// \name QIR/Operations/StandardGates/RxxOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QIRRXXOpTest, QIRTest,
    testing::Values(QIRTestCase{"RXX", MQT_NAMED_BUILDER(rxx<>),
                                MQT_NAMED_BUILDER(rxx<>)},
                    QIRTestCase{"SingleControlledRXX",
                                MQT_NAMED_BUILDER(singleControlledRxx<>),
                                MQT_NAMED_BUILDER(singleControlledRxx<>)},
                    QIRTestCase{"MultipleControlledRXX",
                                MQT_NAMED_BUILDER(multipleControlledRxx<>),
                                MQT_NAMED_BUILDER(multipleControlledRxx<>)}));
/// @}

/// \name QIR/Operations/StandardGates/RyOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QIRRYOpTest, QIRTest,
    testing::Values(QIRTestCase{"RY", MQT_NAMED_BUILDER(ry<>),
                                MQT_NAMED_BUILDER(ry<>)},
                    QIRTestCase{"SingleControlledRY",
                                MQT_NAMED_BUILDER(singleControlledRy<>),
                                MQT_NAMED_BUILDER(singleControlledRy<>)},
                    QIRTestCase{"MultipleControlledRY",
                                MQT_NAMED_BUILDER(multipleControlledRy<>),
                                MQT_NAMED_BUILDER(multipleControlledRy<>)}));
/// @}

/// \name QIR/Operations/StandardGates/RyyOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QIRRYYOpTest, QIRTest,
    testing::Values(QIRTestCase{"RYY", MQT_NAMED_BUILDER(ryy<>),
                                MQT_NAMED_BUILDER(ryy<>)},
                    QIRTestCase{"SingleControlledRYY",
                                MQT_NAMED_BUILDER(singleControlledRyy<>),
                                MQT_NAMED_BUILDER(singleControlledRyy<>)},
                    QIRTestCase{"MultipleControlledRYY",
                                MQT_NAMED_BUILDER(multipleControlledRyy<>),
                                MQT_NAMED_BUILDER(multipleControlledRyy<>)}));
/// @}

/// \name QIR/Operations/StandardGates/RzOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QIRRZOpTest, QIRTest,
    testing::Values(QIRTestCase{"RZ", MQT_NAMED_BUILDER(rz<>),
                                MQT_NAMED_BUILDER(rz<>)},
                    QIRTestCase{"SingleControlledRZ",
                                MQT_NAMED_BUILDER(singleControlledRz<>),
                                MQT_NAMED_BUILDER(singleControlledRz<>)},
                    QIRTestCase{"MultipleControlledRZ",
                                MQT_NAMED_BUILDER(multipleControlledRz<>),
                                MQT_NAMED_BUILDER(multipleControlledRz<>)}));
/// @}

/// \name QIR/Operations/StandardGates/RzxOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QIRRZXOpTest, QIRTest,
    testing::Values(QIRTestCase{"RZX", MQT_NAMED_BUILDER(rzx<>),
                                MQT_NAMED_BUILDER(rzx<>)},
                    QIRTestCase{"SingleControlledRZX",
                                MQT_NAMED_BUILDER(singleControlledRzx<>),
                                MQT_NAMED_BUILDER(singleControlledRzx<>)},
                    QIRTestCase{"MultipleControlledRZX",
                                MQT_NAMED_BUILDER(multipleControlledRzx<>),
                                MQT_NAMED_BUILDER(multipleControlledRzx<>)}));
/// @}

/// \name QIR/Operations/StandardGates/RzzOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QIRRZZOpTest, QIRTest,
    testing::Values(QIRTestCase{"RZZ", MQT_NAMED_BUILDER(rzz<>),
                                MQT_NAMED_BUILDER(rzz<>)},
                    QIRTestCase{"SingleControlledRZZ",
                                MQT_NAMED_BUILDER(singleControlledRzz<>),
                                MQT_NAMED_BUILDER(singleControlledRzz<>)},
                    QIRTestCase{"MultipleControlledRZZ",
                                MQT_NAMED_BUILDER(multipleControlledRzz<>),
                                MQT_NAMED_BUILDER(multipleControlledRzz<>)}));
/// @}

/// \name QIR/Operations/StandardGates/SOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QIRSOpTest, QIRTest,
    testing::Values(
        QIRTestCase{"S", MQT_NAMED_BUILDER(s<>), MQT_NAMED_BUILDER(s<>)},
        QIRTestCase{"SingleControlledS", MQT_NAMED_BUILDER(singleControlledS<>),
                    MQT_NAMED_BUILDER(singleControlledS<>)},
        QIRTestCase{"MultipleControlledS",
                    MQT_NAMED_BUILDER(multipleControlledS<>),
                    MQT_NAMED_BUILDER(multipleControlledS<>)}));
/// @}

/// \name QIR/Operations/StandardGates/SdgOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QIRSdgOpTest, QIRTest,
    testing::Values(QIRTestCase{"Sdg", MQT_NAMED_BUILDER(sdg<>),
                                MQT_NAMED_BUILDER(sdg<>)},
                    QIRTestCase{"SingleControlledSdg",
                                MQT_NAMED_BUILDER(singleControlledSdg<>),
                                MQT_NAMED_BUILDER(singleControlledSdg<>)},
                    QIRTestCase{"MultipleControlledSdg",
                                MQT_NAMED_BUILDER(multipleControlledSdg<>),
                                MQT_NAMED_BUILDER(multipleControlledSdg<>)}));
/// @}

/// \name QIR/Operations/StandardGates/SwapOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QIRSWAPOpTest, QIRTest,
    testing::Values(QIRTestCase{"SWAP", MQT_NAMED_BUILDER(swap<>),
                                MQT_NAMED_BUILDER(swap<>)},
                    QIRTestCase{"SingleControlledSWAP",
                                MQT_NAMED_BUILDER(singleControlledSwap<>),
                                MQT_NAMED_BUILDER(singleControlledSwap<>)},
                    QIRTestCase{"MultipleControlledSWAP",
                                MQT_NAMED_BUILDER(multipleControlledSwap<>),
                                MQT_NAMED_BUILDER(multipleControlledSwap<>)}));
/// @}

/// \name QIR/Operations/StandardGates/SxOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QIRSXOpTest, QIRTest,
    testing::Values(QIRTestCase{"SX", MQT_NAMED_BUILDER(sx<>),
                                MQT_NAMED_BUILDER(sx<>)},
                    QIRTestCase{"SingleControlledSX",
                                MQT_NAMED_BUILDER(singleControlledSx<>),
                                MQT_NAMED_BUILDER(singleControlledSx<>)},
                    QIRTestCase{"MultipleControlledSX",
                                MQT_NAMED_BUILDER(multipleControlledSx<>),
                                MQT_NAMED_BUILDER(multipleControlledSx<>)}));
/// @}

/// \name QIR/Operations/StandardGates/SxdgOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QIRSXdgOpTest, QIRTest,
    testing::Values(QIRTestCase{"SXdg", MQT_NAMED_BUILDER(sxdg<>),
                                MQT_NAMED_BUILDER(sxdg<>)},
                    QIRTestCase{"SingleControlledSXdg",
                                MQT_NAMED_BUILDER(singleControlledSxdg<>),
                                MQT_NAMED_BUILDER(singleControlledSxdg<>)},
                    QIRTestCase{"MultipleControlledSXdg",
                                MQT_NAMED_BUILDER(multipleControlledSxdg<>),
                                MQT_NAMED_BUILDER(multipleControlledSxdg<>)}));
/// @}

/// \name QIR/Operations/StandardGates/TOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QIRTOpTest, QIRTest,
    testing::Values(
        QIRTestCase{"T", MQT_NAMED_BUILDER(t_<>), MQT_NAMED_BUILDER(t_<>)},
        QIRTestCase{"SingleControlledT", MQT_NAMED_BUILDER(singleControlledT<>),
                    MQT_NAMED_BUILDER(singleControlledT<>)},
        QIRTestCase{"MultipleControlledT",
                    MQT_NAMED_BUILDER(multipleControlledT<>),
                    MQT_NAMED_BUILDER(multipleControlledT<>)}));
/// @}

/// \name QIR/Operations/StandardGates/TdgOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QIRTdgOpTest, QIRTest,
    testing::Values(QIRTestCase{"Tdg", MQT_NAMED_BUILDER(tdg<>),
                                MQT_NAMED_BUILDER(tdg<>)},
                    QIRTestCase{"SingleControlledTdg",
                                MQT_NAMED_BUILDER(singleControlledTdg<>),
                                MQT_NAMED_BUILDER(singleControlledTdg<>)},
                    QIRTestCase{"MultipleControlledTdg",
                                MQT_NAMED_BUILDER(multipleControlledTdg<>),
                                MQT_NAMED_BUILDER(multipleControlledTdg<>)}));
/// @}

/// \name QIR/Operations/StandardGates/U2Op.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QIRU2OpTest, QIRTest,
    testing::Values(QIRTestCase{"U2", MQT_NAMED_BUILDER(u2<>),
                                MQT_NAMED_BUILDER(u2<>)},
                    QIRTestCase{"SingleControlledU2",
                                MQT_NAMED_BUILDER(singleControlledU2<>),
                                MQT_NAMED_BUILDER(singleControlledU2<>)},
                    QIRTestCase{"MultipleControlledU2",
                                MQT_NAMED_BUILDER(multipleControlledU2<>),
                                MQT_NAMED_BUILDER(multipleControlledU2<>)}));
/// @}

/// \name QIR/Operations/StandardGates/UOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QIRUOpTest, QIRTest,
    testing::Values(
        QIRTestCase{"U", MQT_NAMED_BUILDER(u<>), MQT_NAMED_BUILDER(u<>)},
        QIRTestCase{"SingleControlledU", MQT_NAMED_BUILDER(singleControlledU<>),
                    MQT_NAMED_BUILDER(singleControlledU<>)},
        QIRTestCase{"MultipleControlledU",
                    MQT_NAMED_BUILDER(multipleControlledU<>),
                    MQT_NAMED_BUILDER(multipleControlledU<>)}));
/// @}

/// \name QIR/Operations/StandardGates/XOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QIRXOpTest, QIRTest,
    testing::Values(
        QIRTestCase{"X", MQT_NAMED_BUILDER(x<>), MQT_NAMED_BUILDER(x<>)},
        QIRTestCase{"SingleControlledX", MQT_NAMED_BUILDER(singleControlledX<>),
                    MQT_NAMED_BUILDER(singleControlledX<>)},
        QIRTestCase{"MultipleControlledX",
                    MQT_NAMED_BUILDER(multipleControlledX<>),
                    MQT_NAMED_BUILDER(multipleControlledX<>)}));
/// @}

/// \name QIR/Operations/StandardGates/XxMinusYyOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QIRXXMinusYYOpTest, QIRTest,
    testing::Values(QIRTestCase{"XXMinusYY", MQT_NAMED_BUILDER(xxMinusYY<>),
                                MQT_NAMED_BUILDER(xxMinusYY<>)},
                    QIRTestCase{"SingleControlledXXMinusYY",
                                MQT_NAMED_BUILDER(singleControlledXxMinusYY<>),
                                MQT_NAMED_BUILDER(singleControlledXxMinusYY<>)},
                    QIRTestCase{
                        "MultipleControlledXXMinusYY",
                        MQT_NAMED_BUILDER(multipleControlledXxMinusYY<>),
                        MQT_NAMED_BUILDER(multipleControlledXxMinusYY<>)}));
/// @}

/// \name QIR/Operations/StandardGates/XxPlusYyOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QIRXXPlusYYOpTest, QIRTest,
    testing::Values(QIRTestCase{"XXPlusYY", MQT_NAMED_BUILDER(xxPlusYY<>),
                                MQT_NAMED_BUILDER(xxPlusYY<>)},
                    QIRTestCase{"SingleControlledXXPlusYY",
                                MQT_NAMED_BUILDER(singleControlledXxPlusYY<>),
                                MQT_NAMED_BUILDER(singleControlledXxPlusYY<>)},
                    QIRTestCase{
                        "MultipleControlledXXPlusYY",
                        MQT_NAMED_BUILDER(multipleControlledXxPlusYY<>),
                        MQT_NAMED_BUILDER(multipleControlledXxPlusYY<>)}));
/// @}

/// \name QIR/Operations/StandardGates/YOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QIRYOpTest, QIRTest,
    testing::Values(
        QIRTestCase{"Y", MQT_NAMED_BUILDER(y<>), MQT_NAMED_BUILDER(y<>)},
        QIRTestCase{"SingleControlledY", MQT_NAMED_BUILDER(singleControlledY<>),
                    MQT_NAMED_BUILDER(singleControlledY<>)},
        QIRTestCase{"MultipleControlledY",
                    MQT_NAMED_BUILDER(multipleControlledY<>),
                    MQT_NAMED_BUILDER(multipleControlledY<>)}));
/// @}

/// \name QIR/Operations/StandardGates/ZOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QIRZOpTest, QIRTest,
    testing::Values(
        QIRTestCase{"Z", MQT_NAMED_BUILDER(z<>), MQT_NAMED_BUILDER(z<>)},
        QIRTestCase{"SingleControlledZ", MQT_NAMED_BUILDER(singleControlledZ<>),
                    MQT_NAMED_BUILDER(singleControlledZ<>)},
        QIRTestCase{"MultipleControlledZ",
                    MQT_NAMED_BUILDER(multipleControlledZ<>),
                    MQT_NAMED_BUILDER(multipleControlledZ<>)}));
/// @}

/// \name QIR/Operations/MeasureOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QIRMeasureOpTest, QIRTest,
    testing::Values(
        QIRTestCase{"SingleMeasurementToSingleBit",
                    MQT_NAMED_BUILDER(singleMeasurementToSingleBit),
                    MQT_NAMED_BUILDER(singleMeasurementToSingleBit)},
        QIRTestCase{"RepeatedMeasurementToSameBit",
                    MQT_NAMED_BUILDER(repeatedMeasurementToSameBit),
                    MQT_NAMED_BUILDER(repeatedMeasurementToSameBit)},
        QIRTestCase{"RepeatedMeasurementToDifferentBits",
                    MQT_NAMED_BUILDER(repeatedMeasurementToDifferentBits),
                    MQT_NAMED_BUILDER(repeatedMeasurementToDifferentBits)}));
/// @}

/// \name QIR/Operations/ResetOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QIRResetOpTest, QIRTest,
    testing::Values(
        QIRTestCase{"ResetQubitWithoutOp",
                    MQT_NAMED_BUILDER(resetQubitWithoutOp<>),
                    MQT_NAMED_BUILDER(resetQubitWithoutOp<>)},
        QIRTestCase{"ResetMultipleQubitsWithoutOp",
                    MQT_NAMED_BUILDER(resetMultipleQubitsWithoutOp<>),
                    MQT_NAMED_BUILDER(resetMultipleQubitsWithoutOp<>)},
        QIRTestCase{"RepeatedResetWithoutOp",
                    MQT_NAMED_BUILDER(repeatedResetWithoutOp<>),
                    MQT_NAMED_BUILDER(repeatedResetWithoutOp<>)},
        QIRTestCase{"ResetQubitAfterSingleOp",
                    MQT_NAMED_BUILDER(resetQubitAfterSingleOp<>),
                    MQT_NAMED_BUILDER(resetQubitAfterSingleOp<>)},
        QIRTestCase{"ResetMultipleQubitsAfterSingleOp",
                    MQT_NAMED_BUILDER(resetMultipleQubitsAfterSingleOp<>),
                    MQT_NAMED_BUILDER(resetMultipleQubitsAfterSingleOp<>)},
        QIRTestCase{"RepeatedResetAfterSingleOp",
                    MQT_NAMED_BUILDER(repeatedResetAfterSingleOp<>),
                    MQT_NAMED_BUILDER(repeatedResetAfterSingleOp<>)}));
/// @}

/// \name QIR/QubitManagement/QubitManagement.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QIRQubitManagementTest, QIRTest,
    testing::Values(
        QIRTestCase{"AllocQubit", MQT_NAMED_BUILDER(allocQubit<>),
                    MQT_NAMED_BUILDER(allocQubit<>)},
        QIRTestCase{"AllocQubitRegister",
                    MQT_NAMED_BUILDER(allocQubitRegister<>),
                    MQT_NAMED_BUILDER(allocQubitRegister<>)},
        QIRTestCase{"AllocMultipleQubitRegisters",
                    MQT_NAMED_BUILDER(allocMultipleQubitRegisters<>),
                    MQT_NAMED_BUILDER(allocMultipleQubitRegisters<>)},
        QIRTestCase{"AllocLargeRegister",
                    MQT_NAMED_BUILDER(allocLargeRegister<>),
                    MQT_NAMED_BUILDER(allocLargeRegister<>)},
        QIRTestCase{"StaticQubits", MQT_NAMED_BUILDER(staticQubits),
                    MQT_NAMED_BUILDER(staticQubits)},
        QIRTestCase{"StaticQubitsWithOps",
                    MQT_NAMED_BUILDER(staticQubitsWithOps),
                    MQT_NAMED_BUILDER(staticQubitsWithOps)},
        QIRTestCase{"StaticQubitsWithParametricOps",
                    MQT_NAMED_BUILDER(staticQubitsWithParametricOps),
                    MQT_NAMED_BUILDER(staticQubitsWithParametricOps)},
        QIRTestCase{"StaticQubitsWithTwoTargetOps",
                    MQT_NAMED_BUILDER(staticQubitsWithTwoTargetOps),
                    MQT_NAMED_BUILDER(staticQubitsWithTwoTargetOps)},
        QIRTestCase{"StaticQubitsWithCtrl",
                    MQT_NAMED_BUILDER(staticQubitsWithCtrl),
                    MQT_NAMED_BUILDER(staticQubitsWithCtrl)},
        QIRTestCase{"StaticQubitsWithInv",
                    MQT_NAMED_BUILDER(staticQubitsWithInv),
                    MQT_NAMED_BUILDER(staticQubitsWithInv)},
        QIRTestCase{"StaticQubitsWithDuplicates",
                    MQT_NAMED_BUILDER(staticQubitsWithDuplicates),
                    MQT_NAMED_BUILDER(staticQubitsCanonical)}));
/// @}
