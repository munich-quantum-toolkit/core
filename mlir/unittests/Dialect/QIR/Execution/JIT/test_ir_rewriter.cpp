/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/Dialect/QIR/Execution/JIT/IRRewriter.h"

#include "gtest/gtest.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

static std::size_t countCallsTo(const llvm::Module& m, llvm::StringRef name) {
  std::size_t count = 0;
  for (const auto& fn : m) {
    for (const auto& bb : fn) {
      for (const auto& inst : bb) {
        const auto* call = llvm::dyn_cast<llvm::CallInst>(&inst);
        if (call == nullptr) {
          continue;
        }
        const auto* callee = call->getCalledFunction();
        if (callee != nullptr && callee->getName() == name) {
          ++count;
        }
      }
    }
  }
  return count;
}

static std::unique_ptr<llvm::Module>
loadIRFile(const std::filesystem::path& path, llvm::LLVMContext& ctx) {
  llvm::SMDiagnostic err;
  auto llvmModule = llvm::parseIRFile(path.string(), err, ctx);
  if (!llvmModule) {
    std::string errStr;
    llvm::raw_string_ostream s(errStr);
    err.print("test_ir_rewriter", s);
    throw std::runtime_error("Failed to parse IR file " + path.string() + ": " +
                             errStr);
  }
  return llvmModule;
}

namespace {

class IRRewriterTest : public testing::TestWithParam<std::string_view> {
protected:
  llvm::LLVMContext ctx_;
};

TEST_P(IRRewriterTest, TruncatesAtIrreversibleBoundary) {
  const std::filesystem::path path =
      std::filesystem::path(QIR_FILES_DIR) / GetParam();
  auto llvmModule = loadIRFile(path, ctx_);

  auto* entryPoint = llvmModule->getFunction("main");
  ASSERT_NE(entryPoint, nullptr);
  ASSERT_GT(countCallsTo(*llvmModule, "__quantum__rt__result_record_output"),
            0U);

  EXPECT_TRUE(qir::prepareForStateExtraction(*entryPoint));
  EXPECT_EQ(countCallsTo(*llvmModule, "__quantum__qis__mz__body"), 0U);
  EXPECT_EQ(countCallsTo(*llvmModule, "__quantum__rt__qubit_release"), 0U);
  EXPECT_EQ(countCallsTo(*llvmModule, "__quantum__rt__result_record_output"),
            0U);
}

INSTANTIATE_TEST_SUITE_P(BellPair, IRRewriterTest,
                         testing::Values("BellPairStatic.ll"));

TEST(IRRewriter, RemovesAllWorkAfterFirstIrreversibleCall) {
  constexpr llvm::StringRef ir = R"(
define i64 @main() #0 {
  call void @prepare()
  call void @measure()
  call void @must_not_run()
  ret i64 0
}
declare void @prepare()
declare void @measure() #1
declare void @must_not_run()
attributes #0 = { "entry_point" "qir_profiles"="base_profile" }
attributes #1 = { "irreversible" }
)";
  llvm::LLVMContext context;
  llvm::SMDiagnostic error;
  auto module = llvm::parseAssemblyString(ir, error, context);
  ASSERT_NE(module, nullptr);
  auto* entryPoint = module->getFunction("main");
  ASSERT_NE(entryPoint, nullptr);

  EXPECT_TRUE(qir::prepareForStateExtraction(*entryPoint));
  EXPECT_EQ(countCallsTo(*module, "prepare"), 1U);
  EXPECT_EQ(countCallsTo(*module, "measure"), 0U);
  EXPECT_EQ(countCallsTo(*module, "must_not_run"), 0U);
}

TEST(IRRewriter, FindsBoundaryAcrossReverseOrderedBlocks) {
  constexpr llvm::StringRef ir = R"(
define i64 @main() #0 {
entry:
  call void @prepare()
  br label %first
last:
  call void @measure()
  ret i64 0
middle:
  call void @measure()
  br label %last
first:
  call void @measure()
  call void @measure()
  br label %middle
}
declare void @prepare()
declare void @measure() #1
attributes #0 = { "entry_point" "qir_profiles"="base_profile" }
attributes #1 = { "irreversible" }
)";
  llvm::LLVMContext context;
  llvm::SMDiagnostic error;
  auto llvmModule = llvm::parseAssemblyString(ir, error, context);
  ASSERT_NE(llvmModule, nullptr);
  ASSERT_FALSE(llvm::verifyModule(*llvmModule));

  EXPECT_TRUE(qir::prepareForStateExtraction(*llvmModule->getFunction("main")));
  EXPECT_EQ(countCallsTo(*llvmModule, "prepare"), 1U);
  EXPECT_EQ(countCallsTo(*llvmModule, "measure"), 0U);
  EXPECT_FALSE(llvm::verifyModule(*llvmModule));
}

TEST(IRRewriter, RejectsIndependentIrreversibleRegions) {
  constexpr llvm::StringRef ir = R"(
define i64 @main() #0 {
entry:
  br i1 true, label %left, label %right
left:
  call void @measure_left()
  ret i64 0
right:
  call void @measure_right()
  ret i64 0
}
declare void @measure_left() #1
declare void @measure_right() #1
attributes #0 = { "entry_point" "qir_profiles"="base_profile" }
attributes #1 = { "irreversible" }
)";
  llvm::LLVMContext context;
  llvm::SMDiagnostic error;
  auto module = llvm::parseAssemblyString(ir, error, context);
  ASSERT_NE(module, nullptr);
  ASSERT_FALSE(llvm::verifyModule(*module));
  auto* entryPoint = module->getFunction("main");
  ASSERT_NE(entryPoint, nullptr);
  std::string before;
  llvm::raw_string_ostream(before) << *module;

  EXPECT_THROW(qir::prepareForStateExtraction(*entryPoint),
               std::invalid_argument);
  std::string after;
  llvm::raw_string_ostream(after) << *module;
  EXPECT_EQ(after, before);
}

TEST(IRRewriter, RequiresBaseProfile) {
  constexpr llvm::StringRef ir = R"(
define i64 @main() #0 { ret i64 0 }
attributes #0 = { "entry_point" }
)";
  llvm::LLVMContext context;
  llvm::SMDiagnostic error;
  auto module = llvm::parseAssemblyString(ir, error, context);
  ASSERT_NE(module, nullptr);
  auto* entryPoint = module->getFunction("main");
  ASSERT_NE(entryPoint, nullptr);

  EXPECT_THROW(qir::prepareForStateExtraction(*entryPoint),
               std::invalid_argument);
}

class QIRSamplingPlan : public testing::TestWithParam<const char*> {
protected:
  auto samplingOutputs(std::string ir) {
    ir.replace(ir.find("base_profile"), std::string_view("base_profile").size(),
               GetParam());
    llvm::LLVMContext context;
    llvm::SMDiagnostic error;
    auto llvmModule = llvm::parseAssemblyString(ir, error, context);
    if (!llvmModule) {
      throw std::runtime_error(error.getMessage().str());
    }
    return qir::getStaticSamplingOutputs(*llvmModule->getFunction("main"));
  }
};

INSTANTIATE_TEST_SUITE_P(Profiles, QIRSamplingPlan,
                         testing::Values("base_profile", "adaptive_profile"));

TEST_P(QIRSamplingPlan, PreservesRepeatedOutputsAndOverwrittenResults) {
  const auto outputs = samplingOutputs(R"(
define i64 @main() #0 {
entry:
  call void @__quantum__rt__initialize(ptr null)
  call void @__quantum__qis__h__body(ptr null)
  br label %measure
measure:
  call void @__quantum__qis__mz__body(ptr null, ptr null)
  call void @__quantum__rt__result_record_output(ptr null, ptr null)
  call void @__quantum__qis__mz__body(ptr inttoptr (i64 2 to ptr), ptr null)
  call void @__quantum__rt__result_record_output(ptr null, ptr null)
  call void @__quantum__rt__result_record_output(ptr null, ptr null)
  ret i64 0
}
declare void @__quantum__rt__initialize(ptr)
declare void @__quantum__qis__h__body(ptr)
declare void @__quantum__qis__mz__body(ptr, ptr)
declare void @__quantum__rt__result_record_output(ptr, ptr)
attributes #0 = { "entry_point" "qir_profiles"="base_profile" }
)");
  ASSERT_TRUE(outputs.has_value());
  EXPECT_EQ(*outputs, (std::vector<uintptr_t>{0, 2, 2}));
}

TEST_P(QIRSamplingPlan, DoesNotDeferMeasurementsBeforeQuantumWork) {
  EXPECT_FALSE(samplingOutputs(R"(
define i64 @main() #0 {
  call void @__quantum__qis__mz__body(ptr null, ptr null)
  call void @__quantum__qis__x__body(ptr null)
  ret i64 0
}
declare void @__quantum__qis__mz__body(ptr, ptr)
declare void @__quantum__qis__x__body(ptr)
attributes #0 = { "entry_point" "qir_profiles"="base_profile" }
)")
                   .has_value());
}

TEST_P(QIRSamplingPlan, RejectsUnknownCallsAndHiddenQuantumEffects) {
  EXPECT_FALSE(samplingOutputs(R"(
define i64 @main() #0 {
  call void @helper()
  ret i64 0
}
define void @helper() {
  call void @__quantum__qis__mz__body(ptr null, ptr null)
  ret void
}
declare void @__quantum__qis__mz__body(ptr, ptr)
attributes #0 = { "entry_point" "qir_profiles"="base_profile" }
)")
                   .has_value());
  EXPECT_FALSE(samplingOutputs(R"(
define i64 @main() #0 {
  call void @external()
  ret i64 0
}
declare void @external()
attributes #0 = { "entry_point" "qir_profiles"="base_profile" }
)")
                   .has_value());
}

TEST_P(QIRSamplingPlan, RejectsControlFlowMemoryAndResets) {
  for (const auto* body : {
           "br label %loop\nloop: br label %loop",
           "br i1 true, label %left, label %right\nleft: ret i64 0\nright: ret "
           "i64 0",
           "store i64 1, ptr @counter\nret i64 0",
           "call void @__quantum__qis__reset__body(ptr null)\nret i64 0",
           "call void @__quantum__qis__mz__body(ptr null, ptr null)\n"
           "%r = call i1 @__quantum__rt__read_result(ptr null)\nret i64 0",
           "call void @__quantum__qis__x__body(ptr null)\n"
           "call void @__quantum__rt__initialize(ptr null)\nret i64 0",
           "ret i64 1",
       }) {
    SCOPED_TRACE(body);
    const std::string ir = std::string(R"(
@counter = global i64 0
define i64 @main() #0 {
)") + body + R"(
}
declare void @__quantum__qis__reset__body(ptr)
declare void @__quantum__qis__mz__body(ptr, ptr)
declare void @__quantum__qis__x__body(ptr)
declare void @__quantum__rt__initialize(ptr)
declare i1 @__quantum__rt__read_result(ptr)
attributes #0 = { "entry_point" "qir_profiles"="base_profile" }
)";
    EXPECT_FALSE(samplingOutputs(ir).has_value());
  }
}

TEST(IRRewriter, RejectsDefinedAndIndirectCallsBeforeChangingIR) {
  for (const auto* body : {
           "call void @readout()",
           "%callee = load ptr, ptr @readout_ptr\ncall void %callee()",
           "call void @__quantum__qis__mz__body(ptr null, ptr null)\n"
           "call void @readout()",
       }) {
    SCOPED_TRACE(body);
    const auto ir = std::string(R"(
@readout_ptr = global ptr @readout
define i64 @main() #0 {
  call void @__quantum__qis__h__body(ptr null)
)") + body + R"(
  ret i64 0
}
define void @readout() {
  call void @__quantum__qis__mz__body(ptr null, ptr null)
  ret void
}
declare void @__quantum__qis__h__body(ptr)
declare void @__quantum__qis__mz__body(ptr, ptr) #1
attributes #0 = { "entry_point" "qir_profiles"="base_profile" }
attributes #1 = { "irreversible" }
)";
    llvm::LLVMContext context;
    llvm::SMDiagnostic error;
    auto llvmModule = llvm::parseAssemblyString(ir, error, context);
    ASSERT_NE(llvmModule, nullptr);
    ASSERT_FALSE(llvm::verifyModule(*llvmModule));
    const auto measurements =
        countCallsTo(*llvmModule, "__quantum__qis__mz__body");
    EXPECT_THROW(
        qir::prepareForStateExtraction(*llvmModule->getFunction("main")),
        std::invalid_argument);
    EXPECT_EQ(countCallsTo(*llvmModule, "__quantum__qis__mz__body"),
              measurements);
    EXPECT_FALSE(llvm::verifyModule(*llvmModule));
  }
}

} // namespace
