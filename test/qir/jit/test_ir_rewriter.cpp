/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QIR/Execution/JIT/IRRewriter.h"

#include <gtest/gtest.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/AsmParser/Parser.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/raw_ostream.h>

#include <cstddef>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

std::size_t countCallsTo(const llvm::Module& m, llvm::StringRef name) {
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

std::unique_ptr<llvm::Module> loadIRFile(const std::filesystem::path& path,
                                         llvm::LLVMContext& ctx) {
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
  auto* entryPoint = module->getFunction("main");
  ASSERT_NE(entryPoint, nullptr);

  EXPECT_THROW(qir::prepareForStateExtraction(*entryPoint),
               std::invalid_argument);
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

} // namespace
