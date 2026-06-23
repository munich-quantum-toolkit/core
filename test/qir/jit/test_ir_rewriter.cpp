/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "qir/jit/IRRewriter.hpp"

#include <gtest/gtest.h>
#include <llvm/ADT/StringRef.h>
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
#include <numeric>
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

std::size_t countCallsToStripTarget(const llvm::Module& m) {
  return std::accumulate(qir::STRIP_TARGETS.begin(), qir::STRIP_TARGETS.end(),
                         std::size_t{},
                         [&m](std::size_t total, const auto& target) {
                           return total + countCallsTo(m, target);
                         });
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

TEST_P(IRRewriterTest, StripMeasurementRelatedCalls) {
  const std::filesystem::path path =
      std::filesystem::path(QIR_FILES_DIR) / GetParam();
  auto llvmModule = loadIRFile(path, ctx_);

  const auto numStripCalls = countCallsToStripTarget(*llvmModule);
  ASSERT_GT(numStripCalls, 0U) << "Module has no calls to strip targets";

  const auto numErased = qir::stripMeasurementRelatedCalls(*llvmModule);

  EXPECT_EQ(numErased, numStripCalls);
  EXPECT_EQ(countCallsToStripTarget(*llvmModule), 0U);
}

INSTANTIATE_TEST_SUITE_P(BellPair, IRRewriterTest,
                         testing::Values("BellPairStatic.ll",
                                         "BellPairDynamic.ll"));

} // namespace
