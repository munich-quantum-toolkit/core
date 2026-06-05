/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/Support/Error.h>

#include <memory>
#include <string>

namespace qir::jit {

class Session {
public:
  using MainFn = int(int, char**);

  explicit Session(llvm::StringRef inputFile);
  Session(llvm::StringRef irBytes, llvm::StringRef bufferName);
  ~Session();
  int run();
  int run(llvm::ArrayRef<std::string> args, llvm::StringRef progName = "");

private:
  llvm::orc::ThreadSafeContext tsCtx_{std::make_unique<llvm::LLVMContext>()};
  llvm::orc::ThreadSafeModule module_;
  std::unique_ptr<llvm::orc::LLJIT> jit_;
  MainFn* mainFn_ = nullptr;

  static void registerRuntimeSymbols();
  static void initNativeTargets();
  llvm::Expected<llvm::orc::ThreadSafeModule>
  loadModuleFromFile(llvm::StringRef irPath);
  llvm::Expected<llvm::orc::ThreadSafeModule>
  loadModuleFromMemory(llvm::StringRef irBytes, llvm::StringRef bufferName);
  void initialize();
  void deinitialize();
};

} // namespace qir::jit
