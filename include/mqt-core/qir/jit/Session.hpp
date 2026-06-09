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

namespace qir {

/**
 * @brief In-process JIT executor for QIR programs.
 * @details The session does the following, in order:
 * - Loads an LLVM module from either an IR file (text or bitcode) or
 * an in-memory buffer,
 * - JIT-compiles it via LLVM's OrcJIT with lazy compilation.
 * - wires up the QIR runtime symbols, and
 * - runs the module's @c main function.
 * A session owns a single LLJIT instance and is not meant to be reused across
 * modules; create a new @ref JitSession for each program.
 */
class JitSession {
public:
  /// Signature of the @c main function produced by QIR-compiled modules.
  using MainFn = int(int, char**);

  /**
   * @brief Build a session by loading IR from a file on disk.
   * @param inputFile Path to a textual IR or bitcode file.
   * @throws std::runtime_error if the file cannot be parsed or the JIT fails
   * to initialize.
   */
  explicit JitSession(llvm::StringRef inputFile);

  /**
   * @brief Build a session by loading IR from a memory buffer.
   * @details Accepts either textual IR or bitcode. The buffer does not have
   * to be null-terminated.
   * @param irBytes Byte view of the IR.
   * @param bufferName Identifier used in diagnostics.
   * @throws std::runtime_error if the IR cannot be parsed or the JIT fails
   * to initialize.
   */
  JitSession(llvm::StringRef irBytes, llvm::StringRef bufferName);

  /// Tears down the LLJIT and any JIT'd resources owned by the session.
  ~JitSession();

  /**
   * @brief Executes the JIT'd @c main function.
   * @param args Argument strings passed as @c argv (excluding @c argv[0]).
   * @param progName Value used as @c argv[0].
   * @return The integer returned by the JIT'd @c main.
   */
  int run(llvm::ArrayRef<std::string> args = {},
          llvm::StringRef progName = "") const;

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

} // namespace qir
