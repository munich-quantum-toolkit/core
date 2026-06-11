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
 * @brief Whether the JIT'd program runs to produce measurement samples or
 * to leave the final quantum state in @ref qir::Runtime for external
 * extraction.
 * @details In @c StateExtraction mode the session strips QIR measurement
 * and result-management calls from the IR before JIT-compiling, so the
 * runtime's quantum state remains intact after @c main returns. Intended
 * for QIR Base Profile programs only.
 */
enum class Execution { Sampling, StateExtraction };

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
   * @param mode Execution mode; see @ref Execution.
   * @throws std::runtime_error if the file cannot be parsed or the JIT fails
   * to initialize.
   */
  explicit JitSession(llvm::StringRef inputFile,
                      Execution mode = Execution::Sampling);

  /**
   * @brief Build a session by loading IR from a memory buffer.
   * @details Accepts either textual IR or bitcode. The buffer does not have
   * to be null-terminated.
   * @param irBytes Byte view of the IR.
   * @param bufferName Identifier used in diagnostics.
   * @param mode Execution mode; see @ref Execution.
   * @throws std::runtime_error if the IR cannot be parsed or the JIT fails
   * to initialize.
   */
  JitSession(llvm::StringRef irBytes, llvm::StringRef bufferName,
             Execution mode = Execution::Sampling);

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

  /// Registers the QIR runtime symbols with @c llvm::sys::DynamicLibrary so the
  /// JIT can resolve them at link time.
  /// Safe to call multiple times; the work runs only on the first call.
  static void registerRuntimeSymbols();

  /// Initializes the native target, asm printer and asm parser.
  /// Safe to call multiple times; the work runs only on the first call.
  static void initNativeTargets();

  /// Parses LLVM IR from @p irPath using the session's thread-safe context.
  llvm::Expected<llvm::orc::ThreadSafeModule>
  loadModuleFromFile(llvm::StringRef irPath);

  /// Parses LLVM IR (textual or bitcode) from @p irBytes using the session's
  /// thread-safe context. @p bufferName is used in diagnostics.
  llvm::Expected<llvm::orc::ThreadSafeModule>
  loadModuleFromMemory(llvm::StringRef irBytes, llvm::StringRef bufferName);

  /// Prepares the session to run the program:
  /// - Validates the loaded module.
  /// - Optionally strips measurement and result management calls
  ///   (for @c Execution::StateExtraction).
  /// - Builds the @c LLJIT instance
  /// - Registers QIR runtime symbols
  /// - Resolves @c main.
  /// @throws std::runtime_error if loading failed or the JIT cannot start.
  void initialize(llvm::Expected<llvm::orc::ThreadSafeModule> llvmModule,
                  Execution mode);

  /// Tears down the @c LLJIT.
  void deinitialize() const;
};

} // namespace qir
