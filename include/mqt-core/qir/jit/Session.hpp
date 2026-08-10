/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

/** @file Session.hpp
 * @brief QIR JIT session interface.
 */

#pragma once

#include <llvm/ADT/StringRef.h>
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/Support/Error.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>

namespace qir {

class Runtime;

/**
 * @brief Whether the JIT'd program runs to produce measurement samples or
 * to leave the final quantum state in @ref qir::Runtime for external
 * extraction.
 * @details In @c StateExtraction mode the session stops a Base Profile entry
 * point at its first irreversible operation before JIT-compiling, so the
 * runtime's quantum state remains intact without executing measurements or
 * output recording. Adaptive Profile entry points are rejected.
 */
enum class Execution { Sampling, StateExtraction };

struct SessionOptions {
  Execution execution = Execution::Sampling;
  std::optional<std::string> entryPoint;
  std::optional<uint64_t> seed;
};

/**
 * @brief In-process JIT executor for QIR programs.
 * @details The session does the following, in order:
 * - Loads an LLVM module from either an IR file (text or bitcode) or
 *   an in-memory buffer,
 * - JIT-compiles it via LLVM's OrcJIT with lazy compilation.
 * - wires up the QIR runtime symbols, and
 * - runs the module function marked as its QIR entry point.
 * A session owns a single LLJIT instance and is not meant to be reused across
 * modules; create a new @ref JitSession for each program.
 */
class JitSession {
public:
  /// QIR 2.1 Base and Adaptive Profile entry-point signature.
  using EntryPointFn = int64_t();

  /**
   * @brief Build a session by loading IR from a file on disk.
   * @param inputFile Path to a textual IR or bitcode file.
   * @param mode Execution mode
   * @throws std::runtime_error if the file cannot be parsed or the JIT fails
   * to initialize.
   */
  explicit JitSession(llvm::StringRef inputFile,
                      const SessionOptions& options = {});

  /**
   * @brief Build a session by loading IR from a memory buffer.
   * @details Accepts either textual IR or bitcode. The buffer does not have
   * to be null-terminated.
   * @param irBytes Byte view of the IR.
   * @param bufferName Identifier used in diagnostics.
   * @param mode Execution mode
   * @throws std::runtime_error if the IR cannot be parsed or the JIT fails
   * to initialize.
   */
  JitSession(llvm::StringRef irBytes, llvm::StringRef bufferName,
             const SessionOptions& options = {});

  /// Tears down the LLJIT and any JIT'd resources owned by the session.
  ~JitSession();

  /**
   * @brief Execute the selected QIR entry point.
   * @return The 64-bit QIR exit code.
   */
  int64_t run();

  [[nodiscard]] auto runtime() -> Runtime&;
  [[nodiscard]] auto runtime() const -> const Runtime&;
  [[nodiscard]] auto entryPointName() const -> const std::string& {
    return entryPointName_;
  }

private:
  llvm::orc::ThreadSafeContext tsCtx_{std::make_unique<llvm::LLVMContext>()};
  llvm::orc::ThreadSafeModule module_;
  std::unique_ptr<Runtime> runtime_;
  std::unique_ptr<llvm::orc::LLJIT> jit_;
  EntryPointFn* entryPointFn_ = nullptr;
  std::string entryPointName_;

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
  /// - Optionally truncates the entry point at its first irreversible operation
  ///   (for @c Execution::StateExtraction).
  /// - Builds the @c LLJIT instance
  /// - Registers QIR runtime symbols
  /// - Resolves the selected QIR entry point.
  /// @throws std::runtime_error if loading failed or the JIT cannot start.
  void initialize(llvm::Expected<llvm::orc::ThreadSafeModule> llvmModule,
                  const SessionOptions& options);

  /// Tears down the @c LLJIT.
  void deinitialize() const;
};

} // namespace qir
