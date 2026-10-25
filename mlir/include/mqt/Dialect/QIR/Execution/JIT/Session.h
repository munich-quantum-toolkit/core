/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

/// @file Session.hpp
/// QIR JIT session interface.

#pragma once

#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace qir {

class Runtime;

/// One terminal shot's output, with Result digit offsets in recording order.
struct SampledOutput {
  std::string text;
  std::vector<uint64_t> resultOffsets;
};

/// Whether the JIT'd program runs to produce measurement samples or
/// to leave the final quantum state in @ref qir::Runtime for external
/// extraction.
///
/// In @c StateExtraction mode the session stops a Base Profile entry
/// point at its first irreversible operation before JIT-compiling, so the
/// runtime's quantum state remains intact without executing measurements or
/// output recording. Adaptive Profile execution starts fresh, defers
/// measurements, preserves released wires, and rejects measurement-dependent
/// computation, resets and operations on measured wires. Classical control flow
/// and direct helpers are supported; recorded outputs are suppressed.
enum class Execution { Sampling, StateExtraction };

/// In-process JIT executor for QIR programs.
///
/// The session does the following, in order:
/// - Loads an LLVM module from an in-memory buffer,
/// - JIT-compiles it via LLVM's OrcJIT,
/// - wires up the QIR runtime symbols, and
/// - runs the module function marked as its QIR entry point.
/// A session owns a single LLJIT instance and is not meant to be reused across
/// modules; create a new @ref JitSession for each program.
class JitSession {
public:
  /// QIR 2.1 Base and Adaptive Profile entry-point signature.
  using EntryPointFn = int64_t();

  /// Build a session by loading IR from a memory buffer.
  ///
  /// Accepts either textual IR or bitcode. The buffer does not have
  /// to be null-terminated.
  /// @param irBytes Byte view of the IR.
  /// @param bufferName Identifier used in diagnostics.
  /// @param execution Execution mode.
  /// @param randomSeed Optional deterministic runtime seed.
  /// @param output Borrowed stream for constructor, entry-point, and destructor
  /// records. It must outlive the session; null disables output.
  /// @throws std::runtime_error if the IR cannot be parsed or the JIT fails
  /// to initialize.
  JitSession(llvm::StringRef irBytes, llvm::StringRef bufferName,
             Execution execution = Execution::Sampling,
             std::optional<uint64_t> randomSeed = std::nullopt,
             llvm::raw_ostream* output = &llvm::outs());

  /// Tears down the LLJIT and any JIT'd resources owned by the session.
  ~JitSession();

  /// Execute the selected QIR entry point.
  ///
  /// @return The 64-bit QIR exit code.
  int64_t run();

  /// Execute a batch, preserving recorded-result order and returning the first
  /// nonzero exit code.
  ///
  /// Eligible static Base or Adaptive programs execute once with deferred
  /// measurements, then sample that state. Other programs execute per shot.
  /// State-extraction sessions cannot be sampled. The supplied vector is
  /// replaced, including for zero shots. If supplied, stateAvailable is set
  /// only when successful terminal sampling leaves an uncollapsed state. The
  /// caller may then use runtime().takeState() before executing the session
  /// again. Results contain recorded Result and Boolean values; numeric records
  /// appear only in the textual output.
  int64_t sample(size_t shots, std::vector<std::string>& results,
                 bool* stateAvailable = nullptr, bool emitHeader = true);

  /// Execute a batch and emit each shot before the next is generated.
  ///
  /// The view is valid only during the callback. If supplied with output
  /// enabled, terminalOutput receives the template instead of writing per-shot
  /// records. Other output goes to the stream; terminalOutput is reset when no
  /// template is produced.
  int64_t sample(size_t shots, llvm::function_ref<void(std::string_view)> emit,
                 bool* stateAvailable = nullptr, bool emitHeader = true,
                 std::optional<SampledOutput>* terminalOutput = nullptr);

  /// Sample a histogram without textual recording or retained shot strings.
  ///
  /// Eligible deterministic programs split shot populations at measurements
  /// and replay their prefixes for sibling paths. Other programs execute per
  /// shot. Counts use QIR recording order; seeded results are repeatable within
  /// this sampling strategy, not across strategies.
  int64_t sampleCounts(size_t shots, std::map<std::string, size_t>& counts,
                       bool* stateAvailable = nullptr);

  /// Whether histogram sampling can group measurement paths.
  [[nodiscard]] bool canSampleBranches() const { return sampleBranches_; }

  /// Whether sampling may share the compiled entry point.
  ///
  /// Eligible modules have only constant globals and calls to the QIR runtime
  /// or LLVM intrinsics.
  [[nodiscard]] bool canShareCompiledCode() const;

  /// Number of direct quantum instruction calls in the loaded module.
  [[nodiscard]] size_t quantumCallSites() const { return quantumCallSites_; }

  /// Create a seeded runtime for one worker.
  ///
  /// Keep this session alive until all workers finish; each runtime and its
  /// output stream belong to one worker. Throws std::logic_error when the
  /// module is not eligible.
  [[nodiscard]] std::unique_ptr<Runtime> makeWorkerRuntime(uint64_t seed) const;

  /// Sample with a worker runtime without changing this session's runtime.
  ///
  /// Concurrent calls require separate runtimes from makeWorkerRuntime().
  /// Throws std::logic_error when a separate runtime is not eligible.
  int64_t sampleWithRuntime(Runtime& runtime, size_t shots,
                            std::vector<std::string>& results,
                            bool emitHeader = true,
                            bool* stateAvailable = nullptr);

  /// Emit each shot with a separate worker runtime.
  /// The view is valid only during the callback.
  int64_t sampleWithRuntime(Runtime& runtime, size_t shots,
                            llvm::function_ref<void(std::string_view)> emit,
                            bool emitHeader = true,
                            bool* stateAvailable = nullptr);

  /// Whether sampling can use one retained terminal state.
  [[nodiscard]] bool canSampleTerminal() const;

  [[nodiscard]] auto runtime() -> Runtime&;

private:
  std::unique_ptr<Runtime> runtime_;
  std::unique_ptr<llvm::orc::LLJIT> jit_;
  EntryPointFn* entryPointFn_ = nullptr;
  std::optional<std::vector<uintptr_t>> samplingOutputs_;
  bool initializesRuntime_ = false;
  bool shareCompiledCode_ = false;
  bool sampleBranches_ = false;
  size_t quantumCallSites_ = 0;
  Execution execution_;

  int64_t runWithRuntime(Runtime& runtime);
  int64_t
  sampleWithRuntimeImpl(Runtime& runtime, size_t shots,
                        llvm::function_ref<void(std::string&&)> emitOwned,
                        llvm::function_ref<void(std::string_view)> emitBorrowed,
                        bool emitHeader, bool* stateAvailable,
                        std::optional<SampledOutput>* terminalOutput);

  /// Initializes the native target, asm printer and asm parser.
  ///
  /// Safe to call multiple times; the work runs only on the first call.
  static void initNativeTargets();

  /// Parses LLVM IR (textual or bitcode) from @p irBytes.
  ///
  /// Uses the session's own thread-safe context. @p bufferName is used in
  /// diagnostics.
  static llvm::Expected<llvm::orc::ThreadSafeModule>
  loadModuleFromMemory(llvm::StringRef irBytes, llvm::StringRef bufferName);

  /// Prepares the session to run the program.
  ///
  /// - Validates the loaded module.
  /// - Optionally truncates the entry point at its first irreversible operation
  ///   (for @c Execution::StateExtraction).
  /// - Builds the @c LLJIT instance
  /// - Registers QIR runtime symbols
  /// - Resolves the selected QIR entry point.
  ///
  /// @throws std::runtime_error if loading failed or the JIT cannot start.
  void initialize(llvm::Expected<llvm::orc::ThreadSafeModule> llvmModule,
                  Execution execution);

  /// Tears down the @c LLJIT.
  void deinitialize() const;
};

} // namespace qir
