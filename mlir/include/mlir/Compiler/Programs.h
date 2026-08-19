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

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace mlir {

class QCProgram;
class QCOProgram;
class JeffProgram;
class OpenQASMProgram;
class QIRProgram;
class CompilerTarget;

/**
 * @brief The QIR profile represented by a QIR program.
 */
enum class QIRProfile : uint8_t {
  /// The QIR Base Profile.
  Base,
  /// The QIR Adaptive Profile.
  Adaptive,
};

/**
 * @brief Formats accepted and produced by the default compiler pipeline.
 */
enum class ProgramFormat : uint8_t {
  /// QC directly after frontend import, without any compiler pass.
  QCImport,
  /// QCO immediately after conversion, before cleanup and optimization.
  QCO,
  /// QCO after the default or user-supplied optimization pipeline.
  QCOOptimized,
  /// QC after the optimized QCO round trip.
  QC,
  /// Portable OpenQASM after the optimized QCO round trip.
  OpenQASM3,
  /// Serializable `jeff` MLIR.
  Jeff,
  /// QIR for the Base Profile.
  QIRBase,
  /// QIR for the Adaptive Profile.
  QIRAdaptive,
};

/**
 * @brief A move-aware MLIR program with a shared dialect context.
 *
 * @details Programs own their module and keep the context alive for its full
 * lifetime. Dialect-changing operations consume an rvalue program, making
 * ownership transfer explicit and avoiding expensive implicit cloning.
 */
class Program {
public:
  Program(const Program&) = delete;
  Program& operator=(const Program&) = delete;
  Program(Program&&) noexcept = default;
  Program& operator=(Program&&) noexcept = default;
  virtual ~Program() = default;

  /// Check whether this program still owns a module.
  [[nodiscard]] bool isValid() const noexcept;

  /// Return the program as textual MLIR.
  [[nodiscard]] std::string str() const;

  /**
   * @brief Borrow the owned MLIR module.
   *
   * @details The returned operation remains valid while this program owns its
   * module. Consuming or destroying the program invalidates the operation.
   */
  [[nodiscard]] ModuleOp module() const;

protected:
  struct Storage {
    std::shared_ptr<MLIRContext> context;
    OwningOpRef<ModuleOp> mod;
  };

  explicit Program(Storage storage);

  /// Return the owned module. Requires a valid program.
  [[nodiscard]] ModuleOp mod() const;

  /// Clone the owned module while sharing its immutable dialect context.
  [[nodiscard]] Storage cloneStorage() const;

  /// Transfer module ownership to a new program.
  [[nodiscard]] Storage releaseStorage() &&;

private:
  Storage storage_;
};

/**
 * @brief An owned OpenQASM source program.
 */
class OpenQASMProgram final {
public:
  explicit OpenQASMProgram(std::string source) : source_(std::move(source)) {}

  /// Return the OpenQASM source.
  [[nodiscard]] const std::string& source() const noexcept;

  /// Return the OpenQASM source.
  [[nodiscard]] const std::string& str() const noexcept;

  /// Write the OpenQASM source to a file.
  [[nodiscard]] bool write(const std::filesystem::path& path) const;

private:
  std::string source_;
};

/**
 * @brief A QC program with reference semantics.
 */
class QCProgram final : public Program {
public:
  explicit QCProgram(Storage storage) : Program(std::move(storage)) {}

  /// Parse QC MLIR assembly.
  [[nodiscard]] static std::optional<QCProgram>
  fromMLIRString(std::string_view source);

  /// Parse QC MLIR assembly from a file.
  [[nodiscard]] static std::optional<QCProgram>
  fromMLIRFile(const std::filesystem::path& path);

  /// Translate OpenQASM 3 source to QC.
  [[nodiscard]] static std::optional<QCProgram>
  fromQASMString(std::string_view source);

  /// Translate an OpenQASM 3 file to QC.
  [[nodiscard]] static std::optional<QCProgram>
  fromQASMFile(const std::filesystem::path& path);

  /**
   * @brief Take ownership of an MLIR module that contains a QC program.
   *
   * @details The context must own every dialect referenced by the module and
   * must remain the module's context. The factory verifies the module and
   * requires at least one operation from the QC dialect.
   */
  [[nodiscard]] static std::optional<QCProgram>
  fromModule(std::shared_ptr<MLIRContext> context,
             OwningOpRef<ModuleOp> moduleOp);
  /// Create an independent QC program copy.
  [[nodiscard]] QCProgram copy() const;

  /// Run the standard QC cleanup passes in place.
  [[nodiscard]] bool cleanup();

  /// Normalize scoped global phases in place.
  [[nodiscard]] bool normalizeGlobalPhases();

  /// Translate this program to portable OpenQASM without consuming it.
  [[nodiscard]] std::optional<OpenQASMProgram> toOpenQASM3() const;

  /// Consume this program and convert it to QCO.
  [[nodiscard]] std::optional<QCOProgram> intoQCO() &&;

  /// Consume this program and lower it to QIR.
  [[nodiscard]] std::optional<QIRProgram> intoQIR(QIRProfile profile) &&;

  /**
   * @brief Count the gates in the program.
   *
   * @details Any operation that implements the `UnitaryOpInterface` is counted.
   * The count includes operations in every structured control-flow region once,
   * regardless of how often the region executes. Operations within modifiers
   * are not counted recursively, and barriers are skipped.
   */
  [[nodiscard]] size_t numGates() const;

  /**
   * @brief Count the single-qubit gates in the program.
   *
   * @details Any operation that implements the `UnitaryOpInterface` and acts on
   * one qubit is counted. The count includes operations in every structured
   * control-flow region once, regardless of how often the region executes.
   * Operations within modifiers are not counted recursively, and barriers are
   * skipped.
   */
  [[nodiscard]] size_t numSingleQubitGates() const;

  /**
   * @brief Count the two-qubit gates in the program.
   *
   * @details Any operation that implements the `UnitaryOpInterface` and acts on
   * two qubits is counted. The count includes operations in every structured
   * control-flow region once, regardless of how often the region executes.
   * Operations within modifiers are not counted recursively, and barriers are
   * skipped.
   */
  [[nodiscard]] size_t numTwoQubitGates() const;
};

/**
 * @brief A QCO program with value semantics.
 */
class QCOProgram final : public Program {
public:
  /// Parse QCO MLIR assembly.
  [[nodiscard]] static std::optional<QCOProgram>
  fromMLIRString(std::string_view source);

  /// Parse QCO MLIR assembly from a file.
  [[nodiscard]] static std::optional<QCOProgram>
  fromMLIRFile(const std::filesystem::path& path);

  /**
   * @brief Take ownership of an MLIR module that contains a QCO program.
   *
   * @details The context must own every dialect referenced by the module and
   * must remain the module's context. The factory verifies the module, requires
   * at least one operation from the QCO dialect, and verifies QCO linearity.
   */
  [[nodiscard]] static std::optional<QCOProgram>
  fromModule(std::shared_ptr<MLIRContext> context,
             OwningOpRef<ModuleOp> moduleOp);

  /// Create an independent QCO program copy.
  [[nodiscard]] QCOProgram copy() const;

  /// Run the standard QCO cleanup passes in place.
  [[nodiscard]] bool cleanup();

  /// Normalize scoped global phases in place.
  [[nodiscard]] bool normalizeGlobalPhases();

  /// Run an MLIR textual QCO pass pipeline in place.
  [[nodiscard]] bool runPassPipeline(std::string_view pipeline,
                                     bool enableTiming = false,
                                     bool enableStatistics = false);

  /// Merge consecutive single-qubit rotation gates.
  [[nodiscard]] bool mergeSingleQubitRotationGates();

  /// Fuse single-qubit unitary runs into the selected Euler basis.
  [[nodiscard]] bool fuseSingleQubitUnitaryRuns(std::string_view basis = "zyz");

  /// Unroll loops containing quantum operations.
  [[nodiscard]] bool unrollQuantumLoops(int64_t factor = -1);

  /// Lift Hadamard gates away from measurements.
  [[nodiscard]] bool liftHadamards();

  /// Reuse independent single-qubit allocations.
  [[nodiscard]] bool reuseQubits();

  /// Prepare the program for qubit reuse and reuse eligible qubits.
  [[nodiscard]] bool runQubitReusePipeline();

  /// Decompose controlled X/Z/SWAP gates, `qco.rccx`, and constant-angle phase
  /// gates that act on at least @p minQubits qubits (@p minQubits must be at
  /// least 3; default 3 means wider than two-qubit).
  [[nodiscard]] bool decomposeMultiControlled(uint64_t minQubits = 3);

  /// Compile this program for a target in place.
  ///
  /// Do not rely on the program contents if compilation fails.
  [[nodiscard]] bool compileForTarget(const CompilerTarget& target,
                                      bool enableTiming = false,
                                      bool enableStatistics = false);

  /// Consume this program and convert it to QC.
  [[nodiscard]] std::optional<QCProgram> intoQC() &&;

  /// Consume this program and convert it to `jeff` MLIR.
  [[nodiscard]] std::optional<JeffProgram> intoJeff() &&;

  /// Return the entry `func.func` (`main` if present, else the first function).
  [[nodiscard]] std::optional<func::FuncOp> entryFunc() const;

private:
  friend class QCProgram;
  friend class JeffProgram;

  explicit QCOProgram(Storage storage) : Program(std::move(storage)) {}
  [[nodiscard]] bool hasValidLinearity() const;
};

/**
 * @brief A serializable `jeff` program.
 */
class JeffProgram final : public Program {
public:
  explicit JeffProgram(Storage storage) : Program(std::move(storage)) {}

  /// Deserialize a `jeff` binary file.
  [[nodiscard]] static std::optional<JeffProgram>
  fromFile(const std::filesystem::path& path);

  /// Deserialize a `jeff` binary buffer.
  [[nodiscard]] static std::optional<JeffProgram>
  fromBytes(std::span<const std::byte> bytes);

  /// Create an independent `jeff` program copy.
  [[nodiscard]] JeffProgram copy() const;

  /// Run the standard `jeff` cleanup passes in place.
  [[nodiscard]] bool cleanup();

  /// Serialize this program to a binary `jeff` buffer.
  [[nodiscard]] std::vector<std::byte> toBytes() const;

  /// Serialize this program to a binary `jeff` file.
  [[nodiscard]] bool write(const std::filesystem::path& path) const;

  /// Consume this program and convert it to QCO.
  [[nodiscard]] std::optional<QCOProgram> intoQCO() &&;
};

/**
 * @brief A QIR program.
 */
class QIRProgram final : public Program {
public:
  QIRProgram(Storage storage, QIRProfile profile);

  /// Create an independent QIR program copy.
  [[nodiscard]] QIRProgram copy() const;

  /// Run QIR cleanup passes in place.
  [[nodiscard]] bool cleanup();

  /// Return the selected QIR profile.
  [[nodiscard]] QIRProfile profile() const noexcept;

  /// Translate this QIR MLIR program to LLVM IR text.
  [[nodiscard]] std::optional<std::string> llvmIR() const;

  /// Translate this QIR program to LLVM bitcode in memory.
  [[nodiscard]] std::optional<std::vector<std::byte>> toBitcode() const;

  /// Translate and write this QIR program as LLVM bitcode.
  [[nodiscard]] bool writeBitcode(const std::filesystem::path& path) const;

private:
  QIRProfile profile_;
};

/// Valid input variants for the default compiler pipeline.
using CompilerInput =
    std::variant<QCProgram, QCOProgram, JeffProgram, OpenQASMProgram>;

/// The program variants returned by the default compiler pipeline.
using CompilerProgram = std::variant<QCProgram, QCOProgram, JeffProgram,
                                     OpenQASMProgram, QIRProgram>;

/**
 * @brief Run the coordinated default compiler pipeline.
 *
 * @details The supplied program is consumed. Call `copy()` before this function
 * when the source program must remain available for another pipeline branch.
 */
[[nodiscard]] std::optional<CompilerProgram>
runDefaultPipeline(CompilerInput&& program, ProgramFormat output,
                   const CompilerTarget* target = nullptr,
                   std::string_view qcoPipeline = "mqt-qco-default",
                   bool enableTiming = false, bool enableStatistics = false);

} // namespace mlir
