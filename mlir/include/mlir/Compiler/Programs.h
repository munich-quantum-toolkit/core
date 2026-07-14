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

#include "mlir/Compiler/CompilerPipeline.h"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <utility>
#include <variant>

namespace qc {
class QuantumComputation;
} // namespace qc

namespace mlir {

class QCProgram;
class QCOProgram;
class JeffProgram;
class QIRProgram;

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
  /// QC after the QCO optimization round trip.
  QC,
  /// Optimized QCO.
  QCO,
  /// Serializable Jeff MLIR.
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

  /** @brief Check whether this program still owns a module. */
  [[nodiscard]] bool isValid() const noexcept;

  /** @brief Return the program as textual MLIR. */
  [[nodiscard]] std::string str() const;

protected:
  struct Storage {
    std::shared_ptr<MLIRContext> context;
    OwningOpRef<ModuleOp> module;
  };

  explicit Program(Storage storage);

  /** @brief Return the owned module, throwing when the program was consumed. */
  [[nodiscard]] ModuleOp module() const;

  /** @brief Clone the owned module while sharing its immutable dialect context.
   */
  [[nodiscard]] Storage cloneStorage() const;

  /** @brief Transfer module ownership to a new program. */
  [[nodiscard]] Storage releaseStorage() &&;

private:
  friend std::variant<QCProgram, QCOProgram, JeffProgram, QIRProgram>
  runDefaultPipeline(
      std::variant<QCProgram, QCOProgram, JeffProgram, QIRProgram>&& program,
      ProgramFormat output, const QuantumCompilerConfig& config);

  Storage storage_;
};

/**
 * @brief A QC-dialect program with reference semantics.
 */
class QCProgram final : public Program {
public:
  explicit QCProgram(Storage storage) : Program(std::move(storage)) {}

  /** @brief Parse QC MLIR assembly. */
  [[nodiscard]] static QCProgram fromMLIRString(const std::string& source);

  /** @brief Parse QC MLIR assembly from a file. */
  [[nodiscard]] static QCProgram
  fromMLIRFile(const std::filesystem::path& path);

  /** @brief Translate OpenQASM 3 source to QC. */
  [[nodiscard]] static QCProgram fromQASMString(const std::string& source);

  /** @brief Translate an OpenQASM 3 file to QC. */
  [[nodiscard]] static QCProgram
  fromQASMFile(const std::filesystem::path& path);

  /** @brief Translate an MQT quantum computation to QC. */
  [[nodiscard]] static QCProgram
  fromQuantumComputation(const ::qc::QuantumComputation& computation);

  /** @brief Create an independent QC program copy. */
  [[nodiscard]] QCProgram copy() const;

  /** @brief Run the standard QC cleanup passes in place. */
  void cleanup();

  /** @brief Consume this program and convert it to QCO. */
  [[nodiscard]] QCOProgram intoQCO() &&;

  /** @brief Consume this program and lower it to QIR. */
  [[nodiscard]] QIRProgram intoQIR(QIRProfile profile) &&;
};

/**
 * @brief A QCO-dialect program with value semantics.
 */
class QCOProgram final : public Program {
public:
  explicit QCOProgram(Storage storage) : Program(std::move(storage)) {}

  /** @brief Parse QCO MLIR assembly. */
  [[nodiscard]] static QCOProgram fromMLIRString(const std::string& source);

  /** @brief Parse QCO MLIR assembly from a file. */
  [[nodiscard]] static QCOProgram
  fromMLIRFile(const std::filesystem::path& path);

  /** @brief Create an independent QCO program copy. */
  [[nodiscard]] QCOProgram copy() const;

  /** @brief Run the standard QCO cleanup passes in place. */
  void cleanup();

  /** @brief Run the standard QCO optimization passes in place. */
  void optimize(bool mergeSingleQubitRotations = true,
                bool enableHadamardLifting = false);

  /** @brief Consume this program and convert it to QC. */
  [[nodiscard]] QCProgram intoQC() &&;

  /** @brief Consume this program and convert it to Jeff MLIR. */
  [[nodiscard]] JeffProgram intoJeff() &&;
};

/**
 * @brief A serializable Jeff-dialect program.
 */
class JeffProgram final : public Program {
public:
  explicit JeffProgram(Storage storage) : Program(std::move(storage)) {}

  /** @brief Deserialize a Jeff binary file. */
  [[nodiscard]] static JeffProgram fromFile(const std::filesystem::path& path);

  /** @brief Deserialize a Jeff binary buffer. */
  [[nodiscard]] static JeffProgram fromBytes(const std::string& bytes);

  /** @brief Create an independent Jeff program copy. */
  [[nodiscard]] JeffProgram copy() const;

  /** @brief Run the standard Jeff cleanup passes in place. */
  void cleanup();

  /** @brief Serialize this program to a binary Jeff buffer. */
  [[nodiscard]] std::string toBytes() const;

  /** @brief Serialize this program to a binary Jeff file. */
  void write(const std::filesystem::path& path) const;

  /** @brief Consume this program and convert it to QCO. */
  [[nodiscard]] QCOProgram intoQCO() &&;
};

/**
 * @brief A QIR-dialect program.
 */
class QIRProgram final : public Program {
public:
  QIRProgram(Storage storage, QIRProfile profile);

  /** @brief Create an independent QIR program copy. */
  [[nodiscard]] QIRProgram copy() const;

  /** @brief Run QIR cleanup passes in place. */
  void cleanup();

  /** @brief Return the selected QIR profile. */
  [[nodiscard]] QIRProfile profile() const noexcept;

  /** @brief Translate this QIR MLIR program to LLVM IR text. */
  [[nodiscard]] std::string llvmIR() const;

private:
  QIRProfile profile_;
};

/** @brief The program variants returned by the default compiler pipeline. */
using CompilerProgram =
    std::variant<QCProgram, QCOProgram, JeffProgram, QIRProgram>;

/**
 * @brief Run the coordinated default compiler pipeline.
 *
 * @details The supplied program is consumed. Call `copy()` before this function
 * when the source program must remain available for another pipeline branch.
 */
[[nodiscard]] CompilerProgram
runDefaultPipeline(CompilerProgram&& program, ProgramFormat output,
                   const QuantumCompilerConfig& config = {});

} // namespace mlir
