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

#include <mlir/Support/LogicalResult.h>

#include <cstdint>
#include <string>

namespace mlir {
class ModuleOp;
class PassManager;

/**
 * @brief Dialect checkpoints of the quantum compiler pipeline
 *
 * @details
 * Identifies the intermediate representations at which the pipeline can be
 * entered or exited. A run may enter at @ref PipelineDialect::QC or
 * @ref PipelineDialect::QCO and exit at any of the four checkpoints.
 */
enum class PipelineDialect : std::uint8_t {
  /// QC dialect (reference semantics)
  QC,
  /// QCO dialect (value semantics)
  QCO,
  /// QIR (Quantum Intermediate Representation)
  QIR,
  /// `jeff` dialect (serializable value semantics)
  Jeff,
};

/**
 * @brief Configuration for the quantum compiler pipeline
 *
 * @details
 * Controls which stages of the compilation pipeline are executed and
 * diagnostic options for profiling and debugging.
 */
struct QuantumCompilerConfig {
  /// Convert to QIR Base Profile at the end of the pipeline
  bool convertToQIRBase = false;

  /// Convert to QIR Adaptive Profile at the end of the pipeline
  bool convertToQIRAdaptive = false;

  /// Record intermediate IR at each stage for debugging/testing
  bool recordIntermediates = false;

  /// Enable pass timing statistics (MLIR builtin)
  bool enableTiming = false;

  /// Enable pass statistics (MLIR builtin)
  bool enableStatistics = false;

  /// Print IR after each stage
  bool printIRAfterAllStages = false;

  /// Disable quaternion-based single-qubit rotation gate merging
  bool disableMergeSingleQubitRotationGates = false;

  /// Enable Hadamard lifting
  bool enableHadamardLifting = false;
};

/**
 * @brief Records the state of IR at various compilation stages
 *
 * @details
 * Stores string representations of the MLIR module at different
 * points in the compilation pipeline. Useful for testing and debugging.
 * All stages are recorded when recordIntermediates is enabled.
 */
struct CompilationRecord {
  std::string afterQCImport;
  std::string afterInitialCanon;
  std::string afterQCOConversion;
  std::string afterQCOCanon;
  std::string afterOptimization;
  std::string afterOptimizationCanon;
  std::string afterQCConversion;
  std::string afterQCCanon;
  std::string afterQIRConversion;
  std::string afterQIRCanon;
  std::string afterJeffConversion;
  std::string afterJeffCanon;
};

/**
 * @brief Main quantum compiler pipeline
 *
 * @details
 * Provides a high-level interface for compiling quantum programs through
 * the MQT compiler infrastructure. A run enters at a source dialect
 * (@ref PipelineDialect::QC or @ref PipelineDialect::QCO) and exits at a
 * target dialect (@ref PipelineDialect::QC, @ref PipelineDialect::QCO,
 * @ref PipelineDialect::QIR, or @ref PipelineDialect::Jeff), executing only
 * the stages in between:
 *
 * 1. QC dialect (reference semantics) - imported from qc::QuantumComputation
 * 2. QC cleanup pipeline
 * 3. QCO dialect (value semantics) - enables SSA-based optimizations
 * 4. QCO cleanup pipeline
 * 5. Quantum optimization passes
 * 6. QCO cleanup pipeline
 * 7a. QC dialect - converted back for backend lowering, then QC cleanup, then
 *     optional QIR lowering and cleanup, or
 * 7b. `jeff` dialect - converted for serialization, then `jeff` cleanup.
 *
 * Following MLIR best practices, simplification and dead-value cleanup are
 * run after each major transformation stage.
 */
class QuantumCompilerPipeline {
public:
  explicit QuantumCompilerPipeline(const QuantumCompilerConfig& config = {})
      : config_(config) {}

  /**
   * @brief Run the complete compilation pipeline on a module
   *
   * @details
   * Convenience wrapper around @ref run that enters at
   * @ref PipelineDialect::QC and exits at @ref PipelineDialect::QIR if a QIR
   * profile is configured, or @ref PipelineDialect::QC otherwise.
   *
   * @param module The MLIR module to compile
   * @param record Optional pointer to record intermediate states
   * @return success() if compilation succeeded, failure() otherwise
   */
  LogicalResult runPipeline(ModuleOp module,
                            CompilationRecord* record = nullptr) const;

  /**
   * @brief Run the compilation stages between two dialect checkpoints
   *
   * @details
   * Executes the enabled compilation stages required to lower @p module from
   * @p from to @p to. If recordIntermediates is enabled in the config,
   * captures an IR snapshot after each stage.
   *
   * The target is required to be a forward lowering target. In particular,
   * QIR requires exactly one QIR profile to be selected in the configuration,
   * while all other targets require no QIR profile.
   *
   * Automatically configures the PassManager with:
   * - Timing statistics if enableTiming is true
   * - Pass statistics if enableStatistics is true
   * - IR printing after each stage if printIRAfterAllStages is true
   *
   * @param module The MLIR module to compile
   * @param from The dialect the module is currently in (QC or QCO)
   * @param to The dialect to lower the module to
   * @param record Optional pointer to record intermediate states
   * @return success() if compilation succeeded, failure() otherwise
   */
  LogicalResult run(ModuleOp module, PipelineDialect from, PipelineDialect to,
                    CompilationRecord* record = nullptr) const;

private:
  /**
   * @brief Configure PassManager with diagnostic options
   *
   * @details
   * Enables timing, statistics, and IR printing based on config flags.
   * Uses MLIR's builtin PassManager configuration methods.
   */
  void configurePassManager(PassManager& pm) const;

  QuantumCompilerConfig config_;
};

/**
 * @brief Utility to capture IR as string
 *
 * @details
 * Prints the MLIR module to a string for recording or comparison.
 *
 * @param module The module to convert to string
 * @return String representation of the IR
 */
std::string captureIR(ModuleOp module);

} // namespace mlir
