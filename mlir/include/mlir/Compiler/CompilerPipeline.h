/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LogicalResult.h>
#include <string>

namespace mlir {
class ModuleOp;

/**
 * @brief Configuration for the quantum compiler pipeline
 *
 * @details
 * Controls which stages of the compilation pipeline are executed and
 * diagnostic options for profiling and debugging.
 */
struct QuantumCompilerConfig {
  /// Run quantum optimization passes (placeholder for future passes)
  bool runOptimization = false;

  /// Convert to QIR at the end of the pipeline
  bool convertToQIR = false;

  /// Record intermediate IR at each stage for debugging/testing
  bool recordIntermediates = false;

  /// Enable pass timing statistics (MLIR builtin)
  bool enableTiming = false;

  /// Enable pass statistics (MLIR builtin)
  bool enableStatistics = false;

  /// Print IR after each pass (MLIR builtin, for debugging)
  bool printIRAfterAll = false;

  /// Print IR after failures only (MLIR builtin)
  bool printIRAfterFailure = false;
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
  std::string afterQuartzImport;
  std::string afterInitialCanon;
  std::string afterFluxConversion;
  std::string afterFluxCanon;
  std::string afterOptimization;
  std::string afterOptimizationCanon;
  std::string afterQuartzConversion;
  std::string afterQuartzCanon;
  std::string afterQIRConversion;
  std::string afterQIRCanon;
};

/**
 * @brief Main quantum compiler pipeline
 *
 * @details
 * Provides a high-level interface for compiling quantum programs through
 * the MQT compiler infrastructure. The pipeline stages are:
 *
 * 1. Quartz dialect (reference semantics) - imported from
 * qc::QuantumComputation
 * 2. Canonicalization + cleanup
 * 3. Flux dialect (value semantics) - enables SSA-based optimizations
 * 4. Canonicalization + cleanup
 * 5. Quantum optimization passes (optional, TODO: to be implemented)
 * 6. Canonicalization + cleanup
 * 7. Quartz dialect - converted back for backend lowering
 * 8. Canonicalization + cleanup
 * 9. QIR (Quantum Intermediate Representation) - optional final lowering
 * 10. Canonicalization + cleanup
 *
 * Following MLIR best practices, canonicalization and dead value removal
 * are always run after each major transformation stage.
 */
class QuantumCompilerPipeline {
public:
  explicit QuantumCompilerPipeline(const QuantumCompilerConfig& config = {})
      : config_(config) {}

  /**
   * @brief Run the complete compilation pipeline on a module
   *
   * @details
   * Executes all enabled compilation stages on the provided MLIR module.
   * If recordIntermediates is enabled in the config, captures IR snapshots
   * at every stage (10 snapshots total for full pipeline).
   *
   * Automatically configures the PassManager with:
   * - Timing statistics if enableTiming is true
   * - Pass statistics if enableStatistics is true
   * - IR printing options if printIRAfterAll or printIRAfterFailure is true
   *
   * @param module The MLIR module to compile
   * @param record Optional pointer to record intermediate states
   * @return success() if compilation succeeded, failure() otherwise
   */
  LogicalResult runPipeline(ModuleOp module,
                            CompilationRecord* record = nullptr) const;

private:
  /**
   * @brief Add canonicalization and cleanup passes
   *
   * @details
   * Always adds the standard MLIR canonicalization pass followed by dead
   * value removal.
   */
  static void addCleanupPasses(PassManager& pm);

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
