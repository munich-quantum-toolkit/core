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

#include <cstdint>
#include <string>

namespace mlir {

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
 * @brief Runtime feature supported by a target for a specific program format.
 */
enum class ProgramFeature : uint8_t {
  /// Measurement followed by further quantum execution or adaptive result use.
  MidCircuitMeasurement,
  /// Continued use of a qubit after it has been measured.
  MeasuredQubitReuse,
  /// Runtime use of a measurement result beyond terminal reporting or return.
  MeasurementResultUse,
  /// Runtime Boolean computation.
  BooleanComputation,
  /// Runtime integer computation.
  IntegerComputation,
  /// Runtime floating-point computation.
  FloatComputation,
  /// Runtime forward branching.
  ForwardBranching,
  /// Runtime counted iteration.
  CountedIteration,
  /// Runtime condition-terminated looping.
  ConditionalLoop,
  /// Runtime multiway branching.
  MultiwayBranching,
  /// Calls to non-entry-point IR functions.
  IRFunctions,
  /// More than one return point in one function.
  MultipleReturnPoints,
  /// Runtime qubit allocation and release.
  DynamicQubitManagement,
  /// Runtime result allocation and release.
  DynamicResultManagement,
  /// Runtime array support.
  Arrays,
};

/// Encoding of a selected target payload.
enum class PayloadEncoding : uint8_t { Text, Binary };

/**
 * @brief Exact identity of a payload accepted by a compiler target.
 */
struct PayloadDescriptor {
  std::string id;
  std::string version;
  std::string profile;
  PayloadEncoding encoding = PayloadEncoding::Text;

  friend bool operator==(const PayloadDescriptor&,
                         const PayloadDescriptor&) = default;
};

/**
 * @brief One runtime capability and its feature-specific value.
 */
struct ProgramCapability {
  ProgramFeature feature;
  uint64_t value = 0U;

  friend bool operator==(const ProgramCapability&,
                         const ProgramCapability&) = default;
};

/// Return whether @p format is a declared program format.
[[nodiscard]] constexpr bool isValidProgramFormat(const ProgramFormat format) {
  switch (format) {
  case ProgramFormat::QCImport:
  case ProgramFormat::QCO:
  case ProgramFormat::QCOOptimized:
  case ProgramFormat::QC:
  case ProgramFormat::OpenQASM3:
  case ProgramFormat::Jeff:
  case ProgramFormat::QIRBase:
  case ProgramFormat::QIRAdaptive:
    return true;
  }
  return false;
}

/// Return whether target compilation accepts @p format as its final payload.
[[nodiscard]] constexpr bool
isTargetCompilationFormat(const ProgramFormat format) {
  switch (format) {
  case ProgramFormat::QCOOptimized:
  case ProgramFormat::QC:
  case ProgramFormat::OpenQASM3:
  case ProgramFormat::QIRBase:
  case ProgramFormat::QIRAdaptive:
    return true;
  case ProgramFormat::QCImport:
  case ProgramFormat::QCO:
  case ProgramFormat::Jeff:
    return false;
  }
  return false;
}

} // namespace mlir
