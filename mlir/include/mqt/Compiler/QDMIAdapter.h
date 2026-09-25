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

#include "mqt/Compiler/CompilationOptions.h"
#include "mqt/Compiler/Programs.h"
#include "mqt/Compiler/Target.h"
#include "mqt/Compiler/TargetEnvironment.h"
#include "mqt/Support/Diagnostics.h"
#include "qdmi/common/Common.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace qdmi {
class Device;
class Job;
} // namespace qdmi

namespace mlir {

/// Snapshot a circuit-model QDMI device as an MQT compiler target.
///
/// The returned target owns all queried metadata and remains valid
/// after the originating device and session have been destroyed. Neutral-atom
/// zone models are not supported. Explicit QDMI site lists must cover every
/// site for one-qubit operations, every undirected topology edge for two-qubit
/// operations, and every ordered tuple of distinct sites for higher arities.
/// Each supported ordered placement carries optional calibration data.
[[nodiscard]] mlir::FailureOr<CompilerTarget>
compilerTargetFromDevice(const qdmi::Device& device);

/// Open a QDMI device by stable ID and snapshot it as a compiler target.
///
/// QDMI failures retain their status in scoped diagnostics. The returned target
/// owns all queried metadata.
[[nodiscard]] mlir::FailureOr<CompilerTarget>
compilerTargetFromDeviceId(std::string_view deviceId);

/// List the stable IDs visible to a fresh QDMI session.
///
/// Discovery failures retain their status in scoped diagnostics.
[[nodiscard]] mlir::FailureOr<std::vector<std::string>>
registeredQDMIDeviceIds();

/// Compiler-supported capabilities of a QDMI program format.
[[nodiscard]] mlir::FailureOr<PayloadSpecification>
payloadSpecificationForProgramFormat(QDMI_Program_Format format);

/// Snapshot the device and select its executable payload before compilation.
/// Preference: Adaptive QIR (binary, text), OpenQASM 3, Base QIR (binary,
/// text).
[[nodiscard]] mlir::FailureOr<TargetEnvironment> targetEnvironmentFromDevice(
    const qdmi::Device& device,
    std::optional<QDMI_Program_Format> format = std::nullopt);

/// Check equality of legality-relevant contracts, ignoring calibration data.
/// Ordered site IDs and ordered operation operands retain their meaning.
[[nodiscard]] mlir::LogicalResult
validateTargetCompatibility(const TargetEnvironment& compiled,
                            const TargetEnvironment& destination);

/// Compiled payload and the target used to check submission compatibility.
class CompiledProgram {
public:
  /// Compile and serialize for one selected hardware/payload contract.
  [[nodiscard]] static mlir::FailureOr<CompiledProgram>
  compile(CompilerInput&& program, const TargetEnvironment& environment,
          const CompilationOptions& options = {});

  [[nodiscard]] const TargetEnvironment& environment() const noexcept {
    return environment_;
  }
  /// Serialized program contents.
  [[nodiscard]] const std::string& payload() const noexcept { return payload_; }
  [[nodiscard]] QDMI_Program_Format programFormat() const noexcept {
    return format_;
  }

private:
  CompiledProgram(TargetEnvironment environment, std::string payload,
                  QDMI_Program_Format format);
  TargetEnvironment environment_;
  std::string payload_;
  QDMI_Program_Format format_;
};

/// Compile for a QDMI device.
[[nodiscard]] mlir::FailureOr<CompiledProgram>
compileProgram(CompilerInput&& program, const qdmi::Device& device,
               std::optional<QDMI_Program_Format> format = std::nullopt,
               const CompilationOptions& options = {});

/// Validate the destination contract before creating and submitting a QDMI job.
[[nodiscard]] mlir::FailureOr<qdmi::Job> submitProgram(
    const qdmi::Device& device, const CompiledProgram& program,
    int64_t numShots = 1024,
    const std::optional<qdmi::CustomJobParameter>& custom1 = std::nullopt,
    const std::optional<qdmi::CustomJobParameter>& custom2 = std::nullopt,
    const std::optional<qdmi::CustomJobParameter>& custom3 = std::nullopt,
    const std::optional<qdmi::CustomJobParameter>& custom4 = std::nullopt,
    const std::optional<qdmi::CustomJobParameter>& custom5 = std::nullopt);

/// Compile and submit source using one snapshot of the destination.
[[nodiscard]] mlir::FailureOr<qdmi::Job> submitProgram(
    const qdmi::Device& device, CompilerInput&& input, int64_t numShots = 1024,
    std::optional<QDMI_Program_Format> format = std::nullopt,
    const std::optional<qdmi::CustomJobParameter>& custom1 = std::nullopt,
    const std::optional<qdmi::CustomJobParameter>& custom2 = std::nullopt,
    const std::optional<qdmi::CustomJobParameter>& custom3 = std::nullopt,
    const std::optional<qdmi::CustomJobParameter>& custom4 = std::nullopt,
    const std::optional<qdmi::CustomJobParameter>& custom5 = std::nullopt,
    const CompilationOptions& options = {});

} // namespace mlir
