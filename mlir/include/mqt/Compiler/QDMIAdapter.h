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

#include "mqt/Compiler/Programs.h"
#include "mqt/Compiler/Target.h"
#include "mqt/Compiler/TargetEnvironment.h"
#include "qdmi/common/Common.hpp"

#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
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
[[nodiscard]] llvm::Expected<CompilerTarget>
compilerTargetFromDevice(const qdmi::Device& device);

/// Open a registered QDMI device and snapshot it as a compiler target.
///
/// This adapter contains exceptions from the QDMI C++ API and returns
/// them as LLVM errors. The returned target owns all queried metadata.
[[nodiscard]] llvm::Expected<CompilerTarget>
compilerTargetFromDeviceId(std::string_view deviceId);

/// List the stable IDs of registered QDMI devices.
///
/// This adapter contains exceptions from QDMI registry discovery and
/// returns them as LLVM errors.
[[nodiscard]] llvm::Expected<std::vector<std::string>>
registeredQDMIDeviceIds();

/// Compiler-supported capabilities of a QDMI program format.
[[nodiscard]] llvm::Expected<PayloadSpecification>
payloadSpecificationForProgramFormat(QDMI_Program_Format format);

/// Snapshot the device and select its executable payload before compilation.
/// Preference: Adaptive QIR (binary, text), OpenQASM 3, Base QIR (binary,
/// text).
[[nodiscard]] llvm::Expected<TargetEnvironment> targetEnvironmentFromDevice(
    const qdmi::Device& device,
    std::optional<QDMI_Program_Format> format = std::nullopt);

/// Check equality of legality-relevant contracts, ignoring calibration data.
/// Ordered site IDs and ordered operation operands retain their meaning.
[[nodiscard]] llvm::Error
validateTargetCompatibility(const TargetEnvironment& compiled,
                            const TargetEnvironment& destination);

/// Compiled payload and the target used to check submission compatibility.
class CompiledProgram {
public:
  /// Compile and serialize for one selected hardware/payload contract.
  [[nodiscard]] static llvm::Expected<CompiledProgram>
  compile(CompilerInput&& program, const TargetEnvironment& environment,
          bool enableTiming = false, bool enableStatistics = false);

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
[[nodiscard]] llvm::Expected<CompiledProgram>
compileProgram(CompilerInput&& program, const qdmi::Device& device,
               std::optional<QDMI_Program_Format> format = std::nullopt,
               bool enableTiming = false, bool enableStatistics = false);

/// Validate the destination contract before creating and submitting a QDMI job.
[[nodiscard]] llvm::Expected<qdmi::Job> submitProgram(
    const qdmi::Device& device, const CompiledProgram& program,
    int64_t numShots = 1024,
    const std::optional<qdmi::CustomJobParameter>& custom1 = std::nullopt,
    const std::optional<qdmi::CustomJobParameter>& custom2 = std::nullopt,
    const std::optional<qdmi::CustomJobParameter>& custom3 = std::nullopt,
    const std::optional<qdmi::CustomJobParameter>& custom4 = std::nullopt,
    const std::optional<qdmi::CustomJobParameter>& custom5 = std::nullopt);

/// Compile and submit source using one snapshot of the destination.
[[nodiscard]] llvm::Expected<qdmi::Job> submitProgram(
    const qdmi::Device& device, CompilerInput&& program,
    int64_t numShots = 1024,
    std::optional<QDMI_Program_Format> format = std::nullopt,
    bool enableTiming = false, bool enableStatistics = false,
    const std::optional<qdmi::CustomJobParameter>& custom1 = std::nullopt,
    const std::optional<qdmi::CustomJobParameter>& custom2 = std::nullopt,
    const std::optional<qdmi::CustomJobParameter>& custom3 = std::nullopt,
    const std::optional<qdmi::CustomJobParameter>& custom4 = std::nullopt,
    const std::optional<qdmi::CustomJobParameter>& custom5 = std::nullopt);

} // namespace mlir
