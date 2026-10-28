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
#include "mqt/Compiler/ResultLayout.h"
#include "mqt/Compiler/Target.h"
#include "mqt/Compiler/TargetEnvironment.h"
#include "qdmi/common/Common.hpp"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace qdmi {
class Device;
class Job;
class Operation;
} // namespace qdmi

namespace mlir {

/// One alternative in the temporary operation-parameter metadata contract.
struct OperationParameterConstraints {
  std::vector<std::optional<double>> fixed;
  std::vector<
      std::optional<CompilerTarget::OperationCapability::ParameterBounds>>
      bounds;
};

/// Read `mqt.operation-parameters.v1` JSON from operation CUSTOM2.
///
/// Empty means absent or foreign metadata. Recognized malformed metadata fails.
/// Remove this compatibility contract when QDMI standardizes parameter domains.
[[nodiscard]] llvm::Expected<std::vector<OperationParameterConstraints>>
operationParameterConstraints(const qdmi::Operation& operation);

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

/// Open a QDMI device by stable ID and snapshot it as a compiler target.
///
/// The returned target owns all queried metadata. QDMI query errors are
/// returned to the caller.
[[nodiscard]] llvm::Expected<CompilerTarget>
compilerTargetFromDeviceId(std::string_view deviceId);

/// List the stable IDs visible to a fresh QDMI session.
///
/// QDMI session discovery errors are returned to the caller.
[[nodiscard]] llvm::Expected<std::vector<std::string>>
registeredQDMIDeviceIds();

/// Compiler-supported capabilities of a QDMI program format.
[[nodiscard]] llvm::Expected<PayloadSpecification>
payloadSpecificationForProgramFormat(QDMI_Program_Format format);

/// Snapshot the device and select its executable payload before compilation.
///
/// Preference: Adaptive QIR (binary, text), OpenQASM 3, IQM JSON, Base QIR
/// (binary, text).
[[nodiscard]] llvm::Expected<TargetEnvironment> targetEnvironmentFromDevice(
    const qdmi::Device& device,
    std::optional<QDMI_Program_Format> format = std::nullopt);

/// Check equality of legality-relevant contracts, ignoring calibration data.
///
/// Ordered site IDs and ordered operation operands retain their meaning.
/// IQM JSON also requires unchanged site names.
[[nodiscard]] llvm::Error
validateTargetCompatibility(const TargetEnvironment& compiled,
                            const TargetEnvironment& destination);

/// Compiled payload and the target used to check submission compatibility.
class CompiledProgram {
public:
  /// Compile and serialize for one selected hardware/payload contract.
  [[nodiscard]] static llvm::Expected<CompiledProgram>
  compile(CompilerInput program, const TargetEnvironment& environment,
          const CompilationOptions& options = {},
          SerializationProfile profile = SerializationProfile::Default);

  /// Serialize an already native program without routing or gate synthesis.
  ///
  /// Receiver profiles normalize terminal measurements and retain the logical
  /// output map. Unsupported feedback or nonterminal readout is rejected.
  [[nodiscard]] static llvm::Expected<CompiledProgram>
  serialize(QCProgram program, const TargetEnvironment& environment,
            SerializationProfile profile = SerializationProfile::Default,
            std::optional<llvm::ArrayRef<size_t>> siteIndices = std::nullopt);

  /// Snapshot exact operation placements and serialize for a QDMI device.
  ///
  /// Optional site indices map positional input wires to ordered target sites.
  /// Omit them when the QC program already contains target site IDs.
  [[nodiscard]] static llvm::Expected<CompiledProgram>
  serialize(QCProgram program, const qdmi::Device& device,
            std::optional<QDMI_Program_Format> format = std::nullopt,
            SerializationProfile profile = SerializationProfile::Default,
            std::optional<llvm::ArrayRef<size_t>> siteIndices = std::nullopt);

  /// Logical binary output reconstruction; absent for an identity layout.
  [[nodiscard]] const std::optional<ResultLayout>&
  resultLayout() const noexcept {
    return resultLayout_;
  }

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
                  QDMI_Program_Format format,
                  std::optional<ResultLayout> resultLayout = std::nullopt);
  TargetEnvironment environment_;
  std::string payload_;
  QDMI_Program_Format format_;
  std::optional<ResultLayout> resultLayout_;
};

/// Compile for a QDMI device.
[[nodiscard]] llvm::Expected<CompiledProgram>
compileProgram(CompilerInput program, const qdmi::Device& device,
               std::optional<QDMI_Program_Format> format = std::nullopt,
               const CompilationOptions& options = {},
               SerializationProfile profile = SerializationProfile::Default);

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
    const qdmi::Device& device, CompilerInput program, int64_t numShots = 1024,
    std::optional<QDMI_Program_Format> format = std::nullopt,
    const std::optional<qdmi::CustomJobParameter>& custom1 = std::nullopt,
    const std::optional<qdmi::CustomJobParameter>& custom2 = std::nullopt,
    const std::optional<qdmi::CustomJobParameter>& custom3 = std::nullopt,
    const std::optional<qdmi::CustomJobParameter>& custom4 = std::nullopt,
    const std::optional<qdmi::CustomJobParameter>& custom5 = std::nullopt,
    const CompilationOptions& options = {});

} // namespace mlir
