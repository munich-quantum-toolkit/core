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

#include "mlir/Compiler/Target.h"

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/Error.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/AnalysisManager.h>

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace mlir {

class MLIRContext;
enum class ProgramFormat : uint8_t;

namespace mqt {
class PayloadSpecAttr;
class TargetEnvAttr;
} // namespace mqt

/// Payload representation encoding.
enum class PayloadEncoding : uint8_t { Text, Binary };

/// Exact identity of one executable payload representation.
struct PayloadFormat {
  std::string id;
  std::string version;
  std::string profile;
  PayloadEncoding encoding = PayloadEncoding::Text;

  friend bool operator==(const PayloadFormat&, const PayloadFormat&) = default;
};

/// One typed constraint on a payload capability.
struct ProgramConstraint {
  std::string id;
  uint64_t value = 0;

  friend bool operator==(const ProgramConstraint&,
                         const ProgramConstraint&) = default;
};

/// One extensible payload execution capability.
struct ProgramCapability {
  std::string id;
  uint64_t value = 0;
  std::vector<ProgramConstraint> constraints;

  friend bool operator==(const ProgramCapability&,
                         const ProgramCapability&) = default;
};

/// Context-free selected payload execution contract.
///
/// Producers must include every effective capability, including
/// payload-format baselines. The knowledge bit states whether the list also
/// contains all optional device capabilities.
class PayloadSpecification {
public:
  /// Create and validate a selected payload specification.
  [[nodiscard]] static llvm::Expected<PayloadSpecification>
  create(PayloadFormat format, std::vector<ProgramCapability> capabilities = {},
         bool optionalCapabilitiesKnown = false);

  /// Reconstruct a context-free value from its typed MLIR attribute.
  [[nodiscard]] static llvm::Expected<PayloadSpecification>
  create(mqt::PayloadSpecAttr attribute);

  /// Return the exact payload format.
  [[nodiscard]] const PayloadFormat& format() const noexcept;

  /// Return the compiler output selected by the payload format.
  [[nodiscard]] llvm::Expected<ProgramFormat> compilerOutput() const;

  /// Return effective payload capabilities in reported order.
  [[nodiscard]] llvm::ArrayRef<ProgramCapability> capabilities() const noexcept;

  /// Return whether optional capability metadata is complete.
  [[nodiscard]] bool optionalCapabilitiesKnown() const noexcept;

  /// Materialize the selected contract as a typed MLIR attribute.
  [[nodiscard]] mqt::PayloadSpecAttr materialize(MLIRContext& context) const;

private:
  PayloadSpecification(PayloadFormat format,
                       std::vector<ProgramCapability> capabilities,
                       bool optionalCapabilitiesKnown);

  PayloadFormat format_;
  std::vector<ProgramCapability> capabilities_;
  bool optionalCapabilitiesKnown_;
};

/// Context-free hardware target and selected payload specification.
class TargetEnvironment {
public:
  TargetEnvironment(const CompilerTarget& target, PayloadSpecification payload);

  /// Reconstruct the context-free pair from its typed MLIR attribute.
  [[nodiscard]] static llvm::Expected<TargetEnvironment>
  create(mqt::TargetEnvAttr attribute);

  /// Return the compiler target.
  [[nodiscard]] const CompilerTarget& target() const noexcept;

  /// Return the selected payload specification.
  [[nodiscard]] const PayloadSpecification&
  payloadSpecification() const noexcept;

  /// Materialize the pair as a typed MLIR attribute.
  [[nodiscard]] mqt::TargetEnvAttr materialize(MLIRContext& context) const;

private:
  CompilerTarget target_;
  PayloadSpecification payloadSpecification_;
};

/// Attach the canonical typed target environment to a module.
void attachTargetEnvironment(ModuleOp moduleOp,
                             const TargetEnvironment& environment);

/// Cached, validated view of a module's canonical target environment.
class TargetEnvironmentAnalysis {
public:
  explicit TargetEnvironmentAnalysis(Operation* operation);

  /// Return whether the module contains a valid target environment.
  [[nodiscard]] explicit operator bool() const noexcept;

  /// Return the target environment. Requires a valid analysis.
  [[nodiscard]] const TargetEnvironment& environment() const noexcept;

  /// Return the validation error, or an empty string for a valid analysis.
  [[nodiscard]] llvm::StringRef error() const noexcept;

  /// Keep the cached values while the canonical attribute is unchanged.
  [[nodiscard]] bool
  isInvalidated(const AnalysisManager::PreservedAnalyses& analyses) const;

private:
  ModuleOp moduleOp_;
  Attribute attribute_;
  std::optional<TargetEnvironment> environment_;
  std::string error_;
};

} // namespace mlir
