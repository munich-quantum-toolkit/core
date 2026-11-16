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

#include "mqt/Compiler/Target.h"
#include "mqt/Dialect/MQT/IR/PayloadVersion.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/AnalysisManager.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

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

/// Text or binary program encoding.
enum class PayloadEncoding : uint8_t { Text, Binary };

/// Selected program format, accepted versions, profile, and encoding.
struct PayloadFormat {
  std::string id;
  std::string version;
  std::string profile;
  PayloadEncoding encoding = PayloadEncoding::Text;

  friend bool operator==(const PayloadFormat&, const PayloadFormat&) = default;
};

/// One limit on a supported program feature.
struct ProgramConstraint {
  /// IDs of control-flow constraints understood by the compiler.
  static constexpr llvm::StringLiteral MAX_NESTING_DEPTH =
      "max-control-flow-nesting-depth";
  static constexpr llvm::StringLiteral MAX_ITERATION_COUNT =
      "max-iteration-count";
  static constexpr llvm::StringLiteral MAX_CASE_COUNT = "max-case-count";
  static constexpr llvm::StringLiteral MAX_CONDITION_WIDTH =
      "max-condition-width";
  static constexpr llvm::StringLiteral REQUIRES_REGISTER_EQUALITY =
      "requires-register-equality";
  static constexpr llvm::StringLiteral UNITARY_ONLY_BRANCHES =
      "unitary-only-branches";

  std::string id;
  uint64_t value = 0;

  friend bool operator==(const ProgramConstraint&,
                         const ProgramConstraint&) = default;
};

/// One program feature supported for execution.
struct ProgramCapability {
  /// IDs of structural control-flow capabilities understood by the compiler.
  static constexpr llvm::StringLiteral FORWARD_BRANCHING = "forward-branching";
  static constexpr llvm::StringLiteral COUNTED_ITERATION = "counted-iteration";
  static constexpr llvm::StringLiteral CONDITIONAL_LOOP = "conditional-loop";
  static constexpr llvm::StringLiteral MULTIWAY_BRANCHING =
      "multiway-branching";

  static constexpr llvm::StringLiteral CLASSICAL_ASSIGNMENT =
      "classical-assignment";

  std::string id;
  uint64_t value = 0;
  std::vector<ProgramConstraint> constraints;

  friend bool operator==(const ProgramCapability&,
                         const ProgramCapability&) = default;
};

/// Whether a capability describes classical control flow or assignment.
[[nodiscard]] bool isControlFlowCapability(llvm::StringRef id);

/// The format and program features selected for execution.
///
/// This value is independent of an MLIR context. It describes one format and
/// the features a program may use, such as branches or loops. The capability
/// list must include the format's baseline features. optionalCapabilitiesKnown
/// states whether the list also contains all optional device capabilities.
class PayloadSpecification {
public:
  /// Return compiler-supported capabilities for OpenQASM 3 or a QIR profile.
  ///
  /// Optional device capabilities remain unknown until supplied by a target.
  [[nodiscard]] static llvm::Expected<PayloadSpecification>
  forProgramFormat(ProgramFormat format,
                   PayloadEncoding encoding = PayloadEncoding::Text);

  /// Create and validate an execution format and its supported features.
  ///
  /// Versions accept numeric ceilings or comma-conjoined >=, <, <= bounds.
  /// Bare versions retain a same-major ceiling; omitted components are zero.
  /// An empty version identifies an unversioned format.
  [[nodiscard]] static llvm::Expected<PayloadSpecification>
  create(PayloadFormat format, std::vector<ProgramCapability> capabilities = {},
         bool optionalCapabilitiesKnown = false);

  /// Reconstruct a context-free value from its typed MLIR attribute.
  [[nodiscard]] static llvm::Expected<PayloadSpecification>
  create(mqt::PayloadSpecAttr attribute);

  /// Return the selected program format and accepted receiver versions.
  [[nodiscard]] const PayloadFormat& format() const noexcept;

  /// Return the parsed receiver interval, or no interval for unversioned
  /// formats.
  [[nodiscard]] const std::optional<mqt::PayloadVersionRange>&
  versionRange() const noexcept;

  /// Return the compiler output selected by the program format.
  [[nodiscard]] llvm::Expected<ProgramFormat> compilerOutput() const;

  /// Return the supported program features in reported order.
  [[nodiscard]] llvm::ArrayRef<ProgramCapability> capabilities() const noexcept;

  /// Return whether optional capability metadata is complete.
  [[nodiscard]] bool optionalCapabilitiesKnown() const noexcept;

  /// Restrict program features to those supported by both specifications.
  ///
  /// Keep this format. Across different formats, only structural control-flow
  /// capabilities are comparable; format-specific capabilities remain
  /// unchanged.
  [[nodiscard]] PayloadSpecification
  restrictCapabilities(const PayloadSpecification& other) const;

  /// Whether generated switches may have arbitrary nesting and case counts.
  [[nodiscard]] bool supportsUnrestrictedMultiwayBranching() const noexcept;

  /// Store the selected format and supported features in a typed MLIR
  /// attribute.
  [[nodiscard]] mqt::PayloadSpecAttr materialize(MLIRContext& context) const;

private:
  PayloadSpecification(PayloadFormat format,
                       std::optional<mqt::PayloadVersionRange> versions,
                       std::vector<ProgramCapability> capabilities,
                       bool optionalCapabilitiesKnown);

  PayloadFormat format_;
  std::optional<mqt::PayloadVersionRange> versionRange_;
  std::vector<ProgramCapability> capabilities_;
  bool optionalCapabilitiesKnown_;
};

/// Hardware target and execution format, independent of an MLIR context.
class TargetEnvironment {
public:
  TargetEnvironment(const CompilerTarget& target, PayloadSpecification payload);

  /// Reconstruct the context-free pair from its typed MLIR attribute.
  [[nodiscard]] static llvm::Expected<TargetEnvironment>
  create(mqt::TargetEnvAttr attribute);

  /// Return the compiler target.
  [[nodiscard]] const CompilerTarget& target() const noexcept;

  /// Return the selected execution format and supported program features.
  [[nodiscard]] const PayloadSpecification&
  payloadSpecification() const noexcept;

  /// Whether indexed qubits can retain runtime addresses through placement.
  ///
  /// Requires Adaptive QIR and an all-to-all target. Site-restricted extras
  /// require a selected synthesis basis usable at unknown sites; advertised
  /// measurement/reset operations must also support unknown sites. The
  /// execution format's other control-flow limits still apply.
  [[nodiscard]] bool supportsIndexedQubits() const noexcept;

  /// Materialize the pair as a typed MLIR attribute.
  [[nodiscard]] mqt::TargetEnvAttr materialize(MLIRContext& context) const;

private:
  CompilerTarget target_;
  PayloadSpecification payloadSpecification_;
};

/// Attach the canonical typed target environment to a module.
///
/// Abandons any execution-environment restoration saved by staged compilation.
void attachTargetEnvironment(ModuleOp moduleOp,
                             const TargetEnvironment& environment);

/// Cached, validated view of a module's canonical target environment.
class TargetEnvironmentAnalysis {
public:
  /// Prefer the prepared environment; otherwise decode the module attribute.
  explicit TargetEnvironmentAnalysis(
      Operation* operation, const TargetEnvironment* environment = nullptr);

  /// Attach and cache a prepared environment without rebuilding its target.
  void initialize(const TargetEnvironment& environment);

  /// Return whether the module contains a valid target environment.
  [[nodiscard]] explicit operator bool() const noexcept;

  /// Return the target environment.
  ///
  /// Requires a valid analysis.
  [[nodiscard]] const TargetEnvironment& environment() const noexcept;

  /// Return the validation error, or an empty string for a valid analysis.
  [[nodiscard]] llvm::StringRef error() const noexcept;

  /// Keep the cached values while the canonical attribute is unchanged.
  [[nodiscard]] bool
  isInvalidated(const AnalysisManager::PreservedAnalyses& analyses) const;

private:
  /// Decode IR only when no prepared environment was supplied.
  void resolve() const;

  ModuleOp moduleOp_;
  Attribute attribute_;
  mutable std::optional<TargetEnvironment> environment_;
  mutable std::string error_;
};

} // namespace mlir
