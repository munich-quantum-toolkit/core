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

#include "mlir/Support/LLVM.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/StringRef.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>

namespace mlir {
class ConversionTarget;
class ModuleOp;
class Operation;
class PayloadSpecification;
class Value;
class RewritePatternSet;
class TargetEnvironmentAnalysis;
namespace scf {
class ForOp;
} // namespace scf
} // namespace mlir

namespace mlir::mqt {

/// Structured control-flow capabilities of a payload.
enum class PayloadControlFeature : uint8_t {
  ForwardBranching,
  CountedIteration,
  ConditionalLoop,
  MultiwayBranching,
  Count,
};

/// Payload limits shared by structured control-flow legalization and unrolling.
class PayloadControlSupport {
public:
  /// Read an explicit payload, or the attached target environment when empty.
  ///
  /// Diagnose invalid specifications and return no support on failure.
  [[nodiscard]] static std::optional<PayloadControlSupport>
  read(ModuleOp moduleOp, const TargetEnvironmentAnalysis& analysis,
       StringRef specification = {});

  [[nodiscard]] static PayloadControlSupport
  forPayload(const PayloadSpecification& payload);

  [[nodiscard]] bool supportsIndexedQubits() const { return indexedQubits_; }

  /// Validate the condition and branch bodies against forward-branching limits.
  [[nodiscard]] bool coversBranch(Operation* operation, Value condition,
                                  uint64_t depth) const;
  [[nodiscard]] bool coversBranch(Operation* operation, Value condition) const;
  [[nodiscard]] bool coversDepth(PayloadControlFeature feature,
                                 Operation* operation) const;
  [[nodiscard]] bool coversDepth(PayloadControlFeature feature,
                                 uint64_t depth) const;
  [[nodiscard]] bool
  coversIteration(PayloadControlFeature feature, Operation* operation,
                  const std::optional<llvm::APInt>& tripCount) const;
  [[nodiscard]] bool
  coversIteration(PayloadControlFeature feature, uint64_t depth,
                  const std::optional<llvm::APInt>& tripCount) const;
  [[nodiscard]] bool coversMultiwayBranching(Operation* operation,
                                             uint64_t caseCount) const;
  [[nodiscard]] bool coversMultiwayBranching(uint64_t depth,
                                             uint64_t caseCount) const;

  /// Check expansion of a switch with at least one explicit case.
  [[nodiscard]] bool canLowerSwitch(Operation* operation,
                                    uint64_t caseCount) const;

private:
  struct CapabilityGroup {
    bool usable = false;
    std::optional<uint64_t> maxNestingDepth;
    std::optional<uint64_t> maxIterationCount;
    std::optional<uint64_t> maxCaseCount;
    std::optional<uint64_t> maxConditionWidth;
    bool requiresRegisterEquality = false;
    bool unitaryOnlyBranches = false;
  };

  [[nodiscard]] const CapabilityGroup& get(PayloadControlFeature feature) const;
  [[nodiscard]] CapabilityGroup& get(PayloadControlFeature feature);
  [[nodiscard]] static uint64_t controlDepth(Operation* operation);
  [[nodiscard]] static bool isStructuredControl(Operation* operation);
  static void applyConstraint(CapabilityGroup& group,
                              PayloadControlFeature feature, StringRef id,
                              uint64_t value);

  std::array<CapabilityGroup, static_cast<size_t>(PayloadControlFeature::Count)>
      groups_{};
  bool indexedQubits_ = false;
};

/// Return the exact trip count of a loop with literal bounds and step.
[[nodiscard]] std::optional<llvm::APInt>
getExactConstantTripCount(scf::ForOp loop);

/// Legalize native QC control flow against an in-memory payload contract.
[[nodiscard]] LogicalResult
legalizePayloadControlFlow(ModuleOp moduleOp,
                           const PayloadSpecification& payload);

/// Add payload legality rules and switch lowering for SCF control flow.
///
/// Reject other branch interfaces except scf.execute_region. Callers can add
/// dialect-specific rules and patterns before applying partial conversion.
/// The support object must outlive the target and patterns.
void populatePayloadControlFlowPatternsAndLegality(
    ConversionTarget& target, RewritePatternSet& patterns,
    const PayloadControlSupport& support);

} // namespace mlir::mqt
