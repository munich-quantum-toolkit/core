/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Compiler/TargetEnvironment.h"

#include "mlir/Compiler/Programs.h"
#include "mlir/Compiler/Target.h"
#include "mlir/Dialect/MQT/IR/MQTAttributes.h"

#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/ADT/Twine.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/Errc.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/VersionTuple.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Pass/AnalysisManager.h>
#include <mlir/Support/LLVM.h>

#include <cassert>
#include <cstdint>
#include <utility>
#include <vector>

namespace mlir {

[[nodiscard]] static llvm::Error invalidPayload(const llvm::Twine& message) {
  return llvm::createStringError(llvm::errc::invalid_argument,
                                 "Invalid payload specification: " + message);
}

[[nodiscard]] static bool isCanonicalVersion(const llvm::StringRef version) {
  llvm::VersionTuple parsed;
  return !parsed.tryParse(version) && parsed.getMinor() &&
         parsed.getSubminor() && !parsed.getBuild() &&
         parsed.getAsString() == version;
}

[[nodiscard]] static bool containsNull(const llvm::StringRef value) {
  return value.contains('\0');
}

llvm::Expected<PayloadSpecification>
PayloadSpecification::create(PayloadFormat format,
                             std::vector<ProgramCapability> capabilities,
                             const bool optionalCapabilitiesKnown) {
  if (format.id.empty() || format.version.empty()) {
    return invalidPayload("Payload format requires an ID and version");
  }
  if (containsNull(format.id) || containsNull(format.version) ||
      containsNull(format.profile)) {
    return invalidPayload(
        "Payload format fields must not contain null characters");
  }
  if (!isCanonicalVersion(format.version)) {
    return invalidPayload(
        "Payload format version must use canonical major.minor.patch");
  }
  switch (format.encoding) {
  case PayloadEncoding::Text:
  case PayloadEncoding::Binary:
    break;
  default:
    return invalidPayload("Payload format encoding is invalid");
  }

  llvm::SmallDenseSet<std::pair<llvm::StringRef, uint64_t>> seenCapabilities;
  seenCapabilities.reserve(capabilities.size());
  for (const ProgramCapability& capability : capabilities) {
    if (capability.id.empty()) {
      return invalidPayload("Program capability ID must not be empty");
    }
    if (containsNull(capability.id)) {
      return invalidPayload(
          "Program capability ID must not contain a null character");
    }
    const auto capabilityKey =
        std::pair(llvm::StringRef(capability.id), capability.value);
    if (!seenCapabilities.insert(capabilityKey).second) {
      return invalidPayload("Payload specification contains a duplicate "
                            "capability ID/value pair");
    }

    llvm::SmallDenseSet<llvm::StringRef> seenConstraints;
    seenConstraints.reserve(capability.constraints.size());
    for (const ProgramConstraint& constraint : capability.constraints) {
      if (constraint.id.empty()) {
        return invalidPayload("Program constraint ID must not be empty");
      }
      if (containsNull(constraint.id)) {
        return invalidPayload(
            "Program constraint ID must not contain a null character");
      }
      if (!seenConstraints.insert(constraint.id).second) {
        return invalidPayload(
            "Program capability contains a duplicate constraint ID");
      }
    }
  }

  return PayloadSpecification(std::move(format), std::move(capabilities),
                              optionalCapabilitiesKnown);
}

llvm::Expected<PayloadSpecification>
PayloadSpecification::create(const mqt::PayloadSpecAttr attribute) {
  if (!attribute) {
    return invalidPayload("Payload specification attribute must not be null");
  }
  const auto formatAttr = attribute.getFormat();
  PayloadEncoding encoding = PayloadEncoding::Text;
  switch (formatAttr.getEncoding()) {
  case mqt::PayloadEncoding::Text:
    encoding = PayloadEncoding::Text;
    break;
  case mqt::PayloadEncoding::Binary:
    encoding = PayloadEncoding::Binary;
    break;
  default:
    return invalidPayload("Payload format encoding is invalid");
  }
  PayloadFormat format{
      .id = formatAttr.getId().getValue().str(),
      .version = formatAttr.getVersion().getValue().str(),
      .profile = formatAttr.getProfile().getValue().str(),
      .encoding = encoding,
  };

  std::vector<ProgramCapability> capabilities;
  capabilities.reserve(attribute.getCapabilities().size());
  for (const mqt::ProgramCapabilityAttr capabilityAttr :
       attribute.getCapabilities()) {
    std::vector<ProgramConstraint> constraints;
    constraints.reserve(capabilityAttr.getConstraints().size());
    for (const mqt::ProgramConstraintAttr constraintAttr :
         capabilityAttr.getConstraints()) {
      constraints.emplace_back(constraintAttr.getId().getValue().str(),
                               constraintAttr.getValue());
    }
    capabilities.push_back({
        .id = capabilityAttr.getId().getValue().str(),
        .value = capabilityAttr.getValue(),
        .constraints = std::move(constraints),
    });
  }
  return create(std::move(format), std::move(capabilities),
                attribute.getOptionalCapabilitiesKnown());
}

PayloadSpecification::PayloadSpecification(
    PayloadFormat format, std::vector<ProgramCapability> capabilities,
    const bool optionalCapabilitiesKnown)
    : format_(std::move(format)), capabilities_(std::move(capabilities)),
      optionalCapabilitiesKnown_(optionalCapabilitiesKnown) {}

const PayloadFormat& PayloadSpecification::format() const noexcept {
  return format_;
}

llvm::Expected<ProgramFormat> PayloadSpecification::compilerOutput() const {
  if (format_.id == "openqasm" && format_.version == "3.0.0" &&
      format_.profile.empty() && format_.encoding == PayloadEncoding::Text) {
    return ProgramFormat::OpenQASM3;
  }
  if (format_.id == "qir" && format_.version == "2.1.0") {
    if (format_.profile == "base") {
      return ProgramFormat::QIRBase;
    }
    if (format_.profile == "adaptive") {
      return ProgramFormat::QIRAdaptive;
    }
  }
  return invalidPayload("MQT Compiler cannot emit the selected payload format");
}

llvm::ArrayRef<ProgramCapability>
PayloadSpecification::capabilities() const noexcept {
  return capabilities_;
}

bool PayloadSpecification::optionalCapabilitiesKnown() const noexcept {
  return optionalCapabilitiesKnown_;
}

mqt::PayloadSpecAttr
PayloadSpecification::materialize(MLIRContext& context) const {
  const auto format = mqt::PayloadFormatAttr::get(
      &context, StringAttr::get(&context, format_.id),
      StringAttr::get(&context, format_.version),
      StringAttr::get(&context, format_.profile),
      format_.encoding == PayloadEncoding::Binary ? mqt::PayloadEncoding::Binary
                                                  : mqt::PayloadEncoding::Text);

  llvm::SmallVector<mqt::ProgramCapabilityAttr> capabilities;
  capabilities.reserve(capabilities_.size());
  for (const ProgramCapability& capability : capabilities_) {
    llvm::SmallVector<mqt::ProgramConstraintAttr> constraints;
    constraints.reserve(capability.constraints.size());
    for (const ProgramConstraint& constraint : capability.constraints) {
      constraints.emplace_back(mqt::ProgramConstraintAttr::get(
          &context, StringAttr::get(&context, constraint.id),
          constraint.value));
    }
    capabilities.emplace_back(mqt::ProgramCapabilityAttr::get(
        &context, StringAttr::get(&context, capability.id), capability.value,
        constraints));
  }
  return mqt::PayloadSpecAttr::get(&context, format, capabilities,
                                   optionalCapabilitiesKnown_);
}

TargetEnvironment::TargetEnvironment(const CompilerTarget& target,
                                     PayloadSpecification payload)
    : target_(target), payloadSpecification_(std::move(payload)) {}

llvm::Expected<TargetEnvironment>
TargetEnvironment::create(const mqt::TargetEnvAttr attribute) {
  if (!attribute) {
    return llvm::createStringError(llvm::errc::invalid_argument,
                                   "Target environment must not be null");
  }
  auto target = CompilerTarget::create(attribute.getCompilationTarget());
  if (!target) {
    return target.takeError();
  }
  auto payload =
      PayloadSpecification::create(attribute.getPayloadSpecification());
  if (!payload) {
    return payload.takeError();
  }
  return TargetEnvironment(*target, std::move(*payload));
}

const CompilerTarget& TargetEnvironment::target() const noexcept {
  return target_;
}

const PayloadSpecification&
TargetEnvironment::payloadSpecification() const noexcept {
  return payloadSpecification_;
}

mqt::TargetEnvAttr TargetEnvironment::materialize(MLIRContext& context) const {
  return mqt::TargetEnvAttr::get(&context, target_.materialize(context),
                                 payloadSpecification_.materialize(context),
                                 {});
}

void attachTargetEnvironment(ModuleOp moduleOp,
                             const TargetEnvironment& environment) {
  MLIRContext& context = *moduleOp.getContext();
  moduleOp->setAttr(mqt::TargetEnvAttr::name, environment.materialize(context));
}

TargetEnvironmentAnalysis::TargetEnvironmentAnalysis(Operation* operation)
    : moduleOp_(cast<ModuleOp>(operation)),
      attribute_(moduleOp_->getAttrOfType<mqt::TargetEnvAttr>(
          mqt::TargetEnvAttr::name)) {
  if (!attribute_) {
    error_ = "module does not contain mqt.target_env";
    return;
  }
  auto environment =
      TargetEnvironment::create(llvm::cast<mqt::TargetEnvAttr>(attribute_));
  if (!environment) {
    error_ = llvm::toString(environment.takeError());
    return;
  }
  environment_.emplace(std::move(*environment));
}

TargetEnvironmentAnalysis::operator bool() const noexcept {
  return environment_.has_value();
}

const TargetEnvironment&
TargetEnvironmentAnalysis::environment() const noexcept {
  assert(environment_.has_value());
  return *environment_;
}

llvm::StringRef TargetEnvironmentAnalysis::error() const noexcept {
  return error_;
}

bool TargetEnvironmentAnalysis::isInvalidated(
    const AnalysisManager::PreservedAnalyses& /*analyses*/) const {
  return moduleOp_->getAttr(mqt::TargetEnvAttr::name) != attribute_;
}

} // namespace mlir
