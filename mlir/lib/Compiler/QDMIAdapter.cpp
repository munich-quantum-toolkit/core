/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Compiler/QDMIAdapter.h"

#include "mlir/Compiler/Target.h"
#include "mlir/Compiler/TargetEnvironment.h"
#include "qdmi/Client.hpp"
#include "qdmi/ProgramFormat.hpp"
#include "qdmi/driver/Driver.hpp"

#include <llvm/ADT/StringRef.h>
#include <llvm/ADT/Twine.h>
#include <llvm/Support/CheckedArithmetic.h>
#include <llvm/Support/Error.h>
#include <qdmi/constants.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iterator>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <unordered_set>
#include <utility>
#include <vector>

namespace mlir {

/// Target facts that QDMI v1.3 cannot encode compactly.
/// TODO(#2093): Remove these compatibility markers when QDMI standardizes
/// explicit unrestricted connectivity, operation applicability, and operation
/// arity ranges.
constexpr std::string_view ALL_TO_ALL_HOMOGENEOUS_METADATA =
    "mqt.compiler-target.v1:all-to-all-homogeneous";
constexpr std::string_view ARBITRARY_POSITIVE_CONTROLS_METADATA =
    "mqt.compiler-target.v1:arbitrary-positive-controls";

[[nodiscard]] static bool
matchesMetadata(const std::optional<std::vector<std::byte>>& metadata,
                const std::string_view expected) {
  const auto expectedBytes =
      std::as_bytes(std::span{expected.data(), expected.size() + 1});
  return metadata && std::ranges::equal(*metadata, expectedBytes);
}

[[nodiscard]] static bool
hasAllToAllHomogeneousMetadata(const qdmi::Device& device) {
  return matchesMetadata(device.queryCustomProperty<std::vector<std::byte>>(
                             qdmi::CustomProperty::Custom1),
                         ALL_TO_ALL_HOMOGENEOUS_METADATA);
}

[[nodiscard]] static bool
hasArbitraryPositiveControlsMetadata(const qdmi::Operation& operation) {
  return matchesMetadata(operation.queryCustomProperty<std::vector<std::byte>>(
                             qdmi::CustomProperty::Custom1),
                         ARBITRARY_POSITIVE_CONTROLS_METADATA);
}

[[nodiscard]] static llvm::Error
requireAdapterInput(bool condition, const llvm::Twine& message) {
  if (!condition) {
    return llvm::createStringError(
        std::make_error_code(std::errc::invalid_argument), message);
  }
  return llvm::Error::success();
}

[[nodiscard]] static llvm::Error
requireCircuitDevice(bool condition, llvm::StringRef deviceName,
                     llvm::StringRef detail) {
  return requireAdapterInput(
      condition, llvm::Twine("QDMI device '") + deviceName +
                     "' cannot be used as an MLIR compiler target: only "
                     "circuit-model devices with one qubit per non-zone site "
                     "are supported (" +
                     detail + ")");
}

[[nodiscard]] static llvm::Error
requireRepresentableOperation(bool condition, llvm::StringRef deviceName,
                              llvm::StringRef operationName,
                              llvm::StringRef detail) {
  return requireAdapterInput(
      condition, llvm::Twine("QDMI device '") + deviceName + "' operation '" +
                     operationName +
                     "' cannot be represented by the MLIR compiler target (" +
                     detail + ")");
}

[[nodiscard]] static llvm::Expected<CompilerTarget::SiteId>
checkedSiteId(size_t index) {
  if (auto error = requireAdapterInput(
          index <= static_cast<uintmax_t>(
                       std::numeric_limits<CompilerTarget::SiteId>::max()),
          "QDMI site index exceeds the nonnegative i64 compiler-target "
          "domain")) {
    return std::move(error);
  }
  return static_cast<CompilerTarget::SiteId>(index);
}

[[nodiscard]] static CompilerTarget::Coupling
canonicalCoupling(CompilerTarget::SiteId first, CompilerTarget::SiteId second) {
  return first < second ? CompilerTarget::Coupling{first, second}
                        : CompilerTarget::Coupling{second, first};
}

[[nodiscard]] static std::optional<size_t>
allToAllCouplingCount(size_t numSites) {
  if (numSites < 2) {
    return 0;
  }
  const auto first = numSites % 2 == 0 ? numSites / 2 : numSites;
  const auto second = numSites % 2 == 0 ? numSites - 1 : (numSites - 1) / 2;
  return llvm::checkedMulUnsigned(first, second);
}

[[nodiscard]] static llvm::Error validateHomogeneousSupport(
    const qdmi::Operation& operation, size_t arity,
    const std::vector<qdmi::Site>& flattenedSites,
    const std::vector<CompilerTarget::Site>& deviceSites,
    const std::optional<std::vector<CompilerTarget::Coupling>>& couplings,
    llvm::StringRef deviceName) {
  const auto operationName = operation.getName();
  if (auto error = requireRepresentableOperation(
          flattenedSites.size() % arity == 0, deviceName, operationName,
          "the reported site list is not divisible by the fixed arity")) {
    return error;
  }
  std::unordered_set<CompilerTarget::SiteId> knownSites;
  knownSites.reserve(deviceSites.size());
  for (const auto& site : deviceSites) {
    knownSites.insert(site.id());
  }

  if (arity > 2) {
    std::set<std::vector<CompilerTarget::SiteId>> supportedTuples;
    for (size_t offset = 0; offset < flattenedSites.size(); offset += arity) {
      std::vector<CompilerTarget::SiteId> tuple;
      tuple.reserve(arity);
      for (size_t index = 0; index < arity; ++index) {
        auto siteId = checkedSiteId(flattenedSites[offset + index].getIndex());
        if (!siteId) {
          return siteId.takeError();
        }
        if (auto error = requireRepresentableOperation(
                knownSites.contains(*siteId) &&
                    !llvm::is_contained(tuple, *siteId),
                deviceName, operationName,
                "each higher-arity site tuple must contain distinct device "
                "sites")) {
          return error;
        }
        tuple.emplace_back(*siteId);
      }
      if (auto error = requireRepresentableOperation(
              supportedTuples.emplace(std::move(tuple)).second, deviceName,
              operationName,
              "the reported higher-arity site tuples must be unique")) {
        return error;
      }
    }

    auto expectedTuples = std::optional<size_t>{1};
    for (size_t index = 0; index < arity && expectedTuples; ++index) {
      if (index >= knownSites.size()) {
        expectedTuples = 0;
        break;
      }
      expectedTuples =
          llvm::checkedMulUnsigned(*expectedTuples, knownSites.size() - index);
    }
    return requireRepresentableOperation(
        expectedTuples && supportedTuples.size() == *expectedTuples, deviceName,
        operationName,
        "support is not homogeneous across all ordered tuples of distinct "
        "device sites");
  }

  if (arity == 1) {
    std::unordered_set<CompilerTarget::SiteId> supportedSites;
    supportedSites.reserve(flattenedSites.size());
    for (const auto& site : flattenedSites) {
      auto siteId = checkedSiteId(site.getIndex());
      if (!siteId) {
        return siteId.takeError();
      }
      const auto inserted = supportedSites.insert(*siteId).second;
      if (auto error = requireRepresentableOperation(
              knownSites.contains(*siteId) && inserted, deviceName,
              operationName,
              "the reported one-qubit sites must be unique device sites")) {
        return error;
      }
    }
    return requireRepresentableOperation(
        supportedSites.size() == knownSites.size(), deviceName, operationName,
        "support is not homogeneous across all device sites");
  }

  std::set<CompilerTarget::Coupling> reportedTuples;
  std::set<CompilerTarget::Coupling> supportedCouplings;
  for (size_t offset = 0; offset < flattenedSites.size(); offset += arity) {
    auto first = checkedSiteId(flattenedSites[offset].getIndex());
    if (!first) {
      return first.takeError();
    }
    auto second = checkedSiteId(flattenedSites[offset + 1].getIndex());
    if (!second) {
      return second.takeError();
    }
    const auto inserted = reportedTuples.insert({*first, *second}).second;
    const auto validTuple = *first != *second && knownSites.contains(*first) &&
                            knownSites.contains(*second) && inserted;
    if (auto error = requireRepresentableOperation(
            validTuple, deviceName, operationName,
            "the reported two-qubit sites must be "
            "unique pairs of device sites")) {
      return error;
    }
    supportedCouplings.insert(canonicalCoupling(*first, *second));
  }

  auto coversTarget = false;
  if (!couplings) {
    const auto expected = allToAllCouplingCount(knownSites.size());
    coversTarget = expected && supportedCouplings.size() == *expected;
  } else {
    std::set<CompilerTarget::Coupling> expectedCouplings;
    for (const auto& [first, second] : *couplings) {
      expectedCouplings.insert(canonicalCoupling(first, second));
    }
    coversTarget =
        supportedCouplings.size() == expectedCouplings.size() &&
        std::ranges::all_of(expectedCouplings, [&](const auto& coupling) {
          return supportedCouplings.contains(coupling);
        });
  }
  return requireRepresentableOperation(
      coversTarget, deviceName, operationName,
      couplings ? "support is not homogeneous across all topology edges"
                : "support is not homogeneous across all-to-all site pairs");
}

[[nodiscard]] static llvm::Expected<std::optional<CompilerTarget::DurationUnit>>
snapshotDurationUnit(const qdmi::Device& device) {
  auto unit = device.getDurationUnit();
  const auto scaleFactor = device.getDurationScaleFactor();
  if (auto error = requireAdapterInput(unit || !scaleFactor,
                                       "QDMI device reports a duration scale "
                                       "factor without a duration unit")) {
    return std::move(error);
  }
  if (!unit) {
    return std::nullopt;
  }
  auto durationUnit = CompilerTarget::DurationUnit::create(
      std::move(*unit), scaleFactor.value_or(1.));
  if (!durationUnit) {
    return durationUnit.takeError();
  }
  return std::optional<CompilerTarget::DurationUnit>(std::move(*durationUnit));
}

[[nodiscard]] static llvm::Expected<std::vector<CompilerTarget::SiteTuple>>
snapshotOperationSites(const qdmi::Operation& operation, size_t arity,
                       const std::vector<qdmi::Site>& flattenedSites,
                       std::optional<uint64_t> defaultDuration,
                       std::optional<double> defaultFidelity, bool variadic) {
  std::vector<CompilerTarget::SiteTuple> result;
  result.reserve(flattenedSites.size() / arity);
  for (size_t offset = 0; offset < flattenedSites.size(); offset += arity) {
    std::vector<qdmi::Site> sites;
    std::vector<CompilerTarget::SiteId> siteIds;
    sites.reserve(arity);
    siteIds.reserve(arity);
    for (size_t index = 0; index < arity; ++index) {
      const auto& site = flattenedSites[offset + index];
      sites.emplace_back(site);
      auto siteId = checkedSiteId(site.getIndex());
      if (!siteId) {
        return siteId.takeError();
      }
      siteIds.emplace_back(*siteId);
    }

    const auto duration = operation.getDuration(sites);
    const auto fidelity = operation.getFidelity(sites);
    const bool hasSiteCalibration =
        duration != defaultDuration || fidelity != defaultFidelity;
    if (!variadic || hasSiteCalibration) {
      auto siteTuple = CompilerTarget::SiteTuple::create(
          std::move(siteIds),
          duration == defaultDuration ? std::nullopt : duration,
          fidelity == defaultFidelity ? std::nullopt : fidelity);
      if (!siteTuple) {
        return siteTuple.takeError();
      }
      result.emplace_back(std::move(*siteTuple));
    }
  }
  return result;
}

[[nodiscard]] static llvm::Expected<CompilerTarget::NativeOperations>
snapshotOperations(
    const std::vector<qdmi::Operation>& operations,
    const std::vector<CompilerTarget::Site>& deviceSites,
    const std::optional<std::vector<CompilerTarget::Coupling>>& couplings,
    llvm::StringRef deviceName, bool homogeneousOperationSupport) {
  std::vector<CompilerTarget::Operation> targetOperations;
  targetOperations.reserve(operations.size());
  for (const auto& operation : operations) {
    if (auto error =
            requireCircuitDevice(!operation.isZoned(), deviceName,
                                 "the device exposes a zoned operation")) {
      return error;
    }
    const auto arity = operation.getQubitsNum();
    if (!arity) {
      continue;
    }
    const auto hasArbitraryPositiveControls =
        hasArbitraryPositiveControlsMetadata(operation);
    if (auto error = requireRepresentableOperation(
            !hasArbitraryPositiveControls ||
                (*arity > 0 && homogeneousOperationSupport),
            deviceName, operation.getName(),
            "arbitrary positive controls require a positive base arity and "
            "homogeneous operation support")) {
      return error;
    }
    const auto flattenedSites = operation.getSites();
    if (*arity > 0 && flattenedSites && flattenedSites->empty()) {
      continue;
    }
    if (auto error = requireRepresentableOperation(
            *arity == 0 || flattenedSites || homogeneousOperationSupport,
            deviceName, operation.getName(),
            "the supported sites are not reported")) {
      return error;
    }
    const auto duration = operation.getDuration();
    const auto fidelity = operation.getFidelity();
    std::vector<CompilerTarget::SiteTuple> siteTuples;
    if (*arity == 0) {
      if (auto error = requireRepresentableOperation(
              !flattenedSites || flattenedSites->empty(), deviceName,
              operation.getName(),
              "a zero-arity operation cannot report supported sites")) {
        return error;
      }
    } else if (flattenedSites) {
      if (auto error =
              validateHomogeneousSupport(operation, *arity, *flattenedSites,
                                         deviceSites, couplings, deviceName)) {
        return error;
      }
      auto tuples =
          snapshotOperationSites(operation, *arity, *flattenedSites, duration,
                                 fidelity, hasArbitraryPositiveControls);
      if (!tuples) {
        return tuples.takeError();
      }
      siteTuples = std::move(*tuples);
    }
    if (auto error = requireRepresentableOperation(
            !hasArbitraryPositiveControls || siteTuples.empty(), deviceName,
            operation.getName(),
            "a variadic operation cannot retain site-specific calibration")) {
      return error;
    }
    const auto targetArity =
        hasArbitraryPositiveControls
            ? CompilerTarget::Operation::Arity::variadic(*arity)
            : CompilerTarget::Operation::Arity::fixed(*arity);
    auto targetOperation = CompilerTarget::Operation::create(
        operation.getName(), targetArity, operation.getParametersNum(),
        std::move(siteTuples), duration, fidelity);
    if (!targetOperation) {
      return targetOperation.takeError();
    }
    targetOperations.emplace_back(std::move(*targetOperation));
  }
  return CompilerTarget::NativeOperations::fromOperations(targetOperations);
}

[[nodiscard]] static llvm::Expected<CompilerTarget>
snapshotCompilerTarget(const qdmi::Device& device) {
  auto deviceName = device.getName();
  const auto hasHomogeneousAllToAllMetadata =
      hasAllToAllHomogeneousMetadata(device);
  const auto deviceSites = device.getSites();
  if (auto error = requireCircuitDevice(
          std::ranges::none_of(deviceSites,
                               [](const auto& site) { return site.isZone(); }),
          deviceName, "the device exposes zone sites")) {
    return error;
  }
  if (auto error = requireCircuitDevice(
          device.getQubitsNum() == deviceSites.size(), deviceName,
          "the qubit count does not match the regular-site count")) {
    return error;
  }

  std::vector<CompilerTarget::Site> sites;
  sites.reserve(deviceSites.size());
  for (const auto& site : deviceSites) {
    auto siteId = checkedSiteId(site.getIndex());
    if (!siteId) {
      return siteId.takeError();
    }
    auto targetSite = CompilerTarget::Site::create(*siteId, site.getName(),
                                                   site.getT1(), site.getT2());
    if (!targetSite) {
      return targetSite.takeError();
    }
    sites.emplace_back(std::move(*targetSite));
  }

  std::optional<std::vector<CompilerTarget::Coupling>> couplings;
  if (const auto deviceCouplings = device.getCouplingMap()) {
    couplings.emplace();
    couplings->reserve(deviceCouplings->size());
    for (const auto& [source, target] : *deviceCouplings) {
      auto sourceId = checkedSiteId(source.getIndex());
      if (!sourceId) {
        return sourceId.takeError();
      }
      auto targetId = checkedSiteId(target.getIndex());
      if (!targetId) {
        return targetId.takeError();
      }
      couplings->emplace_back(*sourceId, *targetId);
    }
  }
  if (auto error = requireAdapterInput(
          couplings || hasHomogeneousAllToAllMetadata || sites.size() == 1,
          llvm::Twine("QDMI device '") + deviceName +
              "' cannot be used as an MLIR compiler target: connectivity is "
              "not reported")) {
    return error;
  }

  auto operations =
      snapshotOperations(device.getOperations(), sites, couplings, deviceName,
                         hasHomogeneousAllToAllMetadata);
  if (!operations) {
    return operations.takeError();
  }
  auto durationUnit = snapshotDurationUnit(device);
  if (!durationUnit) {
    return durationUnit.takeError();
  }
  auto connectivity =
      couplings ? CompilerTarget::Connectivity::fromCouplings(*couplings)
                : CompilerTarget::Connectivity::allToAll();
  return CompilerTarget::create(std::move(deviceName), std::move(sites),
                                std::move(connectivity), std::move(*operations),
                                std::move(*durationUnit));
}

[[nodiscard]] static llvm::Error
invalidFeatureGroup(const std::string_view id, const uint64_t value,
                    const llvm::StringRef detail) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      llvm::Twine("Invalid QDMI program feature group '") + id + "' value " +
          llvm::Twine(value) + ": " + detail);
}

[[nodiscard]] static llvm::Expected<std::vector<ProgramCapability>>
groupProgramFeatures(const std::vector<QDMI_Program_Feature>& features) {
  struct FeatureGroup {
    ProgramCapability capability;
    bool unrestricted = false;
  };

  std::vector<FeatureGroup> groups;
  std::map<std::pair<std::string, uint64_t>, size_t> groupIndices;
  groups.reserve(features.size());

  for (const auto& feature : features) {
    const std::string id{std::data(feature.id)};
    const auto key = std::pair{id, feature.value};
    const auto [position, inserted] =
        groupIndices.try_emplace(key, groups.size());
    if (inserted) {
      groups.push_back({.capability = {.id = id, .value = feature.value}});
    }
    auto& group = groups[position->second];
    const std::string constraintId{std::data(feature.constraint_id)};
    if (constraintId.empty()) {
      if (!inserted || !group.capability.constraints.empty()) {
        return invalidFeatureGroup(
            id, feature.value,
            "an unrestricted group must contain exactly one record");
      }
      group.unrestricted = true;
      continue;
    }
    if (group.unrestricted) {
      return invalidFeatureGroup(
          id, feature.value,
          "an unrestricted record cannot have constrained siblings");
    }
    if (std::ranges::any_of(group.capability.constraints,
                            [&](const ProgramConstraint& constraint) {
                              return constraint.id == constraintId;
                            })) {
      return invalidFeatureGroup(id, feature.value,
                                 "constraint IDs must be unique");
    }
    group.capability.constraints.push_back(
        {.id = constraintId, .value = feature.constraint_value});
  }

  std::vector<ProgramCapability> capabilities;
  capabilities.reserve(groups.size());
  std::ranges::transform(
      groups, std::back_inserter(capabilities),
      [](FeatureGroup& group) { return std::move(group.capability); });
  return capabilities;
}

[[nodiscard]] static std::vector<ProgramCapability>
payloadBaseline(const PayloadFormat& format) {
  if (format.id != "qir" || format.version != "2.1.0" ||
      format.profile != "adaptive") {
    return {};
  }
  return {
      {.id = QDMI_PROGRAM_FEATURE_MID_CIRCUIT_MEASUREMENT},
      {.id = QDMI_PROGRAM_FEATURE_MEASURED_QUBIT_REUSE},
      {.id = QDMI_PROGRAM_FEATURE_MEASUREMENT_RESULT_USE},
      {.id = QDMI_PROGRAM_FEATURE_BOOLEAN_COMPUTATION},
      {.id = QDMI_PROGRAM_FEATURE_FORWARD_BRANCHING},
  };
}

[[nodiscard]] static llvm::Expected<PayloadSpecification>
snapshotPayloadSpecification(const qdmi::Device& device,
                             const QDMI_Program_Format& format) {
  if (!qdmi::isValidProgramFormat(format)) {
    return llvm::createStringError(
        std::make_error_code(std::errc::invalid_argument),
        "Invalid QDMI program format: fields are not canonical");
  }
  const auto supported = device.getSupportedProgramFormats();
  if (std::ranges::none_of(supported, [&](const auto& candidate) {
        return qdmi::equal(candidate, format);
      })) {
    return llvm::createStringError(
        std::make_error_code(std::errc::invalid_argument),
        "QDMI device does not accept the selected program format");
  }

  const auto encoding = format.encoding == QDMI_PROGRAM_ENCODING_TEXT
                            ? PayloadEncoding::Text
                            : PayloadEncoding::Binary;
  PayloadFormat payloadFormat{
      .id = std::data(format.id),
      .version = std::to_string(QDMI_VERSION_MAJOR(format.version)) + "." +
                 std::to_string(QDMI_VERSION_MINOR(format.version)) + "." +
                 std::to_string(QDMI_VERSION_PATCH(format.version)),
      .profile = std::data(format.profile),
      .encoding = encoding,
  };

  auto capabilities = payloadBaseline(payloadFormat);
  const auto optionalFeatures = device.tryGetProgramFeatures(format);
  if (optionalFeatures) {
    auto optionalCapabilities = groupProgramFeatures(*optionalFeatures);
    if (!optionalCapabilities) {
      return optionalCapabilities.takeError();
    }
    capabilities.insert(capabilities.end(),
                        std::make_move_iterator(optionalCapabilities->begin()),
                        std::make_move_iterator(optionalCapabilities->end()));
  }
  return PayloadSpecification::create(std::move(payloadFormat),
                                      std::move(capabilities),
                                      optionalFeatures.has_value());
}

[[nodiscard]] static llvm::Expected<TargetEnvironment>
snapshotTargetEnvironment(const qdmi::Device& device,
                          const QDMI_Program_Format& format) {
  auto target = snapshotCompilerTarget(device);
  if (!target) {
    return target.takeError();
  }
  auto payload = snapshotPayloadSpecification(device, format);
  if (!payload) {
    return payload.takeError();
  }
  return TargetEnvironment(*target, std::move(*payload));
}

[[nodiscard]] static llvm::Error qdmiError(const llvm::Twine& action,
                                           const char* const detail) {
  return llvm::createStringError(std::make_error_code(std::errc::io_error),
                                 action + ": " + detail);
}

[[nodiscard]] static llvm::Error
qdmiError(const llvm::Twine& action, const std::exception_ptr& exception) {
  try {
    std::rethrow_exception(exception);
  } catch (const std::exception& error) {
    return qdmiError(action, error.what());
  } catch (...) {
    return qdmiError(action, "unknown exception");
  }
}

llvm::Expected<CompilerTarget>
compilerTargetFromDevice(const qdmi::Device& device) {
  try {
    return snapshotCompilerTarget(device);
  } catch (...) {
    return qdmiError("Failed to query QDMI device", std::current_exception());
  }
}

llvm::Expected<CompilerTarget>
compilerTargetFromDeviceId(const std::string_view deviceId) {
  const auto action = std::string("Failed to open or query QDMI device '") +
                      std::string(deviceId) + "'";
  try {
    return snapshotCompilerTarget(qdmi::Session::openDevice(deviceId));
  } catch (...) {
    return qdmiError(action, std::current_exception());
  }
}

llvm::Expected<TargetEnvironment>
targetEnvironmentFromDevice(const qdmi::Device& device,
                            const QDMI_Program_Format& format) {
  try {
    return snapshotTargetEnvironment(device, format);
  } catch (...) {
    return qdmiError("Failed to query QDMI device and payload",
                     std::current_exception());
  }
}

llvm::Expected<TargetEnvironment>
targetEnvironmentFromDeviceId(const std::string_view deviceId,
                              const QDMI_Program_Format& format) {
  const auto action = std::string("Failed to open or query QDMI device '") +
                      std::string(deviceId) + "' and payload";
  try {
    return snapshotTargetEnvironment(qdmi::Session::openDevice(deviceId),
                                     format);
  } catch (...) {
    return qdmiError(action, std::current_exception());
  }
}

llvm::Expected<std::vector<std::string>> registeredQDMIDeviceIds() {
  try {
    return qdmi::Driver::get().registeredDeviceIds();
  } catch (...) {
    return qdmiError("Failed to discover registered QDMI devices",
                     std::current_exception());
  }
}

} // namespace mlir
