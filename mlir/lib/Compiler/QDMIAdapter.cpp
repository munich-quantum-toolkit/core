/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/Compiler/QDMIAdapter.h"

#include "mqt/Compiler/Target.h"
#include "mqt/Dialect/MQT/IR/MQTDialect.h"
#include "mqt/Dialect/MQT/Transforms/GlobalPhaseNormalization.h"
#include "mqt/Dialect/MQT/Transforms/Passes.h"
#include "mqt/Dialect/QIR/Utils/QIRUtils.h"
#include "qdmi/QDMI.hpp"

#include "Diagnostics.h"
#include "TerminalReadout.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

using llvm::createStringError;
using llvm::Error;
using llvm::Expected;

namespace mlir {

/// Target facts that QDMI v1.3 cannot encode compactly.
///
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

[[nodiscard]] static Expected<bool>
hasAllToAllHomogeneousMetadata(const qdmi::Device& device) {
  auto metadata = device.queryCustomProperty<std::vector<std::byte>>(
      qdmi::CustomProperty::Custom1);
  if (!metadata) {
    return metadata.takeError();
  }
  return matchesMetadata(*metadata, ALL_TO_ALL_HOMOGENEOUS_METADATA);
}

[[nodiscard]] static Expected<bool>
hasArbitraryPositiveControlsMetadata(const qdmi::Operation& operation) {
  auto metadata = operation.queryCustomProperty<std::vector<std::byte>>(
      qdmi::CustomProperty::Custom1);
  if (!metadata) {
    return metadata.takeError();
  }
  return matchesMetadata(*metadata, ARBITRARY_POSITIVE_CONTROLS_METADATA);
}

Expected<SmallVector<OperationParameterConstraints, 1>>
parseOperationParameterConstraints(StringRef text, size_t numParameters) {
  constexpr StringLiteral schema = "mqt.operation-parameters.v1";
  if (!text.contains(schema)) {
    return SmallVector<OperationParameterConstraints, 1>{};
  }
  const auto invalid = [] {
    return createStringError("Invalid mqt.operation-parameters.v1 metadata");
  };
  if (!text.ends_with('\0')) {
    return invalid();
  }
  auto parsed = llvm::json::parse(text.drop_back());
  if (!parsed) {
    return parsed.takeError();
  }
  const auto* object = parsed->getAsObject();
  if (object == nullptr || object->getString("schema") != schema) {
    return SmallVector<OperationParameterConstraints, 1>{};
  }
  const auto* alternatives = object->getArray("alternatives");
  if (alternatives == nullptr || alternatives->empty()) {
    return invalid();
  }
  SmallVector<OperationParameterConstraints, 1> result;
  result.reserve(alternatives->size());
  for (const auto& alternative : *alternatives) {
    const auto* fields = alternative.getAsObject();
    if (fields == nullptr) {
      return invalid();
    }
    OperationParameterConstraints constraints;
    for (const auto* key : {"fixed", "bounds"}) {
      const auto* value = fields->get(key);
      if (value == nullptr) {
        continue;
      }
      const auto* values = value->getAsArray();
      if (values == nullptr || values->size() != numParameters) {
        return invalid();
      }
      for (const auto& parameter : *values) {
        if (StringRef(key) == "fixed") {
          const auto fixed = parameter.getAsNumber();
          if (!fixed && !parameter.getAsNull()) {
            return invalid();
          }
          constraints.fixed.push_back(fixed);
        } else if (parameter.getAsNull()) {
          constraints.bounds.emplace_back();
        } else {
          const auto* bounds = parameter.getAsArray();
          if (bounds == nullptr || bounds->size() != 2 ||
              !(*bounds)[0].getAsNumber() || !(*bounds)[1].getAsNumber()) {
            return invalid();
          }
          constraints.bounds.emplace_back(std::pair{
              *(*bounds)[0].getAsNumber(), *(*bounds)[1].getAsNumber()});
        }
      }
    }
    auto validated = CompilerTarget::OperationCapability::create(
        "parameters", 0, numParameters, {}, std::nullopt, std::nullopt,
        std::move(constraints.fixed), std::nullopt,
        std::move(constraints.bounds));
    if (!validated) {
      return validated.takeError();
    }
    result.push_back(
        {validated->fixedParameters(), validated->parameterBounds()});
  }
  return result;
}

[[nodiscard]] static Error requireAdapterInput(bool condition,
                                               const llvm::Twine& message) {
  if (!condition) {
    return createStringError(std::make_error_code(std::errc::invalid_argument),
                             message);
  }
  return Error::success();
}

[[nodiscard]] static Error requireCircuitDevice(bool condition,
                                                llvm::StringRef deviceName,
                                                llvm::StringRef detail) {
  return requireAdapterInput(
      condition, llvm::Twine("QDMI device '") + deviceName +
                     "' cannot be used as an MQT compiler target: only "
                     "circuit-model devices with one qubit per non-zone site "
                     "are supported (" +
                     detail + ")");
}

[[nodiscard]] static Error
requireRepresentableOperation(bool condition, llvm::StringRef deviceName,
                              llvm::StringRef operationName,
                              llvm::StringRef detail) {
  return requireAdapterInput(
      condition, llvm::Twine("QDMI device '") + deviceName + "' operation '" +
                     operationName +
                     "' cannot be represented by the MQT compiler target (" +
                     detail + ")");
}

[[nodiscard]] static Expected<CompilerTarget::SiteId>
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

using SiteIndices = DenseMap<QDMI_Site, CompilerTarget::SiteId>;

[[nodiscard]] static Expected<CompilerTarget::SiteId>
snapshotSiteIndex(const qdmi::Site& site, SiteIndices& indices) {
  const auto found = indices.find(site);
  if (found != indices.end()) {
    return found->second;
  }
  auto rawIndex = site.getIndex();
  if (!rawIndex) {
    return rawIndex.takeError();
  }
  auto index = checkedSiteId(*rawIndex);
  if (!index) {
    return index.takeError();
  }
  indices.try_emplace(site, *index);
  return *index;
}

[[nodiscard]] static Error
validateOperationSites(StringRef operationName, size_t arity,
                       const std::vector<qdmi::Site>& flattenedSites,
                       const DenseSet<CompilerTarget::SiteId>& knownSites,
                       StringRef deviceName, SiteIndices& indices) {
  if (auto error = requireRepresentableOperation(
          flattenedSites.size() % arity == 0, deviceName, operationName,
          "the reported site list is not divisible by the fixed arity")) {
    return error;
  }
  if (arity > 2) {
    std::set<std::vector<CompilerTarget::SiteId>> supportedTuples;
    for (size_t offset = 0; offset < flattenedSites.size(); offset += arity) {
      std::vector<CompilerTarget::SiteId> tuple;
      tuple.reserve(arity);
      for (size_t index = 0; index < arity; ++index) {
        auto siteId =
            snapshotSiteIndex(flattenedSites[offset + index], indices);
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

    return Error::success();
  }

  if (arity == 1) {
    DenseSet<CompilerTarget::SiteId> supportedSites;
    supportedSites.reserve(flattenedSites.size());
    for (const auto& site : flattenedSites) {
      auto siteId = snapshotSiteIndex(site, indices);
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
    return Error::success();
  }

  std::set<CompilerTarget::Coupling> reportedTuples;
  for (size_t offset = 0; offset < flattenedSites.size(); offset += arity) {
    auto first = snapshotSiteIndex(flattenedSites[offset], indices);
    if (!first) {
      return first.takeError();
    }
    auto second = snapshotSiteIndex(flattenedSites[offset + 1], indices);
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
  }

  return Error::success();
}

[[nodiscard]] static Expected<std::optional<CompilerTarget::DurationUnit>>
snapshotDurationUnit(const qdmi::Device& device) {
  auto unitResult = device.getDurationUnit();
  if (!unitResult) {
    return unitResult.takeError();
  }
  auto unit = std::move(*unitResult);
  auto scaleFactorResult = device.getDurationScaleFactor();
  if (!scaleFactorResult) {
    return scaleFactorResult.takeError();
  }
  auto scaleFactor = *scaleFactorResult;
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

[[nodiscard]] static Expected<std::vector<CompilerTarget::SiteTuple>>
snapshotOperationSites(const qdmi::Operation& operation, size_t arity,
                       const std::vector<qdmi::Site>& flattenedSites,
                       std::optional<uint64_t> defaultDuration,
                       std::optional<double> defaultFidelity, bool variadic,
                       SiteIndices& indices, bool includeCalibration) {
  std::vector<CompilerTarget::SiteTuple> result;
  result.reserve(flattenedSites.size() / arity);
  std::vector<qdmi::Site> sites;
  sites.reserve(arity);
  for (size_t offset = 0; offset < flattenedSites.size(); offset += arity) {
    sites.clear();
    std::vector<CompilerTarget::SiteId> siteIds;
    siteIds.reserve(arity);
    for (size_t index = 0; index < arity; ++index) {
      const auto& site = flattenedSites[offset + index];
      sites.emplace_back(site);
      auto siteId = snapshotSiteIndex(site, indices);
      if (!siteId) {
        return siteId.takeError();
      }
      siteIds.emplace_back(*siteId);
    }

    auto durationResult = includeCalibration
                              ? operation.getDuration(sites)
                              : Expected<std::optional<uint64_t>>(std::nullopt);
    if (!durationResult) {
      return durationResult.takeError();
    }
    auto duration = *durationResult;
    auto fidelityResult = includeCalibration
                              ? operation.getFidelity(sites)
                              : Expected<std::optional<double>>(std::nullopt);
    if (!fidelityResult) {
      return fidelityResult.takeError();
    }
    auto fidelity = *fidelityResult;
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

[[nodiscard]] static Expected<CompilerTarget::NativeOperations>
snapshotOperations(const std::vector<qdmi::Operation>& operations,
                   const std::vector<CompilerTarget::Site>& deviceSites,
                   StringRef deviceName, bool homogeneousOperationSupport,
                   SiteIndices& indices, bool includeCalibration) {
  DenseSet<CompilerTarget::SiteId> knownSites;
  std::vector<CompilerTarget::OperationCapability> targetOperations;
  targetOperations.reserve(operations.size());
  for (const auto& operation : operations) {
    auto zoned = operation.isZoned();
    if (!zoned) {
      return zoned.takeError();
    }
    if (auto error = requireCircuitDevice(
            !*zoned, deviceName, "the device exposes a zoned operation")) {
      return error;
    }
    auto arityResult = operation.getQubitsNum();
    if (!arityResult) {
      return arityResult.takeError();
    }
    auto arity = *arityResult;
    if (!arity) {
      continue;
    }
    auto operationNameResult = operation.getName();
    if (!operationNameResult) {
      return operationNameResult.takeError();
    }
    auto operationName = std::move(*operationNameResult);
    auto hasArbitraryPositiveControlsResult =
        hasArbitraryPositiveControlsMetadata(operation);
    if (!hasArbitraryPositiveControlsResult) {
      return hasArbitraryPositiveControlsResult.takeError();
    }
    auto hasArbitraryPositiveControls = *hasArbitraryPositiveControlsResult;
    if (auto error = requireRepresentableOperation(
            !hasArbitraryPositiveControls ||
                (*arity > 0 && homogeneousOperationSupport),
            deviceName, operationName,
            "arbitrary positive controls require a positive base arity and "
            "homogeneous operation support")) {
      return error;
    }
    auto flattenedSitesResult = operation.getSites();
    if (!flattenedSitesResult) {
      return flattenedSitesResult.takeError();
    }
    auto flattenedSites = std::move(*flattenedSitesResult);
    if (*arity > 0 && flattenedSites && flattenedSites->empty()) {
      continue;
    }
    if (auto error = requireRepresentableOperation(
            *arity == 0 || flattenedSites || homogeneousOperationSupport,
            deviceName, operationName,
            "the supported sites are not reported")) {
      return error;
    }
    const bool queryCalibration =
        includeCalibration || hasArbitraryPositiveControls;
    auto durationResult = queryCalibration
                              ? operation.getDuration()
                              : Expected<std::optional<uint64_t>>(std::nullopt);
    if (!durationResult) {
      return durationResult.takeError();
    }
    auto duration = *durationResult;
    auto fidelityResult = queryCalibration
                              ? operation.getFidelity()
                              : Expected<std::optional<double>>(std::nullopt);
    if (!fidelityResult) {
      return fidelityResult.takeError();
    }
    auto fidelity = *fidelityResult;
    std::vector<CompilerTarget::SiteTuple> siteTuples;
    if (*arity == 0) {
      if (auto error = requireRepresentableOperation(
              !flattenedSites || flattenedSites->empty(), deviceName,
              operationName,
              "a zero-arity operation cannot report supported sites")) {
        return error;
      }
    } else if (flattenedSites) {
      if (knownSites.empty()) {
        knownSites.reserve(deviceSites.size());
        for (const auto& site : deviceSites) {
          knownSites.insert(site.id());
        }
      }
      if (auto error =
              validateOperationSites(operationName, *arity, *flattenedSites,
                                     knownSites, deviceName, indices)) {
        return error;
      }
      auto tuples = snapshotOperationSites(
          operation, *arity, *flattenedSites, duration, fidelity,
          hasArbitraryPositiveControls, indices, queryCalibration);
      if (!tuples) {
        return tuples.takeError();
      }
      siteTuples = std::move(*tuples);
    }
    if (auto error = requireRepresentableOperation(
            !hasArbitraryPositiveControls || siteTuples.empty(), deviceName,
            operationName,
            "a variadic operation cannot retain site-specific calibration")) {
      return error;
    }
    const auto targetArity =
        hasArbitraryPositiveControls
            ? CompilerTarget::OperationCapability::Arity::variadic(*arity)
            : CompilerTarget::OperationCapability::Arity::fixed(*arity);
    auto numParameters = operation.getParametersNum();
    if (!numParameters) {
      return numParameters.takeError();
    }
    auto metadata = operation.queryCustomProperty<std::vector<std::byte>>(
        qdmi::CustomProperty::Custom2);
    if (!metadata) {
      return metadata.takeError();
    }
    const auto text =
        *metadata
            ? StringRef(reinterpret_cast<const char*>((**metadata).data()),
                        (**metadata).size())
            : StringRef{};
    auto constraints = parseOperationParameterConstraints(text, *numParameters);
    if (!constraints) {
      return constraints.takeError();
    }
    if (constraints->empty()) {
      constraints->emplace_back();
    }
    for (auto& alternative : *constraints) {
      auto targetOperation = CompilerTarget::OperationCapability::create(
          operationName, targetArity, *numParameters, siteTuples,
          includeCalibration ? duration : std::nullopt,
          includeCalibration ? fidelity : std::nullopt,
          std::move(alternative.fixed), std::nullopt,
          std::move(alternative.bounds));
      if (!targetOperation) {
        return targetOperation.takeError();
      }
      targetOperations.emplace_back(std::move(*targetOperation));
    }
  }
  return CompilerTarget::NativeOperations::fromOperations(targetOperations);
}

[[nodiscard]] static Expected<CompilerTarget>
snapshotCompilerTarget(const qdmi::Device& device,
                       bool includeCalibration = true,
                       bool requireSiteNames = false) {
  auto deviceNameResult = device.getName();
  if (!deviceNameResult) {
    return deviceNameResult.takeError();
  }
  auto deviceName = std::move(*deviceNameResult);
  auto hasHomogeneousAllToAllMetadataResult =
      hasAllToAllHomogeneousMetadata(device);
  if (!hasHomogeneousAllToAllMetadataResult) {
    return hasHomogeneousAllToAllMetadataResult.takeError();
  }
  auto hasHomogeneousAllToAllMetadata = *hasHomogeneousAllToAllMetadataResult;
  auto deviceSitesResult = device.getSites();
  if (!deviceSitesResult) {
    return deviceSitesResult.takeError();
  }
  auto deviceSites = std::move(*deviceSitesResult);
  for (const auto& site : deviceSites) {
    auto isZone = site.isZone();
    if (!isZone) {
      return isZone.takeError();
    }
    if (auto error = requireCircuitDevice(!*isZone, deviceName,
                                          "the device exposes zone sites")) {
      return error;
    }
  }
  auto numQubits = device.getQubitsNum();
  if (!numQubits) {
    return numQubits.takeError();
  }
  if (auto error = requireCircuitDevice(
          *numQubits == deviceSites.size(), deviceName,
          "the qubit count does not match the regular-site count")) {
    return error;
  }

  SiteIndices indices;
  indices.reserve(deviceSites.size());
  std::vector<CompilerTarget::Site> sites;
  sites.reserve(deviceSites.size());
  for (const auto& site : deviceSites) {
    auto siteId = snapshotSiteIndex(site, indices);
    if (!siteId) {
      return siteId.takeError();
    }
    auto name = includeCalibration || requireSiteNames
                    ? site.getName()
                    : Expected<std::optional<std::string>>(std::nullopt);
    if (!name) {
      return name.takeError();
    }
    auto t1 = includeCalibration
                  ? site.getT1()
                  : Expected<std::optional<uint64_t>>(std::nullopt);
    if (!t1) {
      return t1.takeError();
    }
    auto t2 = includeCalibration
                  ? site.getT2()
                  : Expected<std::optional<uint64_t>>(std::nullopt);
    if (!t2) {
      return t2.takeError();
    }
    auto targetSite =
        CompilerTarget::Site::create(*siteId, std::move(*name), *t1, *t2);
    if (!targetSite) {
      return targetSite.takeError();
    }
    sites.emplace_back(std::move(*targetSite));
  }

  std::optional<std::vector<CompilerTarget::Coupling>> couplings;
  auto deviceCouplingsResult = device.getCouplingMap();
  if (!deviceCouplingsResult) {
    return deviceCouplingsResult.takeError();
  }
  if (const auto& deviceCouplings = *deviceCouplingsResult) {
    couplings.emplace();
    couplings->reserve(deviceCouplings->size());
    for (const auto& [source, target] : *deviceCouplings) {
      auto sourceId = snapshotSiteIndex(source, indices);
      if (!sourceId) {
        return sourceId.takeError();
      }
      auto targetId = snapshotSiteIndex(target, indices);
      if (!targetId) {
        return targetId.takeError();
      }
      couplings->emplace_back(*sourceId, *targetId);
    }
  }
  if (auto error = requireAdapterInput(
          couplings || hasHomogeneousAllToAllMetadata || sites.size() == 1,
          llvm::Twine("QDMI device '") + deviceName +
              "' cannot be used as an MQT compiler target: connectivity is "
              "not reported")) {
    return error;
  }

  auto deviceOperations = device.getOperations();
  if (!deviceOperations) {
    return deviceOperations.takeError();
  }
  auto operations = snapshotOperations(*deviceOperations, sites, deviceName,
                                       hasHomogeneousAllToAllMetadata, indices,
                                       includeCalibration);
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

Expected<CompilerTarget> compilerTargetFromDevice(const qdmi::Device& device) {
  return snapshotCompilerTarget(device);
}

Expected<CompilerTarget>
compilerTargetFromDeviceId(const std::string_view deviceId) {
  auto device = qdmi::Session::openDevice(deviceId);
  if (!device) {
    return device.takeError();
  }
  return snapshotCompilerTarget(*device);
}

Expected<std::vector<std::string>> registeredQDMIDeviceIds() {
  auto session = qdmi::Session::create();
  if (!session) {
    return session.takeError();
  }
  return session->getDeviceIds();
}

constexpr std::array PROGRAM_FORMAT_PREFERENCE{
    QDMI_PROGRAM_FORMAT_QIRADAPTIVEMODULE,
    QDMI_PROGRAM_FORMAT_QIRADAPTIVESTRING,
    QDMI_PROGRAM_FORMAT_QASM3,
    QDMI_PROGRAM_FORMAT_IQMJSON,
    QDMI_PROGRAM_FORMAT_QIRBASEMODULE,
    QDMI_PROGRAM_FORMAT_QIRBASESTRING,
    QDMI_PROGRAM_FORMAT_QASM2,
};

static Error incompatible(const llvm::Twine& detail) {
  return createStringError(
      std::make_error_code(std::errc::invalid_argument),
      "Compiled program is incompatible with the destination: " + detail +
          "; recompile for this device");
}

static bool sameOperation(const CompilerTarget::OperationCapability& lhs,
                          const CompilerTarget::OperationCapability& rhs) {
  if (lhs.name() != rhs.name() || lhs.canonicalName() != rhs.canonicalName() ||
      lhs.arity() != rhs.arity() ||
      lhs.numParameters() != rhs.numParameters() ||
      lhs.fixedParameters() != rhs.fixedParameters() ||
      lhs.parameterBounds() != rhs.parameterBounds() ||
      lhs.siteTuples().size() != rhs.siteTuples().size()) {
    return false;
  }
  if (std::ranges::equal(lhs.siteTuples(), rhs.siteTuples(),
                         [](const auto& a, const auto& b) {
                           return a.sites() == b.sites();
                         })) {
    return true;
  }
  const auto sortedSites = [](const auto& operation) {
    std::vector<llvm::ArrayRef<CompilerTarget::SiteId>> sites;
    sites.reserve(operation.siteTuples().size());
    for (const auto& tuple : operation.siteTuples()) {
      sites.push_back(tuple.sites());
    }
    std::ranges::sort(sites, [](auto a, auto b) {
      return std::ranges::lexicographical_compare(a, b);
    });
    return sites;
  };
  return sortedSites(lhs) == sortedSites(rhs);
}

static bool sameCapability(const ProgramCapability& lhs,
                           const ProgramCapability& rhs) {
  return lhs.id == rhs.id && lhs.value == rhs.value &&
         std::ranges::is_permutation(lhs.constraints, rhs.constraints);
}

static Expected<QDMI_Program_Format>
qdmiFormatForPayload(const PayloadSpecification& payload) {
  auto output = payload.compilerOutput();
  if (!output) {
    return output.takeError();
  }
  const bool binary = payload.format().encoding == PayloadEncoding::Binary;
  switch (*output) {
  case ProgramFormat::OpenQASM2:
    return QDMI_PROGRAM_FORMAT_QASM2;
  case ProgramFormat::OpenQASM3:
    return QDMI_PROGRAM_FORMAT_QASM3;
  case ProgramFormat::IQMJSON:
    return QDMI_PROGRAM_FORMAT_IQMJSON;
  case ProgramFormat::QIRBase:
    return binary ? QDMI_PROGRAM_FORMAT_QIRBASEMODULE
                  : QDMI_PROGRAM_FORMAT_QIRBASESTRING;
  case ProgramFormat::QIRAdaptive:
    return binary ? QDMI_PROGRAM_FORMAT_QIRADAPTIVEMODULE
                  : QDMI_PROGRAM_FORMAT_QIRADAPTIVESTRING;
  default:
    llvm_unreachable("PayloadSpecification accepted a non-executable format");
  }
}

Expected<PayloadSpecification>
payloadSpecificationForProgramFormat(QDMI_Program_Format format) {
  const auto encoding = qdmi::isBinaryProgramFormat(format)
                            ? PayloadEncoding::Binary
                            : PayloadEncoding::Text;
  switch (format) {
  case QDMI_PROGRAM_FORMAT_IQMJSON:
    return PayloadSpecification::forProgramFormat(ProgramFormat::IQMJSON,
                                                  encoding);
  case QDMI_PROGRAM_FORMAT_QASM2:
    return PayloadSpecification::forProgramFormat(ProgramFormat::OpenQASM2,
                                                  encoding);
  case QDMI_PROGRAM_FORMAT_QASM3:
    return PayloadSpecification::forProgramFormat(ProgramFormat::OpenQASM3,
                                                  encoding);
  case QDMI_PROGRAM_FORMAT_QIRBASEMODULE:
  case QDMI_PROGRAM_FORMAT_QIRBASESTRING:
    return PayloadSpecification::forProgramFormat(ProgramFormat::QIRBase,
                                                  encoding);
  case QDMI_PROGRAM_FORMAT_QIRADAPTIVEMODULE:
  case QDMI_PROGRAM_FORMAT_QIRADAPTIVESTRING:
    return PayloadSpecification::forProgramFormat(ProgramFormat::QIRAdaptive,
                                                  encoding);
  default:
    return createStringError(
        std::make_error_code(std::errc::invalid_argument),
        "MQT compiler cannot emit the requested QDMI program format");
  }
}

static Expected<TargetEnvironment>
snapshotTargetEnvironment(const qdmi::Device& device,
                          std::optional<QDMI_Program_Format> format,
                          bool includeCalibration) {
  auto supportedResult = device.getSupportedProgramFormats();
  if (!supportedResult) {
    return supportedResult.takeError();
  }
  auto supported = std::move(*supportedResult);
  if (format && !llvm::is_contained(supported, *format)) {
    return createStringError(
        std::make_error_code(std::errc::invalid_argument),
        "Device does not support the requested program_format");
  }
  if (!format) {
    for (const auto candidate : PROGRAM_FORMAT_PREFERENCE) {
      if (llvm::is_contained(supported, candidate)) {
        format = candidate;
        break;
      }
    }
  }
  if (!format) {
    return createStringError(
        std::make_error_code(std::errc::invalid_argument),
        "Device has no executable program format supported by the MQT "
        "compiler; hardware-only models require an explicit compiler target "
        "and output");
  }
  auto payload = payloadSpecificationForProgramFormat(*format);
  if (!payload) {
    return payload.takeError();
  }
  auto target = snapshotCompilerTarget(device, includeCalibration,
                                       *format == QDMI_PROGRAM_FORMAT_IQMJSON);
  if (!target) {
    return target.takeError();
  }
  return TargetEnvironment(*target, std::move(*payload));
}

Expected<TargetEnvironment>
targetEnvironmentFromDevice(const qdmi::Device& device,
                            std::optional<QDMI_Program_Format> format) {
  return snapshotTargetEnvironment(device, format, true);
}

Error validateTargetCompatibility(const TargetEnvironment& compiled,
                                  const TargetEnvironment& destination) {
  const auto& lhs = compiled.target();
  const auto& rhs = destination.target();
  if (lhs.siteIds() != rhs.siteIds()) {
    return incompatible("ordered site mapping differs");
  }
  if (compiled.payloadSpecification().format().id == "iqm-json" &&
      !std::ranges::equal(
          lhs.sites(), rhs.sites(),
          [](const auto& a, const auto& b) { return a.name() == b.name(); })) {
    return incompatible("site name mapping differs");
  }
  if (lhs.connectivityKind() != rhs.connectivityKind() ||
      lhs.couplings() != rhs.couplings()) {
    return incompatible("connectivity differs");
  }
  const auto& lhsUnit = lhs.durationUnit();
  const auto& rhsUnit = rhs.durationUnit();
  if (lhsUnit.has_value() != rhsUnit.has_value() ||
      (lhsUnit && (lhsUnit->unit() != rhsUnit->unit() ||
                   lhsUnit->scaleFactor() != rhsUnit->scaleFactor()))) {
    return incompatible("duration units differ");
  }
  if (lhs.nativeOperationsKind() != rhs.nativeOperationsKind() ||
      !std::ranges::is_permutation(lhs.operations(), rhs.operations(),
                                   sameOperation)) {
    return incompatible("native operations or ordered applicability differ");
  }
  const auto& a = compiled.payloadSpecification();
  const auto& b = destination.payloadSpecification();
  const bool versionsCompatible =
      a.versionRange() && b.versionRange()
          ? a.versionRange()->isSubsetOf(*b.versionRange())
          : a.versionRange() == b.versionRange();
  if (a.format().id != b.format().id ||
      a.format().profile != b.format().profile ||
      a.format().encoding != b.format().encoding || !versionsCompatible ||
      !std::ranges::is_permutation(a.capabilities(), b.capabilities(),
                                   sameCapability)) {
    return incompatible("payload format or capabilities differ");
  }
  return Error::success();
}

CompiledProgram::CompiledProgram(TargetEnvironment environment,
                                 std::string payload,
                                 QDMI_Program_Format format,
                                 std::optional<ResultLayout> resultLayout)
    : environment_(std::move(environment)), payload_(std::move(payload)),
      format_(format), resultLayout_(std::move(resultLayout)) {}

static Expected<std::string> serializeQIR(QIRProgram program,
                                          QDMI_Program_Format format) {
  auto entryPoint = qir::getMainFunction(program.module());
  if (!entryPoint || entryPoint.isVarArg() ||
      entryPoint.getNumArguments() != 0 ||
      !entryPoint.getFunctionType().getReturnType().isInteger(64)) {
    return createStringError(
        std::errc::invalid_argument,
        "Compiled QDMI QIR requires an i64 () entry point");
  }
  const auto context = program.context();
  CompilerDiagnostics diagnostics(context.get());
  if (qdmi::isBinaryProgramFormat(format)) {
    std::string bytes;
    llvm::raw_string_ostream stream(bytes);
    if (succeeded(std::move(program).intoBitcode(stream))) {
      return bytes;
    }
  } else if (auto source = std::move(program).intoLLVMIR(); succeeded(source)) {
    return std::move(*source);
  }
  return diagnostics.takeError(
      std::make_error_code(std::errc::invalid_argument));
}

Expected<CompiledProgram> CompiledProgram::compile(
    CompilerInput program, const TargetEnvironment& environment,
    const CompilationOptions& options,
    std::optional<TerminalReadout> terminalReadout, bool physicalQubits) {
  if (terminalReadout || !physicalQubits) {
    auto native = runDefaultPipeline(std::move(program), ProgramFormat::QC,
                                     environment, options);
    if (!native) {
      return native.takeError();
    }
    return serialize(std::move(std::get<QCProgram>(*native)), environment,
                     terminalReadout, physicalQubits);
  }
  auto format = qdmiFormatForPayload(environment.payloadSpecification());
  if (!format) {
    return format.takeError();
  }
  auto result = runDefaultPipeline(std::move(program), environment, options);
  if (!result) {
    return result.takeError();
  }
  if (auto* qasm = std::get_if<OpenQASMProgram>(&*result)) {
    return CompiledProgram(environment, std::move(*qasm).intoSource(), *format);
  }
  if (auto* json = std::get_if<IQMJSONProgram>(&*result)) {
    return CompiledProgram(environment, std::move(*json).intoSource(), *format);
  }
  auto& qir = std::get<QIRProgram>(*result);
  auto payload = serializeQIR(std::move(qir), *format);
  if (!payload) {
    return payload.takeError();
  }
  return CompiledProgram(environment, std::move(*payload), *format);
}

Expected<CompiledProgram> CompiledProgram::serialize(
    QCProgram program, const TargetEnvironment& environment,
    std::optional<TerminalReadout> terminalReadout, bool physicalQubits) {
  auto format = qdmiFormatForPayload(environment.payloadSpecification());
  if (!format) {
    return format.takeError();
  }
  const bool qasm = *format == QDMI_PROGRAM_FORMAT_QASM3;
  if (!physicalQubits && !qasm) {
    return createStringError(std::errc::invalid_argument,
                             "virtual qubit addressing requires OpenQASM 3");
  }
  if (terminalReadout == TerminalReadout::All &&
      llvm::any_of(llvm::enumerate(environment.target().siteIds()),
                   [](auto entry) { return entry.index() != entry.value(); })) {
    return createStringError(
        std::errc::invalid_argument,
        "all-qubit readout requires contiguous target site IDs from zero");
  }
  const auto context = program.context();
  CompilerDiagnostics diagnostics(context.get());
  if (failed(program.cleanup())) {
    return diagnostics.takeError(
        std::make_error_code(std::errc::invalid_argument));
  }
  if (!environment.target().supportsOperation("gphase", 0, 1)) {
    IRRewriter rewriter(context.get());
    mqt::dropGlobalPhases(program.module(), rewriter);
  }
  std::optional<ResultLayout> layout;
  if (terminalReadout) {
    auto normalized =
        normalizeTerminalReadout(program.module(), *terminalReadout);
    if (failed(normalized)) {
      return diagnostics.takeError(
          std::make_error_code(std::errc::invalid_argument));
    }
    if (!normalized->isIdentity()) {
      layout = std::move(*normalized);
    }
  }
  attachTargetEnvironment(program.module(), environment);
  if (qasm || *format == QDMI_PROGRAM_FORMAT_QASM2) {
    auto source = qasm ? std::move(program).intoOpenQASM3(physicalQubits)
                       : std::move(program).intoOpenQASM2();
    if (succeeded(source)) {
      return CompiledProgram(environment, std::move(*source).intoSource(),
                             *format, std::move(layout));
    }
  } else if (*format == QDMI_PROGRAM_FORMAT_IQMJSON) {
    auto source = std::move(program).intoIQMJSON(environment.target());
    if (succeeded(source)) {
      return CompiledProgram(environment, std::move(*source).intoSource(),
                             *format, std::move(layout));
    }
  } else {
    const auto qirProfile = (*format == QDMI_PROGRAM_FORMAT_QIRADAPTIVEMODULE ||
                             *format == QDMI_PROGRAM_FORMAT_QIRADAPTIVESTRING)
                                ? QIRProfile::Adaptive
                                : QIRProfile::Base;
    auto qir = std::move(program).intoQIR(qirProfile);
    if (succeeded(qir)) {
      auto payload = serializeQIR(std::move(*qir), *format);
      if (!payload) {
        return payload.takeError();
      }
      return CompiledProgram(environment, std::move(*payload), *format,
                             std::move(layout));
    }
  }
  return diagnostics.takeError(
      std::make_error_code(std::errc::invalid_argument));
}

Expected<CompiledProgram>
CompiledProgram::serialize(QCProgram program, const qdmi::Device& device,
                           std::optional<QDMI_Program_Format> format,
                           std::optional<TerminalReadout> terminalReadout,
                           bool physicalQubits) {
  auto environment = snapshotTargetEnvironment(device, format, false);
  if (!environment) {
    return environment.takeError();
  }
  return serialize(std::move(program), *environment, terminalReadout,
                   physicalQubits);
}

Expected<CompiledProgram>
compileProgram(CompilerInput program, const qdmi::Device& device,
               std::optional<QDMI_Program_Format> format,
               const CompilationOptions& options,
               std::optional<TerminalReadout> terminalReadout,
               bool physicalQubits) {
  auto environment = targetEnvironmentFromDevice(device, format);
  if (!environment) {
    return environment.takeError();
  }
  return CompiledProgram::compile(std::move(program), *environment, options,
                                  terminalReadout, physicalQubits);
}

static Expected<qdmi::Job>
submitPayload(const qdmi::Device& device, const CompiledProgram& program,
              int64_t numShots,
              const std::optional<qdmi::CustomJobParameter>& custom1,
              const std::optional<qdmi::CustomJobParameter>& custom2,
              const std::optional<qdmi::CustomJobParameter>& custom3,
              const std::optional<qdmi::CustomJobParameter>& custom4,
              const std::optional<qdmi::CustomJobParameter>& custom5) {
  if (qdmi::isBinaryProgramFormat(program.programFormat())) {
    const auto payload = std::as_bytes(std::span(program.payload()));
    return device.submitJob(payload, program.programFormat(),
                            static_cast<size_t>(numShots), custom1, custom2,
                            custom3, custom4, custom5);
  }
  const auto& payload = program.payload();
  return device.submitJob(payload, program.programFormat(),
                          static_cast<size_t>(numShots), custom1, custom2,
                          custom3, custom4, custom5);
}

Expected<qdmi::Job>
submitProgram(const qdmi::Device& device, const CompiledProgram& program,
              int64_t numShots,
              const std::optional<qdmi::CustomJobParameter>& custom1,
              const std::optional<qdmi::CustomJobParameter>& custom2,
              const std::optional<qdmi::CustomJobParameter>& custom3,
              const std::optional<qdmi::CustomJobParameter>& custom4,
              const std::optional<qdmi::CustomJobParameter>& custom5) {
  if (numShots < 0) {
    return createStringError(std::make_error_code(std::errc::invalid_argument),
                             "num_shots must be nonnegative");
  }
  auto destination =
      snapshotTargetEnvironment(device, program.programFormat(), false);
  if (!destination) {
    return destination.takeError();
  }
  if (auto error =
          validateTargetCompatibility(program.environment(), *destination)) {
    return error;
  }
  return submitPayload(device, program, numShots, custom1, custom2, custom3,
                       custom4, custom5);
}

Expected<qdmi::Job>
submitProgram(const qdmi::Device& device, CompilerInput program,
              int64_t numShots, std::optional<QDMI_Program_Format> format,
              const std::optional<qdmi::CustomJobParameter>& custom1,
              const std::optional<qdmi::CustomJobParameter>& custom2,
              const std::optional<qdmi::CustomJobParameter>& custom3,
              const std::optional<qdmi::CustomJobParameter>& custom4,
              const std::optional<qdmi::CustomJobParameter>& custom5,
              const CompilationOptions& options) {
  if (numShots < 0) {
    return createStringError(std::make_error_code(std::errc::invalid_argument),
                             "num_shots must be nonnegative");
  }
  auto compiled = compileProgram(std::move(program), device, format, options);
  if (!compiled) {
    return compiled.takeError();
  }
  if (compiled->resultLayout()) {
    return createStringError(std::errc::invalid_argument,
                             "compile the program explicitly and retain its "
                             "result layout before submission");
  }
  return submitPayload(device, *compiled, numShots, custom1, custom2, custom3,
                       custom4, custom5);
}

llvm::Expected<std::optional<qdmi::Job>>
trySubmitPrograms(const qdmi::Device& device,
                  ArrayRef<const CompiledProgram*> programs, int64_t numShots,
                  const std::optional<qdmi::CustomJobParameter>& custom1,
                  const std::optional<qdmi::CustomJobParameter>& custom2,
                  const std::optional<qdmi::CustomJobParameter>& custom3,
                  const std::optional<qdmi::CustomJobParameter>& custom4,
                  const std::optional<qdmi::CustomJobParameter>& custom5) {
  if (numShots < 0 || programs.empty() ||
      llvm::is_contained(programs, nullptr)) {
    return llvm::createStringError(
        std::make_error_code(std::errc::invalid_argument),
        "A program batch requires non-null programs and nonnegative num_shots");
  }
  const auto format = programs.front()->programFormat();
  auto destination = snapshotTargetEnvironment(device, format, false, false);
  if (!destination) {
    return destination.takeError();
  }
  for (const auto* program : programs) {
    if (program->programFormat() != format) {
      return llvm::createStringError(
          std::make_error_code(std::errc::invalid_argument),
          "A program batch must use the same payload format");
    }
    if (auto error =
            validateTargetCompatibility(program->environment(), *destination)) {
      return error;
    }
  }
  if (qdmi::isBinaryProgramFormat(format)) {
    std::vector<std::span<const std::byte>> payloads;
    payloads.reserve(programs.size());
    for (const auto* program : programs) {
      payloads.push_back(std::as_bytes(std::span(program->payload())));
    }
    return device.trySubmitJob(payloads, format, static_cast<size_t>(numShots),
                               custom1, custom2, custom3, custom4, custom5);
  }
  std::vector<std::string> payloads;
  payloads.reserve(programs.size());
  for (const auto* program : programs) {
    payloads.push_back(program->payload());
  }
  return device.trySubmitJob(payloads, format, static_cast<size_t>(numShots),
                             custom1, custom2, custom3, custom4, custom5);
}

} // namespace mlir
