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
#include "mqt/Dialect/QIR/Utils/QIRUtils.h"
#include "mqt/Support/Diagnostics.h"
#include "qdmi/Client.hpp"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/CheckedArithmetic.h"
#include "llvm/Support/ErrorHandling.h"

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
#include <variant>
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

[[nodiscard]] static LogicalResult
requireAdapterInput(bool condition, const llvm::Twine& message) {
  if (!condition) {
    return ::mqt::emitError(message.str(),
                            ::mqt::ErrorCategory::InvalidArgument);
  }
  return success();
}

[[nodiscard]] static LogicalResult
requireCircuitDevice(bool condition, llvm::StringRef deviceName,
                     llvm::StringRef detail) {
  return requireAdapterInput(
      condition, llvm::Twine("QDMI device '") + deviceName +
                     "' cannot be used as an MQT compiler target: only "
                     "circuit-model devices with one qubit per non-zone site "
                     "are supported (" +
                     detail + ")");
}

[[nodiscard]] static LogicalResult
requireRepresentableOperation(bool condition, llvm::StringRef deviceName,
                              llvm::StringRef operationName,
                              llvm::StringRef detail) {
  return requireAdapterInput(
      condition, llvm::Twine("QDMI device '") + deviceName + "' operation '" +
                     operationName +
                     "' cannot be represented by the MQT compiler target (" +
                     detail + ")");
}

[[nodiscard]] static FailureOr<CompilerTarget::SiteId>
checkedSiteId(size_t index) {
  if (failed(requireAdapterInput(
          index <= static_cast<uintmax_t>(
                       std::numeric_limits<CompilerTarget::SiteId>::max()),
          "QDMI site index exceeds the nonnegative i64 compiler-target "
          "domain"))) {
    return failure();
  }
  return static_cast<CompilerTarget::SiteId>(index);
}

using SiteIndices = DenseMap<QDMI_Site, CompilerTarget::SiteId>;

[[nodiscard]] static FailureOr<CompilerTarget::SiteId>
snapshotSiteIndex(const qdmi::Site& site, SiteIndices& indices) {
  const auto found = indices.find(site);
  if (found != indices.end()) {
    return found->second;
  }
  auto rawIndex = site.getIndex();
  if (failed(rawIndex)) {
    return failure();
  }
  auto index = checkedSiteId(*rawIndex);
  if (failed(index)) {
    return failure();
  }
  indices.try_emplace(site, *index);
  return *index;
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

[[nodiscard]] static LogicalResult validateHomogeneousSupport(
    StringRef operationName, size_t arity,
    const std::vector<qdmi::Site>& flattenedSites,
    const DenseSet<CompilerTarget::SiteId>& knownSites,
    const std::optional<std::set<CompilerTarget::Coupling>>& couplings,
    StringRef deviceName, SiteIndices& indices) {
  if (failed(requireRepresentableOperation(
          flattenedSites.size() % arity == 0, deviceName, operationName,
          "the reported site list is not divisible by the fixed arity"))) {
    return failure();
  }
  if (arity > 2) {
    std::set<std::vector<CompilerTarget::SiteId>> supportedTuples;
    for (size_t offset = 0; offset < flattenedSites.size(); offset += arity) {
      std::vector<CompilerTarget::SiteId> tuple;
      tuple.reserve(arity);
      for (size_t index = 0; index < arity; ++index) {
        auto siteId =
            snapshotSiteIndex(flattenedSites[offset + index], indices);
        if (failed(siteId)) {
          return failure();
        }
        if (failed(requireRepresentableOperation(
                knownSites.contains(*siteId) &&
                    !llvm::is_contained(tuple, *siteId),
                deviceName, operationName,
                "each higher-arity site tuple must contain distinct device "
                "sites"))) {
          return failure();
        }
        tuple.emplace_back(*siteId);
      }
      if (failed(requireRepresentableOperation(
              supportedTuples.emplace(std::move(tuple)).second, deviceName,
              operationName,
              "the reported higher-arity site tuples must be unique"))) {
        return failure();
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
    DenseSet<CompilerTarget::SiteId> supportedSites;
    supportedSites.reserve(flattenedSites.size());
    for (const auto& site : flattenedSites) {
      auto siteId = snapshotSiteIndex(site, indices);
      if (failed(siteId)) {
        return failure();
      }
      const auto inserted = supportedSites.insert(*siteId).second;
      if (failed(requireRepresentableOperation(
              knownSites.contains(*siteId) && inserted, deviceName,
              operationName,
              "the reported one-qubit sites must be unique device sites"))) {
        return failure();
      }
    }
    return requireRepresentableOperation(
        supportedSites.size() == knownSites.size(), deviceName, operationName,
        "support is not homogeneous across all device sites");
  }

  std::set<CompilerTarget::Coupling> reportedTuples;
  std::set<CompilerTarget::Coupling> supportedCouplings;
  for (size_t offset = 0; offset < flattenedSites.size(); offset += arity) {
    auto first = snapshotSiteIndex(flattenedSites[offset], indices);
    if (failed(first)) {
      return failure();
    }
    auto second = snapshotSiteIndex(flattenedSites[offset + 1], indices);
    if (failed(second)) {
      return failure();
    }
    const auto inserted = reportedTuples.insert({*first, *second}).second;
    const auto validTuple = *first != *second && knownSites.contains(*first) &&
                            knownSites.contains(*second) && inserted;
    if (failed(requireRepresentableOperation(
            validTuple, deviceName, operationName,
            "the reported two-qubit sites must be "
            "unique pairs of device sites"))) {
      return failure();
    }
    supportedCouplings.insert(canonicalCoupling(*first, *second));
  }

  auto coversTarget = false;
  if (!couplings) {
    const auto expected = allToAllCouplingCount(knownSites.size());
    coversTarget = expected && supportedCouplings.size() == *expected;
  } else {
    coversTarget = supportedCouplings == *couplings;
  }
  return requireRepresentableOperation(
      coversTarget, deviceName, operationName,
      couplings ? "support is not homogeneous across all topology edges"
                : "support is not homogeneous across all-to-all site pairs");
}

[[nodiscard]] static FailureOr<std::optional<CompilerTarget::DurationUnit>>
snapshotDurationUnit(const qdmi::Device& device) {
  auto unitResult = device.getDurationUnit();
  if (failed(unitResult)) {
    return failure();
  }
  auto unit = std::move(*unitResult);
  auto scaleFactorResult = device.getDurationScaleFactor();
  if (failed(scaleFactorResult)) {
    return failure();
  }
  auto scaleFactor = *scaleFactorResult;
  if (failed(requireAdapterInput(unit || !scaleFactor,
                                 "QDMI device reports a duration scale "
                                 "factor without a duration unit"))) {
    return failure();
  }
  if (!unit) {
    return std::optional<CompilerTarget::DurationUnit>{};
  }
  auto durationUnit = CompilerTarget::DurationUnit::create(
      std::move(*unit), scaleFactor.value_or(1.));
  if (failed(durationUnit)) {
    return failure();
  }
  return std::optional<CompilerTarget::DurationUnit>(std::in_place,
                                                     std::move(*durationUnit));
}

[[nodiscard]] static FailureOr<std::vector<CompilerTarget::SiteTuple>>
snapshotOperationSites(const qdmi::Operation& operation, size_t arity,
                       const std::vector<qdmi::Site>& flattenedSites,
                       std::optional<uint64_t> defaultDuration,
                       std::optional<double> defaultFidelity, bool variadic,
                       const SiteIndices& indices, bool includeCalibration) {
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
      /// Homogeneous-support validation has resolved every reported site.
      siteIds.emplace_back(indices.find(site)->second);
    }

    auto durationResult =
        includeCalibration
            ? operation.getDuration(sites)
            : FailureOr<std::optional<uint64_t>>(std::optional<uint64_t>{});
    if (failed(durationResult)) {
      return failure();
    }
    auto duration = *durationResult;
    auto fidelityResult =
        includeCalibration
            ? operation.getFidelity(sites)
            : FailureOr<std::optional<double>>(std::optional<double>{});
    if (failed(fidelityResult)) {
      return failure();
    }
    auto fidelity = *fidelityResult;
    const bool hasSiteCalibration =
        duration != defaultDuration || fidelity != defaultFidelity;
    if (!variadic || hasSiteCalibration) {
      auto siteTuple = CompilerTarget::SiteTuple::create(
          std::move(siteIds),
          duration == defaultDuration ? std::nullopt : duration,
          fidelity == defaultFidelity ? std::nullopt : fidelity);
      if (failed(siteTuple)) {
        return failure();
      }
      result.emplace_back(std::move(*siteTuple));
    }
  }
  return result;
}

[[nodiscard]] static FailureOr<CompilerTarget::NativeOperations>
snapshotOperations(
    const std::vector<qdmi::Operation>& operations,
    const std::vector<CompilerTarget::Site>& deviceSites,
    const std::optional<std::vector<CompilerTarget::Coupling>>& couplings,
    StringRef deviceName, bool homogeneousOperationSupport,
    SiteIndices& indices, bool includeCalibration) {
  DenseSet<CompilerTarget::SiteId> knownSites;
  std::optional<std::set<CompilerTarget::Coupling>> expectedCouplings;
  if (couplings) {
    expectedCouplings.emplace();
    for (const auto& [first, second] : *couplings) {
      expectedCouplings->insert(canonicalCoupling(first, second));
    }
  }
  std::vector<CompilerTarget::OperationCapability> targetOperations;
  targetOperations.reserve(operations.size());
  for (const auto& operation : operations) {
    auto isZonedResult = operation.isZoned();
    if (failed(isZonedResult)) {
      return failure();
    }
    if (failed(requireCircuitDevice(!*isZonedResult, deviceName,
                                    "the device exposes a zoned operation"))) {
      return failure();
    }
    auto arityResult = operation.getQubitsNum();
    if (failed(arityResult)) {
      return failure();
    }
    auto arity = *arityResult;
    if (!arity) {
      continue;
    }
    auto operationNameResult = operation.getName();
    if (failed(operationNameResult)) {
      return failure();
    }
    auto operationName = std::move(*operationNameResult);
    auto metadataResult = operation.queryCustomProperty<std::vector<std::byte>>(
        qdmi::CustomProperty::Custom1);
    if (failed(metadataResult)) {
      return failure();
    }
    auto metadata = std::move(*metadataResult);
    const auto hasArbitraryPositiveControls =
        matchesMetadata(metadata, ARBITRARY_POSITIVE_CONTROLS_METADATA);
    if (failed(requireRepresentableOperation(
            !hasArbitraryPositiveControls ||
                (*arity > 0 && homogeneousOperationSupport),
            deviceName, operationName,
            "arbitrary positive controls require a positive base arity and "
            "homogeneous operation support"))) {
      return failure();
    }
    auto flattenedSitesResult = operation.getSites();
    if (failed(flattenedSitesResult)) {
      return failure();
    }
    auto flattenedSites = std::move(*flattenedSitesResult);
    if (*arity > 0 && flattenedSites && flattenedSites->empty()) {
      continue;
    }
    if (failed(requireRepresentableOperation(
            *arity == 0 || flattenedSites || homogeneousOperationSupport,
            deviceName, operationName,
            "the supported sites are not reported"))) {
      return failure();
    }
    const bool queryCalibration =
        includeCalibration || hasArbitraryPositiveControls;
    auto durationResult =
        queryCalibration
            ? operation.getDuration()
            : FailureOr<std::optional<uint64_t>>(std::optional<uint64_t>{});
    if (failed(durationResult)) {
      return failure();
    }
    auto duration = *durationResult;
    auto fidelityResult =
        queryCalibration
            ? operation.getFidelity()
            : FailureOr<std::optional<double>>(std::optional<double>{});
    if (failed(fidelityResult)) {
      return failure();
    }
    auto fidelity = *fidelityResult;
    std::vector<CompilerTarget::SiteTuple> siteTuples;
    if (*arity == 0) {
      if (failed(requireRepresentableOperation(
              !flattenedSites || flattenedSites->empty(), deviceName,
              operationName,
              "a zero-arity operation cannot report supported sites"))) {
        return failure();
      }
    } else if (flattenedSites) {
      if (knownSites.empty()) {
        knownSites.reserve(deviceSites.size());
        for (const auto& site : deviceSites) {
          knownSites.insert(site.id());
        }
      }
      if (failed(validateHomogeneousSupport(
              operationName, *arity, *flattenedSites, knownSites,
              expectedCouplings, deviceName, indices))) {
        return failure();
      }
      auto tuples = snapshotOperationSites(
          operation, *arity, *flattenedSites, duration, fidelity,
          hasArbitraryPositiveControls, indices, queryCalibration);
      if (failed(tuples)) {
        return failure();
      }
      siteTuples = std::move(*tuples);
    }
    if (failed(requireRepresentableOperation(
            !hasArbitraryPositiveControls || siteTuples.empty(), deviceName,
            operationName,
            "a variadic operation cannot retain site-specific calibration"))) {
      return failure();
    }
    const auto targetArity =
        hasArbitraryPositiveControls
            ? CompilerTarget::OperationCapability::Arity::variadic(*arity)
            : CompilerTarget::OperationCapability::Arity::fixed(*arity);
    auto numParametersResult = operation.getParametersNum();
    if (failed(numParametersResult)) {
      return failure();
    }
    auto targetOperation = CompilerTarget::OperationCapability::create(
        std::move(operationName), targetArity, *numParametersResult,
        std::move(siteTuples), includeCalibration ? duration : std::nullopt,
        includeCalibration ? fidelity : std::nullopt);
    if (failed(targetOperation)) {
      return failure();
    }
    targetOperations.emplace_back(std::move(*targetOperation));
  }
  return CompilerTarget::NativeOperations::fromOperations(targetOperations);
}

[[nodiscard]] static FailureOr<CompilerTarget>
snapshotCompilerTarget(const qdmi::Device& device,
                       bool includeCalibration = true) {
  auto deviceNameResult = device.getName();
  if (failed(deviceNameResult)) {
    return failure();
  }
  auto deviceName = std::move(*deviceNameResult);
  auto metadataResult = device.queryCustomProperty<std::vector<std::byte>>(
      qdmi::CustomProperty::Custom1);
  if (failed(metadataResult)) {
    return failure();
  }
  auto metadata = std::move(*metadataResult);
  const auto hasHomogeneousAllToAllMetadata =
      matchesMetadata(metadata, ALL_TO_ALL_HOMOGENEOUS_METADATA);
  auto deviceSitesResult = device.getSites();
  if (failed(deviceSitesResult)) {
    return failure();
  }
  auto deviceSites = std::move(*deviceSitesResult);
  for (const auto& site : deviceSites) {
    auto isZoneResult = site.isZone();
    if (failed(isZoneResult)) {
      return failure();
    }
    if (failed(requireCircuitDevice(!*isZoneResult, deviceName,
                                    "the device exposes zone sites"))) {
      return failure();
    }
  }
  auto numQubitsResult = device.getQubitsNum();
  if (failed(numQubitsResult)) {
    return failure();
  }
  if (failed(requireCircuitDevice(
          *numQubitsResult == deviceSites.size(), deviceName,
          "the qubit count does not match the regular-site count"))) {
    return failure();
  }

  SiteIndices indices;
  indices.reserve(deviceSites.size());
  std::vector<CompilerTarget::Site> sites;
  sites.reserve(deviceSites.size());
  for (const auto& site : deviceSites) {
    auto siteId = snapshotSiteIndex(site, indices);
    if (failed(siteId)) {
      return failure();
    }
    auto nameResult = includeCalibration
                          ? site.getName()
                          : FailureOr<std::optional<std::string>>(
                                std::optional<std::string>{});
    if (failed(nameResult)) {
      return failure();
    }
    auto name = std::move(*nameResult);
    auto t1Result =
        includeCalibration
            ? site.getT1()
            : FailureOr<std::optional<uint64_t>>(std::optional<uint64_t>{});
    if (failed(t1Result)) {
      return failure();
    }
    auto t1 = *t1Result;
    auto t2Result =
        includeCalibration
            ? site.getT2()
            : FailureOr<std::optional<uint64_t>>(std::optional<uint64_t>{});
    if (failed(t2Result)) {
      return failure();
    }
    auto t2 = *t2Result;
    auto targetSite =
        CompilerTarget::Site::create(*siteId, std::move(name), t1, t2);
    if (failed(targetSite)) {
      return failure();
    }
    sites.emplace_back(std::move(*targetSite));
  }

  std::optional<std::vector<CompilerTarget::Coupling>> couplings;
  auto deviceCouplingsResult = device.getCouplingMap();
  if (failed(deviceCouplingsResult)) {
    return failure();
  }
  auto deviceCouplings = std::move(*deviceCouplingsResult);
  if (deviceCouplings) {
    couplings.emplace();
    couplings->reserve(deviceCouplings->size());
    for (const auto& [source, target] : *deviceCouplings) {
      auto sourceId = snapshotSiteIndex(source, indices);
      if (failed(sourceId)) {
        return failure();
      }
      auto targetId = snapshotSiteIndex(target, indices);
      if (failed(targetId)) {
        return failure();
      }
      couplings->emplace_back(*sourceId, *targetId);
    }
  }
  if (failed(requireAdapterInput(
          couplings || hasHomogeneousAllToAllMetadata || sites.size() == 1,
          llvm::Twine("QDMI device '") + deviceName +
              "' cannot be used as an MQT compiler target: connectivity is "
              "not reported"))) {
    return failure();
  }

  auto deviceOperationsResult = device.getOperations();
  if (failed(deviceOperationsResult)) {
    return failure();
  }
  auto deviceOperations = std::move(*deviceOperationsResult);
  auto operations = snapshotOperations(
      deviceOperations, sites, couplings, deviceName,
      hasHomogeneousAllToAllMetadata, indices, includeCalibration);
  if (failed(operations)) {
    return failure();
  }
  auto durationUnit = snapshotDurationUnit(device);
  if (failed(durationUnit)) {
    return failure();
  }
  auto connectivity =
      couplings ? CompilerTarget::Connectivity::fromCouplings(*couplings)
                : CompilerTarget::Connectivity::allToAll();
  return CompilerTarget::create(std::move(deviceName), std::move(sites),
                                std::move(connectivity), std::move(*operations),
                                std::move(*durationUnit));
}

FailureOr<CompilerTarget> compilerTargetFromDevice(const qdmi::Device& device) {
  return snapshotCompilerTarget(device);
}

FailureOr<CompilerTarget>
compilerTargetFromDeviceId(const std::string_view deviceId) {
  auto device = qdmi::Session::openDevice(deviceId);
  if (failed(device)) {
    return failure();
  }
  return snapshotCompilerTarget(*device);
}

FailureOr<std::vector<std::string>> registeredQDMIDeviceIds() {
  auto session = qdmi::Session::create();
  if (failed(session)) {
    return failure();
  }
  return session->getDeviceIds();
}

constexpr std::array PROGRAM_FORMAT_PREFERENCE{
    QDMI_PROGRAM_FORMAT_QIRADAPTIVEMODULE,
    QDMI_PROGRAM_FORMAT_QIRADAPTIVESTRING,
    QDMI_PROGRAM_FORMAT_QASM3,
    QDMI_PROGRAM_FORMAT_QIRBASEMODULE,
    QDMI_PROGRAM_FORMAT_QIRBASESTRING,
};

static LogicalResult incompatible(const llvm::Twine& detail) {
  return ::mqt::emitError(
      ("Compiled program is incompatible with the destination: " + detail +
       "; recompile for this device")
          .str(),
      ::mqt::ErrorCategory::InvalidArgument);
}

static bool sameOperation(const CompilerTarget::OperationCapability& lhs,
                          const CompilerTarget::OperationCapability& rhs) {
  if (lhs.canonicalName() != rhs.canonicalName() ||
      lhs.arity() != rhs.arity() ||
      lhs.numParameters() != rhs.numParameters() ||
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

static FailureOr<QDMI_Program_Format>
qdmiFormatForPayload(const PayloadSpecification& payload) {
  auto output = payload.compilerOutput();
  if (failed(output)) {
    return failure();
  }
  const bool binary = payload.format().encoding == PayloadEncoding::Binary;
  switch (*output) {
  case ProgramFormat::OpenQASM3:
    return QDMI_PROGRAM_FORMAT_QASM3;
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

/// Optional QIR module flags and their payload capability IDs.
constexpr std::array QIR_OPTIONAL_CAPABILITIES{
    std::pair{"dynamic_qubit_management", "qir.dynamic-qubit-management"},
    std::pair{"dynamic_result_management", "qir.dynamic-result-management"},
    std::pair{"arrays", "qir.arrays"},
    std::pair{"ir_functions", "qir.ir-functions"},
    std::pair{"multiple_return_points", "qir.multiple-return-points"},
    std::pair{"int_computations", "qir.int-computations"},
    std::pair{"float_computations", "qir.float-computations"},
    std::pair{"multiple_target_branching", "multiway-branching"},
};

static LogicalResult
validateQIRCapabilities(ModuleOp moduleOp,
                        const PayloadSpecification& payload) {
  auto optionalFeatures = llvm::ArrayRef(QIR_OPTIONAL_CAPABILITIES);
  for (auto flags : moduleOp.getOps<LLVM::ModuleFlagsOp>()) {
    for (auto attribute : flags.getFlags()) {
      const auto flag = cast<LLVM::ModuleFlagAttr>(attribute);
      const auto key = flag.getKey().getValue();
      const auto* const feature =
          llvm::find_if(optionalFeatures,
                        [&](const auto& entry) { return key == entry.first; });
      if (feature == optionalFeatures.end()) {
        continue;
      }
      if (const auto value = dyn_cast<IntegerAttr>(flag.getValue());
          value && value.getValue().isZero()) {
        continue;
      }
      const auto* const capability =
          llvm::find_if(payload.capabilities(), [&](const auto& entry) {
            return entry.id == feature->second;
          });
      if (capability != payload.capabilities().end() &&
          capability->value == 0 &&
          (capability->id == ProgramCapability::MULTIWAY_BRANCHING ||
           capability->constraints.empty())) {
        continue;
      }
      return ::mqt::emitError(
          (llvm::Twine("Selected QIR payload requires capability '") +
           feature->second +
           "' that the device contract does not permit; select another "
           "supported program_format")
              .str(),
          ::mqt::ErrorCategory::InvalidArgument);
    }
  }
  return success();
}

FailureOr<PayloadSpecification>
payloadSpecificationForProgramFormat(QDMI_Program_Format format) {
  switch (format) {
  case QDMI_PROGRAM_FORMAT_QASM3:
    return PayloadSpecification::create(
        {
            .id = "openqasm",
            .version = "3.1.0",
            .profile = "",
            .encoding = PayloadEncoding::Text,
        },
        {
            {.id = ProgramCapability::FORWARD_BRANCHING.str()},
            {.id = ProgramCapability::COUNTED_ITERATION.str()},
            {.id = ProgramCapability::CONDITIONAL_LOOP.str()},
            {.id = ProgramCapability::MULTIWAY_BRANCHING.str()},
        });
  case QDMI_PROGRAM_FORMAT_QIRBASEMODULE:
  case QDMI_PROGRAM_FORMAT_QIRBASESTRING:
    return PayloadSpecification::create({
        .id = "qir",
        .version = "2.1.0",
        .profile = "base",
        .encoding = format == QDMI_PROGRAM_FORMAT_QIRBASEMODULE
                        ? PayloadEncoding::Binary
                        : PayloadEncoding::Text,
    });
  case QDMI_PROGRAM_FORMAT_QIRADAPTIVEMODULE:
  case QDMI_PROGRAM_FORMAT_QIRADAPTIVESTRING: {
    std::vector<ProgramCapability> capabilities{
        {.id = ProgramCapability::FORWARD_BRANCHING.str()},
        {.id = ProgramCapability::COUNTED_ITERATION.str()},
        {.id = ProgramCapability::CONDITIONAL_LOOP.str()},
    };
    for (const auto& [flag, id] : QIR_OPTIONAL_CAPABILITIES) {
      capabilities.push_back({.id = id});
    }
    return PayloadSpecification::create(
        {
            .id = "qir",
            .version = "2.1.0",
            .profile = "adaptive",
            .encoding = format == QDMI_PROGRAM_FORMAT_QIRADAPTIVEMODULE
                            ? PayloadEncoding::Binary
                            : PayloadEncoding::Text,
        },
        std::move(capabilities));
  }
  default:
    return ::mqt::emitError(
        "MQT compiler cannot emit the requested QDMI program format",
        ::mqt::ErrorCategory::InvalidArgument);
  }
}

static FailureOr<TargetEnvironment>
snapshotTargetEnvironment(const qdmi::Device& device,
                          std::optional<QDMI_Program_Format> format,
                          bool includeCalibration) {
  auto supportedResult = device.getSupportedProgramFormats();
  if (failed(supportedResult)) {
    return failure();
  }
  auto supported = std::move(*supportedResult);
  if (format && !llvm::is_contained(supported, *format)) {
    return ::mqt::emitError(
        "Device does not support the requested program_format",
        ::mqt::ErrorCategory::InvalidArgument);
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
    return ::mqt::emitError(
        "Device has no executable program format supported by the MQT "
        "compiler; hardware-only models require an explicit compiler target "
        "and output",
        ::mqt::ErrorCategory::InvalidArgument);
  }
  auto payload = payloadSpecificationForProgramFormat(*format);
  if (failed(payload)) {
    return failure();
  }
  auto target = snapshotCompilerTarget(device, includeCalibration);
  if (failed(target)) {
    return failure();
  }
  return TargetEnvironment(*target, std::move(*payload));
}

FailureOr<TargetEnvironment>
targetEnvironmentFromDevice(const qdmi::Device& device,
                            std::optional<QDMI_Program_Format> format) {
  return snapshotTargetEnvironment(device, format, true);
}

LogicalResult
validateTargetCompatibility(const TargetEnvironment& compiled,
                            const TargetEnvironment& destination) {
  const auto& lhs = compiled.target();
  const auto& rhs = destination.target();
  if (lhs.siteIds() != rhs.siteIds()) {
    return incompatible("ordered site mapping differs");
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
  if (a.format() != b.format() ||
      !std::ranges::is_permutation(a.capabilities(), b.capabilities(),
                                   sameCapability)) {
    return incompatible("payload format or capabilities differ");
  }
  return success();
}

CompiledProgram::CompiledProgram(TargetEnvironment environment,
                                 std::string payload,
                                 QDMI_Program_Format format)
    : environment_(std::move(environment)), payload_(std::move(payload)),
      format_(format) {}

FailureOr<CompiledProgram>
CompiledProgram::compile(CompilerInput&& program,
                         const TargetEnvironment& environment,
                         const CompilationOptions& options) {
  auto format = qdmiFormatForPayload(environment.payloadSpecification());
  if (failed(format)) {
    return failure();
  }
  auto result = runDefaultPipeline(std::move(program), environment, options);
  if (!result) {
    return ::mqt::emitError(
        "Compilation failed for selected payload " +
            environment.payloadSpecification().format().id + " " +
            environment.payloadSpecification().format().profile +
            "; see compiler diagnostics for the unsupported construct or "
            "capability",
        ::mqt::ErrorCategory::InvalidArgument);
  }
  if (const auto* qasm = std::get_if<OpenQASMProgram>(&*result)) {
    return CompiledProgram(environment, std::string(qasm->source()), *format);
  }
  const auto& qir = std::get<QIRProgram>(*result);
  auto entryPoint = qir::getMainFunction(qir.module());
  if (entryPoint && !entryPoint.isVarArg() &&
      entryPoint.getNumArguments() == 0 &&
      isa<LLVM::LLVMVoidType>(entryPoint.getFunctionType().getReturnType())) {
    /// A program without a return value completes with success status.
    OpBuilder builder(qir.module().getContext());
    entryPoint.setFunctionType(
        LLVM::LLVMFunctionType::get(builder.getI64Type(), {}));
    entryPoint.walk([&](LLVM::ReturnOp returnOp) {
      builder.setInsertionPoint(returnOp);
      auto zero = LLVM::ConstantOp::create(builder, returnOp.getLoc(),
                                           builder.getI64IntegerAttr(0));
      returnOp->setOperands(zero.getResult());
    });
  }
  if (!entryPoint || entryPoint.isVarArg() ||
      entryPoint.getNumArguments() != 0 ||
      !entryPoint.getFunctionType().getReturnType().isInteger(64)) {
    return ::mqt::emitError(
        "Compiled QDMI QIR requires an i64 () entry point; keep classical "
        "temporaries local or select OpenQASM 3",
        ::mqt::ErrorCategory::InvalidArgument);
  }
  if (failed(validateQIRCapabilities(qir.module(),
                                     environment.payloadSpecification()))) {
    return failure();
  }
  if (qdmi::isBinaryProgramFormat(*format)) {
    if (auto bytes = qir.toBitcode()) {
      return CompiledProgram(
          environment,
          std::string(reinterpret_cast<const char*>(bytes->data()),
                      bytes->size()),
          *format);
    }
  } else if (auto text = qir.llvmIR()) {
    return CompiledProgram(environment, std::move(*text), *format);
  }
  return ::mqt::emitError("Failed to serialize compiled QIR payload",
                          ::mqt::ErrorCategory::InvalidArgument);
}

FailureOr<CompiledProgram>
compileProgram(CompilerInput&& program, const qdmi::Device& device,
               std::optional<QDMI_Program_Format> format,
               const CompilationOptions& options) {
  auto environment = targetEnvironmentFromDevice(device, format);
  if (failed(environment)) {
    return failure();
  }
  return CompiledProgram::compile(std::move(program), *environment, options);
}

static FailureOr<qdmi::Job>
submitPayload(const qdmi::Device& device, const CompiledProgram& program,
              int64_t numShots,
              const std::optional<qdmi::CustomJobParameter>& custom1,
              const std::optional<qdmi::CustomJobParameter>& custom2,
              const std::optional<qdmi::CustomJobParameter>& custom3,
              const std::optional<qdmi::CustomJobParameter>& custom4,
              const std::optional<qdmi::CustomJobParameter>& custom5) {
  ::mqt::ScopedDiagnosticHandler context(
      [](const ::mqt::Diagnostic& diagnostic) {
        auto prefixed = diagnostic;
        prefixed.message =
            "Failed to submit compiled program: " + prefixed.message;
        ::mqt::emitDiagnostic(prefixed);
        return success();
      });
  const auto& payload = program.payload();
  const auto size =
      payload.size() +
      (qdmi::isBinaryProgramFormat(program.programFormat()) ? 0 : 1);
  return device.submitJob(std::as_bytes(std::span(payload.c_str(), size)),
                          program.programFormat(),
                          static_cast<size_t>(numShots), custom1, custom2,
                          custom3, custom4, custom5);
}

FailureOr<qdmi::Job>
submitProgram(const qdmi::Device& device, const CompiledProgram& program,
              int64_t numShots,
              const std::optional<qdmi::CustomJobParameter>& custom1,
              const std::optional<qdmi::CustomJobParameter>& custom2,
              const std::optional<qdmi::CustomJobParameter>& custom3,
              const std::optional<qdmi::CustomJobParameter>& custom4,
              const std::optional<qdmi::CustomJobParameter>& custom5) {
  if (numShots < 0) {
    return ::mqt::emitError("num_shots must be nonnegative",
                            ::mqt::ErrorCategory::InvalidArgument);
  }
  auto destination =
      snapshotTargetEnvironment(device, program.programFormat(), false);
  if (failed(destination)) {
    return failure();
  }
  if (failed(
          validateTargetCompatibility(program.environment(), *destination))) {
    return failure();
  }
  return submitPayload(device, program, numShots, custom1, custom2, custom3,
                       custom4, custom5);
}

FailureOr<qdmi::Job>
submitProgram(const qdmi::Device& device, CompilerInput&& input,
              int64_t numShots, std::optional<QDMI_Program_Format> format,
              const std::optional<qdmi::CustomJobParameter>& custom1,
              const std::optional<qdmi::CustomJobParameter>& custom2,
              const std::optional<qdmi::CustomJobParameter>& custom3,
              const std::optional<qdmi::CustomJobParameter>& custom4,
              const std::optional<qdmi::CustomJobParameter>& custom5,
              const CompilationOptions& options) {
  if (numShots < 0) {
    return ::mqt::emitError("num_shots must be nonnegative",
                            ::mqt::ErrorCategory::InvalidArgument);
  }
  auto compiled = compileProgram(std::move(input), device, format, options);
  if (failed(compiled)) {
    return failure();
  }
  return submitPayload(device, *compiled, numShots, custom1, custom2, custom3,
                       custom4, custom5);
}

} // namespace mlir
