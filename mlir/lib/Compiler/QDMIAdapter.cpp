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
#include "mqt/Compiler/TargetCompilation.h"
#include "mqt/Dialect/QIR/Utils/QIRUtils.h"
#include "qdmi/Client.hpp"
#include "qdmi/driver/Driver.hpp"

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
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
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
                     "' cannot be used as an MQT compiler target: only "
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
                     "' cannot be represented by the MQT compiler target (" +
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

using SiteIndices = DenseMap<QDMI_Site, CompilerTarget::SiteId>;

[[nodiscard]] static llvm::Expected<CompilerTarget::SiteId>
snapshotSiteIndex(const qdmi::Site& site, SiteIndices& indices) {
  const auto found = indices.find(site);
  if (found != indices.end()) {
    return found->second;
  }
  auto index = checkedSiteId(site.getIndex());
  if (!index) {
    return index.takeError();
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

[[nodiscard]] static llvm::Error validateHomogeneousSupport(
    StringRef operationName, size_t arity,
    const std::vector<qdmi::Site>& flattenedSites,
    const DenseSet<CompilerTarget::SiteId>& knownSites,
    const std::optional<std::set<CompilerTarget::Coupling>>& couplings,
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

    const auto duration =
        includeCalibration ? operation.getDuration(sites) : std::nullopt;
    const auto fidelity =
        includeCalibration ? operation.getFidelity(sites) : std::nullopt;
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
    if (auto error =
            requireCircuitDevice(!operation.isZoned(), deviceName,
                                 "the device exposes a zoned operation")) {
      return error;
    }
    const auto arity = operation.getQubitsNum();
    if (!arity) {
      continue;
    }
    auto operationName = operation.getName();
    const auto hasArbitraryPositiveControls =
        hasArbitraryPositiveControlsMetadata(operation);
    if (auto error = requireRepresentableOperation(
            !hasArbitraryPositiveControls ||
                (*arity > 0 && homogeneousOperationSupport),
            deviceName, operationName,
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
            deviceName, operationName,
            "the supported sites are not reported")) {
      return error;
    }
    const bool queryCalibration =
        includeCalibration || hasArbitraryPositiveControls;
    const auto duration =
        queryCalibration ? operation.getDuration() : std::nullopt;
    const auto fidelity =
        queryCalibration ? operation.getFidelity() : std::nullopt;
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
      if (auto error = validateHomogeneousSupport(
              operationName, *arity, *flattenedSites, knownSites,
              expectedCouplings, deviceName, indices)) {
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
    auto targetOperation = CompilerTarget::OperationCapability::create(
        std::move(operationName), targetArity, operation.getParametersNum(),
        std::move(siteTuples), includeCalibration ? duration : std::nullopt,
        includeCalibration ? fidelity : std::nullopt);
    if (!targetOperation) {
      return targetOperation.takeError();
    }
    targetOperations.emplace_back(std::move(*targetOperation));
  }
  return CompilerTarget::NativeOperations::fromOperations(targetOperations);
}

[[nodiscard]] static llvm::Expected<CompilerTarget>
snapshotCompilerTarget(const qdmi::Device& device,
                       bool includeCalibration = true) {
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

  SiteIndices indices;
  indices.reserve(deviceSites.size());
  std::vector<CompilerTarget::Site> sites;
  sites.reserve(deviceSites.size());
  for (const auto& site : deviceSites) {
    auto siteId = snapshotSiteIndex(site, indices);
    if (!siteId) {
      return siteId.takeError();
    }
    auto targetSite = CompilerTarget::Site::create(
        *siteId, includeCalibration ? site.getName() : std::nullopt,
        includeCalibration ? site.getT1() : std::nullopt,
        includeCalibration ? site.getT2() : std::nullopt);
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

  auto operations = snapshotOperations(
      device.getOperations(), sites, couplings, deviceName,
      hasHomogeneousAllToAllMetadata, indices, includeCalibration);
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

llvm::Expected<std::vector<std::string>> registeredQDMIDeviceIds() {
  try {
    return qdmi::Driver::get().registeredDeviceIds();
  } catch (...) {
    return qdmiError("Failed to discover registered QDMI devices",
                     std::current_exception());
  }
}

constexpr std::array PROGRAM_FORMAT_PREFERENCE{
    QDMI_PROGRAM_FORMAT_QIRADAPTIVEMODULE,
    QDMI_PROGRAM_FORMAT_QIRADAPTIVESTRING,
    QDMI_PROGRAM_FORMAT_QASM3,
    QDMI_PROGRAM_FORMAT_QIRBASEMODULE,
    QDMI_PROGRAM_FORMAT_QIRBASESTRING,
};

static llvm::Error incompatible(const llvm::Twine& detail) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      "Compiled program is incompatible with the destination: " + detail +
          "; recompile for this device");
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

static llvm::Expected<QDMI_Program_Format>
qdmiFormatForPayload(const PayloadSpecification& payload) {
  auto output = payload.compilerOutput();
  if (!output) {
    return output.takeError();
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

static llvm::Error
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
      return llvm::createStringError(
          std::make_error_code(std::errc::invalid_argument),
          llvm::Twine("Selected QIR payload requires capability '") +
              feature->second +
              "' that the device contract does not permit; select another "
              "supported program_format");
    }
  }
  return llvm::Error::success();
}

llvm::Expected<PayloadSpecification>
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
    return llvm::createStringError(
        std::make_error_code(std::errc::invalid_argument),
        "MQT compiler cannot emit the requested QDMI program format");
  }
}

static llvm::Expected<TargetEnvironment>
snapshotTargetEnvironment(const qdmi::Device& device,
                          std::optional<QDMI_Program_Format> format,
                          bool includeCalibration) {
  try {
    const auto supported = device.getSupportedProgramFormats();
    if (format && !llvm::is_contained(supported, *format)) {
      return llvm::createStringError(
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
      return llvm::createStringError(
          std::make_error_code(std::errc::invalid_argument),
          "Device has no executable program format supported by the MQT "
          "compiler; hardware-only models require an explicit compiler target "
          "and output");
    }
    auto payload = payloadSpecificationForProgramFormat(*format);
    if (!payload) {
      return payload.takeError();
    }
    auto target = snapshotCompilerTarget(device, includeCalibration);
    if (!target) {
      return target.takeError();
    }
    return TargetEnvironment(*target, std::move(*payload));
  } catch (...) {
    return qdmiError("Failed to query device compilation contract",
                     std::current_exception());
  }
}

llvm::Expected<TargetEnvironment>
targetEnvironmentFromDevice(const qdmi::Device& device,
                            std::optional<QDMI_Program_Format> format) {
  return snapshotTargetEnvironment(device, format, true);
}

llvm::Error validateTargetCompatibility(const TargetEnvironment& compiled,
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
  return llvm::Error::success();
}

CompiledProgram::CompiledProgram(TargetEnvironment environment,
                                 std::string payload,
                                 QDMI_Program_Format format)
    : environment_(std::move(environment)), payload_(std::move(payload)),
      format_(format) {}

llvm::Expected<CompiledProgram> CompiledProgram::compile(
    CompilerInput&& program, const TargetEnvironment& environment,
    bool enableTiming, bool enableStatistics, const MappingOptions& mapping) {
  auto format = qdmiFormatForPayload(environment.payloadSpecification());
  if (!format) {
    return format.takeError();
  }
  auto result = runDefaultPipeline(std::move(program), environment,
                                   enableTiming, enableStatistics, mapping);
  if (!result) {
    return llvm::createStringError(
        std::make_error_code(std::errc::invalid_argument),
        "Compilation failed for selected payload " +
            environment.payloadSpecification().format().id + " " +
            environment.payloadSpecification().format().profile +
            "; see compiler diagnostics for the unsupported construct or "
            "capability");
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
    return llvm::createStringError(
        std::make_error_code(std::errc::invalid_argument),
        "Compiled QDMI QIR requires an i64 () entry point; keep classical "
        "temporaries local or select OpenQASM 3");
  }
  if (auto error = validateQIRCapabilities(
          qir.module(), environment.payloadSpecification())) {
    return error;
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
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      "Failed to serialize compiled QIR payload");
}

llvm::Expected<CompiledProgram>
compileProgram(CompilerInput&& program, const qdmi::Device& device,
               std::optional<QDMI_Program_Format> format, bool enableTiming,
               bool enableStatistics, const MappingOptions& mapping) {
  auto environment = targetEnvironmentFromDevice(device, format);
  if (!environment) {
    return environment.takeError();
  }
  return CompiledProgram::compile(std::move(program), *environment,
                                  enableTiming, enableStatistics, mapping);
}

static llvm::Expected<qdmi::Job>
submitPayload(const qdmi::Device& device, const CompiledProgram& program,
              int64_t numShots,
              const std::optional<qdmi::CustomJobParameter>& custom1,
              const std::optional<qdmi::CustomJobParameter>& custom2,
              const std::optional<qdmi::CustomJobParameter>& custom3,
              const std::optional<qdmi::CustomJobParameter>& custom4,
              const std::optional<qdmi::CustomJobParameter>& custom5) {
  try {
    if (qdmi::isBinaryProgramFormat(program.programFormat())) {
      return device.submitJob(std::as_bytes(std::span(program.payload())),
                              program.programFormat(),
                              static_cast<size_t>(numShots), custom1, custom2,
                              custom3, custom4, custom5);
    }
    return device.submitJob(program.payload(), program.programFormat(),
                            static_cast<size_t>(numShots), custom1, custom2,
                            custom3, custom4, custom5);
  } catch (...) {
    return qdmiError("Failed to submit compiled program",
                     std::current_exception());
  }
}

llvm::Expected<qdmi::Job>
submitProgram(const qdmi::Device& device, const CompiledProgram& program,
              int64_t numShots,
              const std::optional<qdmi::CustomJobParameter>& custom1,
              const std::optional<qdmi::CustomJobParameter>& custom2,
              const std::optional<qdmi::CustomJobParameter>& custom3,
              const std::optional<qdmi::CustomJobParameter>& custom4,
              const std::optional<qdmi::CustomJobParameter>& custom5) {
  if (numShots < 0) {
    return llvm::createStringError(
        std::make_error_code(std::errc::invalid_argument),
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

llvm::Expected<qdmi::Job>
submitProgram(const qdmi::Device& device, CompilerInput&& input,
              int64_t numShots, std::optional<QDMI_Program_Format> format,
              bool enableTiming, bool enableStatistics,
              const std::optional<qdmi::CustomJobParameter>& custom1,
              const std::optional<qdmi::CustomJobParameter>& custom2,
              const std::optional<qdmi::CustomJobParameter>& custom3,
              const std::optional<qdmi::CustomJobParameter>& custom4,
              const std::optional<qdmi::CustomJobParameter>& custom5) {
  if (numShots < 0) {
    return llvm::createStringError(
        std::make_error_code(std::errc::invalid_argument),
        "num_shots must be nonnegative");
  }
  auto compiled = compileProgram(std::move(input), device, format, enableTiming,
                                 enableStatistics);
  if (!compiled) {
    return compiled.takeError();
  }
  return submitPayload(device, *compiled, numShots, custom1, custom2, custom3,
                       custom4, custom5);
}

} // namespace mlir
