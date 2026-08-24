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
#include "qdmi/Client.hpp"
#include "qdmi/driver/Driver.hpp"

#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/ADT/StringSwitch.h>
#include <llvm/ADT/Twine.h>
#include <llvm/Support/CheckedArithmetic.h>
#include <llvm/Support/Error.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace mlir {

[[nodiscard]] static llvm::Error
requireAdapterInput(const bool condition, const llvm::Twine& message) {
  if (!condition) {
    return llvm::createStringError(
        std::make_error_code(std::errc::invalid_argument), message);
  }
  return llvm::Error::success();
}

[[nodiscard]] static llvm::Error
requireCircuitDevice(const bool condition, const llvm::StringRef deviceName,
                     const llvm::StringRef detail) {
  return requireAdapterInput(
      condition, llvm::Twine("QDMI device '") + deviceName +
                     "' cannot be used as an MLIR compiler target: only "
                     "circuit-model devices with one qubit per non-zone site "
                     "are supported (" +
                     detail + ")");
}

[[nodiscard]] static llvm::Error requireHomogeneousOperation(
    const bool condition, const llvm::StringRef deviceName,
    const llvm::StringRef operationName, const llvm::StringRef detail) {
  return requireAdapterInput(
      condition, llvm::Twine("QDMI device '") + deviceName + "' operation '" +
                     operationName +
                     "' cannot be represented by the MLIR compiler target: "
                     "operation support must be homogeneous across the device "
                     "(" +
                     detail + ")");
}

[[nodiscard]] static llvm::Expected<CompilerTarget::SiteId>
checkedSiteId(const size_t index) {
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
canonicalCoupling(const CompilerTarget::SiteId first,
                  const CompilerTarget::SiteId second) {
  return first < second ? CompilerTarget::Coupling{first, second}
                        : CompilerTarget::Coupling{second, first};
}

[[nodiscard]] static std::optional<size_t>
allToAllCouplingCount(const size_t numSites) {
  if (numSites < 2) {
    return 0;
  }
  const auto first = numSites % 2 == 0 ? numSites / 2 : numSites;
  const auto second = numSites % 2 == 0 ? numSites - 1 : (numSites - 1) / 2;
  return llvm::checkedMulUnsigned(first, second);
}

[[nodiscard]] static bool
isSwapInvariantOperation(const llvm::StringRef operationName) {
  const auto canonicalName = operationName.trim().lower();
  return llvm::StringSwitch<bool>(canonicalName)
      .Cases({"cz", "swap", "iswap"}, true)
      .Cases({"rxx", "ryy", "rzz"}, true)
      .Default(false);
}

[[nodiscard]] static llvm::Error validateHomogeneousSupport(
    const qdmi::Operation& operation, const size_t arity,
    const std::vector<qdmi::Site>& flattenedSites,
    const std::vector<CompilerTarget::Site>& deviceSites,
    const std::optional<std::vector<CompilerTarget::Coupling>>& couplings,
    const llvm::StringRef deviceName) {
  const auto operationName = operation.getName();
  if (auto error = requireHomogeneousOperation(
          flattenedSites.size() % arity == 0, deviceName, operationName,
          "the reported site list is not divisible by the fixed arity")) {
    return error;
  }
  if (auto error = requireHomogeneousOperation(
          arity <= 2, deviceName, operationName,
          "explicit site lists are supported only for one- and two-qubit "
          "operations")) {
    return error;
  }

  llvm::DenseSet<CompilerTarget::SiteId> knownSites;
  knownSites.reserve(deviceSites.size());
  for (const auto& site : deviceSites) {
    knownSites.insert(site.id());
  }

  if (arity == 1) {
    llvm::DenseSet<CompilerTarget::SiteId> supportedSites;
    supportedSites.reserve(flattenedSites.size());
    for (const auto& site : flattenedSites) {
      auto siteId = checkedSiteId(site.getIndex());
      if (!siteId) {
        return siteId.takeError();
      }
      const auto inserted = supportedSites.insert(*siteId).second;
      if (auto error = requireHomogeneousOperation(
              knownSites.contains(*siteId) && inserted, deviceName,
              operationName,
              "the reported one-qubit sites must be unique device sites")) {
        return error;
      }
    }
    return requireHomogeneousOperation(
        supportedSites.size() == knownSites.size(), deviceName, operationName,
        "the operation is not available on every device site");
  }

  llvm::DenseSet<CompilerTarget::Coupling> reportedTuples;
  llvm::DenseSet<CompilerTarget::Coupling> supportedCouplings;
  reportedTuples.reserve(flattenedSites.size() / arity);
  supportedCouplings.reserve(flattenedSites.size() / arity);
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
    if (auto error =
            requireHomogeneousOperation(validTuple, deviceName, operationName,
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
    llvm::DenseSet<CompilerTarget::Coupling> expectedCouplings;
    expectedCouplings.reserve(couplings->size());
    for (const auto& [first, second] : *couplings) {
      expectedCouplings.insert(canonicalCoupling(first, second));
    }
    coversTarget =
        supportedCouplings.size() == expectedCouplings.size() &&
        std::ranges::all_of(expectedCouplings, [&](const auto& coupling) {
          return supportedCouplings.contains(coupling);
        });
  }
  if (auto error = requireHomogeneousOperation(
          coversTarget, deviceName, operationName,
          couplings ? "the operation is not available on every topology edge"
                    : "the operation is not available on every all-to-all site "
                      "pair")) {
    return error;
  }

  return requireHomogeneousOperation(
      isSwapInvariantOperation(operationName) ||
          std::ranges::all_of(
              supportedCouplings,
              [&](const auto& coupling) {
                return reportedTuples.contains(coupling) &&
                       reportedTuples.contains(CompilerTarget::Coupling{
                           coupling.second, coupling.first});
              }),
      deviceName, operationName,
      "both orientations must be available on every supported site pair");
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
snapshotSiteTuples(const qdmi::Operation& operation, const size_t arity,
                   const std::vector<qdmi::Site>& flattenedSites,
                   const std::optional<uint64_t> defaultDuration,
                   const std::optional<double> defaultFidelity) {
  std::vector<CompilerTarget::SiteTuple> siteTuples;
  siteTuples.reserve(flattenedSites.size() / arity);
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
    if (duration != defaultDuration || fidelity != defaultFidelity) {
      auto siteTuple = CompilerTarget::SiteTuple::create(std::move(siteIds),
                                                         duration, fidelity);
      if (!siteTuple) {
        return siteTuple.takeError();
      }
      siteTuples.emplace_back(std::move(*siteTuple));
    }
  }
  return siteTuples;
}

[[nodiscard]] static llvm::Expected<CompilerTarget::NativeOperations>
snapshotOperations(
    const std::vector<qdmi::Operation>& operations,
    const std::vector<CompilerTarget::Site>& deviceSites,
    const std::optional<std::vector<CompilerTarget::Coupling>>& couplings,
    const llvm::StringRef deviceName) {
  std::vector<CompilerTarget::Operation> targetOperations;
  targetOperations.reserve(operations.size());
  for (const auto& operation : operations) {
    if (auto error =
            requireCircuitDevice(!operation.isZoned(), deviceName,
                                 "the device exposes a zoned operation")) {
      return error;
    }
    const auto arity = operation.getQubitsNum();
    if (!arity || *arity == 0) {
      continue;
    }
    const auto flattenedSites = operation.getSites();
    if (!flattenedSites) {
      return CompilerTarget::NativeOperations{};
    }
    if (auto error =
            validateHomogeneousSupport(operation, *arity, *flattenedSites,
                                       deviceSites, couplings, deviceName)) {
      return error;
    }
    const auto duration = operation.getDuration();
    const auto fidelity = operation.getFidelity();
    auto siteTuples = snapshotSiteTuples(operation, *arity, *flattenedSites,
                                         duration, fidelity);
    if (!siteTuples) {
      return siteTuples.takeError();
    }
    auto targetOperation = CompilerTarget::Operation::create(
        operation.getName(), *arity, operation.getParametersNum(),
        std::move(*siteTuples), duration, fidelity);
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

  auto operations =
      snapshotOperations(device.getOperations(), sites, couplings, deviceName);
  if (!operations) {
    return operations.takeError();
  }
  auto durationUnit = snapshotDurationUnit(device);
  if (!durationUnit) {
    return durationUnit.takeError();
  }
  auto connectivity =
      couplings ? CompilerTarget::Connectivity::fromCouplings(*couplings)
                : CompilerTarget::Connectivity{};
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

} // namespace mlir
