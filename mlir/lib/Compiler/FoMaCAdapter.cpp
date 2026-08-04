/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Compiler/FoMaCAdapter.h"

#include "fomac/FoMaC.hpp"
#include "mlir/Compiler/Target.h"

#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/ADT/StringSwitch.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mlir {

[[nodiscard]] static CompilerTarget::SiteId checkedSiteId(const size_t index) {
  if (index > static_cast<uintmax_t>(
                  std::numeric_limits<CompilerTarget::SiteId>::max())) {
    throw std::invalid_argument(
        "QDMI site index exceeds the nonnegative i64 compiler-target domain");
  }
  return static_cast<CompilerTarget::SiteId>(index);
}

[[noreturn]] static void
rejectNonCircuitDevice(const llvm::StringRef deviceName,
                       const llvm::StringRef detail) {
  throw std::invalid_argument(
      (std::string("QDMI device '") + deviceName.str() +
       "' cannot be used as an MLIR compiler target: only circuit-model "
       "devices with one qubit per non-zone site are supported (" +
       detail.str() + ")"));
}

[[noreturn]] static void
rejectNonHomogeneousOperation(const llvm::StringRef deviceName,
                              const llvm::StringRef operationName,
                              const llvm::StringRef detail) {
  throw std::invalid_argument(
      (std::string("QDMI device '") + deviceName.str() + "' operation '" +
       operationName.str() +
       "' cannot be represented by the MLIR compiler target: operation "
       "support must be homogeneous across the device (" +
       detail.str() + ")"));
}

[[nodiscard]] static CompilerTarget::Coupling
canonicalCoupling(const CompilerTarget::SiteId first,
                  const CompilerTarget::SiteId second) {
  if (first < second) {
    return {first, second};
  }
  return {second, first};
}

[[nodiscard]] static std::optional<size_t>
allToAllCouplingCount(const size_t numSites) {
  if (numSites < 2) {
    return 0;
  }
  const auto first = numSites % 2 == 0 ? numSites / 2 : numSites;
  const auto second = numSites % 2 == 0 ? numSites - 1 : (numSites - 1) / 2;
  if (first > std::numeric_limits<size_t>::max() / second) {
    return std::nullopt;
  }
  return first * second;
}

[[nodiscard]] static bool
isSwapInvariantOperation(const llvm::StringRef operationName) {
  const auto canonicalName = operationName.trim().lower();
  return llvm::StringSwitch<bool>(canonicalName)
      .Cases("cz", "swap", "iswap", true)
      .Cases("rxx", "ryy", "rzz", true)
      .Default(false);
}

static void validateHomogeneousSupport(
    const fomac::Operation& operation, const size_t arity,
    const std::optional<std::vector<fomac::Site>>& flattenedSites,
    const std::vector<CompilerTarget::Site>& deviceSites,
    const std::optional<std::vector<CompilerTarget::Coupling>>& couplings,
    const llvm::StringRef deviceName) {
  if (!flattenedSites) {
    if (arity == 2 && couplings) {
      rejectNonHomogeneousOperation(
          deviceName, operation.getName(),
          "the device reports an explicit topology but no ordered two-qubit "
          "site support");
    }
    return;
  }
  const auto operationName = operation.getName();
  if (flattenedSites->size() % arity != 0) {
    rejectNonHomogeneousOperation(
        deviceName, operationName,
        "the reported site list is not divisible by the fixed arity");
  }
  if (arity > 2) {
    rejectNonHomogeneousOperation(
        deviceName, operationName,
        "explicit site lists are supported only for one- and two-qubit "
        "operations");
  }

  llvm::DenseSet<CompilerTarget::SiteId> knownSites;
  knownSites.reserve(deviceSites.size());
  for (const auto& site : deviceSites) {
    knownSites.insert(site.id());
  }

  if (arity == 1) {
    llvm::DenseSet<CompilerTarget::SiteId> supportedSites;
    supportedSites.reserve(flattenedSites->size());
    for (const auto& site : *flattenedSites) {
      const auto siteId = checkedSiteId(site.getIndex());
      if (!knownSites.contains(siteId) ||
          !supportedSites.insert(siteId).second) {
        rejectNonHomogeneousOperation(
            deviceName, operationName,
            "the reported one-qubit sites must be unique device sites");
      }
    }
    if (supportedSites.size() != knownSites.size()) {
      rejectNonHomogeneousOperation(
          deviceName, operationName,
          "the operation is not available on every device site");
    }
    return;
  }

  llvm::DenseSet<CompilerTarget::Coupling> reportedTuples;
  llvm::DenseSet<CompilerTarget::Coupling> supportedCouplings;
  reportedTuples.reserve(flattenedSites->size() / arity);
  supportedCouplings.reserve(flattenedSites->size() / arity);
  for (size_t offset = 0; offset < flattenedSites->size(); offset += arity) {
    const auto first = checkedSiteId((*flattenedSites)[offset].getIndex());
    const auto second = checkedSiteId((*flattenedSites)[offset + 1].getIndex());
    if (first == second || !knownSites.contains(first) ||
        !knownSites.contains(second) ||
        !reportedTuples.insert({first, second}).second) {
      rejectNonHomogeneousOperation(
          deviceName, operationName,
          "the reported two-qubit sites must be unique pairs of device sites");
    }
    supportedCouplings.insert(canonicalCoupling(first, second));
  }

  if (!couplings) {
    const auto expected = allToAllCouplingCount(knownSites.size());
    if (!expected || supportedCouplings.size() != *expected) {
      rejectNonHomogeneousOperation(
          deviceName, operationName,
          "the operation is not available on every all-to-all site pair");
    }
  } else {
    llvm::DenseSet<CompilerTarget::Coupling> expectedCouplings;
    expectedCouplings.reserve(couplings->size());
    for (const auto& [first, second] : *couplings) {
      expectedCouplings.insert(canonicalCoupling(first, second));
    }
    if (supportedCouplings.size() != expectedCouplings.size() ||
        !std::ranges::all_of(expectedCouplings, [&](const auto& coupling) {
          return supportedCouplings.contains(coupling);
        })) {
      rejectNonHomogeneousOperation(
          deviceName, operationName,
          "the operation is not available on every topology edge");
    }
  }

  if (!isSwapInvariantOperation(operationName) &&
      !std::ranges::all_of(supportedCouplings, [&](const auto& coupling) {
        return reportedTuples.contains(coupling) &&
               reportedTuples.contains(
                   CompilerTarget::Coupling{coupling.second, coupling.first});
      })) {
    rejectNonHomogeneousOperation(
        deviceName, operationName,
        "both orientations must be available on every supported site pair");
  }
}

[[nodiscard]] static std::optional<CompilerTarget::DurationUnit>
snapshotDurationUnit(const fomac::Device& device) {
  auto unit = device.getDurationUnit();
  const auto scaleFactor = device.getDurationScaleFactor();
  if (!unit) {
    if (scaleFactor) {
      throw std::invalid_argument(
          "QDMI device reports a duration scale factor without a duration "
          "unit");
    }
    return std::nullopt;
  }
  return CompilerTarget::DurationUnit(std::move(*unit),
                                      scaleFactor.value_or(1.));
}

[[nodiscard]] static std::vector<CompilerTarget::SiteTuple> snapshotSiteTuples(
    const fomac::Operation& operation, const size_t arity,
    const std::optional<std::vector<fomac::Site>>& flattenedSites,
    const std::optional<uint64_t> defaultDuration,
    const std::optional<double> defaultFidelity) {
  if (!flattenedSites) {
    return {};
  }

  std::vector<CompilerTarget::SiteTuple> siteTuples;
  siteTuples.reserve(flattenedSites->size() / arity);
  for (size_t offset = 0; offset < flattenedSites->size(); offset += arity) {
    std::vector<fomac::Site> sites;
    std::vector<CompilerTarget::SiteId> siteIds;
    sites.reserve(arity);
    siteIds.reserve(arity);
    for (size_t index = 0; index < arity; ++index) {
      const auto& site = (*flattenedSites)[offset + index];
      sites.emplace_back(site);
      siteIds.emplace_back(checkedSiteId(site.getIndex()));
    }

    const auto duration = operation.getDuration(sites);
    const auto fidelity = operation.getFidelity(sites);
    if (duration != defaultDuration || fidelity != defaultFidelity) {
      siteTuples.emplace_back(std::move(siteIds), duration, fidelity);
    }
  }
  return siteTuples;
}

[[nodiscard]] static std::vector<CompilerTarget::Operation> snapshotOperations(
    const std::vector<fomac::Operation>& operations,
    const std::vector<CompilerTarget::Site>& deviceSites,
    const std::optional<std::vector<CompilerTarget::Coupling>>& couplings,
    const llvm::StringRef deviceName) {
  std::vector<CompilerTarget::Operation> targetOperations;
  targetOperations.reserve(operations.size());
  for (const auto& operation : operations) {
    if (operation.isZoned()) {
      rejectNonCircuitDevice(deviceName,
                             "the device exposes a zoned operation");
    }
    const auto arity = operation.getQubitsNum();
    if (!arity || *arity == 0) {
      continue;
    }
    const auto flattenedSites = operation.getSites();
    validateHomogeneousSupport(operation, *arity, flattenedSites, deviceSites,
                               couplings, deviceName);
    const auto duration = operation.getDuration();
    const auto fidelity = operation.getFidelity();
    targetOperations.emplace_back(
        operation.getName(), *arity, operation.getParametersNum(),
        snapshotSiteTuples(operation, *arity, flattenedSites, duration,
                           fidelity),
        duration, fidelity);
  }
  return targetOperations;
}

CompilerTarget compilerTargetFromDevice(const fomac::Device& device) {
  auto deviceName = device.getName();
  const auto deviceSites = device.getSites();
  if (std::ranges::any_of(deviceSites,
                          [](const auto& site) { return site.isZone(); })) {
    rejectNonCircuitDevice(deviceName, "the device exposes zone sites");
  }
  if (device.getQubitsNum() != deviceSites.size()) {
    rejectNonCircuitDevice(
        deviceName, "the qubit count does not match the regular-site count");
  }

  std::vector<CompilerTarget::Site> sites;
  sites.reserve(deviceSites.size());
  for (const auto& site : deviceSites) {
    sites.emplace_back(checkedSiteId(site.getIndex()), site.getName(),
                       site.getT1(), site.getT2());
  }

  std::optional<std::vector<CompilerTarget::Coupling>> couplings;
  if (const auto deviceCouplings = device.getCouplingMap()) {
    couplings.emplace();
    couplings->reserve(deviceCouplings->size());
    for (const auto& [source, target] : *deviceCouplings) {
      couplings->emplace_back(checkedSiteId(source.getIndex()),
                              checkedSiteId(target.getIndex()));
    }
  }

  auto operations =
      snapshotOperations(device.getOperations(), sites, couplings, deviceName);
  auto durationUnit = snapshotDurationUnit(device);
  return {std::move(deviceName), std::move(sites), std::move(couplings),
          std::move(operations), std::move(durationUnit)};
}

} // namespace mlir
