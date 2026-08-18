/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Compiler/Target.h"

#include "mlir/Dialect/QCO/IR/QCOInterfaces.h"
#include "mlir/Dialect/QCO/IR/QCOOps.h"

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/STLFunctionalExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringMap.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/Error.h>
#include <mlir/IR/Operation.h>
#include <mlir/Support/LLVM.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace mlir {
namespace {

using GateKind = CompilerTarget::GateKind;
using SiteId = CompilerTarget::SiteId;

struct GateSpecification {
  GateKind kind{};
  llvm::StringLiteral name;
  size_t numQubits{};
  size_t numParameters{};
};

constexpr std::array GATE_SPECIFICATIONS{
    GateSpecification{
        .kind = GateKind::U, .name = "u", .numQubits = 1, .numParameters = 3},
    GateSpecification{
        .kind = GateKind::X, .name = "x", .numQubits = 1, .numParameters = 0},
    GateSpecification{
        .kind = GateKind::SX, .name = "sx", .numQubits = 1, .numParameters = 0},
    GateSpecification{
        .kind = GateKind::RZ, .name = "rz", .numQubits = 1, .numParameters = 1},
    GateSpecification{
        .kind = GateKind::RX, .name = "rx", .numQubits = 1, .numParameters = 1},
    GateSpecification{
        .kind = GateKind::RY, .name = "ry", .numQubits = 1, .numParameters = 1},
    GateSpecification{
        .kind = GateKind::R, .name = "r", .numQubits = 1, .numParameters = 2},
    GateSpecification{.kind = GateKind::RXX,
                      .name = "rxx",
                      .numQubits = 2,
                      .numParameters = 1},
    GateSpecification{.kind = GateKind::RYY,
                      .name = "ryy",
                      .numQubits = 2,
                      .numParameters = 1},
    GateSpecification{.kind = GateKind::RZX,
                      .name = "rzx",
                      .numQubits = 2,
                      .numParameters = 1},
    GateSpecification{.kind = GateKind::RZZ,
                      .name = "rzz",
                      .numQubits = 2,
                      .numParameters = 1},
    GateSpecification{.kind = GateKind::ISWAP,
                      .name = "iswap",
                      .numQubits = 2,
                      .numParameters = 0},
    GateSpecification{
        .kind = GateKind::CZ, .name = "cz", .numQubits = 2, .numParameters = 0},
    GateSpecification{
        .kind = GateKind::CX, .name = "cx", .numQubits = 2, .numParameters = 0},
    GateSpecification{.kind = GateKind::ECR,
                      .name = "ecr",
                      .numQubits = 2,
                      .numParameters = 0},
};

} // namespace

[[nodiscard]] static std::string canonicalOperationName(const StringRef name) {
  auto canonical = name.trim().lower();
  if (canonical == "prx") {
    canonical = "r";
  } else if (canonical == "u3") {
    canonical = "u";
  } else if (canonical == "cnot") {
    canonical = "cx";
  }
  return canonical;
}

[[nodiscard]] static llvm::Error invalidTarget(const Twine& message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument), message);
}

[[nodiscard]] static llvm::Error
validatePositiveCoherenceTime(const std::optional<uint64_t> time,
                              const StringRef description) {
  if (time && *time == 0) {
    return invalidTarget(description + " must be positive");
  }
  return llvm::Error::success();
}

[[nodiscard]] static llvm::Error
validateFidelity(const std::optional<double> fidelity,
                 const StringRef description) {
  if (fidelity &&
      (!std::isfinite(*fidelity) || *fidelity < 0. || *fidelity > 1.)) {
    return invalidTarget(description + " must be finite and in [0, 1]");
  }
  return llvm::Error::success();
}

[[nodiscard]] static llvm::Expected<std::vector<CompilerTarget::Site>>
makeDenseSites(const size_t numQubits) {
  if (numQubits == 0) {
    return invalidTarget("Compiler target must contain at least one site");
  }
  constexpr auto maxNumSites =
      static_cast<uintmax_t>(std::numeric_limits<int64_t>::max()) + 1;
  if (static_cast<uintmax_t>(numQubits) > maxNumSites) {
    return invalidTarget(
        "Compiler target qubit count exceeds the nonnegative i64 site domain");
  }

  std::vector<CompilerTarget::Site> sites;
  sites.reserve(numQubits);
  for (size_t id = 0; id < numQubits; ++id) {
    auto site = CompilerTarget::Site::create(static_cast<SiteId>(id));
    if (!site) {
      return site.takeError();
    }
    sites.emplace_back(std::move(*site));
  }
  return sites;
}

llvm::Expected<CompilerTarget::DurationUnit>
CompilerTarget::DurationUnit::create(std::string unit,
                                     const double scaleFactor) {
  if (StringRef(unit).trim().empty()) {
    return invalidTarget("Compiler target duration unit must not be empty");
  }
  if (!std::isfinite(scaleFactor) || scaleFactor <= 0.) {
    return invalidTarget(
        "Compiler target duration scale factor must be positive and finite");
  }
  return DurationUnit(std::move(unit), scaleFactor);
}

CompilerTarget::DurationUnit::DurationUnit(std::string unit,
                                           const double scaleFactor)
    : unit_(std::move(unit)), scaleFactor_(scaleFactor) {}

StringRef CompilerTarget::DurationUnit::unit() const noexcept { return unit_; }

double CompilerTarget::DurationUnit::scaleFactor() const noexcept {
  return scaleFactor_;
}

llvm::Expected<CompilerTarget::Site>
CompilerTarget::Site::create(const SiteId id, std::optional<std::string> name,
                             const std::optional<uint64_t> t1,
                             const std::optional<uint64_t> t2) {
  if (id < 0) {
    return invalidTarget("Compiler target site ID must be nonnegative");
  }
  if (name && name->empty()) {
    return invalidTarget(
        "Compiler target site name must not be empty when present");
  }
  if (auto error =
          validatePositiveCoherenceTime(t1, "Compiler target site T1")) {
    return std::move(error);
  }
  if (auto error =
          validatePositiveCoherenceTime(t2, "Compiler target site T2")) {
    return std::move(error);
  }
  return Site(id, std::move(name), t1, t2);
}

CompilerTarget::Site::Site(const SiteId id, std::optional<std::string> name,
                           const std::optional<uint64_t> t1,
                           const std::optional<uint64_t> t2)
    : id_(id), name_(std::move(name)), t1_(t1), t2_(t2) {}

CompilerTarget::SiteId CompilerTarget::Site::id() const noexcept { return id_; }

std::optional<StringRef> CompilerTarget::Site::name() const noexcept {
  if (!name_) {
    return std::nullopt;
  }
  return *name_;
}

std::optional<uint64_t> CompilerTarget::Site::t1() const noexcept {
  return t1_;
}

std::optional<uint64_t> CompilerTarget::Site::t2() const noexcept {
  return t2_;
}

llvm::Expected<CompilerTarget::SiteTuple>
CompilerTarget::SiteTuple::create(std::vector<SiteId> sites,
                                  const std::optional<uint64_t> duration,
                                  const std::optional<double> fidelity) {
  std::set<SiteId> uniqueSites;
  for (const auto site : sites) {
    if (site < 0) {
      return invalidTarget(
          "Compiler target site tuple contains a negative site ID");
    }
    if (!uniqueSites.insert(site).second) {
      return invalidTarget(
          "Compiler target site tuple contains a duplicate site");
    }
  }
  if (auto error =
          validateFidelity(fidelity, "Compiler target site-tuple fidelity")) {
    return std::move(error);
  }
  return SiteTuple(std::move(sites), duration, fidelity);
}

CompilerTarget::SiteTuple::SiteTuple(std::vector<SiteId> sites,
                                     const std::optional<uint64_t> duration,
                                     const std::optional<double> fidelity)
    : sites_(std::move(sites)), duration_(duration), fidelity_(fidelity) {}

ArrayRef<SiteId> CompilerTarget::SiteTuple::sites() const noexcept {
  return sites_;
}

std::optional<uint64_t> CompilerTarget::SiteTuple::duration() const noexcept {
  return duration_;
}

std::optional<double> CompilerTarget::SiteTuple::fidelity() const noexcept {
  return fidelity_;
}

llvm::Expected<CompilerTarget::Operation> CompilerTarget::Operation::create(
    std::string name, const size_t numQubits, const size_t numParameters,
    std::vector<SiteTuple> siteTuples, const std::optional<uint64_t> duration,
    const std::optional<double> fidelity) {
  auto canonicalName = canonicalOperationName(name);
  if (canonicalName.empty()) {
    return invalidTarget("Compiler target operation name must not be empty");
  }
  if (numQubits == 0) {
    return invalidTarget(
        "Compiler target operation qubit count must be positive");
  }
  if (auto error =
          validateFidelity(fidelity, "Compiler target operation fidelity")) {
    return std::move(error);
  }

  std::set<std::vector<SiteId>> uniqueSiteCombinations;
  for (const auto& siteTuple : siteTuples) {
    if (siteTuple.sites().size() != numQubits) {
      return invalidTarget(
          "Compiler target operation site tuple does not match its arity");
    }
    if (!uniqueSiteCombinations
             .emplace(siteTuple.sites().begin(), siteTuple.sites().end())
             .second) {
      return invalidTarget(
          "Compiler target operation contains a duplicate site tuple");
    }
  }
  return Operation(std::move(name), std::move(canonicalName), numQubits,
                   numParameters, std::move(siteTuples), duration, fidelity);
}

CompilerTarget::Operation::Operation(std::string name,
                                     std::string canonicalName,
                                     const size_t numQubits,
                                     const size_t numParameters,
                                     std::vector<SiteTuple> siteTuples,
                                     const std::optional<uint64_t> duration,
                                     const std::optional<double> fidelity)
    : name_(std::move(name)), canonicalName_(std::move(canonicalName)),
      numQubits_(numQubits), numParameters_(numParameters),
      siteTuples_(std::move(siteTuples)), duration_(duration),
      fidelity_(fidelity) {}

StringRef CompilerTarget::Operation::name() const noexcept { return name_; }

StringRef CompilerTarget::Operation::canonicalName() const noexcept {
  return canonicalName_;
}

size_t CompilerTarget::Operation::numQubits() const noexcept {
  return numQubits_;
}

size_t CompilerTarget::Operation::numParameters() const noexcept {
  return numParameters_;
}

ArrayRef<CompilerTarget::SiteTuple>
CompilerTarget::Operation::siteTuples() const noexcept {
  return siteTuples_;
}

std::optional<uint64_t> CompilerTarget::Operation::duration() const noexcept {
  return duration_;
}

std::optional<double> CompilerTarget::Operation::fidelity() const noexcept {
  return fidelity_;
}

struct CompilerTarget::Storage {
  Storage(std::optional<std::string> targetName, std::vector<Site> targetSites,
          std::optional<std::vector<Coupling>> targetCouplings,
          std::optional<std::vector<Operation>> targetOperations,
          std::optional<DurationUnit> targetDurationUnit,
          std::vector<ClassicalControl> targetClassicalControl);

  [[nodiscard]] static llvm::Expected<std::shared_ptr<const Storage>>
  create(std::optional<std::string> targetName, std::vector<Site> targetSites,
         std::optional<std::vector<Coupling>> targetCouplings,
         std::optional<std::vector<Operation>> targetOperations,
         std::optional<DurationUnit> targetDurationUnit,
         std::vector<ClassicalControl> targetClassicalControl);

  [[nodiscard]] llvm::Error initialize();

  [[nodiscard]] bool
  supportsOperation(StringRef name, size_t numQubits,
                    std::optional<size_t> numParameters) const;
  [[nodiscard]] std::optional<SynthesisBasis> resolveSynthesisBasis() const;

  std::optional<std::string> name;
  std::optional<DurationUnit> durationUnit;
  std::vector<Site> sites;
  SmallVector<SiteId> siteIds;
  DenseMap<SiteId, size_t> siteToVertex;
  std::optional<std::vector<Coupling>> couplings;
  SmallVector<SmallVector<size_t, 4>> adjacency;
  SmallVector<size_t> distances;
  size_t maximumDegree = 0;
  std::optional<std::vector<Operation>> operations;
  llvm::StringMap<SmallVector<size_t, 1>> capabilities;
  std::vector<ClassicalControl> classicalControl;
  SmallVector<GateKind> supportedGates;
  std::optional<SynthesisBasis> basis;
};

CompilerTarget::Storage::Storage(
    std::optional<std::string> targetName, std::vector<Site> targetSites,
    std::optional<std::vector<Coupling>> targetCouplings,
    std::optional<std::vector<Operation>> targetOperations,
    std::optional<DurationUnit> targetDurationUnit,
    std::vector<ClassicalControl> targetClassicalControl)
    : name(std::move(targetName)), durationUnit(std::move(targetDurationUnit)),
      sites(std::move(targetSites)), couplings(std::move(targetCouplings)),
      operations(std::move(targetOperations)),
      classicalControl(std::move(targetClassicalControl)) {}

llvm::Expected<std::shared_ptr<const CompilerTarget::Storage>>
CompilerTarget::Storage::create(
    std::optional<std::string> targetName, std::vector<Site> targetSites,
    std::optional<std::vector<Coupling>> targetCouplings,
    std::optional<std::vector<Operation>> targetOperations,
    std::optional<DurationUnit> targetDurationUnit,
    std::vector<ClassicalControl> targetClassicalControl) {
  auto storage = std::make_shared<Storage>(
      std::move(targetName), std::move(targetSites), std::move(targetCouplings),
      std::move(targetOperations), std::move(targetDurationUnit),
      std::move(targetClassicalControl));
  if (auto error = storage->initialize()) {
    return std::move(error);
  }
  return std::shared_ptr<const Storage>(std::move(storage));
}

llvm::Error CompilerTarget::Storage::initialize() {
  if (name && name->empty()) {
    return invalidTarget("Compiler target name must not be empty when present");
  }
  if (sites.empty()) {
    return invalidTarget("Compiler target must contain at least one site");
  }

  for (const auto capability : classicalControl) {
    switch (capability) {
    case ClassicalControl::Conditional:
    case ClassicalControl::Iteration:
    case ClassicalControl::ConditionalLoop:
    case ClassicalControl::MultiwayBranch:
      break;
    default:
      return invalidTarget(
          "Compiler target contains an unknown classical-control capability");
    }
  }
  std::ranges::sort(classicalControl, [](const auto lhs, const auto rhs) {
    return static_cast<uint8_t>(lhs) < static_cast<uint8_t>(rhs);
  });
  classicalControl.erase(std::ranges::unique(classicalControl).begin(),
                         classicalControl.end());

  siteIds.reserve(sites.size());
  siteToVertex.reserve(sites.size());
  for (const auto [vertex, site] : llvm::enumerate(sites)) {
    if (!siteToVertex.try_emplace(site.id(), vertex).second) {
      return invalidTarget("Compiler target contains duplicate site IDs");
    }
    siteIds.emplace_back(site.id());
  }

  if (couplings) {
    std::set<Coupling> canonicalCouplings;
    for (auto [source, target] : *couplings) {
      if (!siteToVertex.contains(source) || !siteToVertex.contains(target)) {
        return invalidTarget(
            "Compiler target topology references an unknown site");
      }
      if (source == target) {
        return invalidTarget(
            "Compiler target topology contains a self-coupling");
      }
      if (target < source) {
        std::swap(source, target);
      }
      canonicalCouplings.emplace(source, target);
    }
    couplings->assign(canonicalCouplings.begin(), canonicalCouplings.end());

    adjacency.resize(sites.size());
    for (const auto& [source, target] : *couplings) {
      const auto sourceVertex = siteToVertex.at(source);
      const auto targetVertex = siteToVertex.at(target);
      adjacency[sourceVertex].emplace_back(targetVertex);
      adjacency[targetVertex].emplace_back(sourceVertex);
    }
    for (auto& neighbours : adjacency) {
      std::ranges::sort(neighbours);
      maximumDegree = std::max(maximumDegree, neighbours.size());
    }

    if (sites.size() > std::numeric_limits<size_t>::max() / sites.size()) {
      return invalidTarget(
          "Compiler target topology distance matrix is too large");
    }
    constexpr auto unreachable = std::numeric_limits<size_t>::max();
    distances.assign(sites.size() * sites.size(), unreachable);
    for (size_t source = 0; source < sites.size(); ++source) {
      const auto rowOffset = source * sites.size();
      distances[rowOffset + source] = 0;
      SmallVector<size_t> worklist{source};
      for (size_t cursor = 0; cursor < worklist.size(); ++cursor) {
        const auto vertex = worklist[cursor];
        for (const auto neighbour : adjacency[vertex]) {
          auto& distance = distances[rowOffset + neighbour];
          if (distance != unreachable) {
            continue;
          }
          distance = distances[rowOffset + vertex] + 1;
          worklist.emplace_back(neighbour);
        }
      }
      if (llvm::is_contained(
              ArrayRef<size_t>(distances).slice(rowOffset, sites.size()),
              unreachable)) {
        return invalidTarget("Compiler target topology must be connected");
      }
    }
  } else {
    maximumDegree = sites.size() - 1;
  }

  if (operations) {
    for (const auto [index, operation] : llvm::enumerate(*operations)) {
      if (operation.numQubits() > sites.size()) {
        return invalidTarget(
            "Compiler target operation arity exceeds its site count");
      }
      for (const auto& siteTuple : operation.siteTuples()) {
        if (llvm::any_of(siteTuple.sites(), [&](const auto site) {
              return !siteToVertex.contains(site);
            })) {
          return invalidTarget("Compiler target operation site tuple "
                               "references an unknown site");
        }
      }
      capabilities[operation.canonicalName()].emplace_back(index);
    }
  }

  const auto hasSiteTiming = llvm::any_of(sites, [](const auto& site) {
    return site.t1().has_value() || site.t2().has_value();
  });
  const auto hasOperationTiming =
      operations && llvm::any_of(*operations, [](const auto& operation) {
        return operation.duration().has_value() ||
               llvm::any_of(operation.siteTuples(), [](const auto& siteTuple) {
                 return siteTuple.duration().has_value();
               });
      });
  if ((hasSiteTiming || hasOperationTiming) && !durationUnit) {
    return invalidTarget(
        "Compiler target timing metadata requires a duration unit");
  }

  for (const auto& specification : GATE_SPECIFICATIONS) {
    if (supportsOperation(specification.name, specification.numQubits,
                          specification.numParameters)) {
      supportedGates.emplace_back(specification.kind);
    }
  }
  basis = resolveSynthesisBasis();
  return llvm::Error::success();
}

bool CompilerTarget::Storage::supportsOperation(
    const StringRef operationName, const size_t numQubits,
    const std::optional<size_t> numParameters) const {
  const auto canonical = canonicalOperationName(operationName);
  if (canonical.empty() || numQubits == 0 || numQubits > sites.size()) {
    return false;
  }
  if (!operations) {
    return true;
  }
  const auto found = capabilities.find(canonical);
  if (found == capabilities.end()) {
    return false;
  }
  return llvm::any_of(found->second, [&](const auto index) {
    const auto& operation = (*operations)[index];
    return operation.numQubits() == numQubits &&
           (!numParameters || operation.numParameters() == *numParameters);
  });
}

std::optional<CompilerTarget::SynthesisBasis>
CompilerTarget::Storage::resolveSynthesisBasis() const {
  const auto supports = [&](const GateKind gate) {
    return llvm::is_contained(supportedGates, gate);
  };
  std::optional<SingleQubitBasis> singleQubit;
  if (supports(GateKind::U)) {
    singleQubit = SingleQubitBasis::U;
  } else if (supports(GateKind::X) && supports(GateKind::SX) &&
             supports(GateKind::RZ)) {
    singleQubit = SingleQubitBasis::ZSXX;
  } else if (supports(GateKind::R)) {
    singleQubit = SingleQubitBasis::R;
  } else if (supports(GateKind::RX) && supports(GateKind::RZ)) {
    singleQubit = SingleQubitBasis::XZX;
  } else if (supports(GateKind::RX) && supports(GateKind::RY)) {
    singleQubit = SingleQubitBasis::XYX;
  } else if (supports(GateKind::RY) && supports(GateKind::RZ)) {
    singleQubit = SingleQubitBasis::ZYZ;
  }

  constexpr std::array entanglerPreference{
      GateKind::RXX,   GateKind::RYY, GateKind::RZX, GateKind::RZZ,
      GateKind::ISWAP, GateKind::CZ,  GateKind::CX,  GateKind::ECR,
  };
  // NOLINTNEXTLINE(readability-qualified-auto)
  const auto entangler =
      std::ranges::find_if(entanglerPreference, [&](const auto candidate) {
        return supports(candidate);
      });
  if (!singleQubit || entangler == entanglerPreference.end()) {
    return std::nullopt;
  }
  return SynthesisBasis{.singleQubit = *singleQubit, .entangler = *entangler};
}

llvm::Expected<CompilerTarget>
CompilerTarget::create(const size_t numQubits,
                       std::optional<std::vector<Coupling>> couplings,
                       std::optional<std::vector<Operation>> operations,
                       std::optional<DurationUnit> durationUnit) {
  return create(numQubits, std::move(couplings), std::move(operations),
                std::move(durationUnit), {});
}

llvm::Expected<CompilerTarget>
CompilerTarget::create(const size_t numQubits,
                       std::optional<std::vector<Coupling>> couplings,
                       std::optional<std::vector<Operation>> operations,
                       std::optional<DurationUnit> durationUnit,
                       std::vector<ClassicalControl> classicalControl) {
  auto sites = makeDenseSites(numQubits);
  if (!sites) {
    return sites.takeError();
  }
  return createImpl(std::nullopt, std::move(*sites), std::move(couplings),
                    std::move(operations), std::move(durationUnit),
                    std::move(classicalControl));
}

llvm::Expected<CompilerTarget>
CompilerTarget::create(std::string name, const size_t numQubits,
                       std::optional<std::vector<Coupling>> couplings,
                       std::optional<std::vector<Operation>> operations,
                       std::optional<DurationUnit> durationUnit) {
  return create(std::move(name), numQubits, std::move(couplings),
                std::move(operations), std::move(durationUnit), {});
}

llvm::Expected<CompilerTarget>
CompilerTarget::create(std::string name, const size_t numQubits,
                       std::optional<std::vector<Coupling>> couplings,
                       std::optional<std::vector<Operation>> operations,
                       std::optional<DurationUnit> durationUnit,
                       std::vector<ClassicalControl> classicalControl) {
  auto sites = makeDenseSites(numQubits);
  if (!sites) {
    return sites.takeError();
  }
  return createImpl(std::optional<std::string>(std::move(name)),
                    std::move(*sites), std::move(couplings),
                    std::move(operations), std::move(durationUnit),
                    std::move(classicalControl));
}

llvm::Expected<CompilerTarget>
CompilerTarget::create(std::vector<Site> sites,
                       std::optional<std::vector<Coupling>> couplings,
                       std::optional<std::vector<Operation>> operations,
                       std::optional<DurationUnit> durationUnit) {
  return create(std::move(sites), std::move(couplings), std::move(operations),
                std::move(durationUnit), {});
}

llvm::Expected<CompilerTarget>
CompilerTarget::create(std::vector<Site> sites,
                       std::optional<std::vector<Coupling>> couplings,
                       std::optional<std::vector<Operation>> operations,
                       std::optional<DurationUnit> durationUnit,
                       std::vector<ClassicalControl> classicalControl) {
  return createImpl(std::nullopt, std::move(sites), std::move(couplings),
                    std::move(operations), std::move(durationUnit),
                    std::move(classicalControl));
}

llvm::Expected<CompilerTarget>
CompilerTarget::create(std::string name, std::vector<Site> sites,
                       std::optional<std::vector<Coupling>> couplings,
                       std::optional<std::vector<Operation>> operations,
                       std::optional<DurationUnit> durationUnit) {
  return create(std::move(name), std::move(sites), std::move(couplings),
                std::move(operations), std::move(durationUnit), {});
}

llvm::Expected<CompilerTarget>
CompilerTarget::create(std::string name, std::vector<Site> sites,
                       std::optional<std::vector<Coupling>> couplings,
                       std::optional<std::vector<Operation>> operations,
                       std::optional<DurationUnit> durationUnit,
                       std::vector<ClassicalControl> classicalControl) {
  return createImpl(std::optional<std::string>(std::move(name)),
                    std::move(sites), std::move(couplings),
                    std::move(operations), std::move(durationUnit),
                    std::move(classicalControl));
}

llvm::Expected<CompilerTarget>
CompilerTarget::createImpl(std::optional<std::string> name,
                           std::vector<Site> sites,
                           std::optional<std::vector<Coupling>> couplings,
                           std::optional<std::vector<Operation>> operations,
                           std::optional<DurationUnit> durationUnit,
                           std::vector<ClassicalControl> classicalControl) {
  auto storage =
      Storage::create(std::move(name), std::move(sites), std::move(couplings),
                      std::move(operations), std::move(durationUnit),
                      std::move(classicalControl));
  if (!storage) {
    return storage.takeError();
  }
  return CompilerTarget(std::move(*storage));
}

CompilerTarget::CompilerTarget(std::shared_ptr<const Storage> storage)
    : storage_(std::move(storage)) {}

std::optional<StringRef> CompilerTarget::name() const noexcept {
  if (!storage_->name) {
    return std::nullopt;
  }
  return *storage_->name;
}

const std::optional<CompilerTarget::DurationUnit>&
CompilerTarget::durationUnit() const noexcept {
  return storage_->durationUnit;
}

size_t CompilerTarget::numQubits() const noexcept {
  return storage_->sites.size();
}

ArrayRef<CompilerTarget::Site> CompilerTarget::sites() const noexcept {
  return storage_->sites;
}

ArrayRef<SiteId> CompilerTarget::siteIds() const noexcept {
  return storage_->siteIds;
}

std::optional<size_t>
CompilerTarget::vertexForSite(const SiteId site) const noexcept {
  const auto found = storage_->siteToVertex.find(site);
  if (found == storage_->siteToVertex.end()) {
    return std::nullopt;
  }
  return found->second;
}

SiteId CompilerTarget::siteForVertex(const size_t vertex) const {
  assert(vertex < numQubits() && "Compiler target vertex is out of range");
  return storage_->siteIds[vertex];
}

bool CompilerTarget::hasExplicitTopology() const noexcept {
  return storage_->couplings.has_value();
}

ArrayRef<CompilerTarget::Coupling> CompilerTarget::couplings() const noexcept {
  if (!storage_->couplings) {
    return {};
  }
  return *storage_->couplings;
}

bool CompilerTarget::areAdjacent(const size_t source,
                                 const size_t target) const {
  assert(source < numQubits() && target < numQubits() &&
         "Compiler target vertex is out of range");
  if (!hasExplicitTopology()) {
    return source != target;
  }
  return llvm::is_contained(storage_->adjacency[source], target);
}

void CompilerTarget::forEachNeighbour(
    const size_t vertex,
    const llvm::function_ref<void(size_t)> callback) const {
  if (!hasExplicitTopology()) {
    assert(vertex < numQubits() && "Compiler target vertex is out of range");
    for (size_t neighbour = 0; neighbour < numQubits(); ++neighbour) {
      if (neighbour != vertex) {
        callback(neighbour);
      }
    }
    return;
  }
  for (const auto neighbour : explicitNeighbours(vertex)) {
    callback(neighbour);
  }
}

size_t CompilerTarget::distanceBetween(const size_t source,
                                       const size_t target) const {
  assert(source < numQubits() && target < numQubits() &&
         "Compiler target vertex is out of range");
  if (!hasExplicitTopology()) {
    return source == target ? 0 : 1;
  }
  return storage_->distances[(source * numQubits()) + target];
}

ArrayRef<size_t> CompilerTarget::explicitNeighbours(const size_t vertex) const {
  assert(vertex < numQubits() && "Compiler target vertex is out of range");
  return storage_->adjacency[vertex];
}

size_t CompilerTarget::maxDegree() const noexcept {
  return storage_->maximumDegree;
}

bool CompilerTarget::hasExplicitOperations() const noexcept {
  return storage_->operations.has_value();
}

ArrayRef<CompilerTarget::Operation>
CompilerTarget::operations() const noexcept {
  if (!storage_->operations) {
    return {};
  }
  return *storage_->operations;
}

ArrayRef<CompilerTarget::ClassicalControl>
CompilerTarget::classicalControl() const noexcept {
  return storage_->classicalControl;
}

bool CompilerTarget::supportsClassicalControl(
    const ClassicalControl capability) const noexcept {
  return llvm::is_contained(storage_->classicalControl, capability);
}

bool CompilerTarget::supportsOperation(
    const StringRef operationName, const size_t numQubits,
    const std::optional<size_t> numParameters) const {
  return storage_->supportsOperation(operationName, numQubits, numParameters);
}

bool CompilerTarget::supports(::mlir::Operation* operation) const {
  if (operation == nullptr) {
    return false;
  }

  if (auto unitary = dyn_cast<qco::UnitaryOpInterface>(operation)) {
    if (isa<qco::BarrierOp, qco::GPhaseOp>(operation)) {
      return true;
    }
    if (auto controlled = dyn_cast<qco::CtrlOp>(operation);
        controlled && controlled.getNumControls() == 1 &&
        controlled.getNumTargets() == 1 &&
        controlled.getNumBodyUnitaries() == 1) {
      auto* const body = controlled.getBodyUnitary(0).getOperation();
      if (isa<qco::XOp>(body)) {
        return storage_->supportsOperation("cx", 2, 0);
      }
      if (isa<qco::ZOp>(body)) {
        return storage_->supportsOperation("cz", 2, 0);
      }
    }
    return storage_->supportsOperation(unitary.getBaseSymbol(),
                                       unitary.getNumQubits(),
                                       unitary.getNumParams());
  }
  if (isa<qco::MeasureOp>(operation)) {
    return storage_->supportsOperation("measure", 1, 0);
  }
  if (isa<qco::ResetOp>(operation)) {
    return storage_->supportsOperation("reset", 1, 0);
  }
  return false;
}

bool CompilerTarget::supports(const GateKind gate) const {
  return llvm::is_contained(storage_->supportedGates, gate);
}

ArrayRef<GateKind> CompilerTarget::supportedGates() const noexcept {
  return storage_->supportedGates;
}

std::optional<CompilerTarget::SynthesisBasis>
CompilerTarget::synthesisBasis() const noexcept {
  return storage_->basis;
}

} // namespace mlir
