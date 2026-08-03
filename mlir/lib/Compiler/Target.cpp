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
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/STLFunctionalExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringMap.h>
#include <llvm/ADT/StringRef.h>
#include <mlir/IR/Operation.h>
#include <mlir/Support/LLVM.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mlir {
namespace {

using GateKind = CompilerTarget::GateKind;
using SiteId = CompilerTarget::SiteId;

struct GateSpecification {
  GateKind kind;
  llvm::StringLiteral name;
  size_t numQubits;
  size_t numParameters;
  bool symmetric;
};

constexpr std::array GATE_SPECIFICATIONS{
    GateSpecification{.kind = GateKind::U,
                      .name = "u",
                      .numQubits = 1,
                      .numParameters = 3,
                      .symmetric = false},
    GateSpecification{.kind = GateKind::X,
                      .name = "x",
                      .numQubits = 1,
                      .numParameters = 0,
                      .symmetric = false},
    GateSpecification{.kind = GateKind::SX,
                      .name = "sx",
                      .numQubits = 1,
                      .numParameters = 0,
                      .symmetric = false},
    GateSpecification{.kind = GateKind::RZ,
                      .name = "rz",
                      .numQubits = 1,
                      .numParameters = 1,
                      .symmetric = false},
    GateSpecification{.kind = GateKind::RX,
                      .name = "rx",
                      .numQubits = 1,
                      .numParameters = 1,
                      .symmetric = false},
    GateSpecification{.kind = GateKind::RY,
                      .name = "ry",
                      .numQubits = 1,
                      .numParameters = 1,
                      .symmetric = false},
    GateSpecification{.kind = GateKind::R,
                      .name = "r",
                      .numQubits = 1,
                      .numParameters = 2,
                      .symmetric = false},
    GateSpecification{.kind = GateKind::RXX,
                      .name = "rxx",
                      .numQubits = 2,
                      .numParameters = 1,
                      .symmetric = true},
    GateSpecification{.kind = GateKind::RYY,
                      .name = "ryy",
                      .numQubits = 2,
                      .numParameters = 1,
                      .symmetric = true},
    GateSpecification{.kind = GateKind::RZX,
                      .name = "rzx",
                      .numQubits = 2,
                      .numParameters = 1,
                      .symmetric = false},
    GateSpecification{.kind = GateKind::RZZ,
                      .name = "rzz",
                      .numQubits = 2,
                      .numParameters = 1,
                      .symmetric = true},
    GateSpecification{.kind = GateKind::ISWAP,
                      .name = "iswap",
                      .numQubits = 2,
                      .numParameters = 0,
                      .symmetric = true},
    GateSpecification{.kind = GateKind::CZ,
                      .name = "cz",
                      .numQubits = 2,
                      .numParameters = 0,
                      .symmetric = true},
    GateSpecification{.kind = GateKind::CX,
                      .name = "cx",
                      .numQubits = 2,
                      .numParameters = 0,
                      .symmetric = false},
    GateSpecification{.kind = GateKind::ECR,
                      .name = "ecr",
                      .numQubits = 2,
                      .numParameters = 0,
                      .symmetric = false},
};

} // namespace

[[nodiscard]] static const GateSpecification&
gateSpecification(const GateKind gate) {
  const decltype(GATE_SPECIFICATIONS.cbegin()) found =
      std::ranges::find(GATE_SPECIFICATIONS, gate, &GateSpecification::kind);
  if (found == GATE_SPECIFICATIONS.end()) {
    throw std::invalid_argument("Unknown compiler target gate kind");
  }
  return *found;
}

[[nodiscard]] static std::string
canonicalOperationName(const StringRef providerName) {
  auto canonical = providerName.trim().lower();
  if (canonical == "prx") {
    canonical = "r";
  } else if (canonical == "u3") {
    canonical = "u";
  } else if (canonical == "cnot") {
    canonical = "cx";
  }
  return canonical;
}

static void validatePositiveCoherenceTime(const std::optional<uint64_t> time,
                                          const StringRef description) {
  if (time && *time == 0) {
    throw std::invalid_argument((description + " must be positive").str());
  }
}

static void validateFidelity(const std::optional<double> fidelity,
                             const StringRef description) {
  if (fidelity &&
      (!std::isfinite(*fidelity) || *fidelity < 0. || *fidelity > 1.)) {
    throw std::invalid_argument(
        (description + " must be finite and in [0, 1]").str());
  }
}

[[nodiscard]] static std::vector<CompilerTarget::Site>
makeDenseSites(const size_t numQubits) {
  if (numQubits == 0) {
    throw std::invalid_argument(
        "Compiler target must contain at least one site");
  }
  constexpr auto maxNumSites =
      static_cast<uintmax_t>(std::numeric_limits<int64_t>::max()) + 1;
  if (static_cast<uintmax_t>(numQubits) > maxNumSites) {
    throw std::invalid_argument(
        "Compiler target qubit count exceeds the nonnegative i64 site domain");
  }

  std::vector<CompilerTarget::Site> sites;
  sites.reserve(numQubits);
  for (size_t id = 0; id < numQubits; ++id) {
    sites.emplace_back(static_cast<SiteId>(id));
  }
  return sites;
}

CompilerTarget::DurationUnit::DurationUnit(std::string unit,
                                           const double scaleFactor)
    : unit_(std::move(unit)), scaleFactor_(scaleFactor) {
  if (StringRef(unit_).trim().empty()) {
    throw std::invalid_argument(
        "Compiler target duration unit must not be empty");
  }
  if (!std::isfinite(scaleFactor_) || scaleFactor_ <= 0.) {
    throw std::invalid_argument(
        "Compiler target duration scale factor must be positive and finite");
  }
}

StringRef CompilerTarget::DurationUnit::unit() const noexcept { return unit_; }

double CompilerTarget::DurationUnit::scaleFactor() const noexcept {
  return scaleFactor_;
}

CompilerTarget::Site::Site(const SiteId id, std::optional<std::string> name,
                           const std::optional<uint64_t> t1,
                           const std::optional<uint64_t> t2)
    : id_(id), name_(std::move(name)), t1_(t1), t2_(t2) {
  if (id_ < 0) {
    throw std::invalid_argument("Compiler target site ID must be nonnegative");
  }
  if (name_ && name_->empty()) {
    throw std::invalid_argument(
        "Compiler target site name must not be empty when present");
  }
  validatePositiveCoherenceTime(t1_, "Compiler target site T1");
  validatePositiveCoherenceTime(t2_, "Compiler target site T2");
}

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

CompilerTarget::OperationLocus::OperationLocus(
    std::vector<SiteId> sites, const std::optional<uint64_t> duration,
    const std::optional<double> fidelity)
    : sites_(std::move(sites)), duration_(duration), fidelity_(fidelity) {
  DenseSet<SiteId> uniqueSites;
  uniqueSites.reserve(sites_.size());
  for (const auto site : sites_) {
    if (site < 0) {
      throw std::invalid_argument(
          "Compiler target operation locus contains a negative site ID");
    }
    if (!uniqueSites.insert(site).second) {
      throw std::invalid_argument(
          "Compiler target operation locus contains a duplicate site");
    }
  }
  validateFidelity(fidelity_, "Compiler target operation locus fidelity");
}

ArrayRef<SiteId> CompilerTarget::OperationLocus::sites() const noexcept {
  return sites_;
}

std::optional<uint64_t>
CompilerTarget::OperationLocus::duration() const noexcept {
  return duration_;
}

std::optional<double>
CompilerTarget::OperationLocus::fidelity() const noexcept {
  return fidelity_;
}

CompilerTarget::Operation::Operation(
    std::string providerName, const size_t numQubits,
    const size_t numParameters, std::optional<std::vector<OperationLocus>> loci,
    const std::optional<uint64_t> duration,
    const std::optional<double> fidelity)
    : providerName_(std::move(providerName)),
      canonicalName_(canonicalOperationName(providerName_)),
      numQubits_(numQubits), numParameters_(numParameters),
      loci_(std::move(loci)), duration_(duration), fidelity_(fidelity) {
  if (canonicalName_.empty()) {
    throw std::invalid_argument(
        "Compiler target operation name must not be empty");
  }
  if (numQubits_ == 0) {
    throw std::invalid_argument(
        "Compiler target operation qubit count must be positive");
  }
  validateFidelity(fidelity_, "Compiler target operation fidelity");

  if (!loci_) {
    return;
  }
  std::set<std::vector<SiteId>> uniqueLoci;
  for (const auto& locus : *loci_) {
    if (locus.sites().size() != numQubits_) {
      throw std::invalid_argument(
          "Compiler target operation locus does not match its declared arity");
    }
    if (!uniqueLoci.emplace(locus.sites().begin(), locus.sites().end())
             .second) {
      throw std::invalid_argument(
          "Compiler target operation contains a duplicate locus");
    }
  }
}

StringRef CompilerTarget::Operation::providerName() const noexcept {
  return providerName_;
}

StringRef CompilerTarget::Operation::canonicalName() const noexcept {
  return canonicalName_;
}

size_t CompilerTarget::Operation::numQubits() const noexcept {
  return numQubits_;
}

size_t CompilerTarget::Operation::numParameters() const noexcept {
  return numParameters_;
}

bool CompilerTarget::Operation::hasGlobalLoci() const noexcept {
  return !loci_;
}

ArrayRef<CompilerTarget::OperationLocus>
CompilerTarget::Operation::loci() const noexcept {
  if (!loci_) {
    return {};
  }
  return *loci_;
}

std::optional<uint64_t> CompilerTarget::Operation::duration() const noexcept {
  return duration_;
}

std::optional<double> CompilerTarget::Operation::fidelity() const noexcept {
  return fidelity_;
}

bool CompilerTarget::Operation::supports(const ArrayRef<SiteId> locus) const {
  if (locus.size() != numQubits_) {
    return false;
  }
  llvm::SmallDenseSet<SiteId, 4> uniqueSites;
  uniqueSites.reserve(locus.size());
  if (!llvm::all_of(locus, [&](const auto site) {
        return site >= 0 && uniqueSites.insert(site).second;
      })) {
    return false;
  }
  return !loci_ || llvm::any_of(*loci_, [&](const auto& candidate) {
    return std::ranges::equal(candidate.sites(), locus);
  });
}

struct CompilerTarget::Storage {
  Storage(std::optional<std::string> targetName, std::vector<Site> targetSites,
          std::optional<std::vector<Coupling>> targetCouplings,
          std::optional<std::vector<Operation>> targetOperations,
          std::optional<DurationUnit> targetDurationUnit);

  [[nodiscard]] bool validLocus(ArrayRef<SiteId> locus) const;
  [[nodiscard]] bool
  supportsOperation(StringRef name, ArrayRef<SiteId> locus,
                    std::optional<size_t> numParameters) const;
  [[nodiscard]] bool gateIsGloballySupported(GateKind gate) const;
  [[nodiscard]] bool hasGlobalGate(GateKind gate) const;
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
  SmallVector<GateKind> globalGates;
  std::optional<SynthesisBasis> basis;
};

CompilerTarget::Storage::Storage(
    std::optional<std::string> targetName, std::vector<Site> targetSites,
    std::optional<std::vector<Coupling>> targetCouplings,
    std::optional<std::vector<Operation>> targetOperations,
    std::optional<DurationUnit> targetDurationUnit)
    : name(std::move(targetName)), durationUnit(std::move(targetDurationUnit)),
      sites(std::move(targetSites)), couplings(std::move(targetCouplings)),
      operations(std::move(targetOperations)) {
  if (name && name->empty()) {
    throw std::invalid_argument(
        "Compiler target name must not be empty when present");
  }
  if (sites.empty()) {
    throw std::invalid_argument(
        "Compiler target must contain at least one site");
  }

  siteIds.reserve(sites.size());
  siteToVertex.reserve(sites.size());
  for (const auto [vertex, site] : llvm::enumerate(sites)) {
    if (!siteToVertex.try_emplace(site.id(), vertex).second) {
      throw std::invalid_argument(
          "Compiler target contains duplicate site IDs");
    }
    siteIds.emplace_back(site.id());
  }

  if (couplings) {
    std::set<Coupling> canonicalCouplings;
    for (auto [source, target] : *couplings) {
      if (!siteToVertex.contains(source) || !siteToVertex.contains(target)) {
        throw std::invalid_argument(
            "Compiler target topology references an unknown site");
      }
      if (source == target) {
        throw std::invalid_argument(
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
      throw std::invalid_argument(
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
        throw std::invalid_argument(
            "Compiler target topology must be connected");
      }
    }
  } else {
    maximumDegree = sites.size() - 1;
  }

  if (operations) {
    for (const auto [index, operation] : llvm::enumerate(*operations)) {
      for (const auto& locus : operation.loci()) {
        if (llvm::any_of(locus.sites(), [&](const auto site) {
              return !siteToVertex.contains(site);
            })) {
          throw std::invalid_argument(
              "Compiler target operation locus references an unknown site");
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
               llvm::any_of(operation.loci(), [](const auto& locus) {
                 return locus.duration().has_value();
               });
      });
  if ((hasSiteTiming || hasOperationTiming) && !durationUnit) {
    throw std::invalid_argument(
        "Compiler target timing metadata requires a duration unit");
  }

  for (const auto& specification : GATE_SPECIFICATIONS) {
    if (gateIsGloballySupported(specification.kind)) {
      globalGates.emplace_back(specification.kind);
    }
  }
  basis = resolveSynthesisBasis();
}

bool CompilerTarget::Storage::validLocus(const ArrayRef<SiteId> locus) const {
  llvm::SmallDenseSet<SiteId, 4> uniqueSites;
  uniqueSites.reserve(locus.size());
  return llvm::all_of(locus, [&](const auto site) {
    return siteToVertex.contains(site) && uniqueSites.insert(site).second;
  });
}

bool CompilerTarget::Storage::supportsOperation(
    const StringRef operationName, const ArrayRef<SiteId> locus,
    const std::optional<size_t> numParameters) const {
  if (!validLocus(locus)) {
    return false;
  }
  const auto canonical = canonicalOperationName(operationName);
  if (canonical.empty() || locus.empty()) {
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
    return (!numParameters || operation.numParameters() == *numParameters) &&
           operation.supports(locus);
  });
}

bool CompilerTarget::Storage::gateIsGloballySupported(
    const GateKind gate) const {
  const auto& specification = gateSpecification(gate);
  if (specification.numQubits == 1) {
    return llvm::all_of(siteIds, [&](const auto site) {
      const std::array locus{site};
      return supportsOperation(specification.name, locus,
                               specification.numParameters);
    });
  }
  if (sites.size() < 2) {
    return false;
  }

  const auto supportedOnEdge = [&](const SiteId first, const SiteId second) {
    const std::array forward{first, second};
    const std::array reverse{second, first};
    const auto supportsForward = supportsOperation(specification.name, forward,
                                                   specification.numParameters);
    const auto supportsReverse = supportsOperation(specification.name, reverse,
                                                   specification.numParameters);
    return specification.symmetric ? supportsForward || supportsReverse
                                   : supportsForward && supportsReverse;
  };

  if (couplings) {
    return llvm::all_of(*couplings, [&](const auto& coupling) {
      return supportedOnEdge(coupling.first, coupling.second);
    });
  }
  for (size_t first = 0; first < siteIds.size(); ++first) {
    for (size_t second = first + 1; second < siteIds.size(); ++second) {
      if (!supportedOnEdge(siteIds[first], siteIds[second])) {
        return false;
      }
    }
  }
  return true;
}

bool CompilerTarget::Storage::hasGlobalGate(const GateKind gate) const {
  return llvm::is_contained(globalGates, gate);
}

std::optional<CompilerTarget::SynthesisBasis>
CompilerTarget::Storage::resolveSynthesisBasis() const {
  std::optional<SingleQubitBasis> singleQubit;
  if (hasGlobalGate(GateKind::U)) {
    singleQubit = SingleQubitBasis::U;
  } else if (hasGlobalGate(GateKind::X) && hasGlobalGate(GateKind::SX) &&
             hasGlobalGate(GateKind::RZ)) {
    singleQubit = SingleQubitBasis::ZSXX;
  } else if (hasGlobalGate(GateKind::R)) {
    singleQubit = SingleQubitBasis::R;
  } else if (hasGlobalGate(GateKind::RX) && hasGlobalGate(GateKind::RZ)) {
    singleQubit = SingleQubitBasis::XZX;
  } else if (hasGlobalGate(GateKind::RX) && hasGlobalGate(GateKind::RY)) {
    singleQubit = SingleQubitBasis::XYX;
  } else if (hasGlobalGate(GateKind::RY) && hasGlobalGate(GateKind::RZ)) {
    singleQubit = SingleQubitBasis::ZYZ;
  }

  constexpr std::array entanglerPreference{
      GateKind::RXX,   GateKind::RYY, GateKind::RZX, GateKind::RZZ,
      GateKind::ISWAP, GateKind::CZ,  GateKind::CX,  GateKind::ECR,
  };
  const decltype(entanglerPreference.cbegin()) entangler =
      std::ranges::find_if(entanglerPreference, [&](const auto candidate) {
        return hasGlobalGate(candidate);
      });
  if (!singleQubit || entangler == entanglerPreference.end()) {
    return std::nullopt;
  }
  return SynthesisBasis{.singleQubit = *singleQubit, .entangler = *entangler};
}

CompilerTarget::CompilerTarget(const size_t numQubits,
                               std::optional<std::vector<Coupling>> couplings,
                               std::optional<std::vector<Operation>> operations,
                               std::optional<DurationUnit> durationUnit)
    : CompilerTarget(std::nullopt, makeDenseSites(numQubits),
                     std::move(couplings), std::move(operations),
                     std::move(durationUnit), StorageConstructorTag{}) {}

CompilerTarget::CompilerTarget(std::string name, const size_t numQubits,
                               std::optional<std::vector<Coupling>> couplings,
                               std::optional<std::vector<Operation>> operations,
                               std::optional<DurationUnit> durationUnit)
    : CompilerTarget(std::optional<std::string>(std::move(name)),
                     makeDenseSites(numQubits), std::move(couplings),
                     std::move(operations), std::move(durationUnit),
                     StorageConstructorTag{}) {}

CompilerTarget::CompilerTarget(std::vector<Site> sites,
                               std::optional<std::vector<Coupling>> couplings,
                               std::optional<std::vector<Operation>> operations,
                               std::optional<DurationUnit> durationUnit)
    : CompilerTarget(std::nullopt, std::move(sites), std::move(couplings),
                     std::move(operations), std::move(durationUnit),
                     StorageConstructorTag{}) {}

CompilerTarget::CompilerTarget(std::string name, std::vector<Site> sites,
                               std::optional<std::vector<Coupling>> couplings,
                               std::optional<std::vector<Operation>> operations,
                               std::optional<DurationUnit> durationUnit)
    : CompilerTarget(std::optional<std::string>(std::move(name)),
                     std::move(sites), std::move(couplings),
                     std::move(operations), std::move(durationUnit),
                     StorageConstructorTag{}) {}

CompilerTarget::CompilerTarget(
    std::optional<std::string> name, std::vector<Site> sites,
    std::optional<std::vector<Coupling>> couplings,
    std::optional<std::vector<Operation>> operations,
    std::optional<DurationUnit> durationUnit,
    [[maybe_unused]] StorageConstructorTag storageConstructorTag)
    : storage_(std::make_shared<const Storage>(
          std::move(name), std::move(sites), std::move(couplings),
          std::move(operations), std::move(durationUnit))) {}

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
  if (vertex >= numQubits()) {
    throwVertexOutOfRange();
  }
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
  if (source >= numQubits() || target >= numQubits()) {
    throwVertexOutOfRange();
  }
  if (!hasExplicitTopology()) {
    return source != target;
  }
  return llvm::is_contained(storage_->adjacency[source], target);
}

void CompilerTarget::forEachNeighbour(
    const size_t vertex,
    const llvm::function_ref<void(size_t)> callback) const {
  if (!hasExplicitTopology()) {
    if (vertex >= numQubits()) {
      throwVertexOutOfRange();
    }
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
  if (source >= numQubits() || target >= numQubits()) {
    throwVertexOutOfRange();
  }
  if (!hasExplicitTopology()) {
    return source == target ? 0 : 1;
  }
  return storage_->distances[(source * numQubits()) + target];
}

ArrayRef<size_t> CompilerTarget::explicitNeighbours(const size_t vertex) const {
  if (vertex >= numQubits()) {
    throwVertexOutOfRange();
  }
  return storage_->adjacency[vertex];
}

void CompilerTarget::throwVertexOutOfRange() {
  throw std::out_of_range("Compiler target vertex is out of range");
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

bool CompilerTarget::supportsOperation(
    const StringRef operationName, const ArrayRef<SiteId> locus,
    const std::optional<size_t> numParameters) const {
  return storage_->supportsOperation(operationName, locus, numParameters);
}

bool CompilerTarget::supports(::mlir::Operation* operation,
                              const ArrayRef<SiteId> locus) const {
  if (operation == nullptr || !storage_->validLocus(locus)) {
    return false;
  }

  if (auto unitary = dyn_cast<qco::UnitaryOpInterface>(operation)) {
    if (unitary.getNumQubits() != locus.size()) {
      return false;
    }
    if (isa<qco::BarrierOp, qco::GPhaseOp>(operation)) {
      return true;
    }
    if (auto controlled = dyn_cast<qco::CtrlOp>(operation);
        controlled && controlled.getNumControls() == 1 &&
        controlled.getNumTargets() == 1 &&
        controlled.getNumBodyUnitaries() == 1) {
      auto* const body = controlled.getBodyUnitary(0).getOperation();
      if (isa<qco::XOp>(body)) {
        return storage_->supportsOperation("cx", locus, 0);
      }
      if (isa<qco::ZOp>(body)) {
        return storage_->supportsOperation("cz", locus, 0);
      }
    }
    return storage_->supportsOperation(unitary.getBaseSymbol(), locus,
                                       unitary.getNumParams());
  }
  if (isa<qco::MeasureOp>(operation)) {
    return locus.size() == 1 &&
           storage_->supportsOperation("measure", locus, 0);
  }
  if (isa<qco::ResetOp>(operation)) {
    return locus.size() == 1 && storage_->supportsOperation("reset", locus, 0);
  }
  return false;
}

bool CompilerTarget::supports(const GateKind gate,
                              const ArrayRef<SiteId> locus) const {
  const auto& specification = gateSpecification(gate);
  return locus.size() == specification.numQubits &&
         storage_->supportsOperation(specification.name, locus,
                                     specification.numParameters);
}

ArrayRef<GateKind> CompilerTarget::globallySupportedGates() const noexcept {
  return storage_->globalGates;
}

std::optional<CompilerTarget::SynthesisBasis>
CompilerTarget::synthesisBasis() const noexcept {
  return storage_->basis;
}

} // namespace mlir
