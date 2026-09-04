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

#include "mlir/Dialect/MQT/IR/MQTAttributes.h"
#include "mlir/Dialect/QCO/IR/QCOInterfaces.h"
#include "mlir/Dialect/QCO/IR/QCOOps.h"

#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/STLFunctionalExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringMap.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/Error.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/MLIRContext.h>
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
#include <string>
#include <system_error>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace mlir {
namespace {

using GateKind = CompilerTarget::GateKind;
using SiteId = CompilerTarget::SiteId;

struct GateSpecification {
  GateKind kind{};
  llvm::StringLiteral name;
  size_t arity{};
  size_t numParameters{};
};

constexpr std::array GATE_SPECIFICATIONS{
    GateSpecification{
        .kind = GateKind::U, .name = "u", .arity = 1, .numParameters = 3},
    GateSpecification{
        .kind = GateKind::X, .name = "x", .arity = 1, .numParameters = 0},
    GateSpecification{
        .kind = GateKind::SX, .name = "sx", .arity = 1, .numParameters = 0},
    GateSpecification{
        .kind = GateKind::RZ, .name = "rz", .arity = 1, .numParameters = 1},
    GateSpecification{
        .kind = GateKind::RX, .name = "rx", .arity = 1, .numParameters = 1},
    GateSpecification{
        .kind = GateKind::RY, .name = "ry", .arity = 1, .numParameters = 1},
    GateSpecification{
        .kind = GateKind::R, .name = "r", .arity = 1, .numParameters = 2},
    GateSpecification{
        .kind = GateKind::RXX, .name = "rxx", .arity = 2, .numParameters = 1},
    GateSpecification{
        .kind = GateKind::RYY, .name = "ryy", .arity = 2, .numParameters = 1},
    GateSpecification{
        .kind = GateKind::RZX, .name = "rzx", .arity = 2, .numParameters = 1},
    GateSpecification{
        .kind = GateKind::RZZ, .name = "rzz", .arity = 2, .numParameters = 1},
    GateSpecification{.kind = GateKind::ISWAP,
                      .name = "iswap",
                      .arity = 2,
                      .numParameters = 0},
    GateSpecification{
        .kind = GateKind::CZ, .name = "cz", .arity = 2, .numParameters = 0},
    GateSpecification{
        .kind = GateKind::CX, .name = "cx", .arity = 2, .numParameters = 0},
    GateSpecification{
        .kind = GateKind::ECR, .name = "ecr", .arity = 2, .numParameters = 0},
};

} // namespace

[[nodiscard]] static std::string canonicalOperationName(StringRef name) {
  auto canonical = name.trim().lower();
  if (canonical == "prx") {
    canonical = "r";
  } else if (canonical == "i") {
    canonical = "id";
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
validatePositiveCoherenceTime(std::optional<uint64_t> time,
                              StringRef description) {
  if (time && *time == 0) {
    return invalidTarget(description + " must be positive");
  }
  return llvm::Error::success();
}

[[nodiscard]] static llvm::Error
validateFidelity(std::optional<double> fidelity, StringRef description) {
  if (fidelity &&
      (!std::isfinite(*fidelity) || *fidelity < 0. || *fidelity > 1.)) {
    return invalidTarget(description + " must be finite and in [0, 1]");
  }
  return llvm::Error::success();
}

CompilerTarget::Connectivity CompilerTarget::Connectivity::allToAll() {
  return {Kind::AllToAll, {}};
}

CompilerTarget::Connectivity
CompilerTarget::Connectivity::fromCouplings(ArrayRef<Coupling> couplings) {
  return {Kind::Explicit, couplings};
}

CompilerTarget::Connectivity::Kind
CompilerTarget::Connectivity::kind() const noexcept {
  return kind_;
}

ArrayRef<CompilerTarget::Coupling>
CompilerTarget::Connectivity::couplings() const noexcept {
  return couplings_;
}

CompilerTarget::Connectivity::Connectivity(Kind kind,
                                           ArrayRef<Coupling> couplings)
    : kind_(kind), couplings_(couplings) {}

[[nodiscard]] static llvm::Expected<std::vector<CompilerTarget::Site>>
makeDenseSites(size_t numSites) {
  if (numSites == 0) {
    return invalidTarget("Compiler target must contain at least one site");
  }
  constexpr auto maxNumSites =
      static_cast<uintmax_t>(std::numeric_limits<int64_t>::max()) + 1;
  if (static_cast<uintmax_t>(numSites) > maxNumSites) {
    return invalidTarget(
        "Compiler target site count exceeds the nonnegative i64 site domain");
  }

  std::vector<CompilerTarget::Site> sites;
  sites.reserve(numSites);
  for (size_t id = 0; id < numSites; ++id) {
    auto site = CompilerTarget::Site::create(static_cast<SiteId>(id));
    if (!site) {
      return site.takeError();
    }
    sites.emplace_back(std::move(*site));
  }
  return sites;
}

llvm::Expected<CompilerTarget::DurationUnit>
CompilerTarget::DurationUnit::create(std::string unit, double scaleFactor) {
  if (StringRef(unit).trim().empty()) {
    return invalidTarget("Compiler target duration unit must not be empty");
  }
  if (!std::isfinite(scaleFactor) || scaleFactor <= 0.) {
    return invalidTarget(
        "Compiler target duration scale factor must be positive and finite");
  }
  return DurationUnit(std::move(unit), scaleFactor);
}

CompilerTarget::DurationUnit::DurationUnit(std::string unit, double scaleFactor)
    : unit_(std::move(unit)), scaleFactor_(scaleFactor) {}

StringRef CompilerTarget::DurationUnit::unit() const noexcept { return unit_; }

double CompilerTarget::DurationUnit::scaleFactor() const noexcept {
  return scaleFactor_;
}

llvm::Expected<CompilerTarget::Site>
CompilerTarget::Site::create(SiteId id, std::optional<std::string> name,
                             std::optional<uint64_t> t1,
                             std::optional<uint64_t> t2) {
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

CompilerTarget::Site::Site(SiteId id, std::optional<std::string> name,
                           std::optional<uint64_t> t1,
                           std::optional<uint64_t> t2)
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
                                  std::optional<uint64_t> duration,
                                  std::optional<double> fidelity) {
  std::unordered_set<SiteId> uniqueSites;
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
                                     std::optional<uint64_t> duration,
                                     std::optional<double> fidelity)
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

CompilerTarget::Operation::Arity
CompilerTarget::Operation::Arity::fixed(size_t value) noexcept {
  return {Kind::Fixed, value};
}

CompilerTarget::Operation::Arity
CompilerTarget::Operation::Arity::variadic(size_t minimum) noexcept {
  return {Kind::Variadic, minimum};
}

CompilerTarget::Operation::Arity::Kind
CompilerTarget::Operation::Arity::kind() const noexcept {
  return kind_;
}

size_t CompilerTarget::Operation::Arity::value() const noexcept {
  return value_;
}

bool CompilerTarget::Operation::Arity::accepts(size_t width) const noexcept {
  return kind_ == Kind::Variadic ? width >= value_ : width == value_;
}

CompilerTarget::Operation::Arity::Arity(Kind kind, size_t value) noexcept
    : kind_(kind), value_(value) {}

llvm::Expected<CompilerTarget::Operation> CompilerTarget::Operation::create(
    std::string name, size_t arity, size_t numParameters,
    std::vector<SiteTuple> siteTuples, std::optional<uint64_t> duration,
    std::optional<double> fidelity,
    std::optional<std::vector<std::vector<SiteId>>> applicableSiteTuples) {
  return create(std::move(name), Arity::fixed(arity), numParameters,
                std::move(siteTuples), duration, fidelity,
                std::move(applicableSiteTuples));
}

llvm::Expected<CompilerTarget::Operation> CompilerTarget::Operation::create(
    std::string name, Arity arity, size_t numParameters,
    std::vector<SiteTuple> siteTuples, std::optional<uint64_t> duration,
    std::optional<double> fidelity,
    std::optional<std::vector<std::vector<SiteId>>> applicableSiteTuples) {
  auto canonicalName = canonicalOperationName(name);
  if (canonicalName.empty()) {
    return invalidTarget("Compiler target operation name must not be empty");
  }
  if (auto error =
          validateFidelity(fidelity, "Compiler target operation fidelity")) {
    return std::move(error);
  }
  if (arity.kind() == Arity::Kind::Variadic && arity.value() == 0) {
    return invalidTarget(
        "Compiler target operation variadic minimum must be positive");
  }
  if (arity.kind() == Arity::Kind::Variadic && !siteTuples.empty()) {
    return invalidTarget(
        "Compiler target variadic operation cannot contain site tuples");
  }
  if (arity.kind() == Arity::Kind::Fixed && arity.value() == 0 &&
      !siteTuples.empty()) {
    return invalidTarget(
        "Compiler target zero-arity operation cannot contain site tuples");
  }

  SmallVector<ArrayRef<SiteId>> uniqueSiteCombinations;
  for (const auto& siteTuple : siteTuples) {
    if (!arity.accepts(siteTuple.sites().size())) {
      return invalidTarget(
          "Compiler target operation site tuple does not match its arity");
    }
    if (llvm::is_contained(uniqueSiteCombinations, siteTuple.sites())) {
      return invalidTarget(
          "Compiler target operation contains a duplicate site tuple");
    }
    uniqueSiteCombinations.emplace_back(siteTuple.sites());
  }

  SmallVector<ArrayRef<SiteId>> uniqueApplicableSiteCombinations;
  if (applicableSiteTuples) {
    for (const auto& sites : *applicableSiteTuples) {
      if (!arity.accepts(sites.size())) {
        return invalidTarget("Compiler target operation applicable site tuple "
                             "does not match its arity");
      }
      std::unordered_set<SiteId> uniqueSites;
      for (const auto site : sites) {
        if (site < 0) {
          return invalidTarget("Compiler target operation applicable site "
                               "tuple contains a negative site ID");
        }
        if (!uniqueSites.insert(site).second) {
          return invalidTarget("Compiler target operation applicable site "
                               "tuple contains a duplicate site");
        }
      }
      if (llvm::is_contained(uniqueApplicableSiteCombinations,
                             ArrayRef<SiteId>(sites))) {
        return invalidTarget("Compiler target operation contains a duplicate "
                             "applicable site tuple");
      }
      uniqueApplicableSiteCombinations.emplace_back(sites);
    }
    if (llvm::any_of(siteTuples, [&](const auto& siteTuple) {
          return !llvm::is_contained(uniqueApplicableSiteCombinations,
                                     siteTuple.sites());
        })) {
      return invalidTarget("Compiler target operation calibration references "
                           "an inapplicable site tuple");
    }
  }
  return Operation(std::move(name), std::move(canonicalName), arity,
                   numParameters, std::move(siteTuples), duration, fidelity,
                   std::move(applicableSiteTuples));
}

CompilerTarget::Operation::Operation(
    std::string name, std::string canonicalName, Arity arity,
    size_t numParameters, std::vector<SiteTuple> siteTuples,
    std::optional<uint64_t> duration, std::optional<double> fidelity,
    std::optional<std::vector<std::vector<SiteId>>> applicableSiteTuples)
    : name_(std::move(name)), canonicalName_(std::move(canonicalName)),
      arity_(arity), numParameters_(numParameters),
      siteTuples_(std::move(siteTuples)), duration_(duration),
      fidelity_(fidelity),
      applicableSiteTuples_(std::move(applicableSiteTuples)) {}

StringRef CompilerTarget::Operation::name() const noexcept { return name_; }

StringRef CompilerTarget::Operation::canonicalName() const noexcept {
  return canonicalName_;
}

CompilerTarget::Operation::Arity
CompilerTarget::Operation::arity() const noexcept {
  return arity_;
}

size_t CompilerTarget::Operation::numParameters() const noexcept {
  return numParameters_;
}

ArrayRef<CompilerTarget::SiteTuple>
CompilerTarget::Operation::siteTuples() const noexcept {
  return siteTuples_;
}

bool CompilerTarget::Operation::hasExplicitApplicability() const noexcept {
  return applicableSiteTuples_.has_value();
}

ArrayRef<std::vector<SiteId>>
CompilerTarget::Operation::applicableSiteTuples() const noexcept {
  if (!applicableSiteTuples_) {
    return {};
  }
  return *applicableSiteTuples_;
}

std::optional<uint64_t> CompilerTarget::Operation::duration() const noexcept {
  return duration_;
}

std::optional<double> CompilerTarget::Operation::fidelity() const noexcept {
  return fidelity_;
}

CompilerTarget::NativeOperations
CompilerTarget::NativeOperations::unrestricted() {
  return {Kind::Unrestricted, {}};
}

CompilerTarget::NativeOperations
CompilerTarget::NativeOperations::fromOperations(
    ArrayRef<Operation> operations) {
  return {Kind::Explicit, operations};
}

CompilerTarget::NativeOperations::Kind
CompilerTarget::NativeOperations::kind() const noexcept {
  return kind_;
}

ArrayRef<CompilerTarget::Operation>
CompilerTarget::NativeOperations::operations() const noexcept {
  return operations_;
}

CompilerTarget::NativeOperations::NativeOperations(
    Kind kind, ArrayRef<Operation> operations)
    : kind_(kind), operations_(operations) {}

struct CompilerTarget::Storage {
  Storage(std::optional<std::string> targetName, std::vector<Site> targetSites,
          Connectivity::Kind targetConnectivityKind,
          SmallVector<Coupling> targetCouplings,
          NativeOperations::Kind targetNativeOperationsKind,
          SmallVector<Operation> targetOperations,
          std::optional<DurationUnit> targetDurationUnit);

  [[nodiscard]] static llvm::Expected<std::shared_ptr<const Storage>>
  create(std::optional<std::string> targetName, std::vector<Site> targetSites,
         Connectivity::Kind targetConnectivityKind,
         SmallVector<Coupling> targetCouplings,
         NativeOperations::Kind targetNativeOperationsKind,
         SmallVector<Operation> targetOperations,
         std::optional<DurationUnit> targetDurationUnit);

  [[nodiscard]] llvm::Error initialize();

  [[nodiscard]] bool
  isApplicable(size_t operationIndex, size_t arity,
               std::optional<ArrayRef<SiteId>> orderedSites) const;
  [[nodiscard]] bool
  supportsOperation(StringRef name, size_t arity,
                    std::optional<size_t> numParameters,
                    std::optional<ArrayRef<SiteId>> orderedSites = std::nullopt,
                    bool variadicOnly = false) const;
  [[nodiscard]] bool supportsGate(GateKind gate,
                                  ArrayRef<SiteId> orderedSites) const;
  [[nodiscard]] std::optional<SynthesisBasis> resolveSynthesisBasis() const;

  std::optional<std::string> name;
  std::optional<DurationUnit> durationUnit;
  std::vector<Site> sites;
  SmallVector<SiteId> siteIds;
  std::unordered_map<SiteId, size_t> siteToVertex;
  Connectivity::Kind connectivityKind;
  SmallVector<Coupling> couplings;
  SmallVector<SmallVector<size_t, 4>> adjacency;
  SmallVector<size_t> distances;
  size_t maximumDegree = 0;
  NativeOperations::Kind nativeOperationsKind;
  SmallVector<Operation> operations;
  llvm::StringMap<SmallVector<size_t, 1>> capabilities;
  std::vector<std::optional<std::unordered_set<SiteId>>> explicitOneQubitSites;
  std::vector<std::optional<llvm::DenseSet<Coupling>>> explicitTwoQubitSites;
  SmallVector<GateKind> supportedGates;
  std::optional<SynthesisBasis> basis;
};

CompilerTarget::Storage::Storage(
    std::optional<std::string> targetName, std::vector<Site> targetSites,
    Connectivity::Kind targetConnectivityKind,
    SmallVector<Coupling> targetCouplings,
    NativeOperations::Kind targetNativeOperationsKind,
    SmallVector<Operation> targetOperations,
    std::optional<DurationUnit> targetDurationUnit)
    : name(std::move(targetName)), durationUnit(std::move(targetDurationUnit)),
      sites(std::move(targetSites)), connectivityKind(targetConnectivityKind),
      couplings(std::move(targetCouplings)),
      nativeOperationsKind(targetNativeOperationsKind),
      operations(std::move(targetOperations)) {}

llvm::Expected<std::shared_ptr<const CompilerTarget::Storage>>
CompilerTarget::Storage::create(
    std::optional<std::string> targetName, std::vector<Site> targetSites,
    Connectivity::Kind targetConnectivityKind,
    SmallVector<Coupling> targetCouplings,
    NativeOperations::Kind targetNativeOperationsKind,
    SmallVector<Operation> targetOperations,
    std::optional<DurationUnit> targetDurationUnit) {
  auto storage = std::make_shared<Storage>(
      std::move(targetName), std::move(targetSites), targetConnectivityKind,
      std::move(targetCouplings), targetNativeOperationsKind,
      std::move(targetOperations), std::move(targetDurationUnit));
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

  siteIds.reserve(sites.size());
  siteToVertex.reserve(sites.size());
  for (const auto [vertex, site] : llvm::enumerate(sites)) {
    if (!siteToVertex.try_emplace(site.id(), vertex).second) {
      return invalidTarget("Compiler target contains duplicate site IDs");
    }
    siteIds.emplace_back(site.id());
  }

  if (connectivityKind == Connectivity::Kind::Explicit) {
    for (auto& [source, target] : couplings) {
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
    }
    std::ranges::sort(couplings);
    couplings.erase(std::ranges::unique(couplings).begin(), couplings.end());

    adjacency.resize(sites.size());
    for (const auto& [source, target] : couplings) {
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

  if (nativeOperationsKind == NativeOperations::Kind::Explicit) {
    explicitOneQubitSites.resize(operations.size());
    explicitTwoQubitSites.resize(operations.size());
    for (const auto [index, operation] : llvm::enumerate(operations)) {
      if (operation.arity().value() > sites.size()) {
        if (operation.arity().kind() == Operation::Arity::Kind::Variadic) {
          return invalidTarget("Compiler target operation variadic minimum "
                               "exceeds its site count");
        }
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
      if (operation.hasExplicitApplicability()) {
        auto& oneQubitSites = explicitOneQubitSites[index].emplace();
        auto& twoQubitSites = explicitTwoQubitSites[index].emplace();
        oneQubitSites.reserve(operation.applicableSiteTuples().size());
        twoQubitSites.reserve(operation.applicableSiteTuples().size());
        for (const auto& applicableSites : operation.applicableSiteTuples()) {
          if (llvm::any_of(applicableSites, [&](const auto site) {
                return !siteToVertex.contains(site);
              })) {
            return invalidTarget("Compiler target operation applicable site "
                                 "tuple references an unknown site");
          }
          if (applicableSites.size() == 1) {
            oneQubitSites.insert(applicableSites.front());
          } else if (applicableSites.size() == 2) {
            twoQubitSites.insert(
                {applicableSites.front(), applicableSites.back()});
          }
        }
      }
      capabilities[operation.canonicalName()].emplace_back(index);
    }
  }

  const auto hasSiteTiming = llvm::any_of(sites, [](const auto& site) {
    return site.t1().has_value() || site.t2().has_value();
  });
  const auto hasOperationTiming =
      llvm::any_of(operations, [](const auto& operation) {
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
    const bool supportsControlledBase =
        (specification.kind == GateKind::CX &&
         supportsOperation("x", specification.arity,
                           specification.numParameters, std::nullopt,
                           /*variadicOnly=*/true)) ||
        (specification.kind == GateKind::CZ &&
         supportsOperation("z", specification.arity,
                           specification.numParameters, std::nullopt,
                           /*variadicOnly=*/true));
    if (supportsControlledBase ||
        supportsOperation(specification.name, specification.arity,
                          specification.numParameters)) {
      supportedGates.emplace_back(specification.kind);
    }
  }
  basis = resolveSynthesisBasis();
  return llvm::Error::success();
}

bool CompilerTarget::Storage::isApplicable(
    size_t operationIndex, size_t arity,
    std::optional<ArrayRef<SiteId>> orderedSites) const {
  const auto& operation = operations[operationIndex];
  if (!operation.hasExplicitApplicability()) {
    return true;
  }
  if (!orderedSites) {
    if (arity == 1) {
      return !explicitOneQubitSites[operationIndex]->empty();
    }
    if (arity == 2) {
      return !explicitTwoQubitSites[operationIndex]->empty();
    }
    return llvm::any_of(operation.applicableSiteTuples(),
                        [&](const auto& applicableSites) {
                          return applicableSites.size() == arity;
                        });
  }
  if (arity == 1) {
    return explicitOneQubitSites[operationIndex]->contains((*orderedSites)[0]);
  }
  if (arity == 2) {
    return explicitTwoQubitSites[operationIndex]->contains(
        {(*orderedSites)[0], (*orderedSites)[1]});
  }
  return llvm::any_of(
      operation.applicableSiteTuples(), [&](const auto& applicableSites) {
        return ArrayRef<SiteId>(applicableSites) == *orderedSites;
      });
}

bool CompilerTarget::Storage::supportsOperation(
    StringRef operationName, size_t arity, std::optional<size_t> numParameters,
    std::optional<ArrayRef<SiteId>> orderedSites, bool variadicOnly) const {
  const auto canonical = canonicalOperationName(operationName);
  if (canonical.empty() || arity > sites.size() ||
      (orderedSites && orderedSites->size() != arity)) {
    return false;
  }
  if (orderedSites) {
    for (const auto [index, site] : llvm::enumerate(*orderedSites)) {
      if (!siteToVertex.contains(site) ||
          llvm::is_contained(orderedSites->take_front(index), site)) {
        return false;
      }
    }
  }
  if (nativeOperationsKind == NativeOperations::Kind::Unrestricted) {
    return true;
  }
  const auto found = capabilities.find(canonical);
  if (found == capabilities.end()) {
    return false;
  }
  return llvm::any_of(found->second, [&](const auto index) {
    const auto& operation = operations[index];
    return (!variadicOnly ||
            operation.arity().kind() == Operation::Arity::Kind::Variadic) &&
           operation.arity().accepts(arity) &&
           (!numParameters || operation.numParameters() == *numParameters) &&
           isApplicable(index, arity, orderedSites);
  });
}

bool CompilerTarget::Storage::supportsGate(
    GateKind gate, ArrayRef<SiteId> orderedSites) const {
  if ((gate == GateKind::CX &&
       supportsOperation("x", 2, 0, orderedSites, /*variadicOnly=*/true)) ||
      (gate == GateKind::CZ &&
       supportsOperation("z", 2, 0, orderedSites, /*variadicOnly=*/true))) {
    return true;
  }
  const decltype(GATE_SPECIFICATIONS.cbegin()) specification =
      std::ranges::find_if(GATE_SPECIFICATIONS, [&](const auto& candidate) {
        return candidate.kind == gate;
      });
  assert(specification != GATE_SPECIFICATIONS.end() &&
         "unknown compiler target gate");
  return supportsOperation(specification->name, specification->arity,
                           specification->numParameters, orderedSites);
}

std::optional<CompilerTarget::SynthesisBasis>
CompilerTarget::Storage::resolveSynthesisBasis() const {
  const auto supportsEveryPlacement = [&](StringRef operationName, size_t arity,
                                          size_t numParameters,
                                          bool variadicOnly = false) {
    if (nativeOperationsKind == NativeOperations::Kind::Unrestricted) {
      return true;
    }
    const auto found = capabilities.find(operationName);
    if (found == capabilities.end()) {
      return false;
    }
    return llvm::any_of(found->second, [&](const auto index) {
      const auto& operation = operations[index];
      return (!variadicOnly ||
              operation.arity().kind() == Operation::Arity::Kind::Variadic) &&
             operation.arity().accepts(arity) &&
             operation.numParameters() == numParameters &&
             !operation.hasExplicitApplicability();
    });
  };
  const auto supportsOnEverySite = [&](GateKind gate) {
    return llvm::all_of(siteIds, [&](SiteId site) {
      return supportsGate(gate, ArrayRef<SiteId>(&site, 1));
    });
  };
  std::optional<SingleQubitBasis> singleQubit;
  if (supportsOnEverySite(GateKind::U)) {
    singleQubit = SingleQubitBasis::U;
  } else if (supportsOnEverySite(GateKind::X) &&
             supportsOnEverySite(GateKind::SX) &&
             supportsOnEverySite(GateKind::RZ)) {
    singleQubit = SingleQubitBasis::ZSXX;
  } else if (supportsOnEverySite(GateKind::R)) {
    singleQubit = SingleQubitBasis::R;
  } else if (supportsOnEverySite(GateKind::RX) &&
             supportsOnEverySite(GateKind::RZ)) {
    singleQubit = SingleQubitBasis::XZX;
  } else if (supportsOnEverySite(GateKind::RX) &&
             supportsOnEverySite(GateKind::RY)) {
    singleQubit = SingleQubitBasis::XYX;
  } else if (supportsOnEverySite(GateKind::RY) &&
             supportsOnEverySite(GateKind::RZ)) {
    singleQubit = SingleQubitBasis::ZYZ;
  }

  const auto supportsOnEveryCoupling = [&](GateKind gate) {
    if (sites.size() < 2) {
      return false;
    }
    if ((gate == GateKind::CX && supportsEveryPlacement("x", 2, 0, true)) ||
        (gate == GateKind::CZ && supportsEveryPlacement("z", 2, 0, true))) {
      return true;
    }
    const decltype(GATE_SPECIFICATIONS.cbegin()) specification =
        std::ranges::find_if(GATE_SPECIFICATIONS, [&](const auto& candidate) {
          return candidate.kind == gate;
        });
    assert(specification != GATE_SPECIFICATIONS.end() &&
           "unknown compiler target gate");
    if (supportsEveryPlacement(specification->name, specification->arity,
                               specification->numParameters)) {
      return true;
    }
    const auto supportsPair = [&](SiteId source, SiteId target) {
      const std::array forward{source, target};
      const std::array reverse{target, source};
      return supportsGate(gate, forward) || supportsGate(gate, reverse);
    };
    if (connectivityKind == Connectivity::Kind::Explicit) {
      return llvm::all_of(couplings, [&](const auto& coupling) {
        return supportsPair(coupling.first, coupling.second);
      });
    }
    for (size_t source = 0; source < siteIds.size(); ++source) {
      for (size_t target = source + 1; target < siteIds.size(); ++target) {
        if (!supportsPair(siteIds[source], siteIds[target])) {
          return false;
        }
      }
    }
    return true;
  };

  constexpr std::array entanglerPreference{
      GateKind::RXX,   GateKind::RYY, GateKind::RZX, GateKind::RZZ,
      GateKind::ISWAP, GateKind::CZ,  GateKind::CX,  GateKind::ECR,
  };
  const decltype(entanglerPreference.cbegin()) entangler =
      std::ranges::find_if(entanglerPreference, supportsOnEveryCoupling);
  if (!singleQubit || entangler == entanglerPreference.end()) {
    return std::nullopt;
  }
  return SynthesisBasis{.singleQubit = *singleQubit, .entangler = *entangler};
}

llvm::Expected<CompilerTarget>
CompilerTarget::create(size_t numSites, Connectivity connectivity,
                       NativeOperations nativeOperations,
                       std::optional<DurationUnit> durationUnit) {
  auto sites = makeDenseSites(numSites);
  if (!sites) {
    return sites.takeError();
  }
  return createImpl(std::nullopt, std::move(*sites), std::move(connectivity),
                    std::move(nativeOperations), std::move(durationUnit));
}

llvm::Expected<CompilerTarget>
CompilerTarget::create(std::string name, size_t numSites,
                       Connectivity connectivity,
                       NativeOperations nativeOperations,
                       std::optional<DurationUnit> durationUnit) {
  auto sites = makeDenseSites(numSites);
  if (!sites) {
    return sites.takeError();
  }
  return createImpl(std::optional<std::string>(std::move(name)),
                    std::move(*sites), std::move(connectivity),
                    std::move(nativeOperations), std::move(durationUnit));
}

llvm::Expected<CompilerTarget>
CompilerTarget::create(std::vector<Site> sites, Connectivity connectivity,
                       NativeOperations nativeOperations,
                       std::optional<DurationUnit> durationUnit) {
  return createImpl(std::nullopt, std::move(sites), std::move(connectivity),
                    std::move(nativeOperations), std::move(durationUnit));
}

llvm::Expected<CompilerTarget>
CompilerTarget::create(std::string name, std::vector<Site> sites,
                       Connectivity connectivity,
                       NativeOperations nativeOperations,
                       std::optional<DurationUnit> durationUnit) {
  return createImpl(std::optional<std::string>(std::move(name)),
                    std::move(sites), std::move(connectivity),
                    std::move(nativeOperations), std::move(durationUnit));
}

llvm::Expected<CompilerTarget>
CompilerTarget::create(const mqt::CompilationTargetAttr attribute) {
  if (!attribute) {
    return invalidTarget("Compiler target attribute must not be null");
  }
  if (attribute.getConnectivity() != mqt::ConnectivityKind::Explicit &&
      !attribute.getCouplings().empty()) {
    return invalidTarget(
        "Compiler target couplings require explicit connectivity");
  }
  if (attribute.getNativeOperations() != mqt::NativeOperationsKind::Explicit &&
      !attribute.getOperations().empty()) {
    return invalidTarget(
        "Compiler target operations require explicit native operations");
  }

  std::optional<std::string> name;
  if (const auto nameAttr = attribute.getName()) {
    name = nameAttr.getValue().str();
  }

  std::vector<Site> sites;
  sites.reserve(attribute.getSites().size());
  for (const auto siteAttr : attribute.getSites()) {
    std::optional<std::string> siteName;
    if (const auto nameAttr = siteAttr.getName()) {
      siteName = nameAttr.getValue().str();
    }
    auto site = Site::create(siteAttr.getId(), std::move(siteName),
                             siteAttr.getT1(), siteAttr.getT2());
    if (!site) {
      return site.takeError();
    }
    sites.emplace_back(std::move(*site));
  }

  std::optional<DurationUnit> durationUnit;
  if (const auto unitAttr = attribute.getDurationUnit()) {
    auto unit =
        DurationUnit::create(unitAttr.getUnit().getValue().str(),
                             unitAttr.getScaleFactor().getValueAsDouble());
    if (!unit) {
      return unit.takeError();
    }
    durationUnit = std::move(*unit);
  }

  std::vector<Coupling> couplings;
  if (attribute.getConnectivity() == mqt::ConnectivityKind::Explicit) {
    couplings.reserve(attribute.getCouplings().size());
    for (const auto coupling : attribute.getCouplings()) {
      couplings.emplace_back(coupling.getSource(), coupling.getTarget());
    }
  }
  auto connectivity =
      attribute.getConnectivity() == mqt::ConnectivityKind::AllToAll
          ? Connectivity::allToAll()
          : Connectivity::fromCouplings(couplings);

  auto nativeOperations = NativeOperations::unrestricted();
  if (attribute.getNativeOperations() == mqt::NativeOperationsKind::Explicit) {
    std::vector<Operation> operations;
    operations.reserve(attribute.getOperations().size());
    for (const auto operationAttr : attribute.getOperations()) {
      if (operationAttr.getArity().getValue() >
              std::numeric_limits<size_t>::max() ||
          operationAttr.getNumParameters() >
              std::numeric_limits<size_t>::max()) {
        return invalidTarget(
            "Compiler target operation size exceeds the host size domain");
      }
      std::vector<SiteTuple> siteTuples;
      siteTuples.reserve(operationAttr.getSiteTuples().size());
      for (const auto tupleAttr : operationAttr.getSiteTuples()) {
        std::optional<double> fidelity;
        if (const auto fidelityAttr = tupleAttr.getFidelity()) {
          fidelity = fidelityAttr.getValueAsDouble();
        }
        auto siteTuple =
            SiteTuple::create(std::vector<SiteId>(tupleAttr.getSites().begin(),
                                                  tupleAttr.getSites().end()),
                              tupleAttr.getDuration(), fidelity);
        if (!siteTuple) {
          return siteTuple.takeError();
        }
        siteTuples.emplace_back(std::move(*siteTuple));
      }

      std::optional<std::vector<std::vector<SiteId>>> applicableSiteTuples;
      if (operationAttr.getApplicability() ==
          mqt::OperationApplicabilityKind::Explicit) {
        applicableSiteTuples.emplace();
        applicableSiteTuples->reserve(
            operationAttr.getApplicableSiteTuples().size());
        for (const auto tupleAttr : operationAttr.getApplicableSiteTuples()) {
          applicableSiteTuples->emplace_back(tupleAttr.getSites().begin(),
                                             tupleAttr.getSites().end());
        }
      } else if (!operationAttr.getApplicableSiteTuples().empty()) {
        return invalidTarget("Compiler target applicable site tuples require "
                             "explicit operation applicability");
      }

      std::optional<double> fidelity;
      if (const auto fidelityAttr = operationAttr.getFidelity()) {
        fidelity = fidelityAttr.getValueAsDouble();
      }
      const auto arity =
          operationAttr.getArity().getKind() == mqt::OperationArityKind::Fixed
              ? Operation::Arity::fixed(
                    static_cast<size_t>(operationAttr.getArity().getValue()))
              : Operation::Arity::variadic(
                    static_cast<size_t>(operationAttr.getArity().getValue()));
      auto operation = Operation::create(
          operationAttr.getName().getValue().str(), arity,
          static_cast<size_t>(operationAttr.getNumParameters()),
          std::move(siteTuples), operationAttr.getDuration(), fidelity,
          std::move(applicableSiteTuples));
      if (!operation) {
        return operation.takeError();
      }
      operations.emplace_back(std::move(*operation));
    }
    nativeOperations = NativeOperations::fromOperations(operations);
  }

  return createImpl(std::move(name), std::move(sites), std::move(connectivity),
                    std::move(nativeOperations), std::move(durationUnit));
}

llvm::Expected<CompilerTarget>
CompilerTarget::createImpl(std::optional<std::string> name,
                           std::vector<Site> sites, Connectivity connectivity,
                           NativeOperations nativeOperations,
                           std::optional<DurationUnit> durationUnit) {
  auto storage = Storage::create(
      std::move(name), std::move(sites), connectivity.kind_,
      std::move(connectivity.couplings_), nativeOperations.kind_,
      std::move(nativeOperations.operations_), std::move(durationUnit));
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

size_t CompilerTarget::numSites() const noexcept {
  return storage_->sites.size();
}

ArrayRef<CompilerTarget::Site> CompilerTarget::sites() const noexcept {
  return storage_->sites;
}

ArrayRef<SiteId> CompilerTarget::siteIds() const noexcept {
  return storage_->siteIds;
}

std::optional<size_t>
CompilerTarget::vertexForSite(SiteId site) const noexcept {
  const auto found = storage_->siteToVertex.find(site);
  if (found == storage_->siteToVertex.end()) {
    return std::nullopt;
  }
  return found->second;
}

SiteId CompilerTarget::siteForVertex(size_t vertex) const {
  assert(vertex < numSites() && "Compiler target vertex is out of range");
  return storage_->siteIds[vertex];
}

CompilerTarget::Connectivity::Kind
CompilerTarget::connectivityKind() const noexcept {
  return storage_->connectivityKind;
}

ArrayRef<CompilerTarget::Coupling> CompilerTarget::couplings() const noexcept {
  return storage_->couplings;
}

bool CompilerTarget::areAdjacent(size_t source, size_t target) const {
  assert(source < numSites() && target < numSites() &&
         "Compiler target vertex is out of range");
  if (connectivityKind() == Connectivity::Kind::AllToAll) {
    return source != target;
  }
  return llvm::is_contained(storage_->adjacency[source], target);
}

void CompilerTarget::forEachNeighbour(
    size_t vertex, llvm::function_ref<void(size_t)> callback) const {
  if (connectivityKind() == Connectivity::Kind::AllToAll) {
    assert(vertex < numSites() && "Compiler target vertex is out of range");
    for (size_t neighbour = 0; neighbour < numSites(); ++neighbour) {
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

size_t CompilerTarget::distanceBetween(size_t source, size_t target) const {
  assert(source < numSites() && target < numSites() &&
         "Compiler target vertex is out of range");
  if (connectivityKind() == Connectivity::Kind::AllToAll) {
    return source == target ? 0 : 1;
  }
  return storage_->distances[(source * numSites()) + target];
}

ArrayRef<size_t> CompilerTarget::explicitNeighbours(size_t vertex) const {
  assert(vertex < numSites() && "Compiler target vertex is out of range");
  return storage_->adjacency[vertex];
}

size_t CompilerTarget::maxDegree() const noexcept {
  return storage_->maximumDegree;
}

CompilerTarget::NativeOperations::Kind
CompilerTarget::nativeOperationsKind() const noexcept {
  return storage_->nativeOperationsKind;
}

ArrayRef<CompilerTarget::Operation>
CompilerTarget::operations() const noexcept {
  return storage_->operations;
}

bool CompilerTarget::supportsOperation(
    StringRef operationName, size_t arity,
    std::optional<size_t> numParameters) const {
  return storage_->supportsOperation(operationName, arity, numParameters);
}

bool CompilerTarget::supportsOperation(StringRef operationName, size_t arity,
                                       std::optional<size_t> numParameters,
                                       ArrayRef<SiteId> sites) const {
  return storage_->supportsOperation(operationName, arity, numParameters,
                                     sites);
}

bool CompilerTarget::supports(::mlir::Operation* operation) const {
  return supportsImpl(operation, std::nullopt);
}

bool CompilerTarget::supports(::mlir::Operation* operation,
                              ArrayRef<SiteId> sites) const {
  return supportsImpl(operation, sites);
}

bool CompilerTarget::supportsImpl(::mlir::Operation* operation,
                                  std::optional<ArrayRef<SiteId>> sites) const {
  if (operation == nullptr) {
    return false;
  }

  if (auto unitary = dyn_cast<qco::UnitaryOpInterface>(operation)) {
    if (isa<qco::BarrierOp>(operation)) {
      return true;
    }
    if (auto controlled = dyn_cast<qco::CtrlOp>(operation)) {
      if (controlled.getNumControls() == 0 ||
          controlled.getNumBodyUnitaries() != 1) {
        return false;
      }
      auto body = controlled.getBodyUnitary(0);
      if (body.getNumQubits() != controlled.getNumTargets()) {
        return false;
      }
      if (storage_->supportsOperation(body.getBaseSymbol(),
                                      controlled.getNumQubits(),
                                      body.getNumParams(), sites,
                                      /*variadicOnly=*/true)) {
        return true;
      }
      if (controlled.getNumControls() != 1 || controlled.getNumTargets() != 1) {
        return false;
      }
      if (isa<qco::XOp>(body.getOperation())) {
        return storage_->supportsOperation("cx", 2, 0, sites);
      }
      if (isa<qco::ZOp>(body.getOperation())) {
        return storage_->supportsOperation("cz", 2, 0, sites);
      }
      return false;
    }
    return storage_->supportsOperation(unitary.getBaseSymbol(),
                                       unitary.getNumQubits(),
                                       unitary.getNumParams(), sites);
  }
  if (isa<qco::MeasureOp>(operation)) {
    return storage_->supportsOperation("measure", 1, 0, sites);
  }
  if (isa<qco::ResetOp>(operation)) {
    return storage_->supportsOperation("reset", 1, 0, sites);
  }
  return false;
}

bool CompilerTarget::supports(GateKind gate) const {
  return llvm::is_contained(storage_->supportedGates, gate);
}

bool CompilerTarget::supports(GateKind gate, ArrayRef<SiteId> sites) const {
  return storage_->supportsGate(gate, sites);
}

ArrayRef<GateKind> CompilerTarget::supportedGates() const noexcept {
  return storage_->supportedGates;
}

std::optional<CompilerTarget::SynthesisBasis>
CompilerTarget::synthesisBasis() const noexcept {
  return storage_->basis;
}

mqt::CompilationTargetAttr
CompilerTarget::materialize(MLIRContext& context) const {
  Builder builder(&context);

  StringAttr nameAttr;
  if (const auto targetName = name()) {
    nameAttr = builder.getStringAttr(*targetName);
  }

  SmallVector<mqt::SiteAttr> siteAttrs;
  siteAttrs.reserve(sites().size());
  for (const auto& site : sites()) {
    StringAttr siteNameAttr;
    if (const auto siteName = site.name()) {
      siteNameAttr = builder.getStringAttr(*siteName);
    }
    siteAttrs.emplace_back(mqt::SiteAttr::get(&context, site.id(), siteNameAttr,
                                              site.t1(), site.t2()));
  }

  mqt::DurationUnitAttr durationUnitAttr;
  if (const auto& unit = durationUnit()) {
    durationUnitAttr = mqt::DurationUnitAttr::get(
        &context, builder.getStringAttr(unit->unit()),
        builder.getF64FloatAttr(unit->scaleFactor()));
  }

  SmallVector<mqt::CouplingAttr> couplingAttrs;
  couplingAttrs.reserve(couplings().size());
  for (const auto& [source, target] : couplings()) {
    couplingAttrs.emplace_back(
        mqt::CouplingAttr::get(&context, source, target));
  }

  SmallVector<mqt::NativeOperationAttr> operationAttrs;
  operationAttrs.reserve(operations().size());
  for (const auto& operation : operations()) {
    SmallVector<mqt::SiteTupleAttr> siteTupleAttrs;
    siteTupleAttrs.reserve(operation.siteTuples().size());
    for (const auto& siteTuple : operation.siteTuples()) {
      FloatAttr fidelityAttr;
      if (const auto fidelity = siteTuple.fidelity()) {
        fidelityAttr = builder.getF64FloatAttr(*fidelity);
      }
      siteTupleAttrs.emplace_back(mqt::SiteTupleAttr::get(
          &context, siteTuple.sites(), siteTuple.duration(), fidelityAttr));
    }

    SmallVector<mqt::ApplicableSiteTupleAttr> applicableSiteTupleAttrs;
    applicableSiteTupleAttrs.reserve(operation.applicableSiteTuples().size());
    for (const auto& applicableSites : operation.applicableSiteTuples()) {
      applicableSiteTupleAttrs.emplace_back(
          mqt::ApplicableSiteTupleAttr::get(&context, applicableSites));
    }

    FloatAttr fidelityAttr;
    if (const auto fidelity = operation.fidelity()) {
      fidelityAttr = builder.getF64FloatAttr(*fidelity);
    }
    const auto arityKind =
        operation.arity().kind() == Operation::Arity::Kind::Fixed
            ? mqt::OperationArityKind::Fixed
            : mqt::OperationArityKind::Variadic;
    const auto arityAttr = mqt::OperationArityAttr::get(
        &context, arityKind, operation.arity().value());
    const auto applicability =
        operation.hasExplicitApplicability()
            ? mqt::OperationApplicabilityKind::Explicit
            : mqt::OperationApplicabilityKind::Unrestricted;
    operationAttrs.emplace_back(mqt::NativeOperationAttr::get(
        &context, builder.getStringAttr(operation.name()), arityAttr,
        operation.numParameters(), siteTupleAttrs, operation.duration(),
        fidelityAttr, applicability, applicableSiteTupleAttrs));
  }

  const auto connectivity = connectivityKind() == Connectivity::Kind::AllToAll
                                ? mqt::ConnectivityKind::AllToAll
                                : mqt::ConnectivityKind::Explicit;
  const auto nativeOperations =
      nativeOperationsKind() == NativeOperations::Kind::Unrestricted
          ? mqt::NativeOperationsKind::Unrestricted
          : mqt::NativeOperationsKind::Explicit;
  return mqt::CompilationTargetAttr::get(
      &context, nameAttr, siteAttrs, durationUnitAttr, connectivity,
      couplingAttrs, nativeOperations, operationAttrs);
}

} // namespace mlir
