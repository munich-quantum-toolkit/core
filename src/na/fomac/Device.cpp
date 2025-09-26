/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "na/fomac/Device.hpp"

#include "qdmi/FoMaC.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <optional>
#include <ranges>
#include <string>
#include <unordered_map>
#include <vector>

namespace na {

FoMaC::Device::Device(const qdmi::FoMaC::Device& device)
    : qdmi::FoMaC::Device(device) {
  initNameFromDevice();
  initMinAtomDistanceFromDevice();
  initQubitsNumFromDevice();
  initLengthUnitFromDevice();
  initDurationUnitFromDevice();
  initDecoherenceTimesFromDevice();
  initTrapsfromDevice();
  initGlobalSingleQubitOperationsFromDevice();
  initGlobalMultiQubitOperationsFromDevice();
  initLocalSingleQubitOperationsFromDevice();
  initLocalMultiQubitOperationsFromDevice();
  initShuttlingUnitsFromDevice();
}
auto FoMaC::Device::calculateExtentFromSites(
    const std::vector<qdmi::FoMaC::Device::Site>& sites) -> Region {
  auto minX = std::numeric_limits<int64_t>::max();
  auto maxX = std::numeric_limits<int64_t>::min();
  auto minY = std::numeric_limits<int64_t>::max();
  auto maxY = std::numeric_limits<int64_t>::min();
  for (const auto& site : sites) {
    const auto x = *site.getXCoordinate();
    const auto y = *site.getXCoordinate();
    minX = std::min(minX, x);
    maxX = std::max(maxX, x);
    minY = std::min(minY, y);
    maxY = std::max(maxY, y);
  }
  return {.origin = {.x = minX, .y = minY},
          .size = {.width = static_cast<uint64_t>(maxX - minX),
                   .height = static_cast<uint64_t>(maxY - minY)}};
}
auto FoMaC::Device::initNameFromDevice() -> void { name = getName(); }
auto FoMaC::Device::initMinAtomDistanceFromDevice() -> void {
  minAtomDistance = *getMinAtomDistance();
}
auto FoMaC::Device::initQubitsNumFromDevice() -> void {
  numQubits = getQubitsNum();
}
auto FoMaC::Device::initLengthUnitFromDevice() -> void {
  lengthUnit.unit = *qdmi::FoMaC::Device::getLengthUnit();
  lengthUnit.scaleFactor = getLengthScaleFactor().value_or(1.0);
}
auto FoMaC::Device::initDurationUnitFromDevice() -> void {
  durationUnit.unit = *qdmi::FoMaC::Device::getDurationUnit();
  durationUnit.scaleFactor = getDurationScaleFactor().value_or(1.0);
}
auto FoMaC::Device::initDecoherenceTimesFromDevice() -> void {
  // find first non-zone site as reference
  const auto& sites = getSites();
  auto referenceSite = *std::ranges::find_if(
      sites, [](const auto& site) { return !site.isZone().value_or(false); });
  decoherenceTimes.t1 = *referenceSite.getT1();
  decoherenceTimes.t2 = *referenceSite.getT2();
}
auto FoMaC::Device::initTrapsfromDevice() -> void {
  traps.clear();
  // get the non-zone sites of the device
  auto regularSites = getSites() | std::views::filter([](const auto& site) {
                        return !site.isZone().value_or(false);
                      });
  // group sites by their module index
  std::unordered_map<uint64_t, std::vector<Site>> modules;
  for (const auto& site : regularSites) {
    auto it = modules.try_emplace(*site.getModuleIndex()).first;
    it->second.emplace_back(site);
  }
  // iterate over modules
  traps.reserve(modules.size());
  for (const auto& [moduleIdx, moduleSites] : modules) {
    // get submodule sites (submodule 0)
    const auto submodule0Idx = *moduleSites[0].getSubmoduleIndex();
    auto submodule0Sites =
        moduleSites | std::views::filter([=](const auto& site) {
          return *site.getSubmoduleIndex() == submodule0Idx;
        });
    // get reference site (submodule 0) which becomes lattice origin
    const auto& referenceSite0 = *std::ranges::min_element(
        submodule0Sites, [](const auto& a, const auto& b) {
          return std::pair{a.getXCoordinate(), a.getYCoordinate()} <
                 std::pair{b.getXCoordinate(), b.getYCoordinate()};
        });
    const Vector latticeOrigin(*referenceSite0.getXCoordinate(),
                               *referenceSite0.getYCoordinate());
    // get sublattice offsets
    std::vector<Vector> sublatticeOffsets;
    for (const auto& site : submodule0Sites) {
      sublatticeOffsets.emplace_back(*site.getXCoordinate() - latticeOrigin.x,
                                     *site.getYCoordinate() - latticeOrigin.y);
    }
    // reference sites (other submodules)
    std::unordered_map<uint64_t, Site> referenceSites;
    for (const auto& site : moduleSites) {
      if (const auto idx = *site.getSubmoduleIndex(); idx != submodule0Idx) {
        auto [it, success] = referenceSites.try_emplace(idx, site);
        // if already present, keep the one with smaller coordinates
        if (!success) {
          if (std::pair{site.getXCoordinate(), site.getYCoordinate()} <
              std::pair{it->second.getXCoordinate(),
                        it->second.getYCoordinate()}) {
            it->second = site;
          }
        }
      }
    }
    // distance function
    const auto dist = [](const Vector& origin, const Site& site) {
      return std::hypot(*site.getXCoordinate() - origin.x,
                        *site.getYCoordinate() - origin.y);
    };
    // find first lattice vector
    const auto& referenceSite1 =
        std::ranges::min_element(referenceSites, [&](const auto& a,
                                                     const auto& b) {
          return dist(latticeOrigin, a.second) < dist(latticeOrigin, b.second);
        })->second;
    const Vector latticeVector1(
        *referenceSite1.getXCoordinate() - latticeOrigin.x,
        *referenceSite1.getYCoordinate() - latticeOrigin.y);
    // find second lattice vector (non-collinear)
    auto nonCollinearReferenceSites =
        referenceSites |
        std::views::filter([&latticeOrigin, &latticeVector1](const auto& pair) {
          return (*pair.second.getXCoordinate() - latticeOrigin.x) *
                     latticeVector1.y !=
                 (*pair.second.getYCoordinate() - latticeOrigin.y) *
                     latticeVector1.x;
        });
    const auto& referenceSite2 =
        std::ranges::min_element(nonCollinearReferenceSites,
                                 [&](const auto& a, const auto& b) {
                                   return dist(latticeOrigin, a.second) <
                                          dist(latticeOrigin, b.second);
                                 })
            ->second;
    const Vector latticeVector2(
        *referenceSite2.getXCoordinate() - latticeOrigin.x,
        *referenceSite2.getYCoordinate() - latticeOrigin.y);
    const auto& extent = calculateExtentFromSites(moduleSites);
    // ensure canonical order of lattice vectors
    if (latticeVector1 < latticeVector2) {
      traps.emplace_back(latticeOrigin, latticeVector1, latticeVector2,
                         sublatticeOffsets, extent);
    } else {
      traps.emplace_back(latticeOrigin, latticeVector2, latticeVector1,
                         sublatticeOffsets, extent);
    }
  }
}
auto FoMaC::Device::initGlobalSingleQubitOperationsFromDevice() -> void {
  std::ranges::copy(
      getOperations() |
          std::views::filter(
              [](const qdmi::FoMaC::Device::Operation& op) -> bool {
                return op.isZoned() == true && op.getQubitsNum() == 1;
              }) |
          std::views::transform([](const qdmi::FoMaC::Device::Operation& op)
                                    -> GlobalSingleQubitOperation {
            const auto site = op.getSites()->front();
            return GlobalSingleQubitOperation{
                op.getName(),
                {.origin = {.x = *site.getXCoordinate(),
                            .y = *site.getYCoordinate()},
                 .size = {.width = *site.getXExtent(),
                          .height = *site.getYExtent()}},
                *op.getDuration(),
                *op.getFidelity(),
                op.getParametersNum()};
          }),
      std::back_inserter(globalSingleQubitOperations));
}
auto FoMaC::Device::initGlobalMultiQubitOperationsFromDevice() -> void {
  std::ranges::copy(
      getOperations() |
          std::views::filter(
              [](const qdmi::FoMaC::Device::Operation& op) -> bool {
                return op.isZoned() == true && op.getQubitsNum() > 1;
              }) |
          std::views::transform([](const qdmi::FoMaC::Device::Operation& op)
                                    -> GlobalMultiQubitOperation {
            const auto site = op.getSites()->front();
            return GlobalMultiQubitOperation{
                {.name = op.getName(),
                 .region = {.origin = {.x = *site.getXCoordinate(),
                                       .y = *site.getYCoordinate()},
                            .size = Region::Size{.width = *site.getXExtent(),
                                                 .height = *site.getYExtent()}},
                 .duration = *op.getDuration(),
                 .fidelity = *op.getFidelity(),
                 .numParameters = op.getParametersNum()},
                *op.getInteractionRadius(),
                *op.getBlockingRadius(),
                *op.getIdlingFidelity(),
                *op.getQubitsNum()};
          }),
      std::back_inserter(globalMultiQubitOperations));
}
auto FoMaC::Device::initLocalSingleQubitOperationsFromDevice() -> void {
  std::ranges::copy(
      getOperations() |
          std::views::filter(
              [](const qdmi::FoMaC::Device::Operation& op) -> bool {
                return !op.isZoned().value_or(false) && op.getQubitsNum() == 1;
              }) |
          std::views::transform([](const qdmi::FoMaC::Device::Operation& op)
                                    -> LocalSingleQubitOperation {
            return LocalSingleQubitOperation{
                op.getName(), calculateExtentFromSites(*op.getSites()),
                *op.getDuration(), *op.getFidelity(), op.getParametersNum()};
          }),
      std::back_inserter(localSingleQubitOperations));
}
auto FoMaC::Device::initLocalMultiQubitOperationsFromDevice() -> void {
  std::ranges::copy(
      getOperations() |
          std::views::filter(
              [](const qdmi::FoMaC::Device::Operation& op) -> bool {
                return !op.isZoned().value_or(false) && op.getQubitsNum() > 1;
              }) |
          std::views::transform([](const qdmi::FoMaC::Device::Operation& op)
                                    -> LocalMultiQubitOperation {
            return LocalMultiQubitOperation{
                {.name = op.getName(),
                 .region = calculateExtentFromSites(*op.getSites()),
                 .duration = *op.getDuration(),
                 .fidelity = *op.getFidelity(),
                 .numParameters = op.getParametersNum()},
                *op.getInteractionRadius(),
                *op.getBlockingRadius(),
                *op.getQubitsNum()};
          }),
      std::back_inserter(localMultiQubitOperations));
}
auto FoMaC::Device::initShuttlingUnitsFromDevice() -> void {
  std::unordered_map<
      size_t, std::array<std::optional<qdmi::FoMaC::Device::Operation>, 3>>
      shuttlingOpTuples;
  std::ranges::for_each(
      getOperations() |
          std::views::filter(
              [](const qdmi::FoMaC::Device::Operation& op) -> bool {
                return op.isZoned() && !op.getQubitsNum().has_value();
              }),
      [&shuttlingOpTuples](const qdmi::FoMaC::Device::Operation& op) {
        // extract the int from, e.g., `load<0>`, `move<1>`, `store<2>`
        const auto name = op.getName();
        const auto start = name.find('<');
        const auto end = name.find('>');
        const auto id = static_cast<size_t>(
            std::stoi(name.substr(start + 1, end - start - 1)));
        const auto [it, success] = shuttlingOpTuples.try_emplace(id);
        if (name.starts_with("load")) {
          it->second[0] = op;
        } else if (name.starts_with("move")) {
          it->second[1] = op;
        } else { // if (name.starts_with("store"))
          it->second[2] = op;
        }
      });
  std::ranges::copy(
      shuttlingOpTuples |
          std::views::transform([](const auto& pair) -> ShuttlingUnit {
            const auto& [id, triple] = pair;
            const auto& load = *triple[0];
            const auto& move = *triple[1];
            const auto& store = *triple[2];
            const auto site = move.getSites()->front();
            return {.id = id,
                    .region = {.origin = {.x = *site.getXCoordinate(),
                                          .y = *site.getYCoordinate()},
                               .size = {.width = *site.getXExtent(),
                                        .height = *site.getYExtent()}},
                    .loadDuration = *load.getDuration(),
                    .storeDuration = *store.getDuration(),
                    .loadFidelity = *load.getFidelity(),
                    .storeFidelity = *store.getFidelity(),
                    .numParameters = move.getParametersNum(),
                    .meanShuttlingSpeed = *move.getMeanShuttlingSpeed()};
          }),
      std::back_inserter(shuttlingUnits));
}
auto FoMaC::getDevices() -> std::vector<Device> {
  const auto& qdmiDevices = qdmi::FoMaC::getDevices();
  std::vector<Device> devices;
  devices.reserve(qdmiDevices.size());
  std::ranges::transform(
      qdmiDevices, std::back_inserter(devices),
      [](const qdmi::FoMaC::Device& dev) -> Device { return Device(dev); });
  return devices;
}
} // namespace na
