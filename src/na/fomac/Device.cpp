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

namespace na {

FoMaC::Device::Device(const qdmi::FoMaC::Device& device)
    : qdmi::FoMaC::Device(device) {
  initLengthUnitFromDevice();
  initDurationUnitFromDevice();
  initDecoherenceTimesFromDevice();
  initLatticesfromDevice();
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
auto FoMaC::Device::initLatticesfromDevice() -> void {
  traps.clear();
  // get the non-zone sites of the device
  auto regularSites = getSites() | std::views::filter([](const auto& site) {
                        return !site.isZone().value_or(false);
                      });
  // group sites by their module index
  std::unordered_map<uint64_t, std::vector<const Site*>> modules;
  for (const auto& site : regularSites) {
    auto it = modules.try_emplace(*site.getModuleIndex()).first;
    it->second.emplace_back(&site);
  }
  // iterate over modules
  traps.reserve(modules.size());
  for (const auto& [moduleIdx, moduleSites] : modules) {
    // get submodule sites (submodule 0)
    const auto submodule0Idx = *moduleSites[0]->getSubmoduleIndex();
    auto submodule0Sites =
        moduleSites | std::views::filter([=](const auto* site) {
          return *site->getSubmoduleIndex() == submodule0Idx;
        });
    // get reference site (submodule 0) which becomes lattice origin
    const auto* referenceSite0 = *std::ranges::min_element(
        submodule0Sites, [](const auto* a, const auto* b) {
          return std::pair{a->getXCoordinate(), a->getYCoordinate()} <
                 std::pair{b->getXCoordinate(), b->getYCoordinate()};
        });
    const Vector latticeOrigin(*referenceSite0->getXCoordinate(),
                               *referenceSite0->getYCoordinate());
    // get sublattice offsets
    std::vector<Vector> sublatticeOffsets;
    for (const auto* site : submodule0Sites) {
      sublatticeOffsets.emplace_back(*site->getXCoordinate() - latticeOrigin.x,
                                     *site->getYCoordinate() - latticeOrigin.y);
    }
    // reference sites (other submodules)
    std::unordered_map<uint64_t, const Site*> referenceSites;
    for (const auto* site : moduleSites) {
      if (const auto idx = *site->getSubmoduleIndex(); idx != submodule0Idx) {
        auto [it, success] = referenceSites.try_emplace(idx, site);
        // if already present, keep the one with smaller coordinates
        if (!success) {
          if (std::pair{site->getXCoordinate(), site->getYCoordinate()} <
              std::pair{it->second->getXCoordinate(),
                        it->second->getYCoordinate()}) {
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
    const auto referenceSite1 =
        *std::ranges::min_element(referenceSites, [&](const auto& a,
                                                      const auto& b) {
           return dist(latticeOrigin, *a.second) <
                  dist(latticeOrigin, *b.second);
         })->second;
    const Vector latticeVector1(
        *referenceSite1.getXCoordinate() - latticeOrigin.x,
        *referenceSite1.getYCoordinate() - latticeOrigin.y);
    // find second lattice vector (non-collinear)
    auto nonCollinearReferenceSites =
        referenceSites |
        std::views::filter([&latticeOrigin, &latticeVector1](const auto& pair) {
          return (*pair.second->getXCoordinate() - latticeOrigin.x) *
                     latticeVector1.y !=
                 (*pair.second->getYCoordinate() - latticeOrigin.y) *
                     latticeVector1.x;
        });
    auto referenceSite2 =
        *std::ranges::min_element(nonCollinearReferenceSites,
                                  [&](const auto& a, const auto& b) {
                                    return dist(latticeOrigin, *a.second) <
                                           dist(latticeOrigin, *b.second);
                                  })
             ->second;
    const Vector latticeVector2(
        *referenceSite2.getXCoordinate() - latticeOrigin.x,
        *referenceSite2.getYCoordinate() - latticeOrigin.y);
    // find the extent
    int64_t minX = std::numeric_limits<int64_t>::max();
    int64_t maxX = std::numeric_limits<int64_t>::min();
    int64_t minY = std::numeric_limits<int64_t>::max();
    int64_t maxY = std::numeric_limits<int64_t>::min();
    for (const auto* site : moduleSites) {
      const auto x = *site->getXCoordinate();
      const auto y = *site->getXCoordinate();
      minX = std::min(minX, x);
      maxX = std::max(maxX, x);
      minY = std::min(minY, y);
      maxY = std::max(maxY, y);
    }
    const Region extent(Vector(minX, minY),
                        Region::Size(static_cast<uint64_t>(maxX - minX + 1),
                                     static_cast<uint64_t>(maxY - minY + 1)));
    traps.emplace_back(latticeOrigin, latticeVector1, latticeVector2,
                       sublatticeOffsets, extent);
  }
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
