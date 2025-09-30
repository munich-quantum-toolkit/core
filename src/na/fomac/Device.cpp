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

#include "fomac/FoMaC.hpp"
#include "ir/Definitions.hpp"
#include "na/device/Generator.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <limits>
#include <map>
#include <optional>
#include <queue>
#include <ranges>
#include <regex>
#include <spdlog/spdlog.h>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace na {
namespace {
/**
 * @brief Calculate the rectangular extent covering all given FoMaC sites.
 * @param sites is a vector of FoMaC sites
 * @return the extent covering all given sites
 */
auto calculateExtentFromSites(
    const std::vector<fomac::FoMaC::Device::Site>& sites) -> Device::Region {
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
/**
 * @brief Device::Vector does not provide a hash function by default, this is
 * the replacement.
 * @param v is the vector to hash
 * @return the hash value
 */
struct DeviceVectorHash {
  size_t operator()(const Device::Vector& v) const {
    return qc::combineHash(std::hash<int64_t>{}(v.x),
                           std::hash<int64_t>{}(v.y));
  }
};
} // namespace
auto FoMaC::Device::initNameFromDevice() -> void { name = getName(); }
auto FoMaC::Device::initMinAtomDistanceFromDevice() -> bool {
  const auto& d = getMinAtomDistance();
  if (!d.has_value()) {
    SPDLOG_INFO("Minimal atom distance not set");
    return false;
  }
  minAtomDistance = *d;
  return true;
}
auto FoMaC::Device::initQubitsNumFromDevice() -> void {
  numQubits = getQubitsNum();
}
auto FoMaC::Device::initLengthUnitFromDevice() -> bool {
  const auto& u = fomac::FoMaC::Device::getLengthUnit();
  if (!u.has_value()) {
    SPDLOG_INFO("Length unit not set");
    return false;
  }
  lengthUnit.unit = *u;
  lengthUnit.scaleFactor = getLengthScaleFactor().value_or(1.0);
  return true;
}
auto FoMaC::Device::initDurationUnitFromDevice() -> bool {
  const auto& u = fomac::FoMaC::Device::getDurationUnit();
  if (!u.has_value()) {
    SPDLOG_INFO("Duration unit not set");
    return false;
  }
  durationUnit.unit = *u;
  durationUnit.scaleFactor = getDurationScaleFactor().value_or(1.0);
  return true;
}
auto FoMaC::Device::initDecoherenceTimesFromDevice() -> bool {
  // find first non-zone site as reference
  auto regularSites = getSites() | std::views::filter([](const auto& site) {
                        return !site.isZone().value_or(false);
                      });
  uint64_t sumT1 = 0;
  uint64_t sumT2 = 0;
  for (const auto& site : regularSites) {
    const auto& t1 = site.getT1();
    if (!t1.has_value()) {
      SPDLOG_INFO("Regular site missing t1");
      return false;
    }
    const auto& t2 = site.getT2();
    if (!t2.has_value()) {
      SPDLOG_INFO("Regular site missing t2");
      return false;
    }
    sumT1 += *t1;
    sumT2 += *t2;
  }
  const auto count = static_cast<uint64_t>(std::ranges::distance(regularSites));
  decoherenceTimes.t1 = sumT1 / count;
  decoherenceTimes.t2 = sumT2 / count;
  return true;
}
auto FoMaC::Device::initTrapsfromDevice() -> bool {
  traps.clear();
  const auto& sites = getSites();
  if (sites.empty()) {
    SPDLOG_INFO("Device returned no sites");
    return false;
  }
  auto regularSites = sites | std::views::filter([](const Site& site) -> bool {
                        return !site.isZone().value_or(false);
                      });
  if (std::ranges::distance(regularSites) == 0) {
    SPDLOG_INFO("Device has no regular sites");
    return false;
  }
  std::unordered_set<Vector, DeviceVectorHash> retrievedSites;
  std::unordered_map<
      uint64_t,
      std::map<uint64_t, std::priority_queue<Vector, std::vector<Vector>,
                                             std::greater<>>>>
      sitesPerModuleAndSubmodule;
  for (const auto& site : regularSites) {
    const auto& mIdx = site.getModuleIndex();
    if (!mIdx.has_value()) {
      SPDLOG_INFO("Site missing module index");
      return false;
    }
    const auto moduleIt = sitesPerModuleAndSubmodule.try_emplace(*mIdx).first;
    const auto& smIdx = site.getSubmoduleIndex();
    if (!smIdx.has_value()) {
      SPDLOG_INFO("Site missing submodule index");
      return false;
    }
    const auto submoduleIt = moduleIt->second.try_emplace(*smIdx).first;
    const auto& x = site.getXCoordinate();
    if (!x.has_value()) {
      SPDLOG_INFO("Site missing x coordinate");
      return false;
    }
    const auto& y = site.getYCoordinate();
    if (!y.has_value()) {
      SPDLOG_INFO("Site missing y coordinate");
      return false;
    }
    submoduleIt->second.emplace(Vector{*x, *y});
    retrievedSites.emplace(Vector{*x, *y});
  }
  for (const auto& [moduleIdx, sitesPerSubmodule] :
       sitesPerModuleAndSubmodule) {
    // get submodule sites (min. submodule)
    const auto& [minSubmoduleIdx, minSubmoduleSites] =
        *sitesPerSubmodule.cbegin();
    // reference site (min. submodule) which becomes lattice origin
    const auto& latticeOrigin = minSubmoduleSites.top();
    // get sublattice offsets
    std::vector<Vector> sublatticeOffsets;
    std::ranges::for_each(minSubmoduleSites.__get_container(),
                          [&](const auto& v) {
                            sublatticeOffsets.emplace_back(
                                v.x - latticeOrigin.x, v.y - latticeOrigin.y);
                          });
    // find first lattice vector
    auto otherReferenceSites =
        sitesPerSubmodule | std::views::drop(1) | std::views::values |
        std::views::transform([&latticeOrigin](const auto& s) {
          const auto& v = s.top();
          return Vector{v.x - latticeOrigin.x, v.y - latticeOrigin.y};
        });
    const auto& latticeVector1 = *std::ranges::min_element(
        otherReferenceSites, [&](const auto& a, const auto& b) {
          return std::hypot(a.x, a.y) < std::hypot(b.x, b.y);
        });
    // find second lattice vector (non-collinear)
    auto nonCollinearReferenceSites =
        otherReferenceSites |
        std::views::filter([&latticeOrigin, &latticeVector1](const auto& v) {
          return (v.x - latticeOrigin.x) * latticeVector1.y !=
                 (v.y - latticeOrigin.y) * latticeVector1.x;
        });
    const auto& latticeVector2 = *std::ranges::min_element(
        nonCollinearReferenceSites, [&](const auto& a, const auto& b) {
          return std::hypot(a.x, a.y) < std::hypot(b.x, b.y);
        });
    auto minX = std::numeric_limits<std::int64_t>::max();
    auto maxX = std::numeric_limits<std::int64_t>::min();
    auto minY = std::numeric_limits<std::int64_t>::max();
    auto maxY = std::numeric_limits<std::int64_t>::min();
    std::ranges::for_each(
        sitesPerSubmodule | std::views::values, [&](const auto& s) {
          std::ranges::for_each(s.__get_container(), [&](const auto& v) {
            minX = std::min(minX, v.x);
            maxX = std::max(maxX, v.x);
            minY = std::min(minY, v.y);
            maxY = std::max(maxY, v.y);
          });
        });
    const Region extent{.origin = {.x = minX, .y = minY},
                        .size = {.width = static_cast<uint64_t>(maxX - minX),
                                 .height = static_cast<uint64_t>(maxY - minY)}};
    // ensure canonical order of lattice vectors
    if (latticeVector1 < latticeVector2) {
      traps.emplace_back(Lattice{.latticeOrigin = latticeOrigin,
                                 .latticeVector1 = latticeVector1,
                                 .latticeVector2 = latticeVector2,
                                 .sublatticeOffsets = sublatticeOffsets,
                                 .extent = extent});
    } else {
      traps.emplace_back(Lattice{.latticeOrigin = latticeOrigin,
                                 .latticeVector1 = latticeVector2,
                                 .latticeVector2 = latticeVector1,
                                 .sublatticeOffsets = sublatticeOffsets,
                                 .extent = extent});
    }
  }
  std::unordered_set<Vector, DeviceVectorHash> constructedSites;
  forEachRegularSites(
      traps, [&constructedSites](const uint64_t /* unused */, const int64_t x,
                                 const int64_t y, const uint64_t /* unused */,
                                 const uint64_t /* unused */) {
        constructedSites.emplace(x, y);
      });
  return retrievedSites == constructedSites;
}
auto FoMaC::Device::initOperationsFromDevice() -> bool {
  std::map<size_t, std::pair<ShuttlingUnit, std::array<bool, 3>>>
      shuttlingUnitsPerId;
  for (const fomac::FoMaC::Device::Operation& op : getOperations()) {
    const auto zoned = op.isZoned().value_or(false);
    const auto& nq = op.getQubitsNum();
    const auto& name = op.getName();
    const auto& sitesOpt = op.getSites();
    if (!sitesOpt.has_value() || sitesOpt->empty()) {
      SPDLOG_INFO("Operation missing sites");
      return false;
    }
    if (zoned) {
      if (std::ranges::any_of(
              *sitesOpt, [](const fomac::FoMaC::Device::Site& site) -> bool {
                return !site.isZone().value_or(false);
              })) {
        SPDLOG_INFO("Operation marked as zoned but has non-zone sites");
        return false;
      }
      if (sitesOpt->size() > 1) {
        SPDLOG_INFO("Shuttling operation must have one site");
        return false;
      }
      const auto& x = sitesOpt->front().getXCoordinate();
      if (!x.has_value()) {
        SPDLOG_INFO("Site missing x coordinate");
        return false;
      }
      const auto& y = sitesOpt->front().getYCoordinate();
      if (!y.has_value()) {
        SPDLOG_INFO("Site missing y coordinate");
        return false;
      }
      const auto& width = sitesOpt->front().getXExtent();
      if (!width.has_value()) {
        SPDLOG_INFO("Site missing x extent");
        return false;
      }
      const auto& height = sitesOpt->front().getYExtent();
      if (!height.has_value()) {
        SPDLOG_INFO("Site missing y extent");
        return false;
      }
      const Region region{.origin = {.x = *x, .y = *y},
                          .size = {.width = static_cast<uint64_t>(*width),
                                   .height = static_cast<uint64_t>(*height)}};
      if (!nq.has_value()) {
        // shuttling operations
        std::smatch match;
        if (std::regex_match(name, match, std::regex("load<(\\d+)>"))) {
          const auto id = std::stoul(match[1]);
          const auto& d = op.getDuration();
          if (!d.has_value()) {
            SPDLOG_INFO("Load Operation missing duration");
            return false;
          }
          const auto& f = op.getFidelity();
          if (!f.has_value()) {
            SPDLOG_INFO("Load Operation missing fidelity");
            return false;
          }
          const auto& [it, success] = shuttlingUnitsPerId.try_emplace(id);
          auto& [unit, triple] = it->second;
          auto& load = std::get<0>(triple);
          if (load) {
            SPDLOG_INFO("Duplicate load operation for shuttling unit");
            return false;
          }
          load = true;
          if (success) {
            unit.id = id;
            unit.numParameters = op.getParametersNum();
            unit.region = region;
          } else {
            if (unit.numParameters != op.getParametersNum()) {
              SPDLOG_INFO(
                  "Inconsistent number of parameters for shuttling unit");
              return false;
            }
            if (unit.region != region) {
              SPDLOG_INFO("Inconsistent region for shuttling unit");
              return false;
            }
          }
          unit.loadDuration = *d;
          unit.loadFidelity = *f;
        } else if (std::regex_match(name, match, std::regex("move<(\\d+)>"))) {
          const auto id = std::stoul(match[1]);
          const auto& speed = op.getMeanShuttlingSpeed();
          if (!speed.has_value()) {
            SPDLOG_INFO("Move Operation missing mean shuttling speed");
            return false;
          }
          const auto& [it, success] = shuttlingUnitsPerId.try_emplace(id);
          auto& [unit, triple] = it->second;
          auto& move = std::get<1>(triple);
          if (move) {
            SPDLOG_INFO("Duplicate move operation for shuttling unit");
            return false;
          }
          move = true;
          if (success) {
            unit.id = id;
            unit.numParameters = op.getParametersNum();
            unit.region = region;
          } else {
            if (unit.numParameters != op.getParametersNum()) {
              SPDLOG_INFO(
                  "Inconsistent number of parameters for shuttling unit");
              return false;
            }
            if (unit.region != region) {
              SPDLOG_INFO("Inconsistent region for shuttling unit");
              return false;
            }
          }
          unit.meanShuttlingSpeed = *speed;
        } else if (std::regex_match(name, match, std::regex("store<(\\d+)>"))) {
          const auto id = std::stoul(match[1]);
          const auto& d = op.getDuration();
          if (!d.has_value()) {
            SPDLOG_INFO("Store Operation missing duration");
            return false;
          }
          const auto& f = op.getFidelity();
          if (!f.has_value()) {
            SPDLOG_INFO("Store Operation missing fidelity");
            return false;
          }
          const auto& [it, success] = shuttlingUnitsPerId.try_emplace(id);
          auto& [unit, triple] = it->second;
          auto& store = std::get<2>(triple);
          if (store) {
            SPDLOG_INFO("Duplicate store operation for shuttling unit");
            return false;
          }
          store = true;
          if (success) {
            unit.id = id;
            unit.numParameters = op.getParametersNum();
            unit.region = region;
          } else {
            if (unit.numParameters != op.getParametersNum()) {
              SPDLOG_INFO(
                  "Inconsistent number of parameters for shuttling unit");
              return false;
            }
            if (unit.region != region) {
              SPDLOG_INFO("Inconsistent region for shuttling unit");
              return false;
            }
          }
          unit.storeDuration = *d;
          unit.storeFidelity = *f;
        } else {
          SPDLOG_INFO("Invalid shuttling operation name");
          return false;
        }
      } else {
        const auto& d = op.getDuration();
        if (!d.has_value()) {
          SPDLOG_INFO("Store Operation missing duration");
          return false;
        }
        const auto& f = op.getFidelity();
        if (!f.has_value()) {
          SPDLOG_INFO("Store Operation missing fidelity");
          return false;
        }
        if (*nq == 1) {
          // zoned single-qubit operations
          globalSingleQubitOperations.emplace_back(GlobalMultiQubitOperation{
              {.name = name,
               .region = region,
               .duration = *d,
               .fidelity = *f,
               .numParameters = op.getParametersNum()}});
        } else if (*nq == 2) {
          // zoned two-qubit operations
          const auto& ir = op.getInteractionRadius();
          if (!ir.has_value()) {
            SPDLOG_INFO("Two-qubit Operation missing interaction radius");
            return false;
          }
          const auto& br = op.getBlockingRadius();
          if (!br.has_value()) {
            SPDLOG_INFO("Two-qubit Operation missing blocking radius");
            return false;
          }
          const auto& fi = op.getIdlingFidelity();
          if (!fi.has_value()) {
            SPDLOG_INFO("Two-qubit Operation missing idling fidelity");
            return false;
          }
          globalMultiQubitOperations.emplace_back(GlobalMultiQubitOperation{
              {.name = name,
               .region = region,
               .duration = *d,
               .fidelity = *f,
               .numParameters = op.getParametersNum()},
              *ir,
              *br,
              *fi,
              *nq});
        } else {
          SPDLOG_INFO("Number of Qubits must be 1 or 2");
        }
      }
    } else {
      if (!nq.has_value()) {
        SPDLOG_INFO("Operation is missing number of qubits");
        return false;
      }
      const auto& d = op.getDuration();
      if (!d.has_value()) {
        SPDLOG_INFO("Store Operation missing duration");
        return false;
      }
      const auto& f = op.getFidelity();
      if (!f.has_value()) {
        SPDLOG_INFO("Store Operation missing fidelity");
        return false;
      }
      const auto region = calculateExtentFromSites(*sitesOpt);
      if (*nq == 1) {
        localSingleQubitOperations.emplace_back(LocalSingleQubitOperation{
            {name, region, *d, *f, op.getParametersNum()}});
      } else if (*nq == 2) {
        const auto& ir = op.getInteractionRadius();
        if (!ir.has_value()) {
          SPDLOG_INFO("Two-qubit Operation missing interaction radius");
          return false;
        }
        const auto& br = op.getBlockingRadius();
        if (!br.has_value()) {
          SPDLOG_INFO("Two-qubit Operation missing blocking radius");
          return false;
        }
        localMultiQubitOperations.emplace_back(LocalMultiQubitOperation{
            {name, region, *d, *f, op.getParametersNum()}, *ir, *br, *nq});
      } else {
        SPDLOG_INFO("Number of Qubits must be 1 or 2");
      }
    }
  }
  for (const auto& val : shuttlingUnitsPerId | std::views::values) {
    const auto [load, move, store] = val.second;
    if (!(load && move && store)) {
      SPDLOG_INFO("Shuttling unit not complete");
      return false;
    }
    shuttlingUnits.emplace_back(val.first);
  }
  return true;
}
FoMaC::Device::Device(const fomac::FoMaC::Device& device)
    : fomac::FoMaC::Device(device) {}
auto FoMaC::getDevices() -> std::vector<Device> {
  std::vector<Device> devices;
  std::ranges::copy(
      fomac::FoMaC::getDevices() | std::views::transform([](const auto& d) {
        return Device::tryCreateFromDevice(d);
      }) | std::views::filter([](const auto& r) { return r.has_value(); }) |
          std::views::transform([](const auto& r) { return r.value(); }),
      std::back_inserter(devices));
  return devices;
}
} // namespace na
