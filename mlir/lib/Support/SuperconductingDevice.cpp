/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Support/SuperconductingDevice.h"

#include <llvm/ADT/STLExtras.h>
#include <mlir/Support/LLVM.h>

#include <cstddef>
#include <limits>
#include <stdexcept>

using namespace mlir;

size_t SuperconductingDevice::nqubits() const {
  return coupling_.getNumNodes();
}

bool SuperconductingDevice::areAdjacent(size_t u, size_t v) const {
  return dist_[u][v] == 1UL;
}

size_t SuperconductingDevice::distanceBetween(size_t u, size_t v) const {
  const auto dist = dist_[u][v];
  if (dist == std::numeric_limits<size_t>::max()) {
    report_fatal_error("Failed to compute the distance between qubits " +
                       Twine(u) + " and " + Twine(v));
  }
  return dist;
}

SmallVector<size_t> SuperconductingDevice::qubits() const {
  return coupling_.getNodes();
}

ArrayRef<size_t> SuperconductingDevice::neighboursOf(size_t u) const {
  return coupling_.getNeighbours(u);
}

size_t SuperconductingDevice::maxDegree() const {
  return coupling_.getMaxDegree();
}

Graph SuperconductingDevice::getCouplingGraph(
    const std::shared_ptr<fomac::Device>& device) {
  const auto sites = device->getSites();
  const auto siteCoupling = device->getCouplingMap();
  assert(siteCoupling.has_value() &&
         "expected QDMI device with a coupling map");

  // Construct index-type based qubit vector and coupling set from QDMI sites.
  // TODO: Does QDMI assume undirected edges?

  // As we can't guarantee that the site indices form a consecutive range, remap
  // the indices to [0, sites.size()).

  DenseMap<size_t, size_t> mapping;
  mapping.reserve(sites.size());
  for (const auto [i, site] : llvm::enumerate(sites)) {
    mapping.try_emplace(site.getIndex(), i);
  }

  DenseSet<std::pair<size_t, size_t>> coupling;
  coupling.reserve(siteCoupling->size());
  coupling.insert_range(llvm::map_range(*siteCoupling, [&](const auto& pair) {
    const auto& [s0, s1] = pair;
    return std::make_pair(mapping.at(s0.getIndex()), mapping.at(s1.getIndex()));
  }));

  return Graph(to_vector(mapping.values()), coupling);
}
