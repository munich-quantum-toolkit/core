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
  if (dist == UINT64_MAX) {
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
  const auto siteCoupling = device->getCouplingMap();
  if (!siteCoupling) {
    throw std::invalid_argument("Given QDMI device has no coupling map!");
  }

  // Construct index-type based qubit vector and coupling set from QDMI sites.
  // TODO: Does QDMI assume undirected edges?

  DenseSet<std::pair<size_t, size_t>> coupling;
  coupling.reserve(siteCoupling->size());
  coupling.insert_range(llvm::map_range(*siteCoupling, [](const auto& pair) {
    const auto& [s0, s1] = pair;
    return std::make_pair(s0.getIndex(), s1.getIndex());
  }));

  SmallVector<size_t> qubits(
      llvm::map_range(device->getSites(),
                      [](const fomac::Site& site) { return site.getIndex(); }));

  return Graph(qubits, coupling);
}
