/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

#include "fomac/FoMaC.hpp"
#include "mlir/Support/Graph.h"

#include <mlir/Support/LLVM.h>

#include <cstddef>
#include <memory>

namespace mlir {
class SuperconductingDevice {
public:
  /// Construct a superconducting device from qubits and a coupling map.
  /// A coupling is a directed edge from qubit u → v. Thus, for undirected
  /// architectures, for each (u, v) the coupling also contains (v, u).
  SuperconductingDevice(ArrayRef<size_t> qubits,
                        const DenseSet<std::pair<size_t, size_t>>& coupling)
      : coupling_(qubits, coupling), dist_(coupling_.getDistMatrix()),
        device_(nullptr) {}

  /// Construct a superconducting device from a QDMI device.
  explicit SuperconductingDevice(std::shared_ptr<fomac::Device> device)
      : coupling_(getCouplingGraph(device)), dist_(coupling_.getDistMatrix()),
        device_(std::move(device)) {}

  /// Return the device's number of qubits.
  [[nodiscard]] size_t nqubits() const;

  /// Return true if two qubits are adjacent.
  [[nodiscard]] bool areAdjacent(size_t u, size_t v) const;

  /// Return the length of the shortest path between two qubits.
  [[nodiscard]] size_t distanceBetween(size_t u, size_t v) const;

  /// Return the qubit identifiers.
  [[nodiscard]] SmallVector<size_t> qubits() const;

  /// Return all neighbours of a qubit.
  [[nodiscard]] ArrayRef<size_t> neighboursOf(size_t u) const;

  /// Return the max degree (connectivity) of any qubit of the device.
  [[nodiscard]] size_t maxDegree() const;

private:
  /// Construct graph object from QDMI device.
  static Graph getCouplingGraph(const std::shared_ptr<fomac::Device>& device);

  Graph coupling_;
  Graph::DistanceMatrix dist_;
  std::shared_ptr<fomac::Device> device_;
};
} // namespace mlir
