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

#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/Twine.h>
#include <llvm/Support/ErrorHandling.h>
#include <mlir/Support/LLVM.h>

#include <cstddef>
#include <string>
#include <utility>

namespace mlir {

/**
 * @brief A quantum accelerator's architecture.
 */
class [[nodiscard]] Architecture {
public:
  using CouplingSet = mlir::DenseSet<std::pair<std::size_t, std::size_t>>;
  using NeighbourVector = mlir::SmallVector<mlir::SmallVector<std::size_t, 4>>;

  explicit Architecture(std::string name, std::size_t nqubits,
                        CouplingSet couplingSet)
      : name_(std::move(name)), nqubits_(nqubits),
        couplingSet_(std::move(couplingSet)), neighbours_(nqubits),
        dist_(nqubits, mlir::SmallVector<std::size_t>(nqubits, UINT64_MAX)),
        prev_(nqubits, mlir::SmallVector<std::size_t>(nqubits, UINT64_MAX)) {
    floydWarshallWithPathReconstruction();
    collectNeighbours();
  }

  /**
   * @brief Return the architecture's name.
   */
  [[nodiscard]] std::string_view name() const;

  /**
   * @brief Return the architecture's number of qubits.
   */
  [[nodiscard]] std::size_t nqubits() const;

  /**
   * @brief Return true if @p u and @p v are adjacent.
   */
  [[nodiscard]] bool areAdjacent(std::size_t u, std::size_t v) const;

  /**
   * @brief Return the length of the shortest path between @p u and @p v.
   */
  [[nodiscard]] std::size_t distanceBetween(std::size_t u, std::size_t v) const;

  /**
   * @brief Collect all neighbours of @p u.
   */
  [[nodiscard]] mlir::SmallVector<std::size_t, 4>
  neighboursOf(std::size_t u) const;

  /**
   * @brief Return the maximum degree (connectivity) of any qubit in the
   * architecture.
   */
  [[nodiscard]] std::size_t maxDegree() const;

private:
  using Matrix = mlir::SmallVector<mlir::SmallVector<std::size_t, 0>, 0>;

  /**
   * @brief Find all shortest paths in the coupling map between two qubits.
   * @details Vertices are the qubits. Edges connected two qubits. Has a time
   * and memory complexity of O(nqubits^3) and O(nqubits^2), respectively.
   * @link Adapted from https://en.wikipedia.org/wiki/Floyd–Warshall_algorithm
   */
  void floydWarshallWithPathReconstruction();

  /**
   * @brief Collect the neighbours of all qubits.
   * @details Has a time complexity of O(nqubits)
   */
  void collectNeighbours();

  std::string name_;
  std::size_t nqubits_;
  CouplingSet couplingSet_;
  NeighbourVector neighbours_;

  Matrix dist_;
  Matrix prev_;
};

} // namespace mlir
