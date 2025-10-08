/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

#include "ir/Definitions.hpp"

#include <cstddef>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/SmallVector.h>
#include <memory>
#include <string>
#include <string_view>

namespace mqt::ir::opt {

/**
 * @brief Enumerates the available target architectures.
 */
enum class ArchitectureName : std::uint8_t {
  MQTTest,
};

/**
 * @brief A quantum accelerator's architecture.
 * @details Computes all-shortest paths at construction.
 */
class Architecture {
public:
  using CouplingMap = llvm::DenseSet<std::pair<qc::Qubit, qc::Qubit>>;

  explicit Architecture(std::string name, std::size_t nqubits,
                        CouplingMap couplingMap)
      : name_(std::move(name)), nqubits_(nqubits),
        couplingMap_(std::move(couplingMap)),
        dist_(nqubits, llvm::SmallVector<std::size_t>(nqubits, UINT64_MAX)),
        prev_(nqubits, llvm::SmallVector<std::size_t>(nqubits, UINT64_MAX)) {
    floydWarshallWithPathReconstruction();
  }

  /**
   * @brief Return the architecture's name.
   */
  [[nodiscard]] constexpr std::string_view name() const { return name_; }

  /**
   * @brief Return the architecture's number of qubits.
   */
  [[nodiscard]] constexpr std::size_t nqubits() const { return nqubits_; }

  /**
   * @brief Return true if @p u and @p v are adjacent.
   */
  [[nodiscard]] bool areAdjacent(qc::Qubit u, qc::Qubit v) const {
    return couplingMap_.contains({u, v});
  }

  /**
   * @brief Collect the shortest path between @p u and @p v.
   */
  [[nodiscard]] llvm::SmallVector<std::size_t>
  shortestPathBetween(qc::Qubit u, qc::Qubit v) const;

  /**
   * @brief Return the length of the shortest path between @p u and @p v.
   */
  [[nodiscard]] std::size_t distanceBetween(qc::Qubit u, qc::Qubit v) const;

  /**
   * @brief Collect all neighbours of @p u.
   */
  [[nodiscard]] llvm::SmallVector<qc::Qubit> neighboursOf(qc::Qubit u) const;

private:
  using Matrix = llvm::SmallVector<llvm::SmallVector<std::size_t>>;

  /**
   * @brief Find all shortest paths in the coupling map between two qubits.
   * @details Vertices are the qubits. Edges connected two qubits. Has a time
   * and memory complexity of O(nqubits^3) and O(nqubits^2), respectively.
   * @link Adapted from https://en.wikipedia.org/wiki/Floyd–Warshall_algorithm
   */
  void floydWarshallWithPathReconstruction();

  std::string name_;
  std::size_t nqubits_;
  CouplingMap couplingMap_;

  Matrix dist_;
  Matrix prev_;
};

/**
 * @brief Get architecture by its name.
 */
std::unique_ptr<Architecture> getArchitecture(const ArchitectureName& name);

}; // namespace mqt::ir::opt
