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

#include "mlir/Dialect/MQTOpt/Transforms/Transpilation/Architecture.h"
#include "mlir/Dialect/MQTOpt/Transforms/Transpilation/Common.h"
#include "mlir/Dialect/MQTOpt/Transforms/Transpilation/Layout.h"

#include <mlir/Support/LLVM.h>
#include <stdexcept>

namespace mqt::ir::opt {

using PlannerResult = mlir::SmallVector<QubitIndexPair>;

/**
 * @brief A planner determines the sequence of swaps required to route an array
of gates.
*/
struct PlannerBase {
  virtual ~PlannerBase() = default;
  [[nodiscard]] virtual PlannerResult
  plan(const mlir::ArrayRef<QubitIndexPair>& gates,
       const ThinLayout<QubitIndex>& layout, const Architecture& arch) = 0;
};

/**
 * @brief Implements shortest path swapping.
 */
struct NaivePlanner : PlannerBase {

  [[nodiscard]] mlir::SmallVector<QubitIndexPair>
  plan(const mlir::ArrayRef<QubitIndexPair>& gates,
       const ThinLayout<QubitIndex>& layout, const Architecture& arch) final {
    if (gates.size() != 1) {
      throw std::invalid_argument(
          "NaivePlanner expects exactly one gate as input.");
    }

    mlir::SmallVector<QubitIndexPair, 16> swaps;
    for (const auto [prog0, prog1] : gates) {
      const QubitIndex hw0 = layout.getHardwareIndex(prog0);
      const QubitIndex hw1 = layout.getHardwareIndex(prog1);
      const auto path = arch.shortestPathBetween(hw0, hw1);
      for (std::size_t i = 0; i < path.size() - 2; ++i) {
        swaps.emplace_back(path[i], path[i + 1]);
      }
    }
    return swaps;
  }
};

/**
 * @brief Uses A*-search to make all gates executable.
 */
struct QMAPPlanner : PlannerBase {
  [[nodiscard]] mlir::SmallVector<QubitIndexPair>
  plan(const mlir::ArrayRef<QubitIndexPair>& gates,
       const ThinLayout<QubitIndex>& layout, const Architecture& arch) final {
    /// The heuristic cost function counts the number of SWAPs that were
    /// required if we were to route the gate set naively.
    const auto heuristic = [&](const SearchNode& node) {
      double h{};
      for (const auto [prog0, prog1] : gates) {
        const QubitIndex hw0 = node.layout.getHardwareIndex(prog0);
        const QubitIndex hw1 = node.layout.getHardwareIndex(prog1);
        const std::size_t nswaps =
            arch.lengthOfShortestPathBetween(hw0, hw1) - 2;
        h += static_cast<double>(nswaps);
      }
      return h;
    };

    const auto isGoal = [&](const SearchNode& node) {
      return std::ranges::all_of(gates, [&](const QubitIndexPair gate) {
        const auto [prog0, prog1] = gate;
        return arch.areAdjacent(node.layout.getHardwareIndex(prog0),
                                node.layout.getHardwareIndex(prog1));
      });
    };

    /// Initialize queue.
    MinQueue queue{};
    for (const QubitIndexPair swap : collectSWAPs(layout, gates, arch)) {
      SearchNode node({}, swap, layout);
      node.cost = heuristic(node);

      queue.emplace(node);
    }

    /// Iterative searching and expanding.
    while (!queue.empty()) {
      SearchNode curr = queue.top();
      queue.pop();

      if (isGoal(curr)) {
        return curr.seq;
      }

      for (const QubitIndexPair swap : collectSWAPs(curr.layout, gates, arch)) {
        SearchNode node(curr.seq, swap, curr.layout);
        node.depth = curr.depth + 1;
        node.cost = static_cast<double>(node.depth) + heuristic(node);

        queue.emplace(node);
      }
    }

    return {};
  }

private:
  struct SearchNode {
    llvm::SmallVector<QubitIndexPair> seq;
    ThinLayout<QubitIndex> layout;

    double cost{};
    std::size_t depth{};

    SearchNode(llvm::SmallVector<QubitIndexPair> seq, QubitIndexPair swap,
               ThinLayout<QubitIndex> layout)
        : seq(std::move(seq)), layout(std::move(layout)) {
      /// Apply node-specific swap to given layout.
      this->layout.swap(this->layout.getProgramIndex(swap.first),
                        this->layout.getProgramIndex(swap.second));
      /// Add swap to sequence.
      this->seq.push_back(swap);
    }

    bool operator>(const SearchNode& rhs) const { return cost > rhs.cost; }
  };

  using MinQueue =
      std::priority_queue<SearchNode, std::vector<SearchNode>, std::greater<>>;

  static mlir::SmallVector<QubitIndexPair>
  collectSWAPs(const ThinLayout<QubitIndex>& layout,
               const mlir::ArrayRef<QubitIndexPair>& gates,
               const Architecture& arch) {
    llvm::SmallDenseSet<QubitIndexPair, 16> candidates{};

    const auto collect = [&](const QubitIndex p) {
      const std::size_t hw0 = layout.getHardwareIndex(p);
      for (const std::size_t hw1 : arch.neighboursOf(hw0)) {
        /// Ensure consistent hashing/comparison
        const QubitIndexPair swap =
            hw0 < hw1 ? QubitIndexPair{hw0, hw1} : QubitIndexPair{hw1, hw0};
        candidates.insert(swap);
      }
    };

    for (const auto [prog0, prog1] : gates) {
      collect(prog0);
      collect(prog1);
    }

    return {candidates.begin(), candidates.end()};
  }
};

} // namespace mqt::ir::opt
