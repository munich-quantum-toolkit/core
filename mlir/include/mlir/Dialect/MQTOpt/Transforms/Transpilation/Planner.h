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
#include <queue>
#include <stdexcept>
#include <utility>

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
       const ThinLayout<QubitIndex>& layout,
       const Architecture& arch) const = 0;
};

/**
 * @brief Use shortest path swapping to make one gate executable.
 */
struct NaivePlanner final : PlannerBase {
  [[nodiscard]] PlannerResult plan(const mlir::ArrayRef<QubitIndexPair>& gates,
                                   const ThinLayout<QubitIndex>& layout,
                                   const Architecture& arch) const override {
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
 * @brief Use A*-search to make all gates executable.
 */
struct QMAPPlanner final : PlannerBase {
  [[nodiscard]] PlannerResult plan(const mlir::ArrayRef<QubitIndexPair>& gates,
                                   const ThinLayout<QubitIndex>& layout,
                                   const Architecture& arch) const override {
    /// Initialize queue.
    MinQueue frontier{};
    expand(frontier, SearchNode(layout), gates, arch);

    /// Iterative searching and expanding.
    while (!frontier.empty()) {
      SearchNode curr = frontier.top();
      frontier.pop();

      if (curr.isGoal(gates, arch)) {
        return curr.getSequence();
      }

      expand(frontier, curr, gates, arch);
    }

    return {};
  }

private:
  struct SearchNode {
    /**
     * @brief Construct a root node with the given layout. Initialize the
     * sequence with an empty vector and set the cost and depth to zero.
     */
    explicit SearchNode(ThinLayout<QubitIndex> layout)
        : layout_(std::move(layout)) {}

    /**
     * @brief Construct a non-root node from another node. Apply the given
     * swap to the layout of the parent node and reevaluate the cost.
     */
    SearchNode(SearchNode node, QubitIndexPair swap,
               const mlir::ArrayRef<QubitIndexPair>& gates,
               const Architecture& arch)
        : seq_(std::move(node.seq_)), layout_(std::move(node.layout_)),
          depth_(node.depth_ + 1) {
      /// Apply node-specific swap to given layout.
      layout_.swap(layout_.getProgramIndex(swap.first),
                   layout_.getProgramIndex(swap.second));
      /// Add swap to sequence.
      seq_.push_back(swap);

      /// Evaluate cost.
      evaluateCost(gates, arch);
    }

    /**
     * @brief Return true if the current sequence of SWAPs makes all gates
     * executable.
     */
    [[nodiscard]] bool isGoal(const mlir::ArrayRef<QubitIndexPair>& gates,
                              const Architecture& arch) const {
      return std::ranges::all_of(gates, [&](const QubitIndexPair gate) {
        return arch.areAdjacent(layout_.getHardwareIndex(gate.first),
                                layout_.getHardwareIndex(gate.second));
      });
    }

    /**
     * @brief Return the sequence of SWAPs.
     */
    [[nodiscard]] mlir::SmallVector<QubitIndexPair> getSequence() const {
      return seq_;
    }

    /**
     * @brief Return a const reference to the node's layout.
     */
    [[nodiscard]] const ThinLayout<QubitIndex>& getLayout() const {
      return layout_;
    }

    bool operator>(const SearchNode& rhs) const { return f_ > rhs.f_; }

  private:
    void evaluateCost(const mlir::ArrayRef<QubitIndexPair>& gates,
                      const Architecture& arch) {

      /// The path cost function counts the currently required SWAPs to reach
      /// this node.
      const auto g = [&] { return static_cast<double>(depth_); };

      /// The heuristic cost function calculates the nearest neighbour costs.
      /// That is, the amount of SWAPs that a naive router would require.
      const auto h = [&] {
        double h{};
        for (const auto [prog0, prog1] : gates) {
          const QubitIndex hw0 = layout_.getHardwareIndex(prog0);
          const QubitIndex hw1 = layout_.getHardwareIndex(prog1);
          const std::size_t nswaps =
              arch.lengthOfShortestPathBetween(hw0, hw1) - 2;
          h += static_cast<double>(nswaps);
        }
        return h;
      };

      f_ = g() + h();
    }

    mlir::SmallVector<QubitIndexPair> seq_;
    ThinLayout<QubitIndex> layout_;

    double f_{};
    std::size_t depth_{};
  };

  using MinQueue =
      std::priority_queue<SearchNode, mlir::SmallVector<SearchNode>,
                          std::greater<>>;

  static void expand(MinQueue& frontier, const SearchNode& node,
                     const mlir::ArrayRef<QubitIndexPair>& gates,
                     const Architecture& arch) {

    llvm::SmallDenseSet<QubitIndexPair, 16> visited{};
    for (const QubitIndexPair gate : gates) {
      for (const QubitIndex prog : {gate.first, gate.second}) {
        const std::size_t hw0 = node.getLayout().getHardwareIndex(prog);
        for (const std::size_t hw1 : arch.neighboursOf(hw0)) {
          /// Ensure consistent hashing/comparison
          const QubitIndexPair swap =
              hw0 < hw1 ? QubitIndexPair{hw0, hw1} : QubitIndexPair{hw1, hw0};

          if (visited.contains(swap)) {
            continue;
          }

          frontier.emplace(node, swap, gates, arch);
          visited.insert(swap);
        }
      }
    }
  }
};

} // namespace mqt::ir::opt
