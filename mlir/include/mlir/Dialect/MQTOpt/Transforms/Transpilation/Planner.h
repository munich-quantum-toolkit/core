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
#include "mlir/Dialect/MQTOpt/Transforms/Transpilation/Layerizer.h"
#include "mlir/Dialect/MQTOpt/Transforms/Transpilation/Layout.h"

#include <algorithm>
#include <mlir/Support/LLVM.h>
#include <queue>
#include <stdexcept>
#include <utility>
#include <vector>

namespace mqt::ir::opt {

/**
 * @brief A vector of SWAPs.
 */
using PlannerResult = mlir::SmallVector<QubitIndexPair>;

/**
 * @brief A planner determines the sequence of swaps required to route an array
of gates.
*/
struct PlannerBase {
  virtual ~PlannerBase() = default;
  [[nodiscard]] virtual PlannerResult plan(const Layers&,
                                           const ThinLayout<QubitIndex>&,
                                           const Architecture&) const = 0;
};

/**
 * @brief Use shortest path swapping to make one gate executable.
 */
struct NaivePlanner final : PlannerBase {
  [[nodiscard]] PlannerResult plan(const Layers& layers,
                                   const ThinLayout<QubitIndex>& layout,
                                   const Architecture& arch) const override {
    if (layers.size() != 1 && layers.front().size() != 1) {
      throw std::invalid_argument(
          "NaivePlanner expects exactly one layer with one gate");
    }

    mlir::SmallVector<QubitIndexPair, 16> swaps;
    for (const auto [prog0, prog1] : layers.front()) {
      const auto [hw0, hw1] = layout.getHardwareIndices(prog0, prog1);
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
  [[nodiscard]] PlannerResult plan(const Layers& layers,
                                   const ThinLayout<QubitIndex>& layout,
                                   const Architecture& arch) const override {

    /// Initialize queue.
    MinQueue frontier{};
    expand(frontier, SearchNode(layout, arch), layers, arch);

    /// Iterative searching and expanding.
    while (!frontier.empty()) {
      SearchNode curr = frontier.top();
      frontier.pop();

      if (curr.isGoal(layers.front(), arch)) {
        return curr.getSequence();
      }

      expand(frontier, curr, layers, arch);
    }

    return {};
  }

private:
  struct SearchNode {
    /**
     * @brief Construct a root node with the given layout. Initialize the
     * sequence with an empty vector and set the cost to zero.
     */
    explicit SearchNode(ThinLayout<QubitIndex> layout, const Architecture& arch)
        : layout_(std::move(layout)), depthBuckets_(arch.nqubits()) {}

    /**
     * @brief Construct a non-root node from its parent node. Apply the given
     * swap to the layout of the parent node and evaluate the cost.
     */
    SearchNode(const SearchNode& parent, QubitIndexPair swap,
               const Layers& layers, const Architecture& arch)
        : seq_(parent.seq_), layout_(parent.layout_),
          depthBuckets_(parent.depthBuckets_) {
      /// Apply node-specific swap to given layout.
      layout_.swap(layout_.getProgramIndex(swap.first),
                   layout_.getProgramIndex(swap.second));
      /// Add swap to sequence.
      seq_.push_back(swap);

      /// Update degrees.
      depthBuckets_[swap.first]++;
      depthBuckets_[swap.second]++;
      ndepth_ = std::max(
          {depthBuckets_[swap.first], depthBuckets_[swap.second], ndepth_});

      /// Evaluate cost function.
      evaluateCost(layers, arch);
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

    [[nodiscard]] bool operator>(const SearchNode& rhs) const {
      return f_ > rhs.f_;
    }

  private:
    void evaluateCost(const Layers& layers, const Architecture& arch) {
      constexpr float alpha = 1.;
      constexpr float beta = 1.;
      constexpr float lambda = 0.9;

      /// TODO: Hoist outside and make variables configurable.
      mlir::SmallVector<float, 2> lambdas(layers.size());
      lambdas[0] = 1.0;
      for (std::size_t i = 1; i < lambdas.size(); ++i) {
        lambdas[i] = lambdas[i - 1] * lambda;
      }

      /// The path cost function evaluates the weighted sum of the currently
      /// required SWAPs and additionally added depth.
      const auto g = [&] {
        return (alpha * static_cast<float>(seq_.size())) +
               (beta * static_cast<float>(ndepth_));
      };

      /// The heuristic cost function calculates the nearest neighbour costs.
      /// That is, the amount of SWAPs that a naive router would require.
      /// Gamma acts like a decay.
      const auto h = [&] {
        float nn{0};
        for (const auto [i, layer] : llvm::enumerate(layers)) {
          for (const auto [prog0, prog1] : layer) {
            const auto [hw0, hw1] = layout_.getHardwareIndices(prog0, prog1);
            const std::size_t nswaps =
                arch.lengthOfShortestPathBetween(hw0, hw1) - 2;
            nn += lambdas[i] * static_cast<float>(nswaps) /
                  static_cast<float>(layer.size());
          }
        }
        return nn;
      };

      f_ = g() + h();
    }

    mlir::SmallVector<QubitIndexPair> seq_;
    ThinLayout<QubitIndex> layout_;

    mlir::SmallVector<uint16_t> depthBuckets_;
    uint16_t ndepth_{0};

    float f_{0};
  };

  using MinQueue =
      std::priority_queue<SearchNode, std::vector<SearchNode>, std::greater<>>;

  static void expand(MinQueue& frontier, const SearchNode& node,
                     const Layers& layers, const Architecture& arch) {

    llvm::SmallDenseSet<QubitIndexPair, 16> visited{};
    for (const QubitIndexPair gate : layers.front()) {
      for (const QubitIndex prog : {gate.first, gate.second}) {
        const std::size_t hw0 = node.getLayout().getHardwareIndex(prog);
        for (const std::size_t hw1 : arch.neighboursOf(hw0)) {
          /// Ensure consistent hashing/comparison
          const QubitIndexPair swap =
              hw0 < hw1 ? QubitIndexPair{hw0, hw1} : QubitIndexPair{hw1, hw0};

          if (visited.contains(swap)) {
            continue;
          }

          frontier.emplace(node, swap, layers, arch);
          visited.insert(swap);
        }
      }
    }
  }
};

} // namespace mqt::ir::opt
