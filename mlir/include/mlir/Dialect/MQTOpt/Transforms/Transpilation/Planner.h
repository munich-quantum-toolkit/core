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
#include "mlir/Dialect/MQTOpt/Transforms/Transpilation/Scheduler.h"

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
  [[nodiscard]] virtual PlannerResult plan(const Layers&, const ThinLayout&,
                                           const Architecture&) const = 0;
};

/**
 * @brief Use shortest path swapping to make one gate executable.
 */
struct NaivePlanner final : PlannerBase {
  [[nodiscard]] PlannerResult plan(const Layers& layers,
                                   const ThinLayout& layout,
                                   const Architecture& arch) const override {
    if (layers.size() != 1 || layers.front().size() != 1) {
      throw std::invalid_argument(
          "NaivePlanner expects exactly one layer with one gate");
    }

    /// This assumes an avg. of 16 SWAPs per gate.
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
 * @brief Specifies the weights for different terms in the cost function f.
 */
struct HeuristicWeights {
  float alpha;
  float beta;
  mlir::SmallVector<float> lambdas;

  HeuristicWeights(const float alpha, const float beta, const float lambda,
                   const std::size_t nlookahead)
      : alpha(alpha), beta(beta), lambdas(1 + nlookahead) {
    lambdas[0] = 1.0;
    for (std::size_t i = 1; i < lambdas.size(); ++i) {
      lambdas[i] = lambdas[i - 1] * lambda;
    }
  }
};

/**
 * @brief Use A*-search to make all gates executable.
 */
struct QMAPPlanner final : PlannerBase {
  explicit QMAPPlanner(HeuristicWeights weights)
      : weights_(std::move(weights)) {}

  [[nodiscard]] PlannerResult plan(const Layers& layers,
                                   const ThinLayout& layout,
                                   const Architecture& arch) const override {
    /// Initialize queue.
    MinQueue frontier{};
    expand(frontier, SearchNode(layout, arch.nqubits()), layers, arch);

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
    SearchNode(ThinLayout layout, const std::size_t nqubits)
        : layout_(std::move(layout)), depthBuckets_(nqubits) {}

    /**
     * @brief Construct a non-root node from its parent node. Apply the given
     * swap to the layout of the parent node and evaluate the cost.
     */
    SearchNode(const SearchNode& parent, QubitIndexPair swap,
               const Layers& layers, const Architecture& arch,
               const HeuristicWeights& weights)
        : seq_(parent.seq_), layout_(parent.layout_),
          depthBuckets_(parent.depthBuckets_), ndepth_(parent.ndepth_) {
      /// Apply node-specific swap to given layout.
      layout_.swap(layout_.getProgramIndex(swap.first),
                   layout_.getProgramIndex(swap.second));
      /// Add swap to sequence.
      seq_.push_back(swap);

      /// Update degrees.
      const uint16_t start =
          std::max(depthBuckets_[swap.first], depthBuckets_[swap.second]);
      const uint16_t finish = start + 1;
      depthBuckets_[swap.first] = depthBuckets_[swap.second] = finish;
      ndepth_ = std::max(ndepth_, finish);

      /// Evaluate cost function.
      f_ = g(weights) + h(layers, arch, weights); // NOLINT
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
    [[nodiscard]] const ThinLayout& getLayout() const { return layout_; }

    [[nodiscard]] bool operator>(const SearchNode& rhs) const {
      return f_ > rhs.f_;
    }

  private:
    /**
     * @brief Calculate the path cost.
     *
     * The path cost function evaluates the weighted sum of the currently
     * required SWAPs and additionally added depth.
     */
    [[nodiscard]] float g(const HeuristicWeights& weights) {
      return (weights.alpha * static_cast<float>(seq_.size())) +
             (weights.beta * static_cast<float>(ndepth_));
    }

    /**
     * @brief Calculate heuristic cost.
     *
     * The heuristic cost function calculates the nearest neighbour costs.
     * That is, the amount of SWAPs that a naive router would require.
     * Gamma acts like a decay.
     *
     * TODO: Similarly to the LightSABRE algorithm. It should be possible to
     * calculate the heuristic in O(1) by only considering the change that the
     * inserted SWAP causes.
     */
    [[nodiscard]] float h(const Layers& layers, const Architecture& arch,
                          const HeuristicWeights& weights) {
      float nn{0};
      for (const auto [i, layer] : llvm::enumerate(layers)) {
        for (const auto [prog0, prog1] : layer) {
          const auto [hw0, hw1] = layout_.getHardwareIndices(prog0, prog1);
          const std::size_t dist = arch.distanceBetween(hw0, hw1);
          const std::size_t nswaps = dist < 2 ? 0 : dist - 2;
          nn += weights.lambdas[i] * static_cast<float>(nswaps);
        }
      }
      return nn;
    }

    mlir::SmallVector<QubitIndexPair> seq_;
    ThinLayout layout_;

    mlir::SmallVector<uint16_t> depthBuckets_;
    uint16_t ndepth_{0};

    float f_{0};
  };

  using MinQueue =
      std::priority_queue<SearchNode, std::vector<SearchNode>, std::greater<>>;

  /**
   * @brief Expand frontier with all possible neighbouring SWAPs in the current
   * front.
   */
  void expand(MinQueue& frontier, const SearchNode& node, const Layers& layers,
              const Architecture& arch) const {
    llvm::SmallDenseSet<QubitIndexPair, 16> visited{};
    for (const QubitIndexPair gate : layers.front()) {
      for (const auto prog : {gate.first, gate.second}) {
        const auto hw0 = node.getLayout().getHardwareIndex(prog);
        for (const auto hw1 : arch.neighboursOf(hw0)) {
          /// Ensure consistent hashing/comparison
          const QubitIndexPair swap =
              hw0 < hw1 ? QubitIndexPair{hw0, hw1} : QubitIndexPair{hw1, hw0};

          if (visited.contains(swap)) {
            continue;
          }

          frontier.emplace(node, swap, layers, arch, weights_);
          visited.insert(swap);
        }
      }
    }
  }

  HeuristicWeights weights_;
};
} // namespace mqt::ir::opt
